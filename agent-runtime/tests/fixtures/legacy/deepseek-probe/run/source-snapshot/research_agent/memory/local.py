"""Task-related lexical retrieval baseline, separate from sessions and execution state."""

import json
import re
from dataclasses import asdict, dataclass

from research_agent.core.types import dumps
from research_agent.storage.sqlite import now


def terms(text):
    text = text.casefold()
    words = set(re.findall(r"[a-z0-9_]+", text))
    for sequence in re.findall(r"[\u3400-\u9fff]+", text):
        words.update(sequence[i : i + 2] for i in range(max(1, len(sequence) - 1)))
    return words


@dataclass(frozen=True)
class Candidate:
    key: str
    content: str
    sources: list[str]
    kind: str = "fact"  # fact, preference, event
    action: str = "upsert"


class Memory:
    def __init__(self, store):
        self.store = store

    def commit(self, user, candidates: list[Candidate]):
        """Validate a proposed batch before atomically committing it; no implicit per-turn write."""
        self._validate(user, candidates)
        with self.store.transaction() as db:
            self._apply(db, user, candidates)

    def _validate(self, user, candidates):
        seen = set()
        for candidate in candidates:
            if not candidate.key.strip() or len(candidate.key) > 120 or candidate.key in seen:
                raise ValueError("Memory keys must be unique, nonempty and bounded")
            seen.add(candidate.key)
            if candidate.kind not in {"fact", "preference", "event"} or candidate.action not in {
                "upsert",
                "delete",
            }:
                raise ValueError("Invalid memory kind or action")
            if not candidate.sources or len(candidate.sources) > 20:
                raise ValueError("Memory requires bounded source references")
            sources = [self.store.source(user, mid) for mid in candidate.sources]
            if any(source is None for source in sources):
                raise ValueError("Memory source is missing or belongs to another user")
            if candidate.kind != "event" and any(s["message"]["role"] != "user" for s in sources):
                raise ValueError("Stable facts/preferences require user sources")
            if candidate.action == "upsert" and not 0 < len(candidate.content.strip()) <= 2000:
                raise ValueError("Memory content must be nonempty and bounded")

    def _apply(self, db, user, candidates):
        for candidate in candidates:
            body = {
                **asdict(candidate),
                "updated_at": now(),
                "semantic_status": "candidate_unverified",
            }
            if candidate.action == "delete":
                db.execute("DELETE FROM memories WHERE user=? AND key=?", (user, candidate.key))
                # Forget derived content, including old revisions; original sessions are separate.
                db.execute(
                    "DELETE FROM memory_changes WHERE user=? AND key=?", (user, candidate.key)
                )
                body.pop("content")
            else:
                db.execute(
                    "INSERT OR REPLACE INTO memories VALUES (?,?,?)",
                    (user, candidate.key, dumps(body)),
                )
            db.execute(
                "INSERT INTO memory_changes(user,key,action,body,created) VALUES (?,?,?,?,?)",
                (user, candidate.key, candidate.action, dumps(body), now()),
            )

    def commit_extraction(
        self,
        user,
        session,
        candidates,
        *,
        batch_id,
        expected_cursor,
        through_seq,
        expected_memories,
        corrections=(),
        deletions=(),
        details=None,
    ):
        """Reuse commit validation/writes; progress and successful batch state share one transaction.

        Approvals and snapshots come from the trusted application, never model arguments.
        """
        self._validate(user, candidates)
        with self.store.transaction() as db:
            row = db.execute(
                "SELECT last_seq FROM memory_progress WHERE user=? AND session=?", (user, session)
            ).fetchone()
            if (row[0] if row else 0) != expected_cursor:
                raise ValueError("Memory extraction progress changed; batch was not applied")
            existing = {
                row[0]: json.loads(row[1])
                for row in db.execute("SELECT key,body FROM memories WHERE user=?", (user,))
            }
            selected = []
            for candidate in candidates:
                previous = existing.get(candidate.key)
                if previous != expected_memories.get(candidate.key):
                    raise ValueError("Memory changed during extraction; batch was not applied")
                if candidate.action == "delete":
                    if previous is None or candidate.key not in deletions:
                        raise ValueError(
                            "Memory deletion requires an existing key and application approval"
                        )
                elif previous is not None:
                    if (
                        previous["kind"] == candidate.kind
                        and previous["content"] == candidate.content
                    ):
                        continue
                    if candidate.key not in corrections:
                        raise ValueError("Memory correction requires application approval")
                elif any(
                    item["kind"] == candidate.kind
                    and item["content"].strip().casefold() == candidate.content.strip().casefold()
                    for item in [
                        *existing.values(),
                        *[
                            {"kind": c.kind, "content": c.content}
                            for c in selected
                            if c.action != "delete"
                        ],
                    ]
                ):
                    continue
                if previous is not None:
                    old_seq = max(
                        db.execute(
                            "SELECT seq FROM messages WHERE user=? AND id=?", (user, mid)
                        ).fetchone()[0]
                        for mid in previous["sources"]
                    )
                    new_seq = max(
                        db.execute(
                            "SELECT seq FROM messages WHERE user=? AND id=?", (user, mid)
                        ).fetchone()[0]
                        for mid in candidate.sources
                    )
                    if new_seq <= old_seq:
                        raise ValueError("Corrections and deletions require newer user evidence")
                selected.append(candidate)
            self._apply(db, user, selected)
            db.execute(
                "INSERT OR REPLACE INTO memory_progress VALUES (?,?,?)",
                (user, session, through_seq),
            )
            db.execute(
                "UPDATE memory_batches SET status='completed',body=? WHERE id=? AND user=? AND session=?",
                (
                    dumps(
                        {**(details or {}), "changed": len(selected), "through_seq": through_seq}
                    ),
                    batch_id,
                    user,
                    session,
                ),
            )
        return len(selected)

    def retrieve(self, user, query, *, limit=4):
        if not 0 <= limit <= 20:
            raise ValueError("Memory retrieval limit must be 0..20")
        query_terms = terms(query)
        ranked = []
        for row in self.store.db.execute("SELECT body FROM memories WHERE user=?", (user,)):
            memory = json.loads(row[0])
            score = len(query_terms & terms(memory["key"] + " " + memory["content"]))
            if score:
                ranked.append((score, memory["updated_at"], memory["key"], memory))
        ranked.sort(key=lambda item: item[:3], reverse=True)
        return [item[3] for item in ranked[:limit]]
