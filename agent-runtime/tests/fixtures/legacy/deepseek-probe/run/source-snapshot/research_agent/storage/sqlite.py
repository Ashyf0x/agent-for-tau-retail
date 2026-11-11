"""SQLite owns identities, atomic writes and execution evidence."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from research_agent.core.types import SessionBusy, StorageError, dumps


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY, user TEXT, session TEXT, status TEXT, result TEXT,
                created TEXT);
            CREATE UNIQUE INDEX IF NOT EXISTS active_session ON runs(user,session)
                WHERE status='running';
            CREATE TABLE IF NOT EXISTS messages (
                seq INTEGER PRIMARY KEY, id TEXT UNIQUE, user TEXT, session TEXT,
                run TEXT, body TEXT);
            CREATE TABLE IF NOT EXISTS events (
                seq INTEGER PRIMARY KEY, run TEXT, kind TEXT, body TEXT, created TEXT);
            CREATE TABLE IF NOT EXISTS attempts (
                call TEXT, attempt INTEGER, run TEXT, user TEXT, session TEXT, name TEXT,
                read_only INTEGER, status TEXT, body TEXT,
                PRIMARY KEY(call,attempt));
            CREATE TABLE IF NOT EXISTS observations (
                ref TEXT PRIMARY KEY, user TEXT, session TEXT, call TEXT, body TEXT);
            CREATE TABLE IF NOT EXISTS memories (
                user TEXT, key TEXT, body TEXT, PRIMARY KEY(user,key));
            CREATE TABLE IF NOT EXISTS memory_changes (
                seq INTEGER PRIMARY KEY, user TEXT, key TEXT, action TEXT, body TEXT, created TEXT);
            CREATE TABLE IF NOT EXISTS memory_progress (
                user TEXT, session TEXT, last_seq INTEGER, PRIMARY KEY(user,session));
            CREATE TABLE IF NOT EXISTS memory_batches (
                id TEXT PRIMARY KEY, user TEXT, session TEXT, status TEXT, body TEXT, created TEXT);
        """)

    @contextmanager
    def transaction(self):
        try:
            with self.db:
                yield self.db
        except sqlite3.Error as exc:
            raise StorageError(type(exc).__name__) from exc

    def close(self):
        self.db.close()

    def begin(self, user, session, message):
        run = uuid4().hex
        with self.transaction() as db:
            if db.execute(
                "SELECT 1 FROM runs WHERE user=? AND session=? AND status='running'",
                (user, session),
            ).fetchone():
                raise SessionBusy(
                    "Session already running; recover only after its owner has stopped"
                )
            db.execute(
                "INSERT INTO runs VALUES (?,?,?,?,?,?)",
                (run, user, session, "running", dumps({}), now()),
            )
            self._append(db, user, session, run, {"role": "user", "content": message})
        return run

    def _append(self, db, user, session, run, body):
        mid = uuid4().hex
        db.execute(
            "INSERT INTO messages(id,user,session,run,body) VALUES (?,?,?,?,?)",
            (mid, user, session, run, dumps(body)),
        )
        return mid

    def append(self, user, session, run, body):
        with self.transaction() as db:
            return self._append(db, user, session, run, body)

    def messages(self, user, session):
        return [
            {"id": r["id"], "message": json.loads(r["body"])}
            for r in self.db.execute(
                "SELECT id,body FROM messages WHERE user=? AND session=? ORDER BY seq",
                (user, session),
            )
        ]

    def source(self, user, mid):
        row = self.db.execute(
            "SELECT session,body FROM messages WHERE user=? AND id=?", (user, mid)
        ).fetchone()
        return (
            None if row is None else {"session": row["session"], "message": json.loads(row["body"])}
        )

    def event(self, run, kind, body):
        with self.transaction() as db:
            db.execute(
                "INSERT INTO events(run,kind,body,created) VALUES (?,?,?,?)",
                (run, kind, dumps(body), now()),
            )

    def events(self, run):
        return [
            {"kind": r[0], **json.loads(r[1])}
            for r in self.db.execute(
                "SELECT kind,body FROM events WHERE run=? ORDER BY seq", (run,)
            )
        ]

    def finish(self, result):
        with self.transaction() as db:
            db.execute(
                "UPDATE runs SET status=?, result=? WHERE id=? AND status='running'",
                (result.status, dumps(result.to_dict()), result.run_id),
            )
            db.execute(
                "INSERT INTO events(run,kind,body,created) VALUES (?,?,?,?)",
                (result.run_id, "run_end", dumps(result.to_dict()), now()),
            )

    def attempt(self, run, user, session, call, name, read_only, attempt, status, body):
        with self.transaction() as db:
            db.execute(
                "INSERT OR REPLACE INTO attempts VALUES (?,?,?,?,?,?,?,?,?)",
                (call, attempt, run, user, session, name, int(read_only), status, dumps(body)),
            )

    def unresolved(self, user, session):
        return [
            dict(r)
            for r in self.db.execute(
                "SELECT call,name,attempt FROM attempts WHERE user=? AND session=? "
                "AND read_only=0 AND status='unknown'",
                (user, session),
            )
        ]

    def archive(self, user, session, call, value):
        ref = uuid4().hex
        with self.transaction() as db:
            db.execute(
                "INSERT INTO observations VALUES (?,?,?,?,?)",
                (ref, user, session, call, dumps(value)),
            )
        return ref

    def read_observation(self, user, session, ref, offset=0, limit=2000):
        if (
            type(offset) is not int
            or offset < 0
            or type(limit) is not int
            or not 1 <= limit <= 4000
        ):
            raise ValueError("offset must be nonnegative; limit must be 1..4000")
        row = self.db.execute(
            "SELECT call,body FROM observations WHERE ref=? AND user=? AND session=?",
            (ref, user, session),
        ).fetchone()
        if row is None:
            raise KeyError("Observation not found in this session")
        text = row["body"]
        end = min(offset + limit, len(text))
        return {
            "ref": ref,
            "call_id": row["call"],
            "text": text[offset:end],
            "offset": offset,
            "next_offset": end if end < len(text) else None,
            "total_chars": len(text),
        }

    def recover(self, user, session):
        """Explicit crash recovery, never replays a tool. Ensure the old owner stopped first."""
        with self.transaction() as db:
            runs = db.execute(
                "SELECT id FROM runs WHERE user=? AND session=? AND status='running'",
                (user, session),
            ).fetchall()
            for row in runs:
                run = row[0]
                records = self.messages(user, session)
                completed = {r["message"].get("tool_call_id") for r in records}
                for record in records:
                    for call in record["message"].get("tool_calls", []):
                        if call["id"] in completed:
                            continue
                        attempt = db.execute(
                            "SELECT status,body FROM attempts WHERE call=? ORDER BY attempt DESC",
                            (call["id"],),
                        ).fetchone()
                        body = json.loads(attempt["body"]) if attempt else {}
                        body["status"] = attempt["status"] if attempt else "not_dispatched"
                        if body["status"] != "returned":
                            body["error"] = "Interrupted; inspect execution evidence"
                        self._append(
                            db,
                            user,
                            session,
                            run,
                            {
                                "role": "tool",
                                "tool_call_id": call["id"],
                                "content": dumps(body),
                            },
                        )
                result = {"run_id": run, "status": "interrupted", "goal_status": "unverified"}
                db.execute(
                    "UPDATE runs SET status='interrupted',result=? WHERE id=?", (dumps(result), run)
                )
                db.execute(
                    "INSERT INTO events(run,kind,body,created) VALUES (?,?,?,?)",
                    (run, "run_end", dumps(result), now()),
                )
        return len(runs)
