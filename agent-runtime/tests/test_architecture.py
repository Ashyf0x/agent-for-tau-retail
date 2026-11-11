"""Executable architecture invariants for the runtime boundary.

These tests guard *structure*, not behaviour. They never instantiate a model,
open a network connection, or read .env. Their purpose is to make two design
decisions fail loudly if someone drifts away from them:

1. the generic runtime never depends on Retail evaluation wiring or on
   experiment artifacts;
2. the L3 session terminal record is observation-only -- it is never read by
   the code that assembles model input, so it cannot change a request body.
"""

import ast
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
PACKAGE = PROJECT / "agentcore"
EVALS = PROJECT / "evals"


def _python_files(root):
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _imports(path):
    """Return (internal layer names, forbidden external roots) for a module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    internal, external = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        else:
            continue
        for name in names:
            root = name.split(".")[0]
            if root == "agentcore":
                parts = name.split(".")
                internal.add(parts[1] if len(parts) > 1 else "root")
            elif root in {"evals", "experiments"}:
                external.add(root)
    return internal, external


def _method_source(path, class_name, method_name):
    """Return the exact source of one method, including nested lines."""
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    lines = text.splitlines()
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if (isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and child.name == method_name):
                    return "\n".join(lines[child.lineno - 1:child.end_lineno])
    raise AssertionError(f"{class_name}.{method_name} not found in {path}")


def test_runtime_never_imports_eval_wiring_or_experiment_artifacts():
    offenders = {}
    for path in _python_files(PACKAGE):
        _, external = _imports(path)
        if external:
            offenders[str(path.relative_to(PROJECT))] = sorted(external)
    assert offenders == {}, "runtime must stay independent of evals/ and experiments/"


def test_capability_modules_do_not_import_the_application_entry():
    # app/ wires everything together; capability modules must never reach back up.
    offenders = []
    for path in _python_files(PACKAGE):
        layer = path.relative_to(PACKAGE).parts[0]
        if layer in {"", "app"} or len(path.relative_to(PACKAGE).parts) == 1:
            continue
        internal, _ = _imports(path)
        if "app" in internal:
            offenders.append(str(path.relative_to(PROJECT)))
    assert offenders == []


def test_lower_layers_keep_their_dependency_ceiling():
    # storage/context/tools only need core contracts; memory additionally needs
    # storage. Widening any of these couples a capability to orchestration.
    ceiling = {
        "storage": {"root", "core"},
        "context": {"root", "core"},
        "tools": {"root", "core"},
        "memory": {"root", "core", "storage"},
        "observability": {"root", "core", "storage"},
    }
    offenders = {}
    for path in _python_files(PACKAGE):
        rel = path.relative_to(PACKAGE)
        if len(rel.parts) == 1:
            continue
        layer = rel.parts[0]
        if layer not in ceiling:
            continue
        internal, _ = _imports(path)
        extra = internal - ceiling[layer]
        if extra:
            offenders[str(path.relative_to(PROJECT))] = sorted(extra)
    assert offenders == {}


def test_l3_terminal_record_is_invisible_to_model_input_path():
    # If any of these methods referenced the L3 table, the observation layer
    # would become a policy layer and could change the bytes sent to the model.
    for path, cls, method in (
        (PACKAGE / "storage/sqlite.py", "Store", "messages"),
        (PACKAGE / "context/manager.py", "ContextManager", "prepare"),
        (PACKAGE / "core/agent.py", "Agent", "run"),
    ):
        body = _method_source(path, cls, method)
        assert "session_terminations" not in body, f"{cls}.{method}"
        assert "session_terminal" not in body, f"{cls}.{method}"
    assert "session_terminations" not in (PACKAGE / "tools/executor.py").read_text(encoding="utf-8")


def test_l3_write_is_confined_to_storage_and_the_eval_bridge():
    assert "session_terminations" in (PACKAGE / "storage/sqlite.py").read_text(encoding="utf-8")
    for path in _python_files(PACKAGE):
        if path.name == "sqlite.py":
            continue
        text = path.read_text(encoding="utf-8")
        assert "session_terminations" not in text, str(path.relative_to(PROJECT))
        assert "finish_session" not in text, str(path.relative_to(PROJECT))
    assert "finish_session" in (EVALS / "retail_interaction.py").read_text(encoding="utf-8")


def test_offline_preparer_takes_evidence_paths_explicitly():
    source = (EVALS / "decision_snapshots.py").read_text(encoding="utf-8")
    assert "experiments/" not in source, "preparer must not reach into a retired experiment tree"
    assert "audit_path" in source and "runs_root" in source


def test_legacy_fixtures_are_self_contained():
    legacy = Path(__file__).parent / "fixtures" / "legacy"
    required = (
        "decision-study/candidate.json",
        "decision-audit-requests.json",
        "deepseek-probe/run_probe.py",
        "deepseek-protocol/candidate.json",
    )
    missing = [name for name in required if not (legacy / name).is_file()]
    assert missing == []
    jobs = sorted(p for p in (legacy / "decision-source-runs").glob("task-*") if p.is_dir())
    assert len(jobs) == 6
    for job in jobs:
        assert (job / "config.json").is_file(), job.name
        assert (job / "calls.jsonl").is_file(), job.name
        assert any(job.glob("task-*/core.sqlite3")), job.name


T50X3 = PROJECT / "experiments/retail-qwen-tier-order-20261108/runs/t50x3"


@pytest.mark.skipif(not T50X3.exists(), reason="experiment artifacts are gitignored on fresh clones")
def test_authoritative_50x3_evidence_is_intact():
    tasks = [p for p in T50X3.iterdir() if p.is_dir()]
    trials = sum(
        len([x for x in (task / task.name).iterdir() if x.name.startswith("trial-")])
        for task in tasks if (task / task.name).is_dir()
    )
    assert len(tasks) == 50
    assert trials == 150
