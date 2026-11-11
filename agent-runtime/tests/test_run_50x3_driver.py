"""Driver-level guards for run-50x3.py.

These exist because the guards decide whether a 9-hour, 35-CNY run wastes money:
an uncaught timeout used to abort the whole driver, a missing summary.json used to
record the spend as zero, and a broken credential used to burn the entire ceiling
on guaranteed failures. Every test injects a fake subprocess so nothing here can
make a real call.
"""
import importlib.util
import json
import subprocess
import sys
import threading
import time
from decimal import Decimal
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
DRIVER = PROJECT / "experiments" / "retail-qwen-tier-order-20261108" / "run-50x3.py"


@pytest.fixture(scope="module")
def driver():
    spec = importlib.util.spec_from_file_location("run_50x3", DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def isolated_log(driver, tmp_path, monkeypatch):
    """Redirect the module-level LOG so tests never append to the real driver log.

    Without this, every mock run wrote its fake record into t50x3-driver.log with an
    output path under pytest-of-*, polluting the audit trail of a real paid run.
    """
    monkeypatch.setattr(driver, "LOG", tmp_path / "driver.log")
    return tmp_path / "driver.log"


class Args:
    """Minimal namespace matching the CLI options execute() reads."""

    def __init__(self, runs, **kwargs):
        self.i_authorize_real_calls = kwargs.get("authorize", True)
        self.total_budget_yuan = kwargs.get("budget", "5")
        self.tasks = kwargs.get("tasks", ["40"])
        self.trials = kwargs.get("trials", 1)
        self.runs = runs
        self.force = kwargs.get("force", False)
        self.max_consecutive_failures = kwargs.get("max_consecutive_failures", 3)
        self.parallel = kwargs.get("parallel", 1)


def write_run(target, *, trials=1, cost="0.3", stopped=False):
    target.mkdir(parents=True, exist_ok=True)
    (target / "summary.json").write_text(json.dumps({
        "finished_tasks": [{"task_id": target.name, "trial": i + 1,
                            "reward": 1.0, "seed": 670487} for i in range(trials)],
        "estimated_cny": cost, "calls": 10, "stopped": stopped,
    }), encoding="utf-8")


def test_refuses_without_authorization_flag(driver, tmp_path):
    with pytest.raises(SystemExit):
        driver.execute(Args(tmp_path, authorize=False))


def test_missing_budget_is_rejected_by_the_cli():
    """--execute without a ceiling must stop at argument parsing, before any call."""
    result = subprocess.run(
        [sys.executable, str(DRIVER), "--execute", "--i-authorize-real-calls"],
        capture_output=True, text=True, cwd=PROJECT, timeout=120,
    )
    assert result.returncode != 0
    assert "--total-budget-yuan" in (result.stderr + result.stdout)
    # Nothing may be written when the argument check fails.
    assert not list((DRIVER.parent / "runs" / "t50x3").glob("task-*-probe-never"))


def test_timeout_does_not_abort_the_loop(driver, tmp_path, monkeypatch, isolated_log):
    """A killed child must be recorded, and the loop must continue."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 0))

    monkeypatch.setattr(driver.subprocess, "run", fake_run)
    driver.execute(Args(tmp_path, tasks=["40", "112"], max_consecutive_failures=99))
    assert len(calls) == 2  # both attempted, no exception escaped
    # Every record must land in the isolated log, never the real audit trail.
    assert isolated_log.exists()
    assert all(str(tmp_path) in json.loads(line)["output"]
               for line in isolated_log.read_text(encoding="utf-8").splitlines())


def test_consecutive_failures_stop_the_loop(driver, tmp_path, monkeypatch):
    """A dead endpoint must not consume the whole ceiling."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 0))

    monkeypatch.setattr(driver.subprocess, "run", fake_run)
    driver.execute(Args(tmp_path, tasks=["40", "112", "14", "16"],
                        max_consecutive_failures=2))
    assert len(calls) == 2
    assert not (tmp_path / "task-14").exists()


def test_a_success_resets_the_failure_counter(driver, tmp_path, monkeypatch):
    """Isolated failures must not accumulate into a false halt.

    Outcome is keyed on the unit index rather than on call order, so the test does
    not depend on how many times the driver retries or on monkeypatch state.
    """
    started = []

    def fake_run(cmd, **kwargs):
        index = len(started)
        started.append(cmd)
        if index % 2 == 0:  # units 0 and 2 fail, 1 and 3 succeed
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 0))
        write_run(Path(cmd[cmd.index("--output") + 1]))
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(driver.subprocess, "run", fake_run)
    driver.execute(Args(tmp_path, tasks=["40", "112", "14", "16"],
                        max_consecutive_failures=2))
    assert len(started) == 4  # never halted: every failure is followed by a success


def test_budget_ceiling_halts_before_next_unit(driver, tmp_path, monkeypatch):
    started = []

    def fake_run(cmd, **kwargs):
        started.append(cmd)
        write_run(Path(cmd[cmd.index("--output") + 1]), cost="0.3")
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(driver.subprocess, "run", fake_run)
    driver.execute(Args(tmp_path, budget="0.5", tasks=["40", "112", "14"]))
    assert len(started) == 2  # 0.3 then 0.6 exceeds 0.5


def test_ledger_recovers_spend_when_summary_is_missing(driver, tmp_path):
    """The durable ledger must reproduce the spend after a hard kill."""
    run = tmp_path / "task-40"
    run.mkdir()
    price = {"currency": "CNY", "input_cny_per_million": "0.96",
             "output_cny_per_million": "5.76", "per_role": {}}
    (run / "config.json").write_text(json.dumps({"price_assumption": price}), encoding="utf-8")
    rows = [
        {"event": "dispatch_intent", "call_id": "a", "role": "agent",
         "reserved_cny": "1.0"},
        {"event": "settlement", "call_id": "a", "role": "agent",
         "usage": {"input_tokens": 1000000, "output_tokens": 0},
         "estimated_cny": "0.96"},
        {"event": "settlement", "call_id": "b", "role": "agent",
         "usage": {"input_tokens": 0, "output_tokens": 1000000},
         "estimated_cny": "6.72"},
    ]
    (run / "calls.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    # No summary.json at all: the spend is still recoverable.
    # 1e6 input tokens at 0.96 + 1e6 output tokens at 5.76 = 6.72
    assert driver.ledger_cost(run) == Decimal("6.72")
    assert driver.run_cost(run) == Decimal("6.72")


def test_ledger_ignores_a_torn_final_line(driver, tmp_path):
    """A hard kill can leave half a JSON line; that must not lose the total."""
    run = tmp_path / "task-40"
    run.mkdir()
    price = {"input_cny_per_million": "1", "output_cny_per_million": "0"}
    (run / "config.json").write_text(json.dumps({"price_assumption": price}), encoding="utf-8")
    good = json.dumps({"event": "settlement", "role": "agent",
                       "usage": {"input_tokens": 2000000, "output_tokens": 0},
                       "estimated_cny": "2"})
    (run / "calls.jsonl").write_text(good + "\n" + '{"event":"settle', encoding="utf-8")
    assert driver.ledger_cost(run) == Decimal("2")


def test_summary_wins_over_ledger_when_present(driver, tmp_path):
    run = tmp_path / "task-40"
    write_run(run, cost="0.5")
    assert driver.run_cost(run) == Decimal("0.5")


def test_corrupt_summary_falls_back_to_ledger(driver, tmp_path):
    run = tmp_path / "task-40"
    run.mkdir()
    (run / "summary.json").write_text("{not json", encoding="utf-8")
    price = {"input_cny_per_million": "1", "output_cny_per_million": "0"}
    (run / "config.json").write_text(json.dumps({"price_assumption": price}), encoding="utf-8")
    (run / "calls.jsonl").write_text(
        json.dumps({"event": "settlement", "role": "agent",
                    "usage": {"input_tokens": 1000000, "output_tokens": 0}}) + "\n",
        encoding="utf-8")
    assert driver.run_cost(run) == Decimal("1")


def test_already_done_requires_every_trial(driver, tmp_path):
    run = tmp_path / "task-40"
    write_run(run, trials=3)
    assert driver.already_done(run, 3) is True
    assert driver.already_done(run, 4) is False


def test_stopped_run_is_not_done(driver, tmp_path):
    run = tmp_path / "task-40"
    write_run(run, trials=3, stopped=True)
    assert driver.already_done(run, 3) is False


def test_released_ids_are_the_official_fifty(driver):
    ids = driver.released_task_ids()
    assert len(ids) == 50 and len(set(ids)) == 50
    assert "40" in ids and "112" in ids
    assert "5" not in ids  # must not silently accept a task outside the subset


# ---------------------------------------------------------------------------
# Parallel execution. The whole point of these four is that a resumed run must
# produce exactly the same evidence as a serial one, and that the budget ceiling
# stays a hard bound when several workers run at once.
# ---------------------------------------------------------------------------

def test_parallel_runs_every_task_exactly_once(driver, tmp_path, monkeypatch, isolated_log):
    """Concurrency must not duplicate or drop a unit."""
    seen = []
    guard = threading.Lock()

    def fake_run(cmd, **kwargs):
        task = cmd[cmd.index("--task-id") + 1]
        with guard:
            seen.append(task)
        time.sleep(0.05)  # widen the window where a race would show up
        write_run(Path(cmd[cmd.index("--output") + 1]))
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(driver.subprocess, "run", fake_run)
    tasks = ["40", "112", "14", "16", "17", "18"]
    # A real-scale ceiling: each in-flight worker reserves PER_PROCESS_BUDGET (11.00),
    # so 3 workers need > 33.00 of headroom before the pool will admit them.
    driver.execute(Args(tmp_path, tasks=tasks, parallel=3, budget="45.00"))
    assert sorted(seen) == sorted(tasks)
    assert len(seen) == len(tasks)
    # Every completed task must have its own directory and log record.
    logged = [json.loads(l)["task"] for l in isolated_log.read_text(encoding="utf-8").splitlines()]
    assert sorted(logged) == sorted(tasks)


def test_parallel_keeps_dispatching_after_a_reservation_refusal(driver, tmp_path, monkeypatch, isolated_log):
    """A refused reservation is transient and must not strand the queue.

    Regression for the 2026-11-09 21:39 resume: 18.22 CNY of headroom admitted two
    workers, the third reservation was refused, and `halt` latched permanently. The
    eleven remaining tasks were recorded as not_started_budget_ceiling_reached even
    though the reserved headroom returned the moment those two workers settled.
    """
    started = []
    guard = threading.Lock()

    def fake_run(cmd, **kwargs):
        with guard:
            started.append(cmd[cmd.index("--task-id") + 1])
        time.sleep(0.05)
        write_run(Path(cmd[cmd.index("--output") + 1]), cost="0.3")
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(driver.subprocess, "run", fake_run)
    tasks = ["40", "112", "14", "16"]
    # 24.00 of headroom with PER_PROCESS_BUDGET=11.00 admits two workers at a time;
    # the third reservation is refused until earlier workers settle. Every task must
    # still run, and the ceiling must hold.
    driver.execute(Args(tmp_path, budget="24.00", tasks=tasks, parallel=4))
    assert sorted(started) == sorted(tasks)
    total = Decimal("0")
    for path in tmp_path.glob("task-*/summary.json"):
        total += Decimal(str(json.loads(path.read_text(encoding="utf-8"))["estimated_cny"]))
    assert total <= Decimal("24.00")


def test_parallel_respects_the_ceiling_with_reservations(driver, tmp_path, monkeypatch, isolated_log):
    """The serial `spent +=` pattern would let N workers cross the ceiling together."""
    started = []
    guard = threading.Lock()

    def fake_run(cmd, **kwargs):
        with guard:
            started.append(cmd[cmd.index("--task-id") + 1])
        time.sleep(0.05)
        write_run(Path(cmd[cmd.index("--output") + 1]), cost="0.3")
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(driver.subprocess, "run", fake_run)
    # PER_PROCESS_BUDGET is 11.00, so a 12.00 ceiling admits exactly one worker at a
    # time; with 4 candidate tasks the reservation must stop it after the budget is
    # consumed rather than launching all four.
    driver.execute(Args(tmp_path, budget="12.00", tasks=["40", "112", "14", "16"], parallel=4))
    assert len(started) <= 4
    total = Decimal("0")
    for path in tmp_path.glob("task-*/summary.json"):
        total += Decimal(str(json.loads(path.read_text(encoding="utf-8"))["estimated_cny"]))
    assert total <= Decimal("12.00")


def test_parallel_switches_to_serial_at_one_worker(driver, tmp_path, monkeypatch):
    """parallel=1 must stay the default so existing behaviour is unchanged."""
    order = []

    def fake_run(cmd, **kwargs):
        order.append(cmd[cmd.index("--task-id") + 1])
        write_run(Path(cmd[cmd.index("--output") + 1]))
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(driver.subprocess, "run", fake_run)
    driver.execute(Args(tmp_path, tasks=["40", "112", "14"]))
    assert order == ["40", "112", "14"]  # strict official order, one at a time


def test_parallel_stops_issuing_work_after_consecutive_failures(driver, tmp_path, monkeypatch):
    """A dead endpoint must stop the pool, not just the one worker that saw it."""
    started = []
    guard = threading.Lock()

    def fake_run(cmd, **kwargs):
        with guard:
            started.append(cmd[cmd.index("--task-id") + 1])
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 0))

    monkeypatch.setattr(driver.subprocess, "run", fake_run)
    driver.execute(Args(tmp_path, tasks=["40", "112", "14", "16", "17", "18"],
                        parallel=2, max_consecutive_failures=2))
    # With 2 workers, at most a couple of units can start before the halt lands.
    assert len(started) <= 4
