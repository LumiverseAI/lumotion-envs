"""Tier 0 — the machine-readable result record `tools/bench_steps.py` emits.

Engine-free: nothing here builds a world or imports `lm`. What is tested is
the protocol the performance harness depends on — one delimited record per
run, a status that tells the truth about a decline, devices spelled so they
can be compared, and the shared-context variable assigned rather than
defaulted so a pre-existing "0" cannot survive into the measurement.
"""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "tools" / "bench_steps.py"


def load_bench():
    """`tools/` is not a package; load the script by path, as its own users do."""
    spec = importlib.util.spec_from_file_location("bench_steps", BENCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = load_bench()


class FakeDevice:
    def __init__(self, type_, index=None):
        self.type = type_
        self.index = index

    def __str__(self):
        return self.type if self.index is None else f"{self.type}:{self.index}"


class FakeTensor:
    def __init__(self, device):
        self.device = device


def record_from(capsys) -> dict:
    lines = [l for l in capsys.readouterr().out.splitlines()
             if l.startswith(bench.RECORD_PREFIX)]
    assert len(lines) == 1, f"expected exactly one record, got {len(lines)}"
    return json.loads(lines[0][len(bench.RECORD_PREFIX):])


# --- The record the harness parses -------------------------------------------

def test_a_completed_record_carries_every_required_field(capsys):
    bench.emit_record("completed", "Ant", 4096, 200, 50,
                      simulation_device="cuda:0", action_device="cuda:0",
                      shared_context_verified=True, ready=True,
                      policy_sps=110.5, env_sps=452608.0)
    record = record_from(capsys)
    assert record["format_version"] == bench.RESULT_RECORD_FORMAT_VERSION
    assert record["status"] == "completed"
    assert record["simulation_device"] == "cuda:0"
    assert record["action_device"] == "cuda:0"
    assert record["shared_context"] == {"requested": True, "verified_in_use": True}
    assert record["ready"] is True
    assert record["fault"] is None
    assert record["parameters"] == {"task": "Ant", "environment_count": 4096,
                                    "warmup_steps": 50, "step_count": 200}
    assert record["policy_steps_per_second"] == pytest.approx(110.5)
    assert record["env_steps_per_second"] == pytest.approx(452608.0)


def test_the_parameters_object_is_exactly_the_four_declared_ones(capsys):
    """The harness refuses an unknown parameter: a new measurement-affecting
    parameter is a format change on both sides, never a smuggled key."""
    bench.emit_record("completed", "Ant", 8, 3, 1, "cuda:0", "cuda:0", True, True,
                      policy_sps=1.0, env_sps=8.0)
    assert set(record_from(capsys)["parameters"]) == {
        "task", "environment_count", "warmup_steps", "step_count"}


def test_only_a_completed_record_carries_throughput(capsys):
    """A skip or a fault has no throughput to report, and a record that
    carried one would be claiming a measurement it never took."""
    for status in ("skipped", "fault"):
        bench.emit_record(status, "Ant", 4096, 200, 50, "cpu", "cpu", False, False,
                          fault=None if status == "skipped" else "boom")
        record = record_from(capsys)
        assert "policy_steps_per_second" not in record
        assert "env_steps_per_second" not in record


def test_the_requested_flag_reports_the_environment_not_the_intent(capsys, monkeypatch):
    monkeypatch.setenv("LM_PHYSX_SHARE_CUDA_CONTEXT", "0")
    bench.emit_record("skipped", "Ant", 4096, 200, 50, "cpu", "cpu", False, False)
    assert record_from(capsys)["shared_context"] == {"requested": False,
                                                     "verified_in_use": False}


# --- Devices, observed rather than echoed ------------------------------------

def test_a_cuda_device_is_spelled_with_its_ordinal():
    """The harness compares the simulation and action devices for equality and
    refuses a device that merely contains "cuda", so both carry an ordinal."""
    assert bench.device_name(FakeTensor(FakeDevice("cuda", 0))) == "cuda:0"
    assert bench.device_name(FakeTensor(FakeDevice("cuda", 1))) == "cuda:1"
    assert bench.device_name(FakeTensor(FakeDevice("cuda"))) == "cuda:0"
    assert bench.device_name(FakeTensor(FakeDevice("cpu"))) == "cpu"


def test_the_simulation_device_is_read_from_the_sim_state_tensor():
    """Not from SimConfig.device: "auto" that fell back to the host still
    reads "auto"."""
    class Sim:
        device = "auto"

        def acquire_dof_state_tensor(self):
            return FakeTensor(FakeDevice("cpu"))

    class Task:
        sim = Sim()

    assert bench.resolved_simulation_device(Task()) == "cpu"


def test_the_shared_context_is_verified_by_a_real_batch_read():
    """A CUDA tensor alone proves nothing: the direct-GPU read is what only
    succeeds when PhysX shares this process's CUDA primary context."""
    reads = []

    class Sim:
        def acquire_dof_state_tensor(self):
            return FakeTensor(FakeDevice("cuda", 0))

        def refresh_dof_state_tensor(self):
            reads.append(1)

    class Task:
        sim = Sim()

    assert bench.shared_cuda_context_verified(Task()) is True
    assert reads, "verification did not exercise the direct-GPU transport"


def test_a_host_simulation_is_never_a_verified_shared_context():
    class Sim:
        def acquire_dof_state_tensor(self):
            return FakeTensor(FakeDevice("cpu"))

        def refresh_dof_state_tensor(self):
            raise AssertionError("the host path must not be probed")

    class Task:
        sim = Sim()

    assert bench.shared_cuda_context_verified(Task()) is False


def test_a_refused_transport_is_raised_not_reported_as_unverified():
    """A CUDA sim whose direct-GPU transport does not work is a fault, and the
    caller records it as one. Swallowing it here would emit a record that
    looks like an ordinary unverified run."""
    class Sim:
        def acquire_dof_state_tensor(self):
            return FakeTensor(FakeDevice("cuda", 0))

        def refresh_dof_state_tensor(self):
            raise RuntimeError("UNSUPPORTED_ON_PATH")

    class Task:
        sim = Sim()

    with pytest.raises(RuntimeError):
        bench.shared_cuda_context_verified(Task())


# --- The delimiter -----------------------------------------------------------

def test_the_record_prefix_cannot_be_read_as_a_human_result_line():
    """The aggregator scans the child's output for the RESULT line; a record
    prefix containing "RESULT " would be matched instead of the measurement."""
    assert "RESULT " not in bench.RECORD_PREFIX
    record_line = bench.RECORD_PREFIX + '{"status": "completed"}'
    assert bench.RESULT_LINE.search(record_line) is None
    assert bench.RESULT_LINE.search("RESULT task=Ant envs=8") is not None


def test_records_are_split_out_of_the_output_the_failure_path_echoes():
    """A failing run echoes the tail of the child's output. Echoing a record
    line there would put a second copy of it on stdout, and two records for
    one repetition are refused as a duplicate."""
    text = "\n".join(["engine chatter",
                      bench.RECORD_PREFIX + '{"status": "fault"}',
                      "Traceback (most recent call last):"])
    records, rest = bench.split_records(text)
    assert records == [bench.RECORD_PREFIX + '{"status": "fault"}']
    assert bench.RECORD_PREFIX not in rest


def test_the_aggregator_re_emits_the_child_record_exactly_once(monkeypatch, capsys):
    child = subprocess.CompletedProcess(
        args=[], returncode=0,
        stdout="\n".join([bench.RECORD_PREFIX + '{"status": "completed"}',
                          "RESULT task=Ant envs=8 steps=3 wall=1.000s "
                          "policy_sps=3.0 env_sps=24 env_substeps=48 decimation=2"]),
        stderr="")
    monkeypatch.setattr(bench.subprocess, "run", lambda *a, **k: child)
    monkeypatch.setattr(sys, "argv", ["bench_steps.py", "--task", "Ant", "--envs", "8"])
    bench.main()
    out = capsys.readouterr().out
    assert out.count(bench.RECORD_PREFIX) == 1


def test_a_failed_child_still_yields_its_one_record(monkeypatch, capsys):
    child = subprocess.CompletedProcess(
        args=[], returncode=1,
        stdout=bench.RECORD_PREFIX + '{"status": "fault"}',
        stderr="RuntimeError: PhysX scene faulted")
    monkeypatch.setattr(bench.subprocess, "run", lambda *a, **k: child)
    monkeypatch.setattr(sys, "argv", ["bench_steps.py", "--task", "Ant", "--envs", "8"])
    with pytest.raises(SystemExit) as exit_info:
        bench.main()
    assert exit_info.value.code == 1
    assert capsys.readouterr().out.count(bench.RECORD_PREFIX) == 1


# --- The shared-context variable ---------------------------------------------

def test_the_child_inherits_this_process_environment(monkeypatch):
    """The re-execution passes no `env=`, which is the whole reason the
    assignment above reaches the process that actually benchmarks."""
    seen = {}

    def fake_run(*args, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="")

    monkeypatch.setattr(bench.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["bench_steps.py", "--task", "Ant", "--envs", "8"])
    with pytest.raises(SystemExit):
        bench.main()
    assert "env" not in seen, ("the re-execution overrides the environment; the "
                              "shared-context assignment would not reach the child")


def test_the_shared_context_variable_is_assigned_not_defaulted():
    """A setdefault preserves a pre-existing "0", and the benchmark would then
    measure the CPU fallback while reporting a GPU headline."""
    environment = dict(os.environ, LM_PHYSX_SHARE_CUDA_CONTEXT="0")
    completed = subprocess.run(
        [sys.executable, "-c",
         "import importlib.util, os, sys;"
         f"spec = importlib.util.spec_from_file_location('b', r'{BENCH}');"
         "m = importlib.util.module_from_spec(spec);"
         "spec.loader.exec_module(m);"
         "print(os.environ['LM_PHYSX_SHARE_CUDA_CONTEXT'])"],
        capture_output=True, text=True, env=environment, cwd=str(REPO))
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().splitlines()[-1] == "1"
