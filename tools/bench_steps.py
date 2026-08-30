"""Performance micro-bench — env-steps/s vs num_envs, the standing instrument
for the productization plan's Phase 2 (and the honest answer to marketing FPS).

One world per process (engine contract): with several --envs values this script
re-executes itself once per value and aggregates the RESULT lines.

    set LUMENGINE_ROOT=...
    python tools/bench_steps.py --task Ant --envs 256,1024,4096 --steps 200

Reports both policy-steps/s and env-steps/s (= num_envs * policy-steps/s), plus
substeps/s (* control_freq_inv) so comparisons against substep-counting claims
(Genesis) are explicit. Reference target: Isaac Gym class is ~150-500k
env-steps/s on Ant @ 4096 on one GPU.

Two outputs, for two readers:

- the ``RESULT ...`` line, for a human reading a terminal;
- one ``LMPERF-RECORD {json}`` line per benchmarked env count: the versioned
  machine-readable record the performance harness consumes. It states what a
  throughput number cannot state about itself — the status, the devices the
  run actually resolved to, whether the shared CUDA context was verified in
  use, readiness, any fault, and every parameter. Exit status alone can never
  stand for a measurement, since this script exits zero when it declines to
  run, so a decline is a record with ``status: "skipped"``, never a missing
  one.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# ASSIGNED, not defaulted: a pre-existing "0" in the environment would survive
# a setdefault, and the benchmark would then measure the CPU fallback while
# reporting a GPU headline. Set here, before any lumotion import (the task
# registry is imported inside bench_one), so the engine bootstrap's own
# setdefault becomes a no-op — and every re-executed child inherits it, since
# the subprocess call below deliberately passes no `env=` and the child gets
# this process's environment. The opt-out documented for non-benchmark
# consumers is untouched: this assignment lives only here.
os.environ["LM_PHYSX_SHARE_CUDA_CONTEXT"] = "1"

# The record format the performance harness parses. This integer is a
# cross-repository contract (lm-perf, harness/result_record.py,
# RESULT_RECORD_FORMAT_VERSION): both sides move together or not at all.
RESULT_RECORD_FORMAT_VERSION = 1

# One line, one record, one distinctive prefix. A single line survives
# interleaved engine logging where a multi-line fenced block would not, and
# the prefix deliberately carries no "RESULT " substring, so the human-line
# scanner below can never mistake a record for a measurement line.
RECORD_PREFIX = "LMPERF-RECORD "

# Anchored: a human RESULT line starts a line of its own.
RESULT_LINE = re.compile(r"^RESULT .*", re.MULTILINE)


def emit_record(status, task_id, num_envs, steps, warmup_steps,
                simulation_device, action_device, shared_context_verified,
                ready, fault=None, policy_sps=None, env_sps=None):
    """Print the one machine-readable record for this run.

    ``shared_context.requested`` is what this process asked for (the
    assignment above); ``verified_in_use`` is what was proven. The request is
    never echoed into the proof.
    """
    record = {
        "format_version": RESULT_RECORD_FORMAT_VERSION,
        "status": status,
        "simulation_device": simulation_device,
        "action_device": action_device,
        "shared_context": {
            "requested": os.environ.get("LM_PHYSX_SHARE_CUDA_CONTEXT") == "1",
            "verified_in_use": bool(shared_context_verified),
        },
        "ready": bool(ready),
        "fault": fault,
        "parameters": {
            "task": task_id,
            "environment_count": int(num_envs),
            "warmup_steps": int(warmup_steps),
            "step_count": int(steps),
        },
    }
    if status == "completed":
        record["policy_steps_per_second"] = policy_sps
        record["env_steps_per_second"] = env_sps
    sys.stdout.write(RECORD_PREFIX + json.dumps(record, sort_keys=True) + "\n")
    sys.stdout.flush()


def device_name(tensor) -> str:
    """A tensor's device as the harness spells it. A CUDA device always
    carries its ordinal, so the simulation and action devices can be compared
    for agreement rather than merely both containing "cuda"."""
    device = tensor.device
    if device.type == "cuda":
        return f"cuda:{0 if device.index is None else device.index}"
    return str(device)


def resolved_simulation_device(task) -> str:
    """The device the simulation state actually lives on.

    Read from the sim's own state tensor, not from the requested
    ``SimConfig.device``: the facade resolves "auto" at construction and falls
    back to the host silently, so the request proves nothing about the run.
    """
    return device_name(task.sim.acquire_dof_state_tensor())


def shared_cuda_context_verified(task) -> bool:
    """Whether the shared CUDA primary context is genuinely in effect.

    Not "was it asked for". The engine's direct-GPU batch refuses every
    transfer unless PhysX retained this process's CUDA primary context on the
    tensor's own device ordinal, so one real batch read through the public
    facade is the proof: it can only succeed on the shared-context path. A
    simulation that is not on CUDA never takes that path and is never
    verified.

    A refusal is deliberately not swallowed here: a CUDA simulation whose
    direct-GPU transport does not work is a fault, and the caller records it
    as one rather than as a quietly unverified measurement.
    """
    if task.sim.acquire_dof_state_tensor().device.type != "cuda":
        return False
    task.sim.refresh_dof_state_tensor()
    return True


def bench_one(task_id: str, num_envs: int, steps: int, warmup_steps: int) -> None:
    from lumotion_envs.registry import REGISTRY, load_task
    import torch
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        # A decline is stated, not left to be inferred from an exit code that
        # says zero.
        emit_record("skipped", task_id, num_envs, steps, warmup_steps,
                    simulation_device="cpu", action_device="cpu",
                    shared_context_verified=False, ready=False)
        sys.exit(0)

    from lumotion_envs.config import build_config
    task = None
    try:
        # Inside the guard: an unknown task or an engine that will not import
        # is a fault this run has to name, not a bare traceback the harness can
        # only read as "exited non-zero".
        spec = REGISTRY[task_id]
        _, task_cls, _ = load_task(spec)
        task = task_cls(build_config(spec.config_cls, num_envs=num_envs, headless=True))
        for _ in range(4000):
            task.warmup_step()
            task.runner.run()
            if task.ready:
                break
        assert task.ready, "batch never ready"
        task.reset()

        actions = torch.zeros((num_envs, task.num_actions), device=task.device)
        for _ in range(warmup_steps):
            task.step(actions)

        # Observed on the batch that has been stepping, after the warm-up:
        # what the run resolved to, never what it requested.
        simulation_device = resolved_simulation_device(task)
        action_device = device_name(actions)
        shared_context = shared_cuda_context_verified(task)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(steps):
            task.step(actions)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    except BaseException as error:            # noqa: BLE001 - recorded, then re-raised
        # A run that died has a story worth carrying: without a record the
        # harness would only see a non-zero exit and could not say whether the
        # scene faulted or the benchmark never started.
        emit_record("fault", task_id, num_envs, steps, warmup_steps,
                    simulation_device=fault_device(task),
                    action_device=fault_device(task),
                    shared_context_verified=False,
                    ready=bool(task is not None and task.ready),
                    fault=f"{type(error).__name__}: {error}"[:400])
        raise

    decim = getattr(task, "control_freq_inv", 1)
    sps = steps / dt
    print(f"RESULT task={task_id} envs={num_envs} steps={steps} wall={dt:.3f}s "
          f"policy_sps={sps:.1f} env_sps={sps * num_envs:.0f} "
          f"env_substeps={sps * num_envs * decim:.0f} decimation={decim}")
    emit_record("completed", task_id, num_envs, steps, warmup_steps,
                simulation_device=simulation_device, action_device=action_device,
                shared_context_verified=shared_context, ready=task.ready,
                policy_sps=sps, env_sps=sps * num_envs)
    import lumotion as rl
    rl.destroy_world(task.sim, task.runner)


def fault_device(task) -> str:
    """The resolved device for a fault record, or "unknown" when the failure
    came before there was a simulation to ask."""
    try:
        return resolved_simulation_device(task)
    except BaseException:                     # noqa: BLE001 - a fault report never faults
        return "unknown"


def split_records(text: str):
    """Split a child's output into (its record lines, everything else)."""
    records, rest = [], []
    for line in text.splitlines():
        (records if line.startswith(RECORD_PREFIX) else rest).append(line)
    return records, "\n".join(rest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Ant")
    ap.add_argument("--envs", default="256,1024,4096",
                    help="comma-separated num_envs values (one subprocess each)")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=50)
    # PERF GATE: exit 1 if any measured env-steps/s falls below this floor.
    # Calibrate WELL below the thermal band (Ant@4096 swings 410-480k on the
    # reference machine depending on GPU temperature): the gate exists to catch
    # real regressions (a sync-storm reappearing = 8x loss), never thermals.
    ap.add_argument("--min-env-sps", type=float, default=None,
                    help="fail (exit 1) if env-steps/s drops below this floor")
    ap.add_argument("--_one", type=int, help=argparse.SUPPRESS)   # internal: single run
    args = ap.parse_args()

    if args._one is not None:
        bench_one(args.task, args._one, args.steps, args.warmup)
        return

    env_counts = [int(x) for x in args.envs.split(",") if x.strip()]
    rows = []
    bench_failed = False
    for n in env_counts:
        print(f"[bench] {args.task} @ {n} envs ...", flush=True)
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--task", args.task,
             "--steps", str(args.steps), "--warmup", str(args.warmup), "--_one", str(n)],
            capture_output=True, text=True, cwd=str(REPO))
        out = (proc.stdout or "") + (proc.stderr or "")
        # The child's record is this run's record: re-emitted verbatim, exactly
        # once, and taken out of the failure echo below so a failed run cannot
        # appear to have produced two of them.
        records, out = split_records(out)
        for record in records:
            sys.stdout.write(record + "\n")
        sys.stdout.flush()
        m = RESULT_LINE.search(out)
        if proc.returncode != 0 or not m:
            print(f"[bench] {n} envs FAILED:\n" + "\n".join(out.splitlines()[-15:]))
            bench_failed = True
            continue
        print("[bench] " + m.group(0))
        rows.append(m.group(0))

    gate_failed = False
    if rows:
        print("\n=== bench summary ===")
        print(f"{'envs':>8} {'policy sps':>12} {'env-steps/s':>14} {'substeps/s':>14}")
        for r in rows:
            kv = dict(p.split("=") for p in r.split()[1:])
            env_sps = float(kv["env_sps"])
            print(f"{kv['envs']:>8} {float(kv['policy_sps']):>12.1f} "
                  f"{env_sps:>14.0f} {float(kv['env_substeps']):>14.0f}")
            if args.min_env_sps is not None and env_sps < args.min_env_sps:
                print(f"    ^^ PERF GATE FAILED: {env_sps:.0f} < floor {args.min_env_sps:.0f}")
                gate_failed = True
        print("\nScaling read: env-steps/s should GROW with envs until the GPU saturates;"
              "\nan early plateau = CPU/sync-bound (see plan 003, Phase 2).")

    if bench_failed or gate_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
