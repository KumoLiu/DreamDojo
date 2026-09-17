#!/usr/bin/env python3
"""Submit a set of DreamDojo comparison runs to SLURM from one sweep spec.

Each run becomes an independent sbatch job built from dreamdojo_sweep_train.slurm.
Run names must be unique because they set `job.name`, which decides the output
directory; a collision would make two runs overwrite each other's checkpoints.

    python3 scripts/submit_sweep.py /path/to/new_sweep.yaml --dry-run
    python3 scripts/submit_sweep.py /path/to/new_sweep.yaml
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SLURM_TEMPLATE = REPO_ROOT / "dreamdojo_sweep_train.slurm"
# Keys the template reads from the environment; anything else in a run block is
# treated as a Hydra override so typos surface as config errors, not silence.
ENV_KEYS = {
    "experiment": "EXPERIMENT",
    "config_name": "CONFIG_NAME",
    "max_iter": "MAX_ITER",
    "num_gpus": "NUM_GPUS",
    "load_training_state": "LOAD_TRAINING_STATE",
    "user_base": "USER_BASE",
    "repo_root": "REPO_ROOT",
    "dataset_root": "DATASET_ROOT",
    "checkpoint_root": "CHECKPOINT_ROOT",
    "container_checkpoint_path": "CONTAINER_CHECKPOINT_PATH",
    "output_root": "OUTPUT_ROOT",
    "container_image": "CONTAINER_IMAGE",
}
SBATCH_KEYS = {"time", "gpus", "account", "partition", "job_name", "dependency"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path, help="Sweep spec YAML.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the sbatch commands without submitting.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Submit only these run names.",
    )
    return parser.parse_args()


def load_spec(path: Path) -> tuple[dict, list[dict]]:
    spec = yaml.safe_load(path.read_text())
    if not isinstance(spec, dict) or "runs" not in spec:
        raise SystemExit(f"{path}: expected a mapping with a 'runs' list")
    defaults = spec.get("defaults") or {}
    runs = spec["runs"]
    if not runs:
        raise SystemExit(f"{path}: 'runs' is empty")
    return defaults, runs


def merge(defaults: dict, run: dict) -> dict:
    merged = {k: v for k, v in defaults.items() if k != "overrides"}
    merged.update({k: v for k, v in run.items() if k != "overrides"})
    overrides = dict(defaults.get("overrides") or {})
    overrides.update(run.get("overrides") or {})
    merged["overrides"] = overrides
    return merged


def format_override(key: str, value) -> str:
    if isinstance(value, bool):
        value = str(value).lower()
    elif isinstance(value, float):
        # repr() would emit 1e-05, and Hydra is happier with a plain decimal.
        value = f"{value:.10f}".rstrip("0") or "0"
    elif isinstance(value, (list, tuple)):
        value = "[" + ",".join(str(v) for v in value) + "]"
    elif isinstance(value, str) and "," in value:
        # Hydra parses an unquoted comma-separated value as a list, so a string
        # like lora_target_modules="q_proj,k_proj" silently becomes a list.
        raise SystemExit(
            f"Override {key!r} contains commas, which Hydra reads as a list.\n"
            f"Put this value in a dedicated configs/*.yaml instead and point the "
            f"run at it via 'experiment:'."
        )
    return f"{key}={value}"


def build_command(
    run: dict, sweep_name: str, output_dir: Path, *, write_overrides: bool = False
) -> tuple[str, list[str]]:
    name = run.get("name")
    if not name:
        raise SystemExit(f"Every run needs a 'name': {run}")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        raise SystemExit(f"Run name must be filesystem-safe, got {name!r}")
    if "experiment" not in run:
        raise SystemExit(f"Run {name!r} has no 'experiment'")

    run_name = f"{sweep_name}_{name}" if sweep_name else name
    exports = [f"{ENV_KEYS[k]}={run[k]}" for k in ENV_KEYS if k in run]
    overrides = " ".join(
        format_override(k, v) for k, v in sorted(run["overrides"].items())
    )
    exports.append(f"RUN_NAME={run_name}")
    if overrides:
        # sbatch --export splits on commas and is awkward with spaces, so the
        # overrides travel in a runtime file, not a source-tree receipt.
        # Previewing or validating a sweep must not create files.
        override_dir = output_dir / sweep_name
        override_file = override_dir / f"{name}.overrides"
        if write_overrides:
            override_dir.mkdir(parents=True, exist_ok=True)
            override_file.write_text(overrides + "\n")
        exports.append(f"OVERRIDES_FILE={override_file}")

    command = ["sbatch", f"--job-name=dd_{run_name}"]
    if "time" in run:
        command.append(f"--time={run['time']}")
    if "gpus" in run:
        command.append(f"--gpus={run['gpus']}")
    if "account" in run:
        command.append(f"--account={run['account']}")
    if "partition" in run:
        command.append(f"--partition={run['partition']}")
    if "dependency" in run:
        command.append(f"--dependency={run['dependency']}")
    # sbatch --export takes one comma-separated list; values contain spaces, so
    # the whole thing is passed as a single argv entry.
    command.append("--export=ALL," + ",".join(exports))
    command.append(str(SLURM_TEMPLATE))
    return run_name, command


def main() -> None:
    args = parse_args()
    if not SLURM_TEMPLATE.exists():
        raise SystemExit(f"Missing SLURM template: {SLURM_TEMPLATE}")

    defaults, raw_runs = load_spec(args.spec)
    sweep_name = defaults.get("sweep_name", args.spec.stem)

    runs = [merge(defaults, run) for run in raw_runs]
    if args.only:
        wanted = set(args.only)
        runs = [r for r in runs if r.get("name") in wanted]
        missing = wanted - {r.get("name") for r in runs}
        if missing:
            raise SystemExit(f"No such run(s): {', '.join(sorted(missing))}")
    if not runs:
        raise SystemExit("Nothing to submit")

    output_dir = REPO_ROOT / "outputs" / "sweeps"
    commands = [build_command(run, sweep_name, output_dir) for run in runs]

    duplicates = {n for n, _ in commands if [x for x, _ in commands].count(n) > 1}
    if duplicates:
        raise SystemExit(f"Duplicate run names would collide: {sorted(duplicates)}")

    print(f"Sweep '{sweep_name}': {len(commands)} run(s)\n")
    submitted = []
    for run, (run_name, command) in zip(runs, commands):
        print(f"--- {run_name} ---")
        print(shlex.join(command))
        if args.dry_run:
            print()
            continue
        _, command = build_command(run, sweep_name, output_dir, write_overrides=True)
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stdout, end="")
            print(result.stderr, end="", file=sys.stderr)
            raise SystemExit(f"sbatch failed for {run_name}")
        job_id = result.stdout.strip().split()[-1]
        print(f"submitted job {job_id}\n")
        submitted.append((run_name, job_id))

    if args.dry_run:
        print("Dry run: nothing submitted.")
        return

    print("Submitted:")
    for run_name, job_id in submitted:
        print(f"  {job_id}  {run_name}")
    print(f"\nMonitor: squeue -u $USER\nCancel:  scancel {' '.join(j for _, j in submitted)}")


if __name__ == "__main__":
    main()
