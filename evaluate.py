#!/usr/bin/env python3
"""SmolVLA + LIBERO evaluation wrapper.

Detects GPUs, runs lerobot-eval CLI, and prints a success-rate summary table.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch


MODEL_ID = "HuggingFaceVLA/smolvla_libero"
VALID_TASKS = ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]


def print_gpu_info():
    n = torch.cuda.device_count()
    if n == 0:
        print("WARNING: No CUDA GPUs detected!")
        return
    print(f"Detected {n} GPU(s):")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / (1024 ** 3)
        print(f"  [{i}] {props.name} — {mem_gb:.1f} GB")
    print()


def run_eval(task: str, n_episodes: int, batch_size: int,
             task_ids: list[int] | None, output_dir: Path) -> Path:
    """Run lerobot-eval and return path to the results JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "lerobot-eval",
        f"--policy.path={MODEL_ID}",
        "--env.type=libero",
        f"--env.task={task}",
        f"--eval.n_episodes={n_episodes}",
        f"--eval.batch_size={batch_size}",
        f"--output_dir={output_dir}",
    ]
    if task_ids is not None:
        ids = ",".join(str(t) for t in task_ids)
        cmd.append(f"--env.task_ids=[{ids}]")

    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"lerobot-eval exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    # Find the results JSON written by lerobot
    json_files = sorted(output_dir.rglob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not json_files:
        print("ERROR: No results JSON found in output directory.", file=sys.stderr)
        sys.exit(1)
    return json_files[0]


def print_results(json_path: Path):
    """Parse lerobot results JSON and print a summary table."""
    data = json.loads(json_path.read_text())

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    task_rows = []  # list of (label, success_rate)

    # lerobot 0.5.0 format: per_task is a list of {task_group, task_id, metrics}
    if "per_task" in data and isinstance(data["per_task"], list):
        for entry in data["per_task"]:
            tid = entry.get("task_id", "?")
            group = entry.get("task_group", "")
            successes = entry.get("metrics", {}).get("successes", [])
            rate = sum(successes) / len(successes) if successes else 0.0
            task_rows.append((f"{group}/task_{tid}", rate))
    # dict format fallback (older lerobot or other tools)
    elif "per_task" in data and isinstance(data["per_task"], dict):
        for name, info in sorted(data["per_task"].items()):
            rate = info if isinstance(info, (int, float)) else info.get("success_rate", 0)
            task_rows.append((name, rate))
    else:
        # Walk top-level keys for success metrics
        for k, v in data.items():
            if isinstance(v, dict) and "success_rate" in v:
                task_rows.append((k, v["success_rate"]))
            elif "success" in k.lower() and isinstance(v, (int, float)):
                task_rows.append((k, v))

    if task_rows:
        print(f"\n{'Task':<40} {'Success Rate':>12}")
        print("-" * 54)
        for name, rate in task_rows:
            print(f"{name:<40} {rate:>11.1%}")

    # Overall success rate from per_group or overall
    avg_success = None
    if "overall" in data and "pc_success" in data["overall"]:
        avg_success = data["overall"]["pc_success"] / 100.0
    elif "per_group" in data:
        rates = [g["pc_success"] / 100.0 for g in data["per_group"].values() if "pc_success" in g]
        if rates:
            avg_success = sum(rates) / len(rates)
    elif task_rows:
        avg_success = sum(r for _, r in task_rows) / len(task_rows)

    if avg_success is not None:
        print("-" * 54)
        print(f"{'AVERAGE':<40} {avg_success:>11.1%}")

    print("=" * 60)
    print(f"\nFull results: {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SmolVLA on LIBERO benchmarks")
    parser.add_argument("--task", default="libero_10", choices=VALID_TASKS,
                        help="LIBERO task suite (default: libero_10)")
    parser.add_argument("--n-episodes", type=int, default=10,
                        help="Number of evaluation episodes per task (default: 10)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Parallel environments (default: 1, reduce if OOM)")
    parser.add_argument("--task-ids", type=int, nargs="+", default=None,
                        help="Specific task IDs to evaluate (optional)")
    parser.add_argument("--output-dir", type=str, default="/workspace/results",
                        help="Directory to store results")
    args = parser.parse_args()

    print_gpu_info()

    print(f"Model:      {MODEL_ID}")
    print(f"Task:       {args.task}")
    print(f"Episodes:   {args.n_episodes}")
    print(f"Batch size: {args.batch_size}")
    if args.task_ids:
        print(f"Task IDs:   {args.task_ids}")
    print()

    output_dir = Path(args.output_dir)
    json_path = run_eval(args.task, args.n_episodes, args.batch_size,
                         args.task_ids, output_dir)
    print_results(json_path)


if __name__ == "__main__":
    main()
