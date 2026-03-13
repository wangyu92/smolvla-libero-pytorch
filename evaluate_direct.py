#!/usr/bin/env python3
"""Evaluate SmolVLA on LIBERO benchmarks without lerobot-eval CLI.

Directly loads the SmolVLA model and LIBERO MuJoCo environments,
runs inference loops, and reports per-task success rates.
Supports multi-GPU parallel evaluation.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp

from lerobot.envs.libero import (
    TASK_SUITE_MAX_STEPS,
    get_libero_dummy_action,
    get_task_init_states,
)
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.processor.env_processor import LiberoProcessorStep
from lerobot.processor.pipeline import PolicyProcessorPipeline
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

MODEL_ID = "HuggingFaceVLA/smolvla_libero"
VALID_TASKS = ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]

CAMERA_NAMES = ["agentview_image", "robot0_eye_in_hand_image"]
CAMERA_NAME_MAP = {
    "agentview_image": "image",
    "robot0_eye_in_hand_image": "image2",
}

NUM_STEPS_WAIT = 10


def print_gpu_info():
    n = torch.cuda.device_count()
    if n == 0:
        print("WARNING: No CUDA GPUs detected!")
        return
    print(f"Detected {n} GPU(s):")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / (1024**3)
        print(f"  [{i}] {props.name} — {mem_gb:.1f} GB")
    print()


def load_model(model_id: str, device: str) -> SmolVLAPolicy:
    """Load SmolVLA policy from HuggingFace Hub."""
    print(f"[{device}] Loading model: {model_id} ...")
    policy = SmolVLAPolicy.from_pretrained(model_id)
    policy.to(device)
    policy.eval()
    print(f"[{device}] Model loaded.")
    return policy


def create_env(task_suite, task_id: int, img_size: int = 256) -> OffScreenRenderEnv:
    """Create a LIBERO OffScreenRenderEnv for a specific task."""
    task = task_suite.get_task(task_id)
    bddl_file = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=img_size,
        camera_widths=img_size,
    )
    env.reset()
    return env


def format_raw_obs(raw_obs: dict) -> dict:
    """Convert raw robosuite observation to the format expected by preprocess_observation()."""
    images = {}
    for cam_name in CAMERA_NAMES:
        images[CAMERA_NAME_MAP[cam_name]] = raw_obs[cam_name]

    return {
        "pixels": images,
        "robot_state": {
            "eef": {
                "pos": raw_obs["robot0_eef_pos"][np.newaxis],
                "quat": raw_obs["robot0_eef_quat"][np.newaxis],
            },
            "gripper": {
                "qpos": raw_obs["robot0_gripper_qpos"][np.newaxis],
            },
        },
    }


def run_episode(
    policy: SmolVLAPolicy,
    env: OffScreenRenderEnv,
    task_instruction: str,
    init_state: np.ndarray,
    preprocessor,
    postprocessor,
    env_preprocessor,
    max_steps: int,
    control_mode: str = "relative",
) -> bool:
    """Run a single evaluation episode. Returns True if task succeeded."""
    env.reset()
    raw_obs = env.set_init_state(init_state)

    for _ in range(NUM_STEPS_WAIT):
        raw_obs, _, _, _ = env.step(get_libero_dummy_action())

    for robot in env.robots:
        robot.controller.use_delta = (control_mode == "relative")

    policy.reset()

    for step in range(max_steps):
        formatted_obs = format_raw_obs(raw_obs)
        obs = preprocess_observation(formatted_obs)
        obs["task"] = [task_instruction]
        obs = env_preprocessor(obs)
        obs = preprocessor(obs)

        with torch.inference_mode():
            action = policy.select_action(obs)

        action = postprocessor(action)
        action_np = action.squeeze(0).cpu().numpy()
        raw_obs, reward, done, info = env.step(action_np)

        if env.check_success():
            return True
        if done:
            break

    return False


def evaluate_task(
    policy: SmolVLAPolicy,
    task_suite,
    task_suite_name: str,
    task_id: int,
    n_episodes: int,
    preprocessor,
    postprocessor,
    env_preprocessor,
    img_size: int = 256,
    control_mode: str = "relative",
    device_tag: str = "",
) -> dict:
    """Evaluate a single task across multiple episodes."""
    task = task_suite.get_task(task_id)
    instruction = task.language
    max_steps = TASK_SUITE_MAX_STEPS.get(task_suite_name, 500)

    tag = f"[{device_tag}] " if device_tag else ""
    print(f"{tag}{task_suite_name}/Task {task_id}: {instruction}")
    print(f"{tag}  Max steps: {max_steps}, Episodes: {n_episodes}")

    env = create_env(task_suite, task_id, img_size)
    init_states = get_task_init_states(task_suite, task_id)

    successes = []
    for ep in range(n_episodes):
        init_state = init_states[ep % len(init_states)]
        t0 = time.time()
        success = run_episode(
            policy=policy,
            env=env,
            task_instruction=instruction,
            init_state=init_state,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            env_preprocessor=env_preprocessor,
            max_steps=max_steps,
            control_mode=control_mode,
        )
        elapsed = time.time() - t0
        successes.append(success)
        status = "SUCCESS" if success else "FAIL"
        print(f"{tag}  Episode {ep + 1}/{n_episodes}: {status} ({elapsed:.1f}s)")

    env.close()

    rate = sum(successes) / len(successes) if successes else 0.0
    print(f"{tag}  Success rate: {rate:.1%}\n")

    return {
        "task_id": task_id,
        "suite_name": task_suite_name,
        "instruction": instruction,
        "successes": successes,
        "success_rate": rate,
    }


# ─── Worker for multi-GPU parallel evaluation ───

def _worker_fn(gpu_id: int, work_items: list, args, result_queue: mp.Queue):
    """Worker process: loads model on assigned GPU, evaluates all assigned tasks."""
    device = f"cuda:{gpu_id}"
    try:
        policy = load_model(args.model_id, device)

        preprocessor_overrides = {"device_processor": {"device": device}}
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=args.model_id,
            preprocessor_overrides=preprocessor_overrides,
        )
        env_preprocessor = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])

        benchmark_dict = benchmark.get_benchmark_dict()

        for suite_name, task_id in work_items:
            task_suite = benchmark_dict[suite_name]()
            result = evaluate_task(
                policy=policy,
                task_suite=task_suite,
                task_suite_name=suite_name,
                task_id=task_id,
                n_episodes=args.n_episodes,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                env_preprocessor=env_preprocessor,
                img_size=args.img_size,
                control_mode=args.control_mode,
                device_tag=f"GPU{gpu_id}",
            )
            result_queue.put(result)
    except Exception as e:
        import traceback
        print(f"[GPU{gpu_id}] Worker failed: {e}")
        traceback.print_exc()
        # Put error markers for remaining items
        for suite_name, task_id in work_items:
            result_queue.put({
                "task_id": task_id,
                "suite_name": suite_name,
                "instruction": "ERROR",
                "successes": [False] * args.n_episodes,
                "success_rate": 0.0,
                "error": str(e),
            })


# ─── Output formatting ───

def print_suite_results(results: list[dict], suite_name: str):
    """Print a summary table for a single suite."""
    print(f"\n{'Task':<8} {'Instruction':<55} {'Success Rate':>12}")
    print("-" * 77)
    for r in results:
        n_success = sum(r["successes"])
        n_total = len(r["successes"])
        print(
            f"Task {r['task_id']:<3} {r['instruction']:<55} "
            f"{r['success_rate']:>7.1%} ({n_success}/{n_total})"
        )

    avg = sum(r["success_rate"] for r in results) / len(results) if results else 0.0
    print("-" * 77)
    print(f"{'[' + suite_name + '] AVERAGE':<64} {avg:>7.1%}")
    return avg


def print_results(all_suite_results: dict):
    """Print per-suite tables and overall summary."""
    print("\n" + "=" * 100)
    print("EVALUATION RESULTS")
    print("=" * 100)

    for suite_name, suite_data in all_suite_results.items():
        print(f"\n--- {suite_name} ---")
        print_suite_results(suite_data["per_task"], suite_name)

    if len(all_suite_results) > 1:
        print("\n" + "=" * 100)
        print("OVERALL (all suites)")
        print("=" * 100)
        all_rates = []
        for suite_data in all_suite_results.values():
            all_rates.append(suite_data["avg_success_rate"])
        overall_avg = sum(all_rates) / len(all_rates) if all_rates else 0.0
        for suite_name, suite_data in all_suite_results.items():
            print(f"  {suite_name:<20} {suite_data['avg_success_rate']:>7.1%}")
        print(f"  {'OVERALL':<20} {overall_avg:>7.1%}")
        print("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SmolVLA on LIBERO (direct, without lerobot-eval)"
    )
    parser.add_argument(
        "--task", default="all", choices=VALID_TASKS + ["all"],
        help="LIBERO task suite or 'all' to run every suite (default: all)",
    )
    parser.add_argument(
        "--n-episodes", type=int, default=10,
        help="Number of evaluation episodes per task (default: 10)",
    )
    parser.add_argument(
        "--task-ids", type=int, nargs="+", default=None,
        help="Specific task IDs to evaluate (default: all)",
    )
    parser.add_argument(
        "--model-id", type=str, default=MODEL_ID,
        help=f"HuggingFace model ID (default: {MODEL_ID})",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (default: cuda if available, else cpu). Ignored when --num-workers > 1.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="Number of parallel GPU workers (default: 0 = auto-detect GPU count, 1 = sequential)",
    )
    parser.add_argument(
        "--img-size", type=int, default=256,
        help="Camera image size (default: 256)",
    )
    parser.add_argument(
        "--control-mode", type=str, default="relative", choices=["relative", "absolute"],
        help="Robot control mode (default: relative)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./results",
        help="Directory to save results JSON",
    )
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    n_gpus = torch.cuda.device_count()
    if args.num_workers == 0:
        args.num_workers = max(n_gpus, 1)

    print_gpu_info()

    print(f"Model:        {args.model_id}")
    print(f"Task suite:   {args.task}")
    print(f"Episodes:     {args.n_episodes}")
    print(f"Workers:      {args.num_workers}")
    print(f"Control mode: {args.control_mode}")
    if args.task_ids:
        print(f"Task IDs:     {args.task_ids}")
    print()

    # Determine which suites/tasks to run
    suites = VALID_TASKS if args.task == "all" else [args.task]
    benchmark_dict = benchmark.get_benchmark_dict()

    # Build full work list: [(suite_name, task_id), ...]
    work_items = []
    for suite_name in suites:
        task_suite = benchmark_dict[suite_name]()
        if args.task_ids is not None:
            task_ids = sorted(args.task_ids)
            for tid in task_ids:
                if tid < 0 or tid >= task_suite.n_tasks:
                    print(f"ERROR: task_id {tid} out of range [0, {task_suite.n_tasks - 1}] for {suite_name}")
                    return
        else:
            task_ids = list(range(task_suite.n_tasks))
        for tid in task_ids:
            work_items.append((suite_name, tid))

    total_tasks = len(work_items)
    total_episodes = total_tasks * args.n_episodes
    print(f"Total: {total_tasks} tasks, {total_episodes} episodes across {len(suites)} suite(s)\n")

    start_time = time.time()

    if args.num_workers <= 1:
        # ─── Sequential mode ───
        policy = load_model(args.model_id, args.device)
        preprocessor_overrides = {"device_processor": {"device": args.device}}
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=args.model_id,
            preprocessor_overrides=preprocessor_overrides,
        )
        env_preprocessor = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])

        all_results = []
        for suite_name, task_id in work_items:
            task_suite = benchmark_dict[suite_name]()
            result = evaluate_task(
                policy=policy,
                task_suite=task_suite,
                task_suite_name=suite_name,
                task_id=task_id,
                n_episodes=args.n_episodes,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                env_preprocessor=env_preprocessor,
                img_size=args.img_size,
                control_mode=args.control_mode,
            )
            all_results.append(result)
    else:
        # ─── Multi-GPU parallel mode ───
        mp.set_start_method("spawn", force=True)

        num_workers = min(args.num_workers, n_gpus, total_tasks)
        print(f"Launching {num_workers} parallel workers...\n")

        # Round-robin distribute work items to workers
        worker_items = [[] for _ in range(num_workers)]
        for i, item in enumerate(work_items):
            worker_items[i % num_workers].append(item)

        result_queue = mp.Queue()
        processes = []
        for w in range(num_workers):
            gpu_id = w % n_gpus
            p = mp.Process(
                target=_worker_fn,
                args=(gpu_id, worker_items[w], args, result_queue),
            )
            p.start()
            processes.append(p)

        # Collect results
        all_results = []
        for _ in range(total_tasks):
            result = result_queue.get()
            all_results.append(result)

        for p in processes:
            p.join()

    total_time = time.time() - start_time
    print(f"\nTotal evaluation time: {total_time:.1f}s")

    # Organize results by suite
    all_suite_results = {}
    for suite_name in suites:
        suite_results = sorted(
            [r for r in all_results if r["suite_name"] == suite_name],
            key=lambda r: r["task_id"],
        )
        avg_rate = sum(r["success_rate"] for r in suite_results) / len(suite_results) if suite_results else 0.0
        all_suite_results[suite_name] = {
            "per_task": suite_results,
            "avg_success_rate": avg_rate,
            "n_tasks": len(suite_results),
            "n_episodes_total": sum(len(r["successes"]) for r in suite_results),
        }

    # Print summary
    print_results(all_suite_results)

    # Build output JSON
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"eval_direct_{args.task}.json"

    per_suite_json = {}
    for suite_name, suite_data in all_suite_results.items():
        per_suite_json[suite_name] = {
            "per_task": [
                {
                    "task_id": r["task_id"],
                    "instruction": r["instruction"],
                    "successes": r["successes"],
                    "success_rate": r["success_rate"],
                }
                for r in suite_data["per_task"]
            ],
            "avg_success_rate": suite_data["avg_success_rate"],
            "n_tasks": suite_data["n_tasks"],
            "n_episodes_total": suite_data["n_episodes_total"],
        }

    all_avg_rates = [s["avg_success_rate"] for s in all_suite_results.values()]
    overall_avg = sum(all_avg_rates) / len(all_avg_rates) if all_avg_rates else 0.0

    output_data = {
        "model_id": args.model_id,
        "n_episodes": args.n_episodes,
        "num_workers": args.num_workers,
        "total_time_s": total_time,
        "per_suite": per_suite_json,
        "overall": {
            "avg_success_rate": overall_avg,
            "pc_success": overall_avg * 100,
            "n_suites": len(all_suite_results),
            "n_tasks_total": sum(s["n_tasks"] for s in all_suite_results.values()),
            "n_episodes_total": sum(s["n_episodes_total"] for s in all_suite_results.values()),
        },
    }
    output_path.write_text(json.dumps(output_data, indent=2))
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
