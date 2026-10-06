import argparse
from dataclasses import asdict
import csv
import gc
import json
from pathlib import Path

import numpy as np

from evaluate import load_policy_stack, parse_bool, run_episode


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets" / "smolvla_cube_tray"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"
DEFAULT_FINETUNED_POLICY = PROJECT_ROOT / "models" / "smolvla_cube_tray_finetuned"
MODEL_SPECS = (
    ("base", "base_policy_path"),
    ("finetuned", "finetuned_policy_path"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare base and fine-tuned SmolVLA policies on the same independent "
            "CubeTrayEnv seeds."
        )
    )
    parser.add_argument("--base-policy-path", default="lerobot/smolvla_base")
    parser.add_argument(
        "--finetuned-policy-path",
        default=str(DEFAULT_FINETUNED_POLICY),
    )
    parser.add_argument("--dataset-repo-id", default="local/smolvla_cube_tray")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--start-seed", type=int, default=100)
    parser.add_argument("--num-seeds", type=int, default=50)
    parser.add_argument("--n-action-steps", type=int, default=10)
    parser.add_argument("--max-episode-seconds", type=float, default=30.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-name", default="smolvla_base_vs_finetuned")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--video", type=parse_bool, default=False)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Resolve models and tokenizers only from the Hugging Face cache.",
    )
    return parser.parse_args()


def validate_args(args):
    if args.start_seed < 0:
        raise ValueError("--start-seed must be non-negative.")
    if args.num_seeds <= 0:
        raise ValueError("--num-seeds must be positive.")
    if args.n_action_steps <= 0 or args.n_action_steps > 50:
        raise ValueError("--n-action-steps must be in [1, 50].")
    if args.max_episode_seconds <= 0:
        raise ValueError("--max-episode-seconds must be positive.")
    if not (args.dataset_root / "meta" / "info.json").exists():
        raise FileNotFoundError(
            f"LeRobot dataset not found at {args.dataset_root}."
        )


def make_policy_args(args, label, policy_path):
    return argparse.Namespace(
        policy_path=str(policy_path),
        dataset_repo_id=args.dataset_repo_id,
        dataset_root=args.dataset_root,
        seed=args.start_seed,
        n_action_steps=args.n_action_steps,
        max_episode_seconds=args.max_episode_seconds,
        device=args.device,
        run_name=label,
        video=args.video,
        offline=args.offline,
        verbose=False,
    )


def write_episode_metrics(result, episode_dir):
    metrics_path = episode_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(asdict(result), indent=2) + "\n",
        encoding="utf-8",
    )


def clear_device_cache():
    gc.collect()

    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_model(args, experiment_dir, label, policy_path, seeds):
    policy_args = make_policy_args(args, label, policy_path)
    print(f"\nLoading {label} policy: {policy_path}")
    policy, preprocessor, postprocessor, policy_source = load_policy_stack(policy_args)
    results = []

    try:
        for episode_number, seed in enumerate(seeds, start=1):
            policy_args.seed = seed
            episode_dir = experiment_dir / label / f"seed_{seed}"
            result = run_episode(
                policy_args,
                policy,
                preprocessor,
                postprocessor,
                policy_source,
                episode_dir,
            )
            write_episode_metrics(result, episode_dir)
            results.append(result)
            print(
                f"[{label} {episode_number:02d}/{len(seeds):02d}] "
                f"seed={seed} success={result.success} "
                f"time={result.simulated_seconds:.2f}s "
                f"latency={np.mean(result.inference_latencies_ms):.2f}ms "
                f"drop={result.cube_dropped} collision={result.collision}"
            )
    finally:
        policy = None
        preprocessor = None
        postprocessor = None
        clear_device_cache()

    return results


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def aggregate_results(results):
    inference_latencies = [
        latency
        for result in results
        for latency in result.inference_latencies_ms
    ]
    action_latencies = [
        latency
        for result in results
        for latency in result.action_latencies_ms
    ]
    successful_times = [
        result.simulated_seconds for result in results if result.success
    ]
    episode_count = len(results)
    success_count = sum(result.success for result in results)
    drop_count = sum(result.cube_dropped for result in results)
    collision_count = sum(result.collision for result in results)
    return {
        "episode_count": episode_count,
        "success_count": success_count,
        "success_rate": success_count / episode_count,
        "drop_count": drop_count,
        "drop_rate": drop_count / episode_count,
        "collision_count": collision_count,
        "collision_rate": collision_count / episode_count,
        "mean_task_duration_seconds": float(
            np.mean([result.simulated_seconds for result in results])
        ),
        "median_task_duration_seconds": float(
            np.median([result.simulated_seconds for result in results])
        ),
        "mean_successful_task_duration_seconds": (
            float(np.mean(successful_times)) if successful_times else None
        ),
        "mean_episode_wall_seconds": float(
            np.mean([result.wall_seconds for result in results])
        ),
        "total_episode_wall_seconds": float(
            np.sum([result.wall_seconds for result in results])
        ),
        "mean_inference_latency_ms": float(np.mean(inference_latencies)),
        "median_inference_latency_ms": float(np.median(inference_latencies)),
        "p95_inference_latency_ms": percentile(inference_latencies, 95),
        "mean_action_latency_ms": float(np.mean(action_latencies)),
        "p95_action_latency_ms": percentile(action_latencies, 95),
        "mean_collision_events": float(
            np.mean([result.collision_events for result in results])
        ),
        "mean_collision_sim_steps": float(
            np.mean([result.collision_sim_steps for result in results])
        ),
    }


def make_deltas(base_summary, finetuned_summary):
    metric_names = (
        "success_rate",
        "drop_rate",
        "collision_rate",
        "mean_task_duration_seconds",
        "mean_successful_task_duration_seconds",
        "mean_episode_wall_seconds",
        "mean_inference_latency_ms",
        "p95_inference_latency_ms",
    )
    deltas = {}
    for metric_name in metric_names:
        base_value = base_summary[metric_name]
        finetuned_value = finetuned_summary[metric_name]
        deltas[metric_name] = (
            None
            if base_value is None or finetuned_value is None
            else finetuned_value - base_value
        )
    return deltas


def write_episode_csv(results_by_model, output_path):
    fieldnames = [
        "model",
        "seed",
        "success",
        "termination_reason",
        "simulated_seconds",
        "wall_seconds",
        "mean_inference_latency_ms",
        "p95_inference_latency_ms",
        "cube_lifted",
        "cube_dropped",
        "collision",
        "collision_events",
        "collision_sim_steps",
        "tray_xy_error",
        "control_steps",
        "action_chunks",
        "clipped_action_steps",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for label, results in results_by_model.items():
            for result in results:
                writer.writerow(
                    {
                        "model": label,
                        "seed": result.seed,
                        "success": result.success,
                        "termination_reason": result.termination_reason,
                        "simulated_seconds": result.simulated_seconds,
                        "wall_seconds": result.wall_seconds,
                        "mean_inference_latency_ms": np.mean(
                            result.inference_latencies_ms
                        ),
                        "p95_inference_latency_ms": percentile(
                            result.inference_latencies_ms,
                            95,
                        ),
                        "cube_lifted": result.cube_lifted,
                        "cube_dropped": result.cube_dropped,
                        "collision": result.collision,
                        "collision_events": result.collision_events,
                        "collision_sim_steps": result.collision_sim_steps,
                        "tray_xy_error": result.tray_xy_error,
                        "control_steps": result.control_steps,
                        "action_chunks": result.action_chunks,
                        "clipped_action_steps": result.clipped_action_steps,
                    }
                )


def print_summary(summary_by_model):
    print("\n=== SmolVLA comparison ===")
    print(
        f"{'model':<12} {'success':>9} {'task_s':>9} {'latency_ms':>12} "
        f"{'drop':>9} {'collision':>11}"
    )
    for label in ("base", "finetuned"):
        summary = summary_by_model[label]
        print(
            f"{label:<12} "
            f"{summary['success_rate']:>8.1%} "
            f"{summary['mean_task_duration_seconds']:>9.3f} "
            f"{summary['mean_inference_latency_ms']:>12.2f} "
            f"{summary['drop_rate']:>8.1%} "
            f"{summary['collision_rate']:>10.1%}"
        )


def main():
    args = parse_args()
    validate_args(args)
    seeds = list(range(args.start_seed, args.start_seed + args.num_seeds))
    experiment_dir = args.output_dir / (
        f"{args.run_name}_seeds_{seeds[0]}_{seeds[-1]}"
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)

    results_by_model = {}
    for label, path_attribute in MODEL_SPECS:
        results_by_model[label] = run_model(
            args,
            experiment_dir,
            label,
            getattr(args, path_attribute),
            seeds,
        )

    summary_by_model = {
        label: aggregate_results(results)
        for label, results in results_by_model.items()
    }
    summary = {
        "experiment": {
            "seed_start": seeds[0],
            "seed_end": seeds[-1],
            "num_seeds": len(seeds),
            "seeds": seeds,
            "base_policy_path": args.base_policy_path,
            "finetuned_policy_path": args.finetuned_policy_path,
            "dataset_repo_id": args.dataset_repo_id,
            "dataset_root": str(args.dataset_root),
            "n_action_steps": args.n_action_steps,
            "max_episode_seconds": args.max_episode_seconds,
            "device": args.device,
            "video": args.video,
        },
        "metric_definitions": {
            "success_rate": "Episodes with the cube stable in the tray for 2 seconds / all episodes.",
            "task_duration": "Simulated seconds until success or timeout.",
            "inference_latency": "CUDA-synchronized preprocessing, policy forward/chunk generation, postprocessing, and CPU transfer latency.",
            "drop_rate": "Episodes where a cube lifted at least 5 cm later contacts the table or floor / all episodes.",
            "collision_rate": "Episodes with a new robot contact against the table, tray, or floor after reset / all episodes; contacts already present after reset are excluded.",
        },
        "models": summary_by_model,
        "finetuned_minus_base": make_deltas(
            summary_by_model["base"],
            summary_by_model["finetuned"],
        ),
    }

    summary_path = experiment_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    csv_path = experiment_dir / "episodes.csv"
    write_episode_csv(results_by_model, csv_path)
    print_summary(summary_by_model)
    print(f"summary:  {summary_path}")
    print(f"episodes: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
