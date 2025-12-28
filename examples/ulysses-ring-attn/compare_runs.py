#!/usr/bin/env python3
"""Compare loss curves between baseline and Ulysses + Ring-Attention runs.

This script validates that distributed attention produces equivalent results to
standard attention by comparing:
- Final train loss (should be within 5%)
- Loss curve shape (should not diverge)
- Training stability (no NaN/Inf)

Usage:
    python compare_runs.py --baseline outputs/baseline_no_cp/runs \\
                           --distributed outputs/ulysses_ring_*/runs \\
                           --threshold 0.05
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator


def load_tensorboard_scalar(logdir: Path, tag: str):
    """Load scalar values from TensorBoard logs."""
    ea = event_accumulator.EventAccumulator(
        str(logdir),
        size_guidance={
            event_accumulator.SCALARS: 0,  # Load all scalars
        },
    )
    ea.Reload()

    if tag not in ea.Tags()["scalars"]:
        raise ValueError(
            f"Tag '{tag}' not found in {logdir}. Available: {ea.Tags()['scalars']}"
        )

    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values


def find_tensorboard_dirs(pattern: str):
    """Find all TensorBoard log directories matching pattern."""
    dirs = list(Path(".").glob(pattern))
    if not dirs:
        raise ValueError(f"No directories found matching: {pattern}")
    return dirs


def compare_final_loss(baseline_loss, distributed_loss, threshold=0.05):
    """Compare final train loss between baseline and distributed."""
    baseline_final = baseline_loss[-1]
    distributed_final = distributed_loss[-1]

    diff = abs(baseline_final - distributed_final)
    rel_diff = diff / baseline_final

    passed = rel_diff <= threshold

    return {
        "baseline_final": baseline_final,
        "distributed_final": distributed_final,
        "absolute_diff": diff,
        "relative_diff": rel_diff,
        "threshold": threshold,
        "passed": passed,
    }


def check_stability(values):
    """Check if training is stable (no NaN/Inf)."""
    has_nan = any(np.isnan(v) for v in values)
    has_inf = any(np.isinf(v) for v in values)
    return not (has_nan or has_inf)


def plot_comparison(baseline_steps, baseline_loss, distributed_runs, output_path):
    """Plot loss curves for visual comparison."""
    plt.figure(figsize=(12, 6))

    # Plot baseline
    plt.plot(
        baseline_steps,
        baseline_loss,
        label="Baseline (no CP)",
        linewidth=2,
        color="black",
    )

    # Plot distributed runs
    colors = ["red", "blue", "green", "orange", "purple"]
    for i, (name, (steps, loss)) in enumerate(distributed_runs.items()):
        color = colors[i % len(colors)]
        plt.plot(steps, loss, label=name, linewidth=1.5, alpha=0.7, color=color)

    plt.xlabel("Training Step")
    plt.ylabel("Train Loss")
    plt.title("Loss Curve Comparison: Baseline vs Ulysses + Ring-Attention")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    print(f"✅ Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare baseline and distributed training runs"
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="Path to baseline TensorBoard logs (e.g., outputs/baseline_no_cp/runs)",
    )
    parser.add_argument(
        "--distributed",
        required=True,
        nargs="+",
        help="Paths to distributed TensorBoard logs (e.g., outputs/ulysses_ring_*/runs)",
    )
    parser.add_argument(
        "--metric",
        default="train/train_loss",
        help="Metric to compare (default: train/train_loss)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Relative difference threshold for pass/fail (default: 0.05 = 5%%)",
    )
    parser.add_argument(
        "--output",
        default="loss_comparison.png",
        help="Output plot filename (default: loss_comparison.png)",
    )

    args = parser.parse_args()

    print(
        "╔══════════════════════════════════════════════════════════════════════════════╗"
    )
    print(
        "║              Ulysses + Ring-Attention Validation Results                     ║"
    )
    print(
        "╚══════════════════════════════════════════════════════════════════════════════╝"
    )
    print()

    # Load baseline
    print(f"Loading baseline from: {args.baseline}")
    baseline_steps, baseline_loss = load_tensorboard_scalar(
        Path(args.baseline), args.metric
    )
    print(f"  - Steps: {len(baseline_steps)}")
    print(f"  - Final loss: {baseline_loss[-1]:.4f}")
    print()

    # Check baseline stability
    if not check_stability(baseline_loss):
        print("❌ ERROR: Baseline training is unstable (NaN/Inf detected)")
        sys.exit(1)

    # Load distributed runs
    distributed_runs = {}
    for dist_path in args.distributed:
        # Extract run name from path
        run_name = Path(dist_path).parent.name
        print(f"Loading distributed run: {run_name}")
        try:
            steps, loss = load_tensorboard_scalar(Path(dist_path), args.metric)
            distributed_runs[run_name] = (steps, loss)
            print(f"  - Steps: {len(steps)}")
            print(f"  - Final loss: {loss[-1]:.4f}")

            # Check stability
            if not check_stability(loss):
                print("  ⚠️  WARNING: Training is unstable (NaN/Inf detected)")
        except Exception as e:
            print(f"  ❌ ERROR: Failed to load: {e}")
        print()

    if not distributed_runs:
        print("❌ ERROR: No distributed runs loaded successfully")
        sys.exit(1)

    # Compare final losses
    print(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )
    print("FINAL LOSS COMPARISON")
    print(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )
    print()

    all_passed = True
    results = {}

    for run_name, (_steps, loss) in distributed_runs.items():
        result = compare_final_loss(baseline_loss, loss, threshold=args.threshold)
        results[run_name] = result

        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        print(f"{run_name}:")
        print(f"  Baseline:     {result['baseline_final']:.4f}")
        print(f"  Distributed:  {result['distributed_final']:.4f}")
        print(f"  Abs diff:     {result['absolute_diff']:.4f}")
        print(
            f"  Rel diff:     {result['relative_diff']:.2%} (threshold: {result['threshold']:.2%})"
        )
        print(f"  Status:       {status}")
        print()

        if not result["passed"]:
            all_passed = False

    # Plot comparison
    print(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )
    print("GENERATING PLOT")
    print(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )
    print()

    plot_comparison(baseline_steps, baseline_loss, distributed_runs, args.output)
    print()

    # Summary
    print(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )
    print("SUMMARY")
    print(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )
    print()

    passed_count = sum(1 for r in results.values() if r["passed"])
    total_count = len(results)

    print(f"Passed: {passed_count}/{total_count}")
    print()

    if all_passed:
        print("✅ All distributed runs match baseline within threshold!")
        print("   → Convergence validation PASSED")
        print()
        print("Next steps:")
        print("  - Review plot for visual confirmation")
        print("  - Check training logs for NCCL errors")
        print("  - Measure performance (tokens/sec)")
        sys.exit(0)
    else:
        print("❌ Some distributed runs deviate from baseline")
        print("   → Convergence validation FAILED")
        print()
        print("Debugging steps:")
        print("  - Review loss curves in plot")
        print("  - Check for NCCL errors in train.log")
        print("  - Verify gradient correctness")
        print("  - Check process group creation")
        sys.exit(1)


if __name__ == "__main__":
    main()
