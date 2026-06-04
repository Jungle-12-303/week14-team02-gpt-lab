# -*- coding: utf-8 -*-
"""Automatic research loop for [삼각 관계] attention variants."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VARIANTS = (
    "standard,"
    "triangular_v0,"
    "triangular_v1_pair_value,"
    "triangular_v2_scaled,"
    "triangular_v3_topk,"
    "triangular_v4_gated_pair,"
    "triangular_v5_gated_topk"
)

STRATEGY_DESCRIPTIONS = {
    "standard": "기존 pairwise attention 기준선.",
    "triangular_v0": "score[i,j,k]만 삼각 관계로 두고 value는 marginal 평균으로 유지하는 baseline.",
    "triangular_v1_pair_value": "score와 output value 모두 (j,k) 조합을 보게 만드는 pair-value 전략.",
    "triangular_v2_scaled": "pair-value에 learned logit scale과 entropy 진단을 더한 softmax 안정화 전략.",
    "triangular_v3_topk": "query별 top-k 후보 안에서만 (j,k)를 만들어 O(T*K^2)로 줄이는 비용 절감 전략.",
    "triangular_v4_gated_pair": "pair-value 경로를 learned gate로 천천히 여는 안정화 전략.",
    "triangular_v5_gated_topk": "gated pair-value와 top-k 후보 선택을 결합한 비용/안정성 절충 전략.",
}

TRIANGULAR_STRATEGY_ORDER = [
    "triangular_v2_scaled",
    "triangular_v4_gated_pair",
    "triangular_v1_pair_value",
    "triangular_v3_topk",
    "triangular_v5_gated_topk",
    "triangular_v0",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the [삼각 관계] automatic research loop.")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seeds", type=str, default="123")
    parser.add_argument("--variants", type=str, default=DEFAULT_VARIANTS)
    parser.add_argument("--synthetic-batch-size", type=int, default=64)
    parser.add_argument("--synthetic-eval-batches", type=int, default=20)
    parser.add_argument("--synthetic-facts", type=int, default=2)
    parser.add_argument(
        "--synthetic-loss-mode",
        choices=("answer", "answer_relation"),
        default="answer",
    )
    parser.add_argument("--synthetic-relation-loss-weight", type=float, default=0.5)
    parser.add_argument("--nsmc-context-length", type=int, default=32)
    parser.add_argument("--nsmc-batch-size", type=int, default=8)
    parser.add_argument("--nsmc-eval-batches", type=int, default=8)
    parser.add_argument("--nsmc-train-char-limit", type=int, default=1000000)
    parser.add_argument("--nsmc-val-char-limit", type=int, default=100000)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--max-selected-variants", type=int, default=2)
    parser.add_argument("--synthetic-success-margin", type=float, default=0.02)
    parser.add_argument("--nsmc-success-margin", type=float, default=0.0)
    parser.add_argument("--cost-ratio-warning", type=float, default=3.0)
    parser.add_argument("--stop-on-success", action="store_true")
    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument(
        "--report-md",
        type=Path,
        default=ROOT / "reports" / "triangular_relation_research.md",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports" / "auto_triangular_research",
    )
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"command failed with exit code {result.returncode}: {' '.join(command)}")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_variant_names(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def group_results(results: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(result["variant"], []).append(result)
    return grouped


def select_synthetic_candidates(
    synthetic_json: Path,
    max_selected: int,
) -> list[str]:
    data = load_json(synthetic_json)
    grouped = group_results(data["results"])
    standard_best = max(result["best_accuracy"] for result in grouped["standard"])
    triangular = [
        {
            "variant": variant,
            "best_accuracy": max(result["best_accuracy"] for result in results),
            "final_accuracy": sum(result["final_accuracy"] for result in results) / len(results),
            "ms_per_step": sum(result["ms_per_step"] for result in results) / len(results),
        }
        for variant, results in grouped.items()
        if variant != "standard"
    ]
    winners = [
        result
        for result in triangular
        if result["best_accuracy"] >= standard_best
    ]
    pool = winners if winners else triangular
    pool = sorted(
        pool,
        key=lambda result: (
            result["best_accuracy"],
            result["final_accuracy"],
            -result["ms_per_step"],
        ),
        reverse=True,
    )

    selected = []
    for result in pool:
        variant = result["variant"]
        if variant not in selected:
            selected.append(variant)
        if len(selected) >= max_selected:
            break
    return selected


def summarize_synthetic(path: Path) -> dict[str, Any]:
    data = load_json(path)
    grouped = group_results(data["results"])

    summary = {}
    for variant, results in grouped.items():
        summary[variant] = {
            "best_accuracy": max(item["best_accuracy"] for item in results),
            "final_accuracy_mean": sum(item["final_accuracy"] for item in results) / len(results),
            "final_loss_mean": sum(item["final_loss"] for item in results) / len(results),
            "ms_per_step_mean": sum(item["ms_per_step"] for item in results) / len(results),
        }
    return summary


def summarize_nsmc(path: Path) -> dict[str, Any]:
    data = load_json(path)
    grouped = group_results(data["results"])

    summary = {}
    for variant, results in grouped.items():
        summary[variant] = {
            "best_val_loss": min(item["best_val_loss"] for item in results),
            "final_val_loss_mean": sum(item["final_val_loss"] for item in results) / len(results),
            "final_train_loss_mean": sum(item["final_train_loss"] for item in results) / len(results),
            "ms_per_step_mean": sum(item["ms_per_step"] for item in results) / len(results),
        }
    return summary


def evaluate_success(
    args: argparse.Namespace,
    synthetic_summary: dict[str, Any],
    nsmc_summary: dict[str, Any],
) -> dict[str, Any]:
    standard_synthetic = synthetic_summary["standard"]
    standard_nsmc = nsmc_summary["standard"]
    standard_cost = standard_nsmc["ms_per_step_mean"]
    candidates = []
    for variant, syn_item in synthetic_summary.items():
        if variant == "standard" or variant not in nsmc_summary:
            continue
        nsmc_item = nsmc_summary[variant]
        synthetic_gain = syn_item["best_accuracy"] - standard_synthetic["best_accuracy"]
        nsmc_gain = standard_nsmc["best_val_loss"] - nsmc_item["best_val_loss"]
        cost_ratio = nsmc_item["ms_per_step_mean"] / standard_cost if standard_cost else float("inf")
        candidates.append(
            {
                "variant": variant,
                "synthetic_gain": synthetic_gain,
                "nsmc_gain": nsmc_gain,
                "cost_ratio": cost_ratio,
                "synthetic_pass": synthetic_gain >= args.synthetic_success_margin,
                "nsmc_pass": nsmc_gain >= args.nsmc_success_margin,
                "cost_warning": cost_ratio > args.cost_ratio_warning,
            }
        )

    candidates.sort(
        key=lambda item: (
            item["synthetic_pass"],
            item["nsmc_pass"],
            item["synthetic_gain"],
            item["nsmc_gain"],
            -item["cost_ratio"],
        ),
        reverse=True,
    )
    return {
        "satisfied": any(item["synthetic_pass"] and item["nsmc_pass"] for item in candidates),
        "candidates": candidates,
    }


def next_variant_pool(selected_variants: list[str], previous_variants: str) -> str:
    next_variants = ["standard"]
    for variant in selected_variants:
        if variant not in next_variants:
            next_variants.append(variant)
    for variant in TRIANGULAR_STRATEGY_ORDER:
        if variant not in next_variants:
            next_variants.append(variant)
        if len(next_variants) >= 1 + max(3, len(selected_variants) + 1):
            break
    for variant in parse_variant_names(previous_variants):
        if variant not in next_variants and len(next_variants) < 5:
            next_variants.append(variant)
    return ",".join(next_variants)


def append_loop_report(
    args: argparse.Namespace,
    run_id: str,
    iteration: int,
    variants: str,
    synthetic_json: Path,
    nsmc_json: Path,
    selected_variants: list[str],
) -> None:
    synthetic_summary = summarize_synthetic(synthetic_json)
    nsmc_summary = summarize_nsmc(nsmc_json)
    success = evaluate_success(args, synthetic_summary, nsmc_summary)

    lines = [
        "",
        f"## Auto Research Loop: {run_id} iteration {iteration}",
        "",
        "### Loop Policy",
        "",
        f"- steps: `{args.steps}`",
        f"- eval_every: `{args.eval_every}`",
        f"- seeds: `{args.seeds}`",
        f"- synthetic facts: `{args.synthetic_facts}`",
        f"- synthetic loss mode: `{args.synthetic_loss_mode}`",
        "- 700 step 이후 변화가 보이도록 100 step 단위로 history를 기록한다.",
        "- synthetic task에서 표준 attention 이상인 [삼각 관계] variant를 NSMC 후보로 선발한다.",
        f"- 만족 기준: synthetic best accuracy가 standard보다 `{args.synthetic_success_margin:.3f}` 이상 높고, NSMC best val loss가 standard보다 `{args.nsmc_success_margin:.4f}` 이상 개선되어야 한다.",
        f"- 비용 경고 기준: NSMC ms/step이 standard의 `{args.cost_ratio_warning:.1f}`배를 넘으면 비용 병목으로 표시한다.",
        "",
        "### Strategy Under Test",
        "",
    ]
    for variant in parse_variant_names(variants):
        if variant in STRATEGY_DESCRIPTIONS:
            lines.append(f"- `{variant}`: {STRATEGY_DESCRIPTIONS[variant]}")

    lines.extend(
        [
            "",
            "### Synthetic Screening Summary",
            "",
            "| variant | best accuracy | final accuracy mean | final loss mean | ms/step mean |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for variant, item in sorted(synthetic_summary.items()):
        lines.append(
            f"| {variant} | {item['best_accuracy']:.3f} | "
            f"{item['final_accuracy_mean']:.3f} | {item['final_loss_mean']:.4f} | "
            f"{item['ms_per_step_mean']:.2f} |"
        )

    lines.extend(
        [
            "",
            "### Selected Variants",
            "",
            "- " + ", ".join(selected_variants),
            "",
            "### NSMC Validation Summary",
            "",
            "| variant | best val loss | final val loss mean | final train loss mean | ms/step mean |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for variant, item in sorted(nsmc_summary.items()):
        lines.append(
            f"| {variant} | {item['best_val_loss']:.4f} | "
            f"{item['final_val_loss_mean']:.4f} | {item['final_train_loss_mean']:.4f} | "
            f"{item['ms_per_step_mean']:.2f} |"
        )

    lines.extend(
        [
            "",
            "### Success Evaluation",
            "",
            "| variant | synthetic gain | NSMC val gain | cost ratio | status |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for item in success["candidates"]:
        statuses = []
        statuses.append("synthetic pass" if item["synthetic_pass"] else "synthetic fail")
        statuses.append("NSMC pass" if item["nsmc_pass"] else "NSMC fail")
        if item["cost_warning"]:
            statuses.append("cost warning")
        lines.append(
            f"| {item['variant']} | {item['synthetic_gain']:.3f} | "
            f"{item['nsmc_gain']:.4f} | {item['cost_ratio']:.2f}x | "
            f"{', '.join(statuses)} |"
        )

    lines.extend(
        [
            "",
            "### Next Automatic Decision",
            "",
            "- 만족 기준을 통과한 variant가 있으면 multi-seed 장기 검증 후보로 유지한다.",
            "- synthetic gain은 있으나 기준에 못 미치면 더 강한 relation task 또는 gated pair/top-k 변형을 다음 후보군에 넣는다.",
            "- NSMC가 앞서지만 비용 경고가 뜨면 top-k/gated-topk 최적화를 우선한다.",
            f"- loop status: `{'satisfied' if success['satisfied'] else 'not_satisfied'}`",
            "",
            f"- synthetic JSON: `{synthetic_json}`",
            f"- NSMC JSON: `{nsmc_json}`",
            "",
        ]
    )
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    with args.report_md.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_iteration(args: argparse.Namespace, run_id: str, iteration: int, variants: str) -> tuple[str, bool]:
    iteration_dir = args.output_dir / run_id / f"iter_{iteration:02d}"
    iteration_dir.mkdir(parents=True, exist_ok=True)

    synthetic_json = iteration_dir / "synthetic.json"
    synthetic_csv = iteration_dir / "synthetic.csv"
    synthetic_plot = iteration_dir / "synthetic_accuracy.png"
    synthetic_command = [
        sys.executable,
        "scripts/run_triangular_synthetic.py",
        "--steps",
        str(args.steps),
        "--eval-every",
        str(args.eval_every),
        "--eval-batches",
        str(args.synthetic_eval_batches),
        "--batch-size",
        str(args.synthetic_batch_size),
        "--facts",
        str(args.synthetic_facts),
        "--loss-mode",
        args.synthetic_loss_mode,
        "--relation-loss-weight",
        str(args.synthetic_relation_loss_weight),
        "--variants",
        variants,
        "--seeds",
        args.seeds,
        "--top-k",
        str(args.top_k),
        "--experiment-name",
        f"Auto {run_id} synthetic iteration {iteration}",
        "--output-json",
        str(synthetic_json),
        "--csv-path",
        str(synthetic_csv),
        "--plot-path",
        str(synthetic_plot),
        "--report-md",
        str(args.report_md),
        "--append-report",
    ]
    run_command(synthetic_command)

    selected = select_synthetic_candidates(synthetic_json, args.max_selected_variants)
    nsmc_variants = ["standard"] + selected
    nsmc_variant_arg = ",".join(dict.fromkeys(nsmc_variants))

    nsmc_json = iteration_dir / "nsmc.json"
    nsmc_csv = iteration_dir / "nsmc.csv"
    nsmc_plot = iteration_dir / "nsmc_loss.png"
    nsmc_command = [
        sys.executable,
        "scripts/compare_attention_models.py",
        "--context-length",
        str(args.nsmc_context_length),
        "--batch-size",
        str(args.nsmc_batch_size),
        "--steps",
        str(args.steps),
        "--eval-every",
        str(args.eval_every),
        "--eval-batches",
        str(args.nsmc_eval_batches),
        "--train-char-limit",
        str(args.nsmc_train_char_limit),
        "--val-char-limit",
        str(args.nsmc_val_char_limit),
        "--variants",
        nsmc_variant_arg,
        "--seeds",
        args.seeds,
        "--top-k",
        str(args.top_k),
        "--experiment-name",
        f"Auto {run_id} NSMC iteration {iteration}",
        "--output-json",
        str(nsmc_json),
        "--csv-path",
        str(nsmc_csv),
        "--plot-path",
        str(nsmc_plot),
        "--report-md",
        str(args.report_md),
        "--append-report",
    ]
    run_command(nsmc_command)
    append_loop_report(args, run_id, iteration, variants, synthetic_json, nsmc_json, selected)
    success = evaluate_success(
        args,
        summarize_synthetic(synthetic_json),
        summarize_nsmc(nsmc_json),
    )
    return next_variant_pool(selected, variants), bool(success["satisfied"])


def main() -> None:
    args = parse_args()
    if args.iterations <= 0:
        raise ValueError("iterations must be positive")
    if args.steps < 1000 and not args.allow_short:
        raise ValueError("automatic research loop requires --steps >= 1000")
    if args.eval_every <= 0:
        raise ValueError("eval-every must be positive")

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    variants = args.variants
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for iteration in range(1, args.iterations + 1):
        variants, satisfied = run_iteration(args, run_id, iteration, variants)
        if satisfied and args.stop_on_success:
            print(f"\nStopping early: success criteria satisfied at iteration {iteration}")
            break

    print(f"\nAuto research loop complete: {run_id}")
    print(f"Artifacts: {args.output_dir / run_id}")
    print(f"Report: {args.report_md}")


if __name__ == "__main__":
    main()
