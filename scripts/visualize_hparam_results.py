#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Visualize train/validation loss gaps from hyperparameter search results."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_CSV = REPO_ROOT / "results" / "hparam_search" / "summary.csv"
DEFAULT_HISTORY_CSV = REPO_ROOT / "results" / "hparam_search" / "history.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "hparam_search" / "figures"

HPARAM_KEYS = [
    "batch_size",
    "drop_rate",
    "learning_rate",
    "context_length",
    "n_layers",
    "emb_dim",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create matplotlib charts for hparam train/val loss gaps."
    )
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--history-csv", type=Path, default=DEFAULT_HISTORY_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-n", type=int, default=15)
    return parser.parse_args()


def parse_number(value: str) -> int | float:
    number = float(value)
    return int(number) if number.is_integer() else number


def load_rows(summary_csv: Path) -> list[dict[str, Any]]:
    if not summary_csv.exists():
        raise FileNotFoundError(f"summary CSV not found: {summary_csv}")

    rows: list[dict[str, Any]] = []
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = set(HPARAM_KEYS + ["run_id", "final_train_loss", "final_val_loss"])
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"summary CSV is missing columns: {sorted(missing)}")

        for row in reader:
            parsed = dict(row)
            for key in HPARAM_KEYS:
                parsed[key] = parse_number(parsed[key])
            for key in ["final_train_loss", "final_val_loss", "best_val_loss"]:
                parsed[key] = float(parsed[key])
            parsed["loss_gap"] = parsed["final_val_loss"] - parsed["final_train_loss"]
            rows.append(parsed)

    if not rows:
        raise RuntimeError(f"summary CSV has no rows: {summary_csv}")
    return rows


def load_history_rows(history_csv: Path) -> list[dict[str, Any]]:
    if not history_csv.exists():
        return []

    rows: list[dict[str, Any]] = []
    with history_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = set(HPARAM_KEYS + ["run_id", "epoch", "train_loss", "val_loss"])
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"history CSV is missing columns: {sorted(missing)}")

        for row in reader:
            parsed = dict(row)
            for key in HPARAM_KEYS:
                parsed[key] = parse_number(parsed[key])
            parsed["epoch"] = int(float(parsed["epoch"]))
            parsed["step"] = int(float(parsed["step"])) if parsed.get("step") else 0
            parsed["tokens_seen"] = int(float(parsed["tokens_seen"])) if parsed.get("tokens_seen") else 0
            parsed["train_loss"] = float(parsed["train_loss"])
            parsed["val_loss"] = float(parsed["val_loss"])
            parsed["loss_gap"] = float(parsed["loss_gap"])
            rows.append(parsed)

    return rows


def hparam_label(row: dict[str, Any]) -> str:
    return (
        f"bs={row['batch_size']} dr={row['drop_rate']} "
        f"lr={row['learning_rate']:.0e} ctx={row['context_length']} "
        f"L={row['n_layers']} d={row['emb_dim']}"
    )


def fmt_value(value: Any) -> str:
    if isinstance(value, float) and value < 0.01:
        return f"{value:.0e}"
    return f"{value:g}" if isinstance(value, float) else str(value)


def mean_by(rows: list[dict[str, Any]], key: str, value_key: str) -> list[tuple[Any, float, int]]:
    groups: dict[Any, list[float]] = defaultdict(list)
    for row in rows:
        groups[row[key]].append(float(row[value_key]))
    return sorted(
        ((value, sum(values) / len(values), len(values)) for value, values in groups.items()),
        key=lambda item: item[0],
    )


def save_train_val_gap_by_run(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    ranked = sorted(rows, key=lambda row: row["final_val_loss"])
    x = list(range(1, len(ranked) + 1))
    train_losses = [row["final_train_loss"] for row in ranked]
    val_losses = [row["final_val_loss"] for row in ranked]
    gaps = [row["loss_gap"] for row in ranked]

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True, height_ratios=[2, 1])
    fig.suptitle("Train vs Validation Loss by Hyperparameter Run", fontsize=16, fontweight="bold")

    axes[0].plot(x, train_losses, marker="o", markersize=3, linewidth=1.2, label="Train loss")
    axes[0].plot(x, val_losses, marker="o", markersize=3, linewidth=1.2, label="Val loss")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Runs sorted by validation loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    colors = ["#C84B31" if gap > 0 else "#2E8B57" for gap in gaps]
    axes[1].bar(x, gaps, color=colors)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xlabel("Run rank by validation loss")
    axes[1].set_ylabel("Val - train")
    axes[1].set_title("Generalization gap")
    axes[1].grid(axis="y", alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = output_dir / "train_val_gap_by_run.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_top_overfit_cases(rows: list[dict[str, Any]], output_dir: Path, top_n: int) -> Path:
    top = sorted(rows, key=lambda row: row["loss_gap"], reverse=True)[:top_n]
    labels = [hparam_label(row) for row in top]
    y = list(range(len(top)))

    fig, ax = plt.subplots(figsize=(14, max(5, top_n * 0.42)))
    ax.barh(y, [row["loss_gap"] for row in top], color="#C84B31")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Validation loss - train loss")
    ax.set_title(f"Top {top_n} Largest Train/Validation Gaps")
    ax.grid(axis="x", alpha=0.25)
    for index, row in enumerate(top):
        ax.text(
            row["loss_gap"] + 0.004,
            index,
            f"train={row['final_train_loss']:.3f}, val={row['final_val_loss']:.3f}",
            va="center",
            fontsize=8,
        )

    plt.tight_layout()
    path = output_dir / "largest_overfit_gaps.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_train_vs_val_scatter(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    colors = {1e-4: "#999999", 3e-4: "#2E8B57", 5e-4: "#C84B31"}
    fig, ax = plt.subplots(figsize=(8, 8))

    for lr in sorted({row["learning_rate"] for row in rows}):
        group = [row for row in rows if row["learning_rate"] == lr]
        ax.scatter(
            [row["final_train_loss"] for row in group],
            [row["final_val_loss"] for row in group],
            s=[max(28, row["batch_size"] * 10) for row in group],
            alpha=0.75,
            label=f"lr={lr:.0e}",
            color=colors.get(lr),
            edgecolors="white",
            linewidths=0.5,
        )

    all_losses = [loss for row in rows for loss in [row["final_train_loss"], row["final_val_loss"]]]
    low, high = min(all_losses), max(all_losses)
    margin = (high - low) * 0.06
    ax.plot([low - margin, high + margin], [low - margin, high + margin], "--", color="black", linewidth=1)
    ax.set_xlim(low - margin, high + margin)
    ax.set_ylim(low - margin, high + margin)
    ax.set_xlabel("Final train loss")
    ax.set_ylabel("Final validation loss")
    ax.set_title("Train vs Validation Loss Scatter")
    ax.legend()
    ax.grid(alpha=0.25)

    largest_gap = max(rows, key=lambda row: row["loss_gap"])
    best_val = min(rows, key=lambda row: row["final_val_loss"])
    annotations = [("largest gap", largest_gap)]
    if best_val["run_id"] == largest_gap["run_id"]:
        annotations = [("best val / largest gap", best_val)]
    else:
        annotations.append(("best val", best_val))

    for label, row in annotations:
        ax.annotate(
            label,
            (row["final_train_loss"], row["final_val_loss"]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=8,
            arrowprops={"arrowstyle": "->", "linewidth": 0.8},
        )

    plt.tight_layout()
    path = output_dir / "train_vs_val_scatter.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_gap_by_hparam(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Mean Generalization Gap by Hyperparameter", fontsize=16, fontweight="bold")

    for ax, key in zip(axes.ravel(), HPARAM_KEYS):
        items = mean_by(rows, key, "loss_gap")
        labels = [fmt_value(value) for value, _, _ in items]
        means = [mean for _, mean, _ in items]
        colors = ["#C84B31" if mean > 0 else "#2E8B57" for mean in means]
        ax.bar(labels, means, color=colors)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(key)
        ax.set_ylabel("Mean val - train")
        ax.grid(axis="y", alpha=0.25)
        ymin = min(0.0, min(means)) - 0.005
        ymax = max(means) + 0.03
        ax.set_ylim(ymin, ymax)
        for index, (_, mean, count) in enumerate(items):
            ax.text(index, mean + 0.004, f"{mean:.3f}\nn={count}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = output_dir / "gap_by_hparam.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_epoch_history_curves(
    summary_rows: list[dict[str, Any]],
    history_rows: list[dict[str, Any]],
    output_dir: Path,
    top_n: int,
) -> Path | None:
    if not history_rows:
        return None

    top_run_ids = [row["run_id"] for row in sorted(summary_rows, key=lambda row: row["final_val_loss"])[:top_n]]
    history_by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in history_rows:
        if row["run_id"] in top_run_ids:
            history_by_run[row["run_id"]].append(row)

    if not history_by_run:
        return None

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, height_ratios=[2, 1])
    fig.suptitle(f"Epoch Loss Curves for Top {len(history_by_run)} Runs", fontsize=16, fontweight="bold")

    for rank, run_id in enumerate(top_run_ids, start=1):
        run_history = sorted(history_by_run.get(run_id, []), key=lambda row: row["epoch"])
        if not run_history:
            continue
        epochs = [row["epoch"] for row in run_history]
        train_losses = [row["train_loss"] for row in run_history]
        val_losses = [row["val_loss"] for row in run_history]
        gaps = [row["loss_gap"] for row in run_history]
        label = f"#{rank} {run_id}"
        axes[0].plot(epochs, train_losses, marker="o", linewidth=1.1, alpha=0.75, label=f"{label} train")
        axes[0].plot(epochs, val_losses, marker="s", linewidth=1.1, alpha=0.75, linestyle="--", label=f"{label} val")
        axes[1].plot(epochs, gaps, marker="o", linewidth=1.1, alpha=0.75, label=label)

    axes[0].set_ylabel("Loss")
    axes[0].set_title("Train/validation loss by epoch")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7, ncols=2)

    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val - train")
    axes[1].set_title("Generalization gap by epoch")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=7, ncols=2)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = output_dir / "epoch_loss_curves.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def print_summary(rows: list[dict[str, Any]]) -> None:
    gaps = [row["loss_gap"] for row in rows]
    print(f"rows={len(rows)}")
    print(f"gap_mean={sum(gaps) / len(gaps):.4f}")
    print(f"gap_min={min(gaps):.4f}")
    print(f"gap_max={max(gaps):.4f}")
    print("\nLargest gaps:")
    for row in sorted(rows, key=lambda item: item["loss_gap"], reverse=True)[:5]:
        print(f"{row['loss_gap']:.4f} {hparam_label(row)}")
    print("\nSmallest gaps:")
    for row in sorted(rows, key=lambda item: item["loss_gap"])[:5]:
        print(f"{row['loss_gap']:.4f} {hparam_label(row)}")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.summary_csv)
    history_rows = load_history_rows(args.history_csv)
    paths = [
        save_train_val_gap_by_run(rows, args.output_dir),
        save_top_overfit_cases(rows, args.output_dir, args.top_n),
        save_train_vs_val_scatter(rows, args.output_dir),
        save_gap_by_hparam(rows, args.output_dir),
    ]
    history_path = save_epoch_history_curves(rows, history_rows, args.output_dir, min(5, args.top_n))
    if history_path is not None:
        paths.append(history_path)

    print_summary(rows)
    print(f"\nhistory_rows={len(history_rows)}")
    print("\nSaved figures:")
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
