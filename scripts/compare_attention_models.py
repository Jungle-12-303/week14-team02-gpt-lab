# -*- coding: utf-8 -*-
"""Compare standard attention and [삼각 관계] variants on NSMC language modeling."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from attention import TriangularRelationAttention
from bpe import BPETokenizer
from dataset import create_dataloader
from model import GPTModel
from train import calc_loss_loader, generate


VARIANT_CONFIGS: dict[str, dict[str, Any]] = {
    "standard": {"attention_type": "standard"},
    "standard_embedding": {
        "attention_type": "standard",
        "position_mode": "embedding",
    },
    "standard_encoding": {
        "attention_type": "standard",
        "position_mode": "encoding",
    },
    "triangular_v0": {
        "attention_type": "triangular",
        "triangular_value_mode": "marginal",
        "triangular_logit_scale": "fixed",
        "triangular_candidate_mode": "full",
    },
    "triangular_v1_pair_value": {
        "attention_type": "triangular",
        "triangular_value_mode": "pair",
        "triangular_logit_scale": "fixed",
        "triangular_candidate_mode": "full",
    },
    "triangular_v2_scaled": {
        "attention_type": "triangular",
        "triangular_value_mode": "pair",
        "triangular_logit_scale": "learned",
        "triangular_candidate_mode": "full",
    },
    "triangular_v3_topk": {
        "attention_type": "triangular",
        "triangular_value_mode": "pair",
        "triangular_logit_scale": "learned",
        "triangular_candidate_mode": "topk",
    },
    "triangular_v4_gated_pair": {
        "attention_type": "triangular",
        "triangular_value_mode": "pair_gated",
        "triangular_logit_scale": "learned",
        "triangular_candidate_mode": "full",
    },
    "triangular_v5_gated_topk": {
        "attention_type": "triangular",
        "triangular_value_mode": "pair_gated",
        "triangular_logit_scale": "learned",
        "triangular_candidate_mode": "topk",
    },
    "triangular_final": {
        "attention_type": "triangular",
        "triangular_value_mode": "pair_gated",
        "triangular_logit_scale": "learned",
        "triangular_candidate_mode": "topk",
    },
    "triangular_embedding": {
        "attention_type": "triangular",
        "position_mode": "embedding",
        "triangular_value_mode": "pair_gated",
        "triangular_logit_scale": "learned",
        "triangular_candidate_mode": "topk",
    },
    "triangular_encoding": {
        "attention_type": "triangular",
        "position_mode": "encoding",
        "triangular_value_mode": "pair_gated",
        "triangular_logit_scale": "learned",
        "triangular_candidate_mode": "topk",
    },
    "alternating_standard_triangular": {
        "attention_type": "alternating",
        "triangular_value_mode": "pair_gated",
        "triangular_logit_scale": "learned",
        "triangular_candidate_mode": "topk",
    },
    "alternating_embedding": {
        "attention_type": "alternating",
        "position_mode": "embedding",
        "triangular_value_mode": "pair_gated",
        "triangular_logit_scale": "learned",
        "triangular_candidate_mode": "topk",
    },
    "alternating_encoding": {
        "attention_type": "alternating",
        "position_mode": "encoding",
        "triangular_value_mode": "pair_gated",
        "triangular_logit_scale": "learned",
        "triangular_candidate_mode": "topk",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an apples-to-apples NSMC LM comparison of attention variants.",
    )
    parser.add_argument("--context-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--train-char-limit", type=int, default=1000000)
    parser.add_argument("--val-char-limit", type=int, default=100000)
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seeds", type=str, default="123")
    parser.add_argument(
        "--variants",
        type=str,
        default="standard,triangular_v0,triangular_v1_pair_value,triangular_v2_scaled,triangular_v3_topk,triangular_v4_gated_pair,triangular_v5_gated_topk,triangular_final,alternating_standard_triangular",
    )
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--prompt", type=str, default="영화")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--experiment-name", type=str, default="NSMC LM")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "reports" / "attention_compare.json",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=ROOT / "reports" / "attention_compare.csv",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=ROOT / "reports" / "attention_compare_loss.png",
    )
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--history-csv", type=Path, default=None)
    parser.add_argument("--mean-plot-path", type=Path, default=None)
    parser.add_argument("--cost-quality-plot-path", type=Path, default=None)
    parser.add_argument("--zoom-plot-path", type=Path, default=None)
    parser.add_argument("--delta-plot-path", type=Path, default=None)
    parser.add_argument("--best-loss-bar-path", type=Path, default=None)
    parser.add_argument("--position-effect-plot-path", type=Path, default=None)
    parser.add_argument("--relation-stats-plot-path", type=Path, default=None)
    parser.add_argument("--final-report-md", type=Path, default=None)
    parser.add_argument(
        "--report-md",
        type=Path,
        default=ROOT / "reports" / "triangular_relation_research.md",
    )
    parser.add_argument("--append-report", action="store_true")
    return parser.parse_args()


def parse_int_list(raw: str) -> list[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return [int(item) for item in values]


def parse_variant_list(raw: str) -> list[str]:
    variants = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in variants if item not in VARIANT_CONFIGS]
    if unknown:
        raise ValueError(f"unknown variants: {unknown}")
    return variants


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_tokenizer(path: Path) -> BPETokenizer:
    tokenizer = BPETokenizer()
    tokenizer.load(path)
    return tokenizer


def load_token_ids(tokenizer: BPETokenizer, path: Path, char_limit: int) -> list[int]:
    text = path.read_text(encoding="utf-8")[:char_limit]
    return tokenizer.encode(text)


def safe_decode(tokenizer: BPETokenizer, ids: list[int]) -> str:
    byte_values = []
    special_ids = {
        tokenizer.get_pad_id(),
        tokenizer.get_unk_id(),
        tokenizer.get_bos_id(),
        tokenizer.get_eos_id(),
    }
    for token_id in ids:
        if token_id in special_ids:
            continue
        byte_values.extend(tokenizer.token_to_bytes(token_id))
    return bytes(byte_values).decode("utf-8", errors="replace")


def build_config(
    args: argparse.Namespace,
    vocab_size: int,
    variant: str,
) -> dict[str, Any]:
    config = {
        "vocab_size": vocab_size,
        "context_length": args.context_length,
        "emb_dim": args.emb_dim,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "drop_rate": 0.0,
        "qkv_bias": False,
    }
    config.update(VARIANT_CONFIGS[variant])
    if config.get("triangular_candidate_mode") == "topk":
        config["triangular_top_k"] = args.top_k
    return config


def parameter_count(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def score_elements_per_layer(args: argparse.Namespace, variant: str) -> int:
    variant_config = VARIANT_CONFIGS[variant]
    standard_elements = args.batch_size * args.n_heads * args.context_length * args.context_length
    if variant_config.get("attention_type") == "standard":
        return standard_elements
    if variant_config.get("triangular_candidate_mode") == "topk":
        k = min(args.top_k, args.context_length)
        triangular_elements = args.batch_size * args.n_heads * args.context_length * k * k
    else:
        triangular_elements = args.batch_size * args.n_heads * args.context_length**3
    if variant_config.get("attention_type") == "alternating":
        return (standard_elements + triangular_elements) // 2
    return triangular_elements


def score_elements_per_model(args: argparse.Namespace, variant: str) -> int:
    variant_config = VARIANT_CONFIGS[variant]
    standard_elements = args.batch_size * args.n_heads * args.context_length * args.context_length
    if variant_config.get("attention_type") == "standard":
        return standard_elements * args.n_layers

    if variant_config.get("triangular_candidate_mode") == "topk":
        k = min(args.top_k, args.context_length)
        triangular_elements = args.batch_size * args.n_heads * args.context_length * k * k
    else:
        triangular_elements = args.batch_size * args.n_heads * args.context_length**3

    if variant_config.get("attention_type") != "alternating":
        return triangular_elements * args.n_layers

    total = 0
    for layer_idx in range(args.n_layers):
        total += standard_elements if layer_idx % 2 == 0 else triangular_elements
    return total


def collect_relation_stats(model: torch.nn.Module) -> dict[str, float]:
    collected: dict[str, list[float]] = defaultdict(list)
    for module in model.modules():
        if not isinstance(module, TriangularRelationAttention):
            continue
        stats = getattr(module, "last_relation_stats", None)
        if not stats:
            continue
        for key, value in stats.items():
            collected[key].append(float(value.detach().cpu().item()))
    return {key: mean(values) for key, values in collected.items() if values}


def evaluate_losses(
    model: GPTModel,
    train_loader,
    val_loader,
    device: torch.device,
    eval_batches: int,
) -> tuple[float, float, dict[str, float]]:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_batches)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_batches)
    stats = collect_relation_stats(model)
    if was_training:
        model.train()
    return train_loss, val_loss, stats


def run_one(
    variant: str,
    seed: int,
    args: argparse.Namespace,
    tokenizer: BPETokenizer,
    train_ids: list[int],
    val_ids: list[int],
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(seed)

    config = build_config(args, len(tokenizer.id_to_token), variant)
    model = GPTModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"\n[{variant} seed={seed}] training start", flush=True)

    train_loader = create_dataloader(
        train_ids,
        context_length=args.context_length,
        batch_size=args.batch_size,
        stride=args.context_length,
        shuffle=False,
        drop_last=True,
    )
    val_loader = create_dataloader(
        val_ids,
        context_length=args.context_length,
        batch_size=args.batch_size,
        stride=args.context_length,
        shuffle=False,
        drop_last=True,
    )

    initial_train_loss, initial_val_loss, initial_stats = evaluate_losses(
        model,
        train_loader,
        val_loader,
        device,
        args.eval_batches,
    )
    history = [
        {
            "step": 0,
            "step_loss": None,
            "train_loss": initial_train_loss,
            "val_loss": initial_val_loss,
            "relation_stats": initial_stats,
        }
    ]

    model.train()
    losses = []
    start_time = time.perf_counter()
    actual_steps = 0
    while actual_steps < args.steps:
        progressed = False
        for input_batch, target_batch in train_loader:
            progressed = True
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            optimizer.zero_grad()
            loss, _ = model(input_batch, targets=target_batch)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            actual_steps += 1

            if actual_steps % args.eval_every == 0:
                train_loss, val_loss, stats = evaluate_losses(
                    model,
                    train_loader,
                    val_loader,
                    device,
                    args.eval_batches,
                )
                history.append(
                    {
                        "step": actual_steps,
                        "step_loss": losses[-1],
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "relation_stats": stats,
                    }
                )
                print(
                    f"[{variant} seed={seed}] step {actual_steps:>5} "
                    f"step_loss={losses[-1]:.4f} train={train_loss:.4f} val={val_loss:.4f}",
                    flush=True,
                )

            if actual_steps >= args.steps:
                break
        if not progressed:
            raise RuntimeError("train_loader produced no batches")

    elapsed = time.perf_counter() - start_time
    final_train_loss, final_val_loss, final_stats = evaluate_losses(
        model,
        train_loader,
        val_loader,
        device,
        args.eval_batches,
    )
    if history[-1]["step"] != actual_steps:
        history.append(
            {
                "step": actual_steps,
                "step_loss": losses[-1] if losses else None,
                "train_loss": final_train_loss,
                "val_loss": final_val_loss,
                "relation_stats": final_stats,
            }
        )

    model.eval()
    prompt_ids = tokenizer.encode(args.prompt)
    idx = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        generated_ids = generate(
            model,
            idx,
            max_new_tokens=args.max_new_tokens,
            context_size=args.context_length,
            temperature=0.0,
        )
    generated_text = safe_decode(tokenizer, generated_ids.squeeze(0).tolist())

    return {
        "variant": variant,
        "seed": seed,
        "attention_type": config.get("attention_type"),
        "position_mode": config.get("position_mode", "embedding"),
        "params": parameter_count(model),
        "score_elements_per_layer": score_elements_per_layer(args, variant),
        "score_elements_per_model": score_elements_per_model(args, variant),
        "steps": actual_steps,
        "initial_train_loss": initial_train_loss,
        "initial_val_loss": initial_val_loss,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "best_val_loss": min([point["val_loss"] for point in history] + [final_val_loss]),
        "last_step_loss": losses[-1] if losses else None,
        "final_relation_stats": final_stats,
        "history": history,
        "elapsed_sec": elapsed,
        "ms_per_step": (elapsed / actual_steps * 1000.0) if actual_steps else None,
        "generated_text": generated_text.replace("\n", " "),
    }


def write_csv(results: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant",
        "seed",
        "attention_type",
        "position_mode",
        "params",
        "score_elements_per_layer",
        "score_elements_per_model",
        "steps",
        "initial_train_loss",
        "final_train_loss",
        "initial_val_loss",
        "final_val_loss",
        "best_val_loss",
        "ms_per_step",
        "relation_entropy",
        "relation_max_weight",
        "relation_logit_mean",
        "relation_logit_std",
        "logit_scale",
        "pair_gate",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            stats = result.get("final_relation_stats", {})
            writer.writerow(
                {
                    **{key: result.get(key) for key in fieldnames if key in result},
                    "relation_entropy": stats.get("relation_entropy"),
                    "relation_max_weight": stats.get("relation_max_weight"),
                    "relation_logit_mean": stats.get("relation_logit_mean"),
                    "relation_logit_std": stats.get("relation_logit_std"),
                    "logit_scale": stats.get("logit_scale"),
                    "pair_gate": stats.get("pair_gate"),
                }
            )


def save_loss_plot(results: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 6))
    for result in results:
        steps = [point["step"] for point in result["history"]]
        val_losses = [point["val_loss"] for point in result["history"]]
        label = f"{display_label(result['variant'])} s{result['seed']}"
        plt.plot(steps, val_losses, marker="o", linewidth=1.1, label=label)
    plt.xlabel("Training step")
    plt.ylabel("Validation loss")
    plt.title("NSMC validation loss by model")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def group_results(results: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(result["variant"], []).append(result)
    return grouped


def summarize_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for variant, items in group_results(results).items():
        best_vals = [item["best_val_loss"] for item in items]
        final_vals = [item["final_val_loss"] for item in items]
        ms_values = [item["ms_per_step"] for item in items]
        score_model_values = [item["score_elements_per_model"] for item in items]
        params_values = [item["params"] for item in items]
        variant_config = VARIANT_CONFIGS[variant]
        summary.append(
            {
                "variant": variant,
                "attention_type": variant_config.get("attention_type", "standard"),
                "position_mode": variant_config.get("position_mode", "embedding"),
                "seeds": len(items),
                "best_val_mean": mean(best_vals),
                "best_val_std": pstdev(best_vals),
                "final_val_mean": mean(final_vals),
                "final_val_std": pstdev(final_vals),
                "ms_per_step_mean": mean(ms_values),
                "ms_per_step_std": pstdev(ms_values),
                "score_elements_per_model_mean": mean(score_model_values),
                "params_mean": mean(params_values),
            }
        )
    summary.sort(key=lambda item: item["best_val_mean"])
    return summary


def write_summary_csv(summary: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant",
        "attention_type",
        "position_mode",
        "seeds",
        "best_val_mean",
        "best_val_std",
        "final_val_mean",
        "final_val_std",
        "ms_per_step_mean",
        "ms_per_step_std",
        "score_elements_per_model_mean",
        "params_mean",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in summary:
            writer.writerow(item)


def display_label(variant: str) -> str:
    labels = {
        "standard_embedding": "standard / embedding",
        "triangular_embedding": "triangular / embedding",
        "alternating_embedding": "alternating / embedding",
        "standard_encoding": "standard / encoding",
        "triangular_encoding": "triangular / encoding",
        "alternating_encoding": "alternating / encoding",
        "triangular_final": "triangular_final",
        "alternating_standard_triangular": "alternating",
    }
    return labels.get(variant, variant)


def ordered_variants(results: list[dict[str, Any]]) -> list[str]:
    variants = []
    for result in results:
        if result["variant"] not in variants:
            variants.append(result["variant"])
    return variants


def ordered_summary(
    summary: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summary_by_variant = {item["variant"]: item for item in summary}
    return [
        summary_by_variant[variant]
        for variant in ordered_variants(results)
        if variant in summary_by_variant
    ]


def mean_history_by_variant(results: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    histories = {}
    for variant, items in group_results(results).items():
        steps = [point["step"] for point in items[0]["history"]]
        variant_history = []
        for idx, step in enumerate(steps):
            stats_by_key: dict[str, list[float]] = defaultdict(list)
            for item in items:
                point = item["history"][idx]
                for key, value in point.get("relation_stats", {}).items():
                    if value is not None:
                        stats_by_key[key].append(float(value))
            variant_history.append(
                {
                    "step": step,
                    "step_loss": mean(
                        point["step_loss"]
                        for point in (item["history"][idx] for item in items)
                        if point["step_loss"] is not None
                    )
                    if any(item["history"][idx]["step_loss"] is not None for item in items)
                    else None,
                    "train_loss": mean(item["history"][idx]["train_loss"] for item in items),
                    "val_loss": mean(item["history"][idx]["val_loss"] for item in items),
                    "relation_stats": {
                        key: mean(values) for key, values in stats_by_key.items()
                    },
                }
            )
        histories[variant] = variant_history
    return histories


def write_history_csv(results: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant",
        "seed",
        "attention_type",
        "position_mode",
        "step",
        "step_loss",
        "train_loss",
        "val_loss",
        "relation_entropy",
        "relation_max_weight",
        "relation_logit_mean",
        "relation_logit_std",
        "logit_scale",
        "pair_gate",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            for point in result["history"]:
                stats = point.get("relation_stats", {})
                writer.writerow(
                    {
                        "variant": result["variant"],
                        "seed": result["seed"],
                        "attention_type": result.get("attention_type"),
                        "position_mode": result.get("position_mode"),
                        "step": point["step"],
                        "step_loss": point["step_loss"],
                        "train_loss": point["train_loss"],
                        "val_loss": point["val_loss"],
                        "relation_entropy": stats.get("relation_entropy"),
                        "relation_max_weight": stats.get("relation_max_weight"),
                        "relation_logit_mean": stats.get("relation_logit_mean"),
                        "relation_logit_std": stats.get("relation_logit_std"),
                        "logit_scale": stats.get("logit_scale"),
                        "pair_gate": stats.get("pair_gate"),
                    }
                )


def derived_history_csv_path(csv_path: Path) -> Path:
    stem = csv_path.stem
    if stem.endswith("_comparison"):
        stem = stem[: -len("_comparison")]
    return csv_path.with_name(f"{stem}_history.csv")


def derived_plot_path(plot_path: Path, suffix: str) -> Path:
    stem = plot_path.stem
    if stem.endswith("_loss_full"):
        stem = stem[: -len("_loss_full")]
    elif stem.endswith("_loss"):
        stem = stem[: -len("_loss")]
    return plot_path.with_name(f"{stem}_{suffix}.png")


def save_mean_loss_plot(results: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 6))
    for variant, history in mean_history_by_variant(results).items():
        steps = [point["step"] for point in history]
        mean_losses = [point["val_loss"] for point in history]
        plt.plot(steps, mean_losses, marker="o", linewidth=1.5, label=display_label(variant))
    plt.xlabel("Training step")
    plt.ylabel("Mean validation loss")
    plt.title("Mean NSMC validation loss across seeds")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_zoom_loss_plot(results: list[dict[str, Any]], path: Path, min_step: int = 700) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 6))
    for variant, history in mean_history_by_variant(results).items():
        filtered = [point for point in history if point["step"] >= min_step]
        if not filtered:
            filtered = history
        steps = [point["step"] for point in filtered]
        val_losses = [point["val_loss"] for point in filtered]
        plt.plot(steps, val_losses, marker="o", linewidth=1.5, label=display_label(variant))
    plt.xlabel("Training step")
    plt.ylabel("Mean validation loss")
    plt.title(f"Validation loss zoom after step {min_step}")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_delta_loss_plot(results: list[dict[str, Any]], path: Path) -> None:
    histories = mean_history_by_variant(results)
    baseline_variant = (
        "standard_embedding"
        if "standard_embedding" in histories
        else next(
            (
                variant
                for variant in histories
                if VARIANT_CONFIGS[variant].get("attention_type") == "standard"
            ),
            next(iter(histories)),
        )
    )
    baseline_by_step = {
        point["step"]: point["val_loss"] for point in histories[baseline_variant]
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 6))
    for variant, history in histories.items():
        steps = []
        deltas = []
        for point in history:
            if point["step"] not in baseline_by_step:
                continue
            steps.append(point["step"])
            deltas.append(point["val_loss"] - baseline_by_step[point["step"]])
        plt.plot(steps, deltas, marker="o", linewidth=1.5, label=display_label(variant))
    plt.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    plt.xlabel("Training step")
    plt.ylabel(f"Validation loss delta vs {display_label(baseline_variant)}")
    plt.title("Validation loss delta against the embedding standard baseline")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_best_loss_bar_plot(
    summary: list[dict[str, Any]],
    results: list[dict[str, Any]],
    path: Path,
) -> None:
    ordered = ordered_summary(summary, results)
    labels = [display_label(item["variant"]) for item in ordered]
    values = [item["best_val_mean"] for item in ordered]

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 6))
    bars = plt.bar(labels, values)
    plt.ylabel("Best validation loss")
    plt.title("Best validation loss by model")
    plt.xticks(rotation=25, ha="right")
    y_min = min(values) - 0.002
    y_max = max(values) + 0.002
    plt.ylim(y_min, y_max)
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_position_effect_plot(summary: list[dict[str, Any]], path: Path) -> None:
    summary_by_variant = {item["variant"]: item for item in summary}
    pairs = [
        ("standard", "standard_embedding", "standard_encoding"),
        ("triangular", "triangular_embedding", "triangular_encoding"),
        ("alternating", "alternating_embedding", "alternating_encoding"),
    ]
    labels = []
    deltas = []
    for label, embedding_variant, encoding_variant in pairs:
        if embedding_variant not in summary_by_variant or encoding_variant not in summary_by_variant:
            continue
        labels.append(label)
        deltas.append(
            summary_by_variant[encoding_variant]["best_val_mean"]
            - summary_by_variant[embedding_variant]["best_val_mean"]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, deltas)
    plt.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    plt.ylabel("Encoding best loss - embedding best loss")
    plt.title("Position encoding effect by attention family")
    for bar, value in zip(bars, deltas):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:+.4f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
        )
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_relation_stats_plot(results: list[dict[str, Any]], path: Path) -> None:
    histories = {
        variant: history
        for variant, history in mean_history_by_variant(results).items()
        if any(point.get("relation_stats") for point in history)
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for variant, history in histories.items():
        steps = [point["step"] for point in history]
        entropy = [
            point.get("relation_stats", {}).get("relation_entropy")
            for point in history
        ]
        max_weight = [
            point.get("relation_stats", {}).get("relation_max_weight")
            for point in history
        ]
        if any(value is not None for value in entropy):
            axes[0].plot(steps, entropy, marker="o", linewidth=1.4, label=display_label(variant))
        if any(value is not None for value in max_weight):
            axes[1].plot(steps, max_weight, marker="o", linewidth=1.4, label=display_label(variant))
    axes[0].set_title("Relation weight entropy")
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Entropy")
    axes[1].set_title("Relation max weight")
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Mean max weight")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_cost_quality_plot(summary: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 6))
    for item in summary:
        plt.scatter(item["ms_per_step_mean"], item["best_val_mean"], s=80)
        plt.annotate(
            display_label(item["variant"]),
            (item["ms_per_step_mean"], item["best_val_mean"]),
            textcoords="offset points",
            xytext=(7, 5),
            fontsize=8,
        )
    plt.xlabel("Mean ms/step")
    plt.ylabel("Mean best validation loss")
    plt.title("NSMC cost vs quality")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def is_position_mode_comparison(results: list[dict[str, Any]]) -> bool:
    variants = {result["variant"] for result in results}
    required = {
        "standard_embedding",
        "triangular_embedding",
        "alternating_embedding",
        "standard_encoding",
        "triangular_encoding",
        "alternating_encoding",
    }
    return required.issubset(variants)


def write_position_mode_report(
    results: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
    paths: dict[str, Path],
) -> None:
    summary_by_variant = {item["variant"]: item for item in summary}
    ordered = ordered_summary(summary, results)
    best = min(summary, key=lambda item: item["best_val_mean"])

    raw_rows = []
    for result in results:
        stats = result.get("final_relation_stats", {})
        entropy = stats.get("relation_entropy")
        entropy_text = f"{entropy:.3f}" if entropy is not None else "-"
        raw_rows.append(
            f"| {result['variant']} | {result['attention_type']} | {result['position_mode']} | "
            f"{result['seed']} | {result['params']} | {result['final_val_loss']:.4f} | "
            f"{result['best_val_loss']:.4f} | {result['ms_per_step']:.2f} | {entropy_text} |"
        )

    summary_rows = []
    for item in ordered:
        summary_rows.append(
            f"| {item['variant']} | {item['attention_type']} | {item['position_mode']} | "
            f"{item['best_val_mean']:.4f} | {item['final_val_mean']:.4f} | "
            f"{item['ms_per_step_mean']:.2f} | {int(item['params_mean'])} | "
            f"{int(item['score_elements_per_model_mean'])} |"
        )

    position_pairs = [
        ("standard", "standard_embedding", "standard_encoding"),
        ("triangular", "triangular_embedding", "triangular_encoding"),
        ("alternating", "alternating_embedding", "alternating_encoding"),
    ]
    position_rows = []
    insight_lines = [
        f"- best validation loss 기준 최상위 모델은 `{best['variant']}`이다.",
    ]
    param_delta = (
        summary_by_variant["standard_embedding"]["params_mean"]
        - summary_by_variant["standard_encoding"]["params_mean"]
    )
    insight_lines.append(
        f"- encoding variant는 learned position table을 학습하지 않아 같은 attention family에서 "
        f"parameter가 `{int(param_delta)}`개 적다."
    )
    for label, embedding_variant, encoding_variant in position_pairs:
        embedding = summary_by_variant[embedding_variant]
        encoding = summary_by_variant[encoding_variant]
        delta = encoding["best_val_mean"] - embedding["best_val_mean"]
        direction = "encoding better" if delta < 0 else "embedding better"
        position_rows.append(
            f"| {label} | {embedding['best_val_mean']:.4f} | "
            f"{encoding['best_val_mean']:.4f} | {delta:+.4f} | {direction} |"
        )
        if label == "triangular":
            if delta < 0:
                insight_lines.append(
                    f"- 핵심 가설과 맞게 `[삼각 관계]`는 fixed encoding에서 best loss가 "
                    f"`{abs(delta):.4f}` 낮았다."
                )
            else:
                insight_lines.append(
                    f"- 핵심 가설과 달리 `[삼각 관계]`는 learned embedding에서 best loss가 "
                    f"`{abs(delta):.4f}` 낮았다."
                )
        else:
            insight_lines.append(
                f"- `{label}` 계열의 encoding-embedding best loss 차이는 `{delta:+.4f}`이다."
            )

    standard_delta = (
        summary_by_variant["standard_encoding"]["best_val_mean"]
        - summary_by_variant["standard_embedding"]["best_val_mean"]
    )
    triangular_delta = (
        summary_by_variant["triangular_encoding"]["best_val_mean"]
        - summary_by_variant["triangular_embedding"]["best_val_mean"]
    )
    if triangular_delta < standard_delta:
        insight_lines.append(
            "- 위치 encoding 효과는 standard보다 `[삼각 관계]`에서 더 우호적으로 나타났다."
        )
    else:
        insight_lines.append(
            "- 위치 encoding 효과가 `[삼각 관계]`에만 특별히 우호적이라고 보기는 어렵다."
        )
    insight_lines.append(
        "- 공식 비교는 seed 123 단일 실행이므로, 효과 방향은 강하게 보이지만 통계적 확정은 추가 seed 반복이 필요하다."
    )

    figure_keys = [
        ("Full loss curves", "loss_plot"),
        ("Zoom loss curves after 700", "zoom_plot"),
        ("Delta loss curves", "delta_plot"),
        ("Best loss bar", "best_loss_bar"),
        ("Cost-quality scatter", "cost_quality_plot"),
        ("Position effect", "position_effect_plot"),
        ("Relation stats", "relation_stats_plot"),
    ]

    lines = [
        "# [삼각 관계] Position Mode NSMC Report",
        "",
        "## Background",
        "",
        "이번 실험은 위치 정보를 learned position embedding으로 줄 때와 fixed sin/cos positional encoding으로 줄 때를 비교한다. `[삼각 관계]`는 `score[i,j,k]`로 세 토큰 조합을 직접 만들기 때문에, 위치 신호가 hidden state에 섞이는 방식의 영향을 standard attention보다 크게 받을 수 있다는 가설을 둔다.",
        "",
        "비교 대상은 standard, `[삼각 관계]`, alternating을 각각 embedding/encoding 위치 방식으로 만든 정확히 6개 모델이다.",
        "",
        "## Setup",
        "",
        f"- device: `{device}`",
        f"- variants: `{args.variants}`",
        f"- seeds: `{args.seeds}`",
        f"- steps: `{args.steps}`",
        f"- eval_every: `{args.eval_every}`",
        f"- context_length: `{args.context_length}`",
        f"- batch_size: `{args.batch_size}`",
        f"- top_k: `{args.top_k}`",
        "- triangular final: `pair_gated + learned logit scale + topk`",
        "- alternating order: `standard -> triangular`",
        "",
        "## Raw Results",
        "",
        "| variant | attention | position | seed | params | final val | best val | ms/step | entropy |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        *raw_rows,
        "",
        "## Summary",
        "",
        "| variant | attention | position | best val | final val | ms/step | params | score elements/model |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        *summary_rows,
        "",
        "## Position Effect",
        "",
        "`encoding - embedding`이 음수이면 fixed positional encoding이 더 낮은 best validation loss를 냈다는 뜻이다.",
        "",
        "| attention family | embedding best | encoding best | encoding - embedding | winner |",
        "| --- | ---: | ---: | ---: | --- |",
        *position_rows,
        "",
        "## Figures",
        "",
    ]
    for label, key in figure_keys:
        if key not in paths:
            continue
        lines.append(f"- {label}: `{paths[key]}`")
    lines.append(f"- History CSV: `{paths['history_csv']}`")
    lines.append(f"- Summary CSV: `{paths['summary_csv']}`")
    lines.append("")
    for label, key in figure_keys:
        if key not in paths:
            continue
        lines.extend([f"![{label}]({paths[key].name})", ""])

    lines.extend(
        [
            "## Interpretation",
            "",
            *insight_lines,
            "",
            "확대 그래프는 700 step 이후의 후반 학습 구간에서 여섯 곡선이 실제로 벌어지는지 확인하기 위한 것이다. delta 그래프는 `standard_embedding`을 기준으로 각 모델의 상대적 손익을 보여주며, position-effect 그래프는 같은 attention family 안에서 위치 방식만 바꾼 차이를 압축해서 보여준다.",
            "",
            "## Next Direction",
            "",
            "- 이번처럼 encoding이 전체적으로 유리하고 `[삼각 관계]`에서 폭이 가장 크면, 3 seed 반복으로 방향성을 확인한 뒤 relative position bias 또는 RoPE 계열을 삼각 relation score에 추가한다.",
            "- 모든 attention family에서 encoding 효과가 계속 비슷하게 유지되면, 위치 방식은 모델별 특성이 아니라 작은 모델/NSMC 설정의 일반 안정화 효과로 해석한다.",
            "- encoding 이득이 없으면, `[삼각 관계]`의 병목은 위치 신호보다 top-k 후보 품질, pair-value gate, 또는 relation softmax 온도 쪽으로 본다.",
            "",
        ]
    )

    args.final_report_md.parent.mkdir(parents=True, exist_ok=True)
    args.final_report_md.write_text("\n".join(lines), encoding="utf-8")


def write_final_report(
    results: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
    paths: dict[str, Path],
) -> None:
    if args.final_report_md is None:
        return
    if is_position_mode_comparison(results):
        write_position_mode_report(results, summary, args, device, paths)
        return

    summary_by_variant = {item["variant"]: item for item in summary}
    standard = summary_by_variant.get("standard")
    best = summary[0]
    triangular = summary_by_variant.get("triangular_final") or summary_by_variant.get("triangular_v5_gated_topk")
    alternating = summary_by_variant.get("alternating_standard_triangular")

    raw_rows = []
    for result in results:
        raw_rows.append(
            f"| {result['variant']} | {result['seed']} | "
            f"{result['final_val_loss']:.4f} | {result['best_val_loss']:.4f} | "
            f"{result['ms_per_step']:.2f} | {result['score_elements_per_model']} |"
        )

    summary_rows = []
    for item in summary:
        gain = ""
        if standard and item["variant"] != "standard":
            gain_value = standard["best_val_mean"] - item["best_val_mean"]
            gain = "0.0000" if abs(gain_value) < 0.00005 else fmt(gain_value, 4)
        summary_rows.append(
            f"| {item['variant']} | {item['seeds']} | "
            f"{item['best_val_mean']:.4f} | {item['best_val_std']:.4f} | "
            f"{item['final_val_mean']:.4f} | {item['ms_per_step_mean']:.2f} | "
            f"{int(item['score_elements_per_model_mean'])} | {gain or '-'} |"
        )

    insight_lines = [
        f"- 평균 best val loss 기준 최상위 모델은 `{best['variant']}`이다.",
    ]
    if standard and triangular:
        triangular_gain = standard["best_val_mean"] - triangular["best_val_mean"]
        if triangular_gain >= 0:
            insight_lines.append(
                f"- `triangular_final`은 standard 대비 평균 best val loss를 "
                f"`{triangular_gain:.4f}`만큼 낮췄다."
            )
        else:
            insight_lines.append(
                f"- `triangular_final`은 standard 대비 평균 best val loss가 "
                f"`{abs(triangular_gain):.4f}`만큼 높았다."
            )
    if standard and alternating:
        alternating_gain = standard["best_val_mean"] - alternating["best_val_mean"]
        if alternating_gain >= 0:
            insight_lines.append(
                f"- `alternating_standard_triangular`은 standard 대비 평균 best val loss를 "
                f"`{alternating_gain:.4f}`만큼 낮췄다."
            )
        else:
            insight_lines.append(
                f"- `alternating_standard_triangular`은 standard 대비 평균 best val loss가 "
                f"`{abs(alternating_gain):.4f}`만큼 높아, 이번 설정에서는 성능 절충점으로 보기 어렵다."
            )
    if triangular and alternating:
        insight_lines.append(
            f"- 혼합 모델은 순수 [삼각 관계] 대비 비용과 성능의 절충점인지 확인하는 기준점이다. "
            f"평균 ms/step은 `{alternating['ms_per_step_mean']:.2f}`이고 "
            f"순수 [삼각 관계]는 `{triangular['ms_per_step_mean']:.2f}`이다."
        )

    lines = [
        "# [삼각 관계] Final NSMC Report",
        "",
        "## Background",
        "",
        "[삼각 관계]는 기존 attention의 `score[i,j]`를 `score[i,j,k]`로 확장해 세 토큰 조합을 직접 보려는 실험이다. 최종 후보는 pair-value 경로를 learned gate로 조절하고, top-k 후보 안에서만 삼각 관계를 계산해 full `O(T^3)` 비용을 줄인다.",
        "",
        "기대는 두 가지였다. 첫째, 세 토큰 관계를 직접 보는 inductive bias가 NSMC LM에서도 작은 loss 개선을 만들 수 있는지 확인한다. 둘째, 표준 attention과 [삼각 관계]를 섞은 모델이 성능과 비용 사이의 절충점을 만들 수 있는지 본다.",
        "",
        "## Setup",
        "",
        f"- device: `{device}`",
        f"- variants: `{args.variants}`",
        f"- seeds: `{args.seeds}`",
        f"- steps: `{args.steps}`",
        f"- context_length: `{args.context_length}`",
        f"- batch_size: `{args.batch_size}`",
        f"- top_k: `{args.top_k}`",
        "- alternating order: `standard -> triangular`",
        "",
        "## Raw Results",
        "",
        "| variant | seed | final val | best val | ms/step | score elements/model |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        *raw_rows,
        "",
        "## Aggregate Results",
        "",
        "| variant | seeds | best val mean | best val std | final val mean | ms/step mean | score elements/model | best-val gain vs standard |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *summary_rows,
        "",
        "## Figures",
        "",
        f"- Seed loss curves: `{paths['loss_plot']}`",
        f"- Mean loss curves: `{paths['mean_plot']}`",
        f"- Cost-quality scatter: `{paths['cost_quality_plot']}`",
        f"- Summary CSV: `{paths['summary_csv']}`",
        "",
        f"![Seed loss curves]({paths['loss_plot'].name})",
        "",
        f"![Mean loss curves]({paths['mean_plot'].name})",
        "",
        f"![Cost-quality scatter]({paths['cost_quality_plot'].name})",
        "",
        "## Interpretation",
        "",
        *insight_lines,
        "",
        "## Next Direction",
        "",
        "- multi-seed 결과에서 [삼각 관계] 또는 혼합 모델의 개선이 유지되면 top-k gather path 최적화를 우선한다.",
        "- 혼합 모델이 순수 [삼각 관계]보다 비슷한 성능에 낮은 비용을 보이면, 층별 배치 비율을 `standard, triangular, standard, triangular` 식으로 더 깊은 모델에서 탐색한다.",
        "- 개선이 seed 평균에서 사라지면 synthetic 성공은 relation supervision 조건부 효과로 제한해 해석한다.",
        "",
    ]

    args.final_report_md.parent.mkdir(parents=True, exist_ok=True)
    args.final_report_md.write_text("\n".join(lines), encoding="utf-8")


def append_aggregate_report(
    summary: list[dict[str, Any]],
    args: argparse.Namespace,
    paths: dict[str, Path],
) -> None:
    if not args.append_report:
        return

    lines = [
        "",
        f"## Final Aggregate Summary: {args.experiment_name}",
        "",
        "| variant | seeds | best val mean | best val std | final val mean | ms/step mean | score elements/model |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summary:
        lines.append(
            f"| {item['variant']} | {item['seeds']} | "
            f"{item['best_val_mean']:.4f} | {item['best_val_std']:.4f} | "
            f"{item['final_val_mean']:.4f} | {item['ms_per_step_mean']:.2f} | "
            f"{int(item['score_elements_per_model_mean'])} |"
        )
    lines.extend(
        [
            "",
            f"- history CSV: `{paths['history_csv']}`",
            f"- summary CSV: `{paths['summary_csv']}`",
            f"- full loss plot: `{paths['loss_plot']}`",
            f"- mean loss plot: `{paths['mean_plot']}`",
            f"- zoom loss plot: `{paths['zoom_plot']}`",
            f"- delta loss plot: `{paths['delta_plot']}`",
            f"- cost-quality plot: `{paths['cost_quality_plot']}`",
            f"- best loss bar plot: `{paths['best_loss_bar']}`",
            f"- position effect plot: `{paths['position_effect_plot']}`",
            f"- relation stats plot: `{paths['relation_stats_plot']}`",
        ]
    )
    if args.final_report_md is not None:
        lines.append(f"- final report: `{args.final_report_md}`")
    lines.append("")

    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    with args.report_md.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def print_results(results: list[dict[str, Any]]) -> None:
    print("\n== Attention comparison ==")
    print(
        "variant                    seed  params  score_elems/layer  "
        "final_train  final_val  best_val  ms/step  entropy"
    )
    for result in results:
        stats = result.get("final_relation_stats", {})
        entropy = stats.get("relation_entropy")
        entropy_text = f"{entropy:.3f}" if entropy is not None else "-"
        print(
            f"{result['variant']:<26} "
            f"{result['seed']:>4} "
            f"{result['params']:>7} "
            f"{result['score_elements_per_layer']:>18} "
            f"{result['final_train_loss']:>11.4f} "
            f"{result['final_val_loss']:>9.4f} "
            f"{result['best_val_loss']:>9.4f} "
            f"{result['ms_per_step']:>8.2f} "
            f"{entropy_text:>8}"
        )


def append_report(
    results: list[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "",
        f"## Experiment: {args.experiment_name}",
        "",
        "### Command",
        "",
        "```bash",
        "python scripts/compare_attention_models.py "
        f"--context-length {args.context_length} --batch-size {args.batch_size} "
        f"--steps {args.steps} --eval-every {args.eval_every} "
        f"--eval-batches {args.eval_batches} --seeds {args.seeds} "
        f"--variants {args.variants}",
        "```",
        "",
        "### Setup",
        "",
        f"- device: `{device}`",
        f"- context_length: `{args.context_length}`",
        f"- batch_size: `{args.batch_size}`",
        f"- steps: `{args.steps}`",
        f"- seeds: `{args.seeds}`",
        "",
        "### Results",
        "",
        "| variant | seed | final val | best val | ms/step | entropy | score elements/layer |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        stats = result.get("final_relation_stats", {})
        entropy = stats.get("relation_entropy")
        entropy_text = f"{entropy:.3f}" if entropy is not None else "-"
        lines.append(
            f"| {result['variant']} | {result['seed']} | "
            f"{result['final_val_loss']:.4f} | {result['best_val_loss']:.4f} | "
            f"{result['ms_per_step']:.2f} | {entropy_text} | "
            f"{result['score_elements_per_layer']} |"
        )
    if is_position_mode_comparison(results):
        interpretation = [
            "- 이번 비교는 attention family 3개와 위치 방식 2개를 교차한 6모델 실험이다.",
            "- `standard_embedding`을 loss delta 기준선으로 두고, 같은 attention family 안에서는 encoding과 embedding의 best loss 차이를 본다.",
            "- `[삼각 관계]`와 alternating은 final top-k 설정을 사용하므로 비용은 full `O(T^3)`가 아니라 `O(T*K^2)` 경로다.",
        ]
    else:
        interpretation = [
            "- 표준 attention은 비용 기준선이다.",
            "- [삼각 관계] full variant는 세 토큰 조합을 직접 보지만 비용은 `O(T^3)`이다.",
            "- top-k variant는 후보 토큰을 줄여 `O(T*K^2)` 경로를 검증한다.",
            "- 다음 단계는 synthetic 삼각 관계 task에서 구조적 이득을 검증하는 것이다.",
        ]
    lines.extend(["", "### Interpretation", "", *interpretation, ""])
    mode = "a" if args.report_md.exists() else "w"
    with args.report_md.open(mode, encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    args = parse_args()
    if args.context_length <= 0:
        raise ValueError("context-length must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    if args.eval_every <= 0:
        raise ValueError("eval-every must be positive")
    if args.top_k <= 0:
        raise ValueError("top-k must be positive")

    seeds = parse_int_list(args.seeds)
    variants = parse_variant_list(args.variants)
    device = select_device()

    tokenizer = load_tokenizer(ROOT / "data" / "vocab_bpe_basic_3000.json")
    train_ids = load_token_ids(tokenizer, ROOT / "data" / "nsmc_lm_train.txt", args.train_char_limit)
    val_ids = load_token_ids(tokenizer, ROOT / "data" / "nsmc_lm_val.txt", args.val_char_limit)

    results = []
    for seed in seeds:
        for variant in variants:
            results.append(run_one(variant, seed, args, tokenizer, train_ids, val_ids, device))

    print_results(results)
    save_loss_plot(results, args.plot_path)
    write_csv(results, args.csv_path)
    summary = summarize_results(results)

    summary_csv = args.summary_csv or args.csv_path.with_name(
        f"{args.csv_path.stem}_summary.csv"
    )
    history_csv = args.history_csv or derived_history_csv_path(args.csv_path)
    mean_plot_path = args.mean_plot_path or derived_plot_path(args.plot_path, "loss_mean")
    cost_quality_plot_path = args.cost_quality_plot_path or derived_plot_path(
        args.plot_path,
        "cost_quality",
    )
    zoom_plot_path = args.zoom_plot_path or derived_plot_path(args.plot_path, "loss_zoom_700")
    delta_plot_path = args.delta_plot_path or derived_plot_path(args.plot_path, "loss_delta")
    best_loss_bar_path = args.best_loss_bar_path or derived_plot_path(args.plot_path, "best_loss_bar")
    position_effect_plot_path = args.position_effect_plot_path or derived_plot_path(args.plot_path, "position_effect")
    relation_stats_plot_path = args.relation_stats_plot_path or derived_plot_path(args.plot_path, "relation_stats")
    paths = {
        "loss_plot": args.plot_path,
        "history_csv": history_csv,
        "summary_csv": summary_csv,
        "mean_plot": mean_plot_path,
        "cost_quality_plot": cost_quality_plot_path,
        "zoom_plot": zoom_plot_path,
        "delta_plot": delta_plot_path,
        "best_loss_bar": best_loss_bar_path,
        "position_effect_plot": position_effect_plot_path,
        "relation_stats_plot": relation_stats_plot_path,
    }
    write_summary_csv(summary, summary_csv)
    write_history_csv(results, history_csv)
    save_mean_loss_plot(results, mean_plot_path)
    save_cost_quality_plot(summary, cost_quality_plot_path)
    save_zoom_loss_plot(results, zoom_plot_path)
    save_delta_loss_plot(results, delta_plot_path)
    save_best_loss_bar_plot(summary, results, best_loss_bar_path)
    save_position_effect_plot(summary, position_effect_plot_path)
    save_relation_stats_plot(results, relation_stats_plot_path)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            {
                "device": str(device),
                "config": vars(args)
                | {
                    "output_json": str(args.output_json),
                    "csv_path": str(args.csv_path),
                    "plot_path": str(args.plot_path),
                    "summary_csv": str(summary_csv),
                    "history_csv": str(history_csv),
                    "mean_plot_path": str(mean_plot_path),
                    "cost_quality_plot_path": str(cost_quality_plot_path),
                    "zoom_plot_path": str(zoom_plot_path),
                    "delta_plot_path": str(delta_plot_path),
                    "best_loss_bar_path": str(best_loss_bar_path),
                    "position_effect_plot_path": str(position_effect_plot_path),
                    "relation_stats_plot_path": str(relation_stats_plot_path),
                    "report_md": str(args.report_md),
                    "final_report_md": str(args.final_report_md) if args.final_report_md else None,
                },
                "summary": summary,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if args.append_report:
        append_report(results, args, device)
        append_aggregate_report(summary, args, paths)
    write_final_report(results, summary, args, device, paths)

    print(f"\nSaved JSON: {args.output_json}")
    print(f"Saved CSV: {args.csv_path}")
    print(f"Saved history CSV: {history_csv}")
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved plot: {args.plot_path}")
    print(f"Saved mean plot: {mean_plot_path}")
    print(f"Saved zoom plot: {zoom_plot_path}")
    print(f"Saved delta plot: {delta_plot_path}")
    print(f"Saved best loss bar plot: {best_loss_bar_path}")
    print(f"Saved cost-quality plot: {cost_quality_plot_path}")
    print(f"Saved position effect plot: {position_effect_plot_path}")
    print(f"Saved relation stats plot: {relation_stats_plot_path}")
    if args.append_report:
        print(f"Updated report: {args.report_md}")
    if args.final_report_md:
        print(f"Saved final report: {args.final_report_md}")


if __name__ == "__main__":
    main()
