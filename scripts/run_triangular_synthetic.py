# -*- coding: utf-8 -*-
"""Synthetic give(subject, object, recipient) task for [삼각 관계] attention."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from attention import TriangularRelationAttention
from model import GPTModel


BOS = 1
GIVE = 2
TO = 3
SEP = 4
QSUBJ = 5
QOBJ = 6
ANS = 7
ENTITY_BASE = 10
OBJECT_BASE = 30
VOCAB_SIZE = 64

VARIANT_CONFIGS: dict[str, dict[str, Any]] = {
    "standard": {"attention_type": "standard"},
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
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic [삼각 관계] relation experiment.")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entities", type=int, default=8)
    parser.add_argument("--objects", type=int, default=8)
    parser.add_argument("--facts", type=int, default=2)
    parser.add_argument(
        "--loss-mode",
        choices=("answer", "answer_relation"),
        default="answer",
    )
    parser.add_argument("--relation-loss-weight", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seeds", type=str, default="123")
    parser.add_argument(
        "--variants",
        type=str,
        default="standard,triangular_v0,triangular_v1_pair_value,triangular_v2_scaled,triangular_v3_topk,triangular_v4_gated_pair,triangular_v5_gated_topk",
    )
    parser.add_argument("--experiment-name", type=str, default="Synthetic give relation")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "reports" / "triangular_synthetic.json",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=ROOT / "reports" / "triangular_synthetic.csv",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=ROOT / "reports" / "triangular_synthetic_accuracy.png",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=ROOT / "reports" / "triangular_relation_research.md",
    )
    parser.add_argument("--append-report", action="store_true")
    return parser.parse_args()


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_variant_list(raw: str) -> list[str]:
    variants = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [variant for variant in variants if variant not in VARIANT_CONFIGS]
    if unknown:
        raise ValueError(f"unknown variants: {unknown}")
    return variants


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_config(args: argparse.Namespace, variant: str, context_length: int) -> dict[str, Any]:
    config = {
        "vocab_size": VOCAB_SIZE,
        "context_length": context_length,
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


def make_batch(
    args: argparse.Namespace,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequences = []
    answers = []
    for _ in range(batch_size):
        base_s = ENTITY_BASE + torch.randint(args.entities, (1,), generator=generator).item()
        base_o = OBJECT_BASE + torch.randint(args.objects, (1,), generator=generator).item()
        base_r = ENTITY_BASE + torch.randint(args.entities, (1,), generator=generator).item()
        facts = [(base_s, base_o, base_r)]
        used_pairs = {(base_s, base_o)}

        while len(facts) < args.facts:
            mode = torch.randint(3, (1,), generator=generator).item()
            if mode == 0:
                subject = base_s
                object_ = OBJECT_BASE + torch.randint(args.objects, (1,), generator=generator).item()
            elif mode == 1:
                subject = ENTITY_BASE + torch.randint(args.entities, (1,), generator=generator).item()
                object_ = base_o
            else:
                subject = ENTITY_BASE + torch.randint(args.entities, (1,), generator=generator).item()
                object_ = OBJECT_BASE + torch.randint(args.objects, (1,), generator=generator).item()

            if (subject, object_) in used_pairs:
                continue
            used_pairs.add((subject, object_))
            recipient = ENTITY_BASE + torch.randint(args.entities, (1,), generator=generator).item()
            facts.append((subject, object_, recipient))

        query_idx = torch.randint(len(facts), (1,), generator=generator).item()
        query_subject, query_object, answer = facts[query_idx]

        sequence = [BOS]
        for subject, object_, recipient in facts:
            sequence.extend([subject, GIVE, object_, TO, recipient, SEP])
        sequence.extend([QSUBJ, query_subject, QOBJ, query_object, ANS, answer])
        sequences.append(sequence)
        answers.append(answer)

    tokens = torch.tensor(sequences, dtype=torch.long, device=device)
    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]
    answers_tensor = torch.tensor(answers, dtype=torch.long, device=device)
    return inputs, targets, answers_tensor


def parameter_count(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def score_elements_per_layer(args: argparse.Namespace, variant: str, context_length: int) -> int:
    if variant == "standard":
        return args.batch_size * args.n_heads * context_length * context_length
    if VARIANT_CONFIGS[variant].get("triangular_candidate_mode") == "topk":
        k = min(args.top_k, context_length)
        return args.batch_size * args.n_heads * context_length * k * k
    return args.batch_size * args.n_heads * context_length**3


def relation_prediction_positions(args: argparse.Namespace) -> list[int]:
    return [4 + 6 * fact_idx for fact_idx in range(args.facts)]


def synthetic_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    answers: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    answer_loss = torch.nn.functional.cross_entropy(logits[:, -1, :], answers)
    if args.loss_mode == "answer":
        return answer_loss

    positions = relation_prediction_positions(args)
    relation_logits = logits[:, positions, :].reshape(-1, logits.size(-1))
    relation_targets = targets[:, positions].reshape(-1)
    relation_loss = torch.nn.functional.cross_entropy(relation_logits, relation_targets)
    return answer_loss + args.relation_loss_weight * relation_loss


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


def evaluate(
    model: GPTModel,
    args: argparse.Namespace,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[float, float, dict[str, float]]:
    was_training = model.training
    model.eval()
    losses = []
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(args.eval_batches):
            inputs, targets, answers = make_batch(args, args.batch_size, device, generator)
            logits = model(inputs)
            loss = synthetic_loss(logits, targets, answers, args)
            losses.append(loss.item())
            preds = logits[:, -1, :].argmax(dim=-1)
            correct += (preds == answers).sum().item()
            total += answers.numel()
    stats = collect_relation_stats(model)
    if was_training:
        model.train()
    return mean(losses), correct / total, stats


def run_one(
    variant: str,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    context_length = 6 + args.facts * 6
    config = build_config(args, variant, context_length)
    model = GPTModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_generator = torch.Generator(device="cpu").manual_seed(seed)
    eval_generator = torch.Generator(device="cpu").manual_seed(seed + 10_000)
    print(f"\n[{variant} seed={seed}] synthetic training start", flush=True)

    initial_loss, initial_acc, initial_stats = evaluate(model, args, device, eval_generator)
    history = [
        {
            "step": 0,
            "loss": initial_loss,
            "accuracy": initial_acc,
            "relation_stats": initial_stats,
        }
    ]
    start_time = time.perf_counter()
    last_loss = None
    for step in range(1, args.steps + 1):
        model.train()
        inputs, targets, answers = make_batch(args, args.batch_size, device, train_generator)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = synthetic_loss(logits, targets, answers, args)
        loss.backward()
        optimizer.step()
        last_loss = loss.item()

        if step % args.eval_every == 0:
            eval_loss, accuracy, stats = evaluate(model, args, device, eval_generator)
            history.append(
                {
                    "step": step,
                    "loss": eval_loss,
                    "accuracy": accuracy,
                    "relation_stats": stats,
                }
            )
            print(
                f"[{variant} seed={seed}] step {step:>5} "
                f"train_loss={last_loss:.4f} eval_loss={eval_loss:.4f} acc={accuracy:.3f}",
                flush=True,
            )

    elapsed = time.perf_counter() - start_time
    final_loss, final_acc, final_stats = evaluate(model, args, device, eval_generator)
    if history[-1]["step"] != args.steps:
        history.append(
            {
                "step": args.steps,
                "loss": final_loss,
                "accuracy": final_acc,
                "relation_stats": final_stats,
            }
        )

    return {
        "variant": variant,
        "seed": seed,
        "params": parameter_count(model),
        "score_elements_per_layer": score_elements_per_layer(args, variant, context_length),
        "steps": args.steps,
        "initial_loss": initial_loss,
        "initial_accuracy": initial_acc,
        "final_loss": final_loss,
        "final_accuracy": final_acc,
        "best_accuracy": max([point["accuracy"] for point in history] + [final_acc]),
        "last_step_loss": last_loss,
        "final_relation_stats": final_stats,
        "history": history,
        "elapsed_sec": elapsed,
        "ms_per_step": elapsed / args.steps * 1000.0,
    }


def print_results(results: list[dict[str, Any]]) -> None:
    print("\n== Synthetic triangular relation comparison ==")
    print("variant                    seed  final_loss  final_acc  best_acc  ms/step  entropy")
    for result in results:
        entropy = result.get("final_relation_stats", {}).get("relation_entropy")
        entropy_text = f"{entropy:.3f}" if entropy is not None else "-"
        print(
            f"{result['variant']:<26} {result['seed']:>4} "
            f"{result['final_loss']:>10.4f} {result['final_accuracy']:>10.3f} "
            f"{result['best_accuracy']:>9.3f} {result['ms_per_step']:>8.2f} "
            f"{entropy_text:>8}"
        )


def write_csv(results: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant",
        "seed",
        "params",
        "score_elements_per_layer",
        "steps",
        "initial_loss",
        "initial_accuracy",
        "final_loss",
        "final_accuracy",
        "best_accuracy",
        "ms_per_step",
        "relation_entropy",
        "relation_max_weight",
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
                }
            )


def save_plot(results: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 6))
    for result in results:
        steps = [point["step"] for point in result["history"]]
        accs = [point["accuracy"] for point in result["history"]]
        plt.plot(steps, accs, marker="o", linewidth=1.1, label=f"{result['variant']} s{result['seed']}")
    plt.xlabel("Training step")
    plt.ylabel("Answer accuracy")
    plt.title("Synthetic give(subject, object, recipient) task")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def append_report(results: list[dict[str, Any]], args: argparse.Namespace, device: torch.device) -> None:
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "",
        f"## Experiment: {args.experiment_name}",
        "",
        "### Task",
        "",
        f"`give(subject, object, recipient)` fact `{args.facts}`개를 제시하고, query의 `(subject, object)` 조합에 맞는 recipient를 마지막 token으로 맞힌다. distractor는 subject 또는 object를 공유하므로 단일 token shortcut을 어렵게 만든다. loss_mode는 `{args.loss_mode}`이다.",
        "",
        "### Results",
        "",
        "| variant | seed | final accuracy | best accuracy | final loss | ms/step | entropy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        entropy = result.get("final_relation_stats", {}).get("relation_entropy")
        entropy_text = f"{entropy:.3f}" if entropy is not None else "-"
        lines.append(
            f"| {result['variant']} | {result['seed']} | "
            f"{result['final_accuracy']:.3f} | {result['best_accuracy']:.3f} | "
            f"{result['final_loss']:.4f} | {result['ms_per_step']:.2f} | {entropy_text} |"
        )
    lines.extend(
        [
            "",
            "### Interpretation",
            "",
            "- 이 task는 [삼각 관계]가 목표로 하는 세 토큰 조합 인식을 직접 요구한다.",
            "- 표준 attention 대비 accuracy가 높으면 구조적 이득 후보로 기록한다.",
            "- 다음 iteration에서는 가장 좋은 variant를 NSMC LM으로 재검증한다.",
            "",
        ]
    )
    mode = "a" if args.report_md.exists() else "w"
    with args.report_md.open(mode, encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.eval_every <= 0 or args.batch_size <= 0:
        raise ValueError("steps, eval-every, and batch-size must be positive")
    if args.entities < 3 or args.objects < 3:
        raise ValueError("entities and objects must be at least 3")
    if args.facts < 2:
        raise ValueError("facts must be at least 2")
    if args.relation_loss_weight < 0:
        raise ValueError("relation-loss-weight must be non-negative")

    seeds = parse_int_list(args.seeds)
    variants = parse_variant_list(args.variants)
    device = select_device()

    results = []
    for seed in seeds:
        for variant in variants:
            results.append(run_one(variant, seed, args, device))

    print_results(results)
    save_plot(results, args.plot_path)
    write_csv(results, args.csv_path)
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
                    "report_md": str(args.report_md),
                },
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if args.append_report:
        append_report(results, args, device)

    print(f"\nSaved JSON: {args.output_json}")
    print(f"Saved CSV: {args.csv_path}")
    print(f"Saved plot: {args.plot_path}")
    if args.append_report:
        print(f"Updated report: {args.report_md}")


if __name__ == "__main__":
    main()
