#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run the report pretraining setup from gpt-lab.ipynb."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bpe import BPETokenizer
from dataset import create_dataloader
from model import GPTModel
from train import calc_loss_batch, evaluate_model, generate


SEED = 42
REPORT_BPE = {
    "vocab_size": 3000,
    "corpus_limit": 1_500_000,
}
REPORT_MODEL_CONFIG = {
    "vocab_size": REPORT_BPE["vocab_size"],
    "context_length": 128,
    "emb_dim": 192,
    "n_heads": 4,
    "n_layers": 4,
    "drop_rate": 0.1,
    "qkv_bias": False,
}
REPORT_PRETRAIN = {
    "batch_size": 8,
    "num_epochs": 1,
    "lr": 3e-4,
    "weight_decay": 0.01,
    "eval_freq": 100,
    "eval_iter": 20,
    "start_context": "이 영화는",
    "max_new_tokens": 50,
    "temperature": 0.8,
    "top_k": 40,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the gpt-lab.ipynb report pretraining configuration."
    )
    parser.add_argument(
        "--device",
        choices=["auto", "mps", "cuda", "cpu"],
        default="auto",
    )
    parser.add_argument(
        "--train-char-limit",
        type=int,
        default=1_500_000,
    )
    parser.add_argument(
        "--val-char-limit",
        type=int,
        default=160_000,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results" / "report_pretrain_conda",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "report_basic_last_conda.pt",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if name == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("Requested --device mps, but MPS is not available.")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but CUDA is not available.")
    return torch.device(name)


def fmt_seconds(seconds: float) -> str:
    return f"{seconds:.1f}초" if seconds < 60 else f"{seconds / 60:.1f}분"


def compact_text(text: str, max_len: int = 500) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def decode_token_ids_safe(tokenizer: BPETokenizer, token_ids: list[int]) -> str:
    try:
        return tokenizer.decode(token_ids)
    except UnicodeDecodeError:
        special_ids = {
            tokenizer.get_pad_id(),
            tokenizer.get_unk_id(),
            tokenizer.get_bos_id(),
            tokenizer.get_eos_id(),
        }
        byte_values = []
        for token_id in token_ids:
            if token_id in special_ids:
                continue
            try:
                byte_values.extend(tokenizer.token_to_bytes(token_id))
            except (KeyError, ValueError):
                continue
        return bytes(byte_values).decode("utf-8", errors="replace")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def save_history_csv(path: Path, history: list[dict[str, float | int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "step",
                "tokens_seen",
                "train_loss",
                "val_loss",
                "elapsed_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(history)


def save_loss_plot(path: Path, history: list[dict[str, float | int]]) -> None:
    steps = [row["step"] for row in history]
    train_losses = [row["train_loss"] for row in history]
    val_losses = [row["val_loss"] for row in history]
    gaps = [val - train for train, val in zip(train_losses, val_losses)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    axes[0].plot(steps, train_losses, marker="o", label="Train loss")
    axes[0].plot(steps, val_losses, marker="o", label="Val loss")
    axes[0].set_title("Generative GPT Loss Curve")
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Cross entropy loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(steps, gaps, marker="o", color="#E45756", label="Val - Train")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_title("Generalization Gap")
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Loss gap")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_markdown_summary(
    path: Path,
    *,
    device: torch.device,
    model_param_count: int,
    train_tokens: int,
    val_tokens: int,
    train_batches: int,
    val_batches: int,
    final_row: dict[str, float | int],
    elapsed: float,
    checkpoint_path: Path,
    samples: list[dict[str, str]],
) -> None:
    sample_lines = "\n".join(
        f"- `{sample['prompt']}`: {sample['compact']}" for sample in samples
    )
    path.write_text(
        (
            "# Conda Pretrain Summary\n\n"
            f"- device: {device}\n"
            f"- torch: {torch.__version__}\n"
            f"- model params: {model_param_count:,}\n"
            f"- train tokens: {train_tokens:,}\n"
            f"- val tokens: {val_tokens:,}\n"
            f"- train batches: {train_batches:,}\n"
            f"- val batches: {val_batches:,}\n"
            f"- final train loss: {final_row['train_loss']:.4f}\n"
            f"- final val loss: {final_row['val_loss']:.4f}\n"
            f"- final gap(val-train): "
            f"{final_row['val_loss'] - final_row['train_loss']:.4f}\n"
            f"- tokens seen: {final_row['tokens_seen']:,}\n"
            f"- elapsed: {fmt_seconds(elapsed)}\n"
            f"- checkpoint: {checkpoint_path}\n\n"
            "## Samples\n\n"
            f"{sample_lines}\n"
        ),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = resolve_device(args.device)

    paths = {
        "vocab": REPO_ROOT / "data" / "vocab_bpe_basic_3000.json",
        "train_text": REPO_ROOT / "data" / "nsmc_lm_train.txt",
        "val_text": REPO_ROOT / "data" / "nsmc_lm_val.txt",
    }
    for label, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{label}: {path}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"device: {device}", flush=True)
    print(f"torch: {torch.__version__}", flush=True)
    print(f"vocab: {paths['vocab']}", flush=True)

    tokenizer = BPETokenizer(vocab_size=REPORT_BPE["vocab_size"])
    started = time.perf_counter()
    tokenizer.load(paths["vocab"])
    bpe_elapsed = time.perf_counter() - started

    model_config = dict(REPORT_MODEL_CONFIG)
    model_config["vocab_size"] = len(tokenizer.id_to_token)

    train_text = paths["train_text"].read_text(encoding="utf-8")[
        : args.train_char_limit
    ]
    val_text = paths["val_text"].read_text(encoding="utf-8")[: args.val_char_limit]
    if not train_text.strip() or not val_text.strip():
        raise RuntimeError("LM train/val corpus is empty.")

    started = time.perf_counter()
    train_token_ids = tokenizer.encode(train_text)
    val_token_ids = tokenizer.encode(val_text)
    tokenize_elapsed = time.perf_counter() - started

    train_loader = create_dataloader(
        train_token_ids,
        context_length=model_config["context_length"],
        batch_size=REPORT_PRETRAIN["batch_size"],
        stride=model_config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )
    val_loader = create_dataloader(
        val_token_ids,
        context_length=model_config["context_length"],
        batch_size=REPORT_PRETRAIN["batch_size"],
        stride=model_config["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )
    if len(train_loader) == 0 or len(val_loader) == 0:
        raise RuntimeError("DataLoader is empty.")

    model = GPTModel(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=REPORT_PRETRAIN["lr"],
        weight_decay=REPORT_PRETRAIN["weight_decay"],
    )
    model_param_count = count_parameters(model)

    print(f"actual vocab size: {len(tokenizer.id_to_token):,}", flush=True)
    print(f"train chars: {len(train_text):,}", flush=True)
    print(f"val chars: {len(val_text):,}", flush=True)
    print(f"train tokens: {len(train_token_ids):,}", flush=True)
    print(f"val tokens: {len(val_token_ids):,}", flush=True)
    print(f"train batches: {len(train_loader):,}", flush=True)
    print(f"val batches: {len(val_loader):,}", flush=True)
    print(f"model parameters: {model_param_count:,}", flush=True)
    print(f"tokenization elapsed: {fmt_seconds(tokenize_elapsed)}", flush=True)

    history: list[dict[str, float | int]] = []
    tokens_seen = 0
    global_step = 0
    pretrain_started = time.perf_counter()

    for epoch in range(1, REPORT_PRETRAIN["num_epochs"] + 1):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % REPORT_PRETRAIN["eval_freq"] == 0:
                train_loss, val_loss = evaluate_model(
                    model,
                    train_loader,
                    val_loader,
                    device,
                    REPORT_PRETRAIN["eval_iter"],
                )
                row = {
                    "epoch": epoch,
                    "step": global_step,
                    "tokens_seen": tokens_seen,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "elapsed_sec": time.perf_counter() - pretrain_started,
                }
                history.append(row)
                print(
                    f"epoch={epoch} step={global_step} "
                    f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                    f"elapsed={fmt_seconds(row['elapsed_sec'])}",
                    flush=True,
                )

        train_loss, val_loss = evaluate_model(
            model,
            train_loader,
            val_loader,
            device,
            REPORT_PRETRAIN["eval_iter"],
        )
        row = {
            "epoch": epoch,
            "step": global_step,
            "tokens_seen": tokens_seen,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "elapsed_sec": time.perf_counter() - pretrain_started,
        }
        history.append(row)
        print(
            f"epoch_end={epoch} step={global_step} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"elapsed={fmt_seconds(row['elapsed_sec'])}",
            flush=True,
        )

    pretrain_elapsed = time.perf_counter() - pretrain_started
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "pretrain": REPORT_PRETRAIN,
            "history": history,
            "tokens_seen": tokens_seen,
            "global_step": global_step,
            "device": str(device),
            "seed": SEED,
        },
        args.checkpoint_path,
    )

    samples = []
    model.eval()
    for prompt in ["이 영화는", "배우들의 연기는", "스토리는"]:
        start_ids = torch.tensor(
            tokenizer.encode(prompt),
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        with torch.no_grad():
            sampled_ids = generate(
                model=model,
                idx=start_ids,
                max_new_tokens=REPORT_PRETRAIN["max_new_tokens"],
                context_size=model_config["context_length"],
                temperature=REPORT_PRETRAIN["temperature"],
                top_k=REPORT_PRETRAIN["top_k"],
                eos_id=tokenizer.get_eos_id(),
            )
        sample_text = decode_token_ids_safe(
            tokenizer,
            sampled_ids.squeeze(0).tolist(),
        )
        sample = {
            "prompt": prompt,
            "text": sample_text,
            "compact": compact_text(sample_text),
        }
        samples.append(sample)
        print("=" * 80, flush=True)
        print(f"prompt: {prompt}", flush=True)
        print(sample["compact"], flush=True)

    final_row = history[-1]
    summary = {
        "seed": SEED,
        "device": str(device),
        "torch_version": torch.__version__,
        "paths": {key: str(path) for key, path in paths.items()},
        "bpe": {
            **REPORT_BPE,
            "actual_vocab_size": len(tokenizer.id_to_token),
            "load_elapsed_sec": bpe_elapsed,
        },
        "model_config": model_config,
        "pretrain": REPORT_PRETRAIN,
        "train_char_limit": args.train_char_limit,
        "val_char_limit": args.val_char_limit,
        "train_chars": len(train_text),
        "val_chars": len(val_text),
        "train_tokens": len(train_token_ids),
        "val_tokens": len(val_token_ids),
        "train_batches": len(train_loader),
        "val_batches": len(val_loader),
        "model_param_count": model_param_count,
        "tokenize_elapsed_sec": tokenize_elapsed,
        "pretrain_elapsed_sec": pretrain_elapsed,
        "final": final_row,
        "gap_val_minus_train": final_row["val_loss"] - final_row["train_loss"],
        "checkpoint": str(args.checkpoint_path),
        "samples": samples,
    }

    summary_path = args.results_dir / "summary.json"
    history_csv_path = args.results_dir / "history.csv"
    samples_path = args.results_dir / "samples.json"
    plot_path = args.results_dir / "loss_curve.png"
    markdown_path = args.results_dir / "summary.md"

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    samples_path.write_text(
        json.dumps(samples, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_history_csv(history_csv_path, history)
    save_loss_plot(plot_path, history)
    save_markdown_summary(
        markdown_path,
        device=device,
        model_param_count=model_param_count,
        train_tokens=len(train_token_ids),
        val_tokens=len(val_token_ids),
        train_batches=len(train_loader),
        val_batches=len(val_loader),
        final_row=final_row,
        elapsed=pretrain_elapsed,
        checkpoint_path=args.checkpoint_path,
        samples=samples,
    )

    print(f"summary: {summary_path}", flush=True)
    print(f"history_csv: {history_csv_path}", flush=True)
    print(f"samples: {samples_path}", flush=True)
    print(f"plot: {plot_path}", flush=True)
    print(f"markdown: {markdown_path}", flush=True)
    print(f"checkpoint saved: {args.checkpoint_path}", flush=True)
    print(f"pretrain elapsed: {fmt_seconds(pretrain_elapsed)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
