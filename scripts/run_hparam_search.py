#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run a small deterministic hyperparameter search for the mini GPT LM."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import random
import sys
import time
from pathlib import Path
from typing import Any


if sys.version_info < (3, 11):
    raise SystemExit(
        "This project requires Python 3.11+. Activate the gpt-lab environment, "
        "then run: python scripts/run_hparam_search.py --budget medium"
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch

from bpe import BPETokenizer
from dataset import create_dataloader
from model import GPTModel
from train import calc_loss_batch, evaluate_model, generate


CANDIDATES = {
    "batch_size": [2, 4, 8, 16],
    "drop_rate": [0.0, 0.1, 0.2],
    "learning_rate": [1e-4, 3e-4, 5e-4],
    "context_length": [64, 128],
    "n_layers": [1, 2, 4],
    "emb_dim": [64, 128, 192],
}

FOCUSED_CANDIDATES = {
    "batch_size": [2, 4, 8, 16],
    "drop_rate": [0.1, 0.15, 0.2],
    "learning_rate": [3e-4],
    "context_length": [64, 128],
    "n_layers": [4],
    "emb_dim": [192],
}

BASELINE_HPARAMS = {
    "batch_size": 8,
    "drop_rate": 0.1,
    "learning_rate": 3e-4,
    "context_length": 128,
    "n_layers": 4,
    "emb_dim": 192,
}

BUDGET_SIZES = {
    "focused": None,
    "light": 24,
    "medium": 64,
    "full": None,
}

DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "hparam_search"
DEFAULT_VOCAB_PATH = REPO_ROOT / "data" / "vocab_bpe_basic_3000.json"
DEFAULT_TRAIN_TEXT_PATH = REPO_ROOT / "data" / "nsmc_lm_train.txt"
DEFAULT_VAL_TEXT_PATH = REPO_ROOT / "data" / "nsmc_lm_val.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic hyperparameter search for mini GPT pretraining."
    )
    parser.add_argument("--budget", choices=sorted(BUDGET_SIZES), default="medium")
    parser.add_argument("--num-runs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--vocab-path", type=Path, default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--train-text-path", type=Path, default=DEFAULT_TRAIN_TEXT_PATH)
    parser.add_argument("--val-text-path", type=Path, default=DEFAULT_VAL_TEXT_PATH)
    parser.add_argument("--train-char-limit", type=int, default=300_000)
    parser.add_argument("--val-char-limit", type=int, default=40_000)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--eval-iter", type=int, default=20)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--start-context", type=str, default="이 영화는")
    parser.add_argument(
        "--backfill-history",
        action="store_true",
        help="Create history.jsonl/history.csv from existing completed runs when possible.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but CUDA is not available.")
    if name == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("Requested --device mps, but MPS is not available.")
    return torch.device(name)


def candidate_grid(candidates: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(candidates)
    values = [candidates[key] for key in keys]
    return [dict(zip(keys, combination)) for combination in itertools.product(*values)]


def full_grid() -> list[dict[str, Any]]:
    return candidate_grid(CANDIDATES)


def focused_grid() -> list[dict[str, Any]]:
    return candidate_grid(FOCUSED_CANDIDATES)


def run_id_for(hparams: dict[str, Any]) -> str:
    payload = json.dumps(hparams, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"run_{digest}"


def select_hparams(budget: str, num_runs: int | None, seed: int) -> list[dict[str, Any]]:
    grid = focused_grid() if budget == "focused" else full_grid()
    size = len(grid) if BUDGET_SIZES[budget] is None else BUDGET_SIZES[budget]
    if num_runs is not None:
        size = num_runs
    size = min(size, len(grid))

    rng = random.Random(seed)
    baseline = dict(BASELINE_HPARAMS)
    pool = [item for item in grid if item != baseline]

    selected = [baseline]
    if size > 1:
        selected.extend(rng.sample(pool, k=size - 1))
    return selected[:size]


def load_completed_run_ids(runs_path: Path) -> set[str]:
    completed: set[str] = set()
    if not runs_path.exists():
        return completed

    with runs_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") == "completed" and row.get("run_id"):
                completed.add(str(row["run_id"]))
    return completed


def load_completed_history_run_ids(history_path: Path) -> set[str]:
    completed: set[str] = set()
    if not history_path.exists():
        return completed

    with history_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("run_id"):
                completed.add(str(row["run_id"]))
    return completed


def flatten_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    hparams = row.get("hyperparams", {})
    flat = {
        "run_id": row.get("run_id", ""),
        "status": row.get("status", ""),
        "seed": row.get("seed", ""),
        "device": row.get("device", ""),
        "train_char_limit": row.get("train_char_limit", ""),
        "val_char_limit": row.get("val_char_limit", ""),
        "num_epochs": row.get("num_epochs", ""),
        "model_param_count": row.get("model_param_count", ""),
        "train_batches": row.get("train_batches", ""),
        "val_batches": row.get("val_batches", ""),
        "final_train_loss": row.get("final_train_loss", ""),
        "final_val_loss": row.get("final_val_loss", ""),
        "best_val_loss": row.get("best_val_loss", ""),
        "elapsed_seconds": row.get("elapsed_seconds", ""),
        "tokens_seen": row.get("tokens_seen", ""),
        "generated_sample": row.get("generated_sample", ""),
        "error_message": row.get("error_message", ""),
    }
    for key in CANDIDATES:
        flat[key] = hparams.get(key, "")
    return flat


def flatten_history_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    hparams = row.get("hyperparams", {})
    flat = {
        "run_id": row.get("run_id", ""),
        "seed": row.get("seed", ""),
        "device": row.get("device", ""),
        "epoch": row.get("epoch", ""),
        "step": row.get("step", ""),
        "tokens_seen": row.get("tokens_seen", ""),
        "train_loss": row.get("train_loss", ""),
        "val_loss": row.get("val_loss", ""),
        "loss_gap": row.get("loss_gap", ""),
    }
    for key in CANDIDATES:
        flat[key] = hparams.get(key, "")
    return flat


def rebuild_summary_csv(runs_path: Path, summary_path: Path) -> None:
    rows: dict[str, dict[str, Any]] = {}
    if runs_path.exists():
        with runs_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("status") == "completed":
                    rows[row["run_id"]] = row

    completed_rows = list(rows.values())
    completed_rows.sort(key=lambda row: float(row.get("best_val_loss", float("inf"))))

    fieldnames = [
        "run_id",
        "status",
        "seed",
        "device",
        "batch_size",
        "drop_rate",
        "learning_rate",
        "context_length",
        "n_layers",
        "emb_dim",
        "train_char_limit",
        "val_char_limit",
        "num_epochs",
        "model_param_count",
        "train_batches",
        "val_batches",
        "final_train_loss",
        "final_val_loss",
        "best_val_loss",
        "elapsed_seconds",
        "tokens_seen",
        "generated_sample",
        "error_message",
    ]

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in completed_rows:
            writer.writerow(flatten_for_csv(row))


def rebuild_history_csv(history_path: Path, history_csv_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    if history_path.exists():
        with history_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rows.append(json.loads(line))

    rows.sort(
        key=lambda row: (
            row.get("run_id", ""),
            int(row.get("epoch", 0)),
            int(row.get("step", 0)),
        )
    )

    fieldnames = [
        "run_id",
        "seed",
        "device",
        "batch_size",
        "drop_rate",
        "learning_rate",
        "context_length",
        "n_layers",
        "emb_dim",
        "epoch",
        "step",
        "tokens_seen",
        "train_loss",
        "val_loss",
        "loss_gap",
    ]

    history_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with history_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(flatten_history_for_csv(row))


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def release_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


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
        byte_values: list[int] = []
        for token_id in token_ids:
            if token_id in special_ids:
                continue
            try:
                byte_values.extend(tokenizer.token_to_bytes(token_id))
            except (KeyError, ValueError):
                continue
        return bytes(byte_values).decode("utf-8", errors="replace")


def validate_inputs(args: argparse.Namespace) -> None:
    require_positive("train-char-limit", args.train_char_limit)
    require_positive("val-char-limit", args.val_char_limit)
    require_positive("num-epochs", args.num_epochs)
    require_positive("eval-iter", args.eval_iter)
    require_positive("max-new-tokens", args.max_new_tokens)
    if args.num_runs is not None:
        require_positive("num-runs", args.num_runs)

    required_paths = [
        ("vocab", args.vocab_path),
        ("LM train text", args.train_text_path),
        ("LM val text", args.val_text_path),
    ]
    missing = [f"{label}: {path}" for label, path in required_paths if not path.exists()]
    if missing:
        message = "\n".join(missing)
        raise FileNotFoundError(
            "Required data files are missing. Run `python download_data.py` first "
            "and make sure the BPE vocab exists.\n" + message
        )


def make_model_config(
    hparams: dict[str, Any],
    vocab_size: int,
    n_heads: int,
) -> dict[str, Any]:
    emb_dim = int(hparams["emb_dim"])
    if emb_dim % n_heads != 0:
        raise ValueError(f"emb_dim={emb_dim} must be divisible by n_heads={n_heads}")
    return {
        "vocab_size": vocab_size,
        "context_length": int(hparams["context_length"]),
        "emb_dim": emb_dim,
        "n_heads": n_heads,
        "n_layers": int(hparams["n_layers"]),
        "drop_rate": float(hparams["drop_rate"]),
        "qkv_bias": False,
    }


def run_one(
    hparams: dict[str, Any],
    args: argparse.Namespace,
    tokenizer: BPETokenizer,
    train_token_ids: list[int],
    val_token_ids: list[int],
    device: torch.device,
) -> dict[str, Any]:
    started = time.perf_counter()
    run_id = run_id_for(hparams)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_config = make_model_config(
        hparams=hparams,
        vocab_size=len(tokenizer.id_to_token),
        n_heads=args.n_heads,
    )
    context_length = model_config["context_length"]
    batch_size = int(hparams["batch_size"])

    train_loader = create_dataloader(
        train_token_ids,
        context_length=context_length,
        batch_size=batch_size,
        stride=context_length,
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )
    val_loader = create_dataloader(
        val_token_ids,
        context_length=context_length,
        batch_size=batch_size,
        stride=context_length,
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )
    if len(train_loader) == 0 or len(val_loader) == 0:
        raise RuntimeError(
            "DataLoader is empty. Increase train/val char limits or reduce context_length."
        )

    model = GPTModel(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(hparams["learning_rate"]),
        weight_decay=args.weight_decay,
    )

    history = []
    tokens_seen = 0
    global_step = 0

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1

        train_loss, val_loss = evaluate_model(
            model,
            train_loader,
            val_loader,
            device,
            args.eval_iter,
        )
        history.append(
            {
                "epoch": epoch,
                "step": global_step,
                "tokens_seen": tokens_seen,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

    final = history[-1]
    best_val_loss = min(row["val_loss"] for row in history)

    model.eval()
    prompt_ids = torch.tensor(
        tokenizer.encode(args.start_context),
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    with torch.no_grad():
        sampled_ids = generate(
            model=model,
            idx=prompt_ids,
            max_new_tokens=args.max_new_tokens,
            context_size=context_length,
            temperature=args.temperature,
            top_k=args.top_k,
            eos_id=tokenizer.get_eos_id(),
        )
    generated_sample = decode_token_ids_safe(
        tokenizer,
        sampled_ids.squeeze(0).tolist(),
    )

    elapsed = time.perf_counter() - started
    return {
        "run_id": run_id,
        "status": "completed",
        "seed": args.seed,
        "device": str(device),
        "hyperparams": hparams,
        "train_char_limit": args.train_char_limit,
        "val_char_limit": args.val_char_limit,
        "num_epochs": args.num_epochs,
        "model_param_count": count_parameters(model),
        "train_batches": len(train_loader),
        "val_batches": len(val_loader),
        "final_train_loss": final["train_loss"],
        "final_val_loss": final["val_loss"],
        "best_val_loss": best_val_loss,
        "elapsed_seconds": elapsed,
        "tokens_seen": tokens_seen,
        "generated_sample": " ".join(generated_sample.split()),
        "error_message": "",
        "history": history,
    }


def history_rows_from_run(row: dict[str, Any]) -> list[dict[str, Any]]:
    history_rows = []
    for history_item in row.get("history", []):
        train_loss = history_item["train_loss"]
        val_loss = history_item["val_loss"]
        history_rows.append(
            {
                "run_id": row["run_id"],
                "seed": row["seed"],
                "device": row["device"],
                "hyperparams": row["hyperparams"],
                "epoch": history_item["epoch"],
                "step": history_item["step"],
                "tokens_seen": history_item["tokens_seen"],
                "train_loss": train_loss,
                "val_loss": val_loss,
                "loss_gap": val_loss - train_loss,
            }
        )
    return history_rows


def strip_history(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    row.pop("history", None)
    return row


def backfill_history_from_runs(
    runs_path: Path,
    history_path: Path,
    history_csv_path: Path,
) -> int:
    existing = load_completed_history_run_ids(history_path)
    written = 0
    if not runs_path.exists():
        raise FileNotFoundError(f"runs file not found: {runs_path}")

    with runs_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") != "completed":
                continue
            run_id = str(row.get("run_id", ""))
            if not run_id or run_id in existing:
                continue

            if row.get("history"):
                history_rows = history_rows_from_run(row)
            else:
                train_loss = float(row["final_train_loss"])
                val_loss = float(row["final_val_loss"])
                history_rows = [
                    {
                        "run_id": run_id,
                        "seed": row["seed"],
                        "device": row["device"],
                        "hyperparams": row["hyperparams"],
                        "epoch": int(row.get("num_epochs", 1)),
                        "step": row.get("train_batches", ""),
                        "tokens_seen": row.get("tokens_seen", ""),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "loss_gap": val_loss - train_loss,
                    }
                ]

            for history_row in history_rows:
                append_jsonl(history_path, history_row)
                written += 1
            existing.add(run_id)

    rebuild_history_csv(history_path, history_csv_path)
    return written


def print_dry_run(selected: list[dict[str, Any]], args: argparse.Namespace) -> None:
    baseline_run_id = run_id_for(BASELINE_HPARAMS)
    selected_ids = {run_id_for(hparams) for hparams in selected}
    print(f"budget={args.budget}")
    print(f"num_runs={len(selected)}")
    print(f"grid_size={len(focused_grid()) if args.budget == 'focused' else len(full_grid())}")
    print(f"baseline_included={baseline_run_id in selected_ids}")
    print(f"baseline_run_id={baseline_run_id}")
    for index, hparams in enumerate(selected, start=1):
        print(f"{index:03d} {run_id_for(hparams)} {json.dumps(hparams, sort_keys=True)}")


def main() -> int:
    args = parse_args()
    selected = select_hparams(args.budget, args.num_runs, args.seed)

    runs_path = args.results_dir / "runs.jsonl"
    summary_path = args.results_dir / "summary.csv"
    history_path = args.results_dir / "history.jsonl"
    history_csv_path = args.results_dir / "history.csv"

    if args.backfill_history:
        written = backfill_history_from_runs(runs_path, history_path, history_csv_path)
        rebuild_summary_csv(runs_path, summary_path)
        print(f"backfilled_history_rows={written}")
        print(f"history: {history_path}")
        print(f"history_csv: {history_csv_path}")
        return 0

    if args.dry_run:
        print_dry_run(selected, args)
        return 0

    validate_inputs(args)
    device = resolve_device(args.device)

    tokenizer = BPETokenizer(vocab_size=3000)
    tokenizer.load(args.vocab_path)

    train_text = args.train_text_path.read_text(encoding="utf-8")[: args.train_char_limit]
    val_text = args.val_text_path.read_text(encoding="utf-8")[: args.val_char_limit]
    if not train_text.strip() or not val_text.strip():
        raise RuntimeError("LM train/val text is empty after applying char limits.")

    print("Encoding train/val text...")
    train_token_ids = tokenizer.encode(train_text)
    val_token_ids = tokenizer.encode(val_text)
    print(f"device={device}")
    print(f"selected_runs={len(selected)}")
    print(f"train_tokens={len(train_token_ids):,}")
    print(f"val_tokens={len(val_token_ids):,}")

    completed_run_ids = load_completed_run_ids(runs_path)
    completed_history_run_ids = load_completed_history_run_ids(history_path)

    for index, hparams in enumerate(selected, start=1):
        run_id = run_id_for(hparams)
        if run_id in completed_run_ids:
            print(f"[{index}/{len(selected)}] skip completed {run_id}")
            continue

        print(f"[{index}/{len(selected)}] start {run_id} {hparams}")
        try:
            row = run_one(
                hparams=hparams,
                args=args,
                tokenizer=tokenizer,
                train_token_ids=train_token_ids,
                val_token_ids=val_token_ids,
                device=device,
            )
            print(
                f"[{index}/{len(selected)}] done {run_id} "
                f"val_loss={row['final_val_loss']:.4f} "
                f"elapsed={row['elapsed_seconds']:.1f}s"
            )
            if run_id not in completed_history_run_ids:
                for history_row in history_rows_from_run(row):
                    append_jsonl(history_path, history_row)
                completed_history_run_ids.add(run_id)
            row = strip_history(row)
            completed_run_ids.add(run_id)
        except Exception as exc:
            row = {
                "run_id": run_id,
                "status": "failed",
                "seed": args.seed,
                "device": str(device),
                "hyperparams": hparams,
                "train_char_limit": args.train_char_limit,
                "val_char_limit": args.val_char_limit,
                "num_epochs": args.num_epochs,
                "model_param_count": "",
                "train_batches": "",
                "val_batches": "",
                "final_train_loss": "",
                "final_val_loss": "",
                "best_val_loss": "",
                "elapsed_seconds": "",
                "tokens_seen": "",
                "generated_sample": "",
                "error_message": repr(exc),
            }
            print(f"[{index}/{len(selected)}] failed {run_id}: {exc}")

        append_jsonl(runs_path, row)
        rebuild_summary_csv(runs_path, summary_path)
        rebuild_history_csv(history_path, history_csv_path)
        release_device_cache(device)

    rebuild_summary_csv(runs_path, summary_path)
    rebuild_history_csv(history_path, history_csv_path)
    print(f"runs: {runs_path}")
    print(f"summary: {summary_path}")
    print(f"history: {history_path}")
    print(f"history_csv: {history_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
