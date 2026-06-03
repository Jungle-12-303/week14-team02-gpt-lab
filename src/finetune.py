# -*- coding: utf-8 -*-
"""NSMC 감성 분류 미세 조정 과제 템플릿."""

from pathlib import Path
import csv
import json
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset

try:
    from .model import GPTModel
except ImportError:
    from model import GPTModel


def make_sentiment_dataset(
    train_tsv_path: str | Path,
    test_tsv_path: str | Path | None = None,
    val_ratio: float = 0.08,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    TODO: NSMC TSV를 읽어 train/validation/test 감성 분류 데이터를 만듭니다.

    반환 형식:
        [{"text": "리뷰", "label": 0 또는 1}, ...]
    """
    def read_nsmc_tsv(path: str | Path) -> list[dict]:
        rows = []
        with Path(path).open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                text = (row.get("document") or "").strip()
                label = row.get("label")
                if not text or label not in {"0", "1"}:
                    continue
                rows.append({"text": text, "label": int(label)})
        return rows

    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be greater than or equal to 0 and less than 1.")

    train_data = read_nsmc_tsv(train_tsv_path)
    rng = random.Random(seed)
    rng.shuffle(train_data)

    val_size = int(len(train_data) * val_ratio)
    val_data = train_data[:val_size]
    train_data = train_data[val_size:]

    test_data = read_nsmc_tsv(test_tsv_path) if test_tsv_path is not None else []

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        splits = {
            "nsmc_sentiment_train.jsonl": train_data,
            "nsmc_sentiment_val.jsonl": val_data,
            "nsmc_sentiment_test.jsonl": test_data,
        }
        for filename, data in splits.items():
            with (output_path / filename).open("w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return train_data, val_data, test_data


class ReviewSentimentDataset(Dataset):
    """감성 분류용 Dataset. 리뷰 하나와 label 하나를 반환합니다."""

    def __init__(
        self,
        data: list[dict],
        tokenizer,
        max_length: int = 128,
        pad_id: int | None = None,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.get_pad_id() if pad_id is None else pad_id

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """TODO: text를 encode하고 max_length까지 자르거나 padding한 뒤 label과 함께 반환합니다."""
        item = self.data[idx]
        token_ids = self.tokenizer.encode(item["text"], add_bos_eos=True)
        token_ids = token_ids[: self.max_length]
        token_ids += [self.pad_id] * (self.max_length - len(token_ids))

        input_ids = torch.tensor(token_ids, dtype=torch.long)
        label = int(item["label"])
        return input_ids, label


class GPTForSequenceClassification(nn.Module):
    """
    GPT backbone 위에 감성 분류용 Linear head를 붙인 모델.

    주의: LM head는 다음 토큰 예측용입니다. 감성 분류는 hidden state 위에 별도 classifier를 붙입니다.
    """

    def __init__(
        self,
        gpt_model: GPTModel,
        num_labels: int = 2,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.gpt = gpt_model
        self.num_labels = num_labels
        emb_dim = gpt_model.config.get("emb_dim", gpt_model.config.get("d_model"))
        if emb_dim is None:
            raise KeyError("gpt_model.config must contain 'emb_dim' or 'd_model'")
        self.dropout = nn.Dropout(drop_rate)
        self.classifier = nn.Linear(emb_dim, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: GPT hidden state에서 문장 대표 벡터를 뽑아 분류 logits를 만듭니다.

        labels가 있으면 (loss, logits), 없으면 logits를 반환합니다.
        """
        _, seq_len = input_ids.size()
        tok_embeds = self.gpt.tok_emb(input_ids)
        pos_embeds = self.gpt.pos_emb(torch.arange(seq_len, device=input_ids.device))

        x = tok_embeds + pos_embeds
        x = self.gpt.drop_emb(x)
        x = self.gpt.trf_blocks(x)
        x = self.gpt.final_norm(x)

        pooled = x[:, -1, :]
        logits = self.classifier(self.dropout(pooled))

        if labels is None:
            return logits

        loss = nn.CrossEntropyLoss()(logits, labels)
        return loss, logits

def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    outputs = model(input_batch)
    logits = outputs[1] if isinstance(outputs, tuple) else outputs
    if logits.dim() == 3:
        logits = logits[:, -1, :]
    loss = nn.CrossEntropyLoss()(logits, target_batch)
    return loss

def calc_accuracy_loader(
    data_loader,
    model: nn.Module,
    device: torch.device,
    num_batches: int | None = None,
) -> float:

    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:        
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)
        with torch.no_grad():
            outputs = model(input_batch)
            logits = outputs[1] if isinstance(outputs, tuple) else outputs
            if logits.dim() == 3:
                logits = logits[:, -1, :]
        predictions = torch.argmax(logits, dim=-1)
        correct_predictions += (predictions == target_batch).sum().item()
        num_examples += target_batch.size(0)

    accuracy = correct_predictions / num_examples if num_examples > 0 else 0
    return accuracy

def train_epoch_sentiment(
    model: GPTForSequenceClassification,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """TODO: 감성 분류 모델을 1 epoch 훈련하고 (평균 loss, accuracy)를 반환합니다."""

    model.train()
    total_loss = 0.0
    correct_predictions = 0
    num_examples = 0

    for input_batch, target_batch in train_loader:
        optimizer.zero_grad()
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        loss.backward()
        optimizer.step()

        batch_size = target_batch.size(0)
        total_loss += loss.item() * batch_size

        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)
        with torch.no_grad():
            outputs = model(input_batch)
            logits = outputs[1] if isinstance(outputs, tuple) else outputs
            if logits.dim() == 3:
                logits = logits[:, -1, :]
            predictions = torch.argmax(logits, dim=-1)
        correct_predictions += (predictions == target_batch).sum().item()
        num_examples += batch_size

    avg_loss = total_loss / num_examples if num_examples > 0 else float("nan")
    accuracy = correct_predictions / num_examples if num_examples > 0 else 0.0
    return avg_loss, accuracy

def evaluate_sentiment(
    model: GPTForSequenceClassification,
    data_loader,
    device: torch.device,
) -> tuple[float, float]:
    """TODO: 감성 분류 모델을 평가하고 (평균 loss, accuracy)를 반환합니다."""
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    num_examples = 0

    with torch.no_grad():
        for input_batch, target_batch in data_loader:
            loss = calc_loss_batch(input_batch, target_batch, model, device)

            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            outputs = model(input_batch)
            logits = outputs[1] if isinstance(outputs, tuple) else outputs
            if logits.dim() == 3:
                logits = logits[:, -1, :]
            predictions = torch.argmax(logits, dim=-1)

            batch_size = target_batch.size(0)
            total_loss += loss.item() * batch_size
            correct_predictions += (predictions == target_batch).sum().item()
            num_examples += batch_size

    avg_loss = total_loss / num_examples if num_examples > 0 else float("nan")
    accuracy = correct_predictions / num_examples if num_examples > 0 else 0.0
    model.train()
    return avg_loss, accuracy
