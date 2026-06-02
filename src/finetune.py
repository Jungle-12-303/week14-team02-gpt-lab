# -*- coding: utf-8 -*-
"""NSMC 감성 분류 미세 조정 과제 템플릿."""

import csv
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset

try:
    from .model import GPTModel
except ImportError:
    from model import GPTModel


# NSMC TSV를 읽어 train/validation/test 감성 분류 데이터를 만듭니다.
def make_sentiment_dataset(
    train_tsv_path: str | Path,
    test_tsv_path: str | Path | None = None,
    val_ratio: float = 0.08,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    반환 형식:
        [{"text": "리뷰", "label": 0 또는 1}, ...]
    """
    def read_nsmc_tsv(path: str | Path) -> list[dict]:
        rows = []

        with Path(path).open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")

            for row in reader:
                text = row.get("document") or ""
                text = " ".join(text.split())
                label = row.get("label")

                if not text or label not in {"0", "1"}:
                    continue

                rows.append({
                    "text": text,
                    "label": int(label)
                })

        return rows

    train_rows = read_nsmc_tsv(train_tsv_path)              # train TSV -> list[dict]

    rng = random.Random(seed)
    rng.shuffle(train_rows)

    if val_ratio <= 0:
        val_size = 0
    else:
        val_size = max(1, int(len(train_rows) * val_ratio))

    val_data = train_rows[:val_size]
    train_data = train_rows[val_size:]

    if test_tsv_path is None:
        test_data = []
    else:
        test_data = read_nsmc_tsv(test_tsv_path)

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        files = {
            "nsmc_sentiment_train.jsonl": train_data,
            "nsmc_sentiment_val.jsonl": val_data,
            "nsmc_sentiment_test.jsonl": test_data
        }

        for filename, rows in files.items():
            with (output_path / filename).open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return train_data, val_data, test_data


# 감성 분류용 Dataset. 리뷰 하나와 label 하나를 반환합니다.
class ReviewSentimentDataset(Dataset):

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

    # text를 encode하고 max_length까지 자르거나 padding한 뒤 label과 함께 반환합니다.
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.data[idx]

        token_ids = self.tokenizer.encode(row["text"], add_bos_eos=True)
        token_ids = token_ids[:self.max_length]

        if len(token_ids) < self.max_length:
            token_ids = token_ids + [self.pad_id] * (self.max_length - len(token_ids))

        input_ids = torch.tensor(token_ids, dtype=torch.long)           # (max_length,)
        label = int(row["label"])

        return input_ids, label               # (batch_size, max_length), (batch_size,)


class GPTForSequenceClassification(nn.Module):
    """
    GPT backbone 위에 감성 분류용 Linear head를 붙인 모델.

    주의: LM head는 다음 토큰 예측용입니다. 감성 분류는 hidden state 위에 별도 classifier를 붙입니다.
    """
    # dropout과 classifier를 정의하세요. classifier 입력 차원은 gpt_model.config["emb_dim"]입니다.
    def __init__(
        self,
        gpt_model: GPTModel,
        num_labels: int = 2,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.gpt = gpt_model
        self.num_labels = num_labels
        self.pad_id = 0                         # 현재 BPE <pad> ID

        self.dropout = nn.Dropout(drop_rate)
        self.classifier = nn.Linear(
            gpt_model.config["emb_dim"],        # 입력 차원: hidden shape 마지막 차원
            num_labels                          # 출력 차원: logits shape 마지막 차원
        )

    # GPT hidden state에서 문장 대표 벡터를 뽑아 분류 logits를 만듭니다.
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        labels가 있으면 (loss, logits), 없으면 logits를 반환합니다.
        """
        x = self.gpt.embedding(input_ids)                                       # (batch_size, seq_len, emb_dim)
        x = self.gpt.blocks(x)                                                  # (batch_size, seq_len, emb_dim)
        x = self.gpt.final_norm(x)                                              # (batch_size, seq_len, emb_dim)

        non_pad_mask = input_ids.ne(self.pad_id)                                # (batch_size, seq_len)
        last_token_idx = non_pad_mask.sum(dim=1).clamp(min=1) - 1               # (batch_size,)
        batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)    # (batch_size,)

        pooled = x[batch_idx, last_token_idx]               # 각 샘플의 마지막 non-pad hidden state (batch_size, emb_dim)
        pooled = self.dropout(pooled)                       # (batch_size, emb_dim)
        logits = self.classifier(pooled)                    # (batch_size, num_labels)

        if labels is None:
            return logits                                   # (batch_size, num_labels)

        labels = labels.to(input_ids.device).long()         # (batch_size,)
        loss = torch.nn.functional.cross_entropy(logits, labels)

        return loss, logits

# 감성 분류 모델을 1 epoch 훈련하고 (평균 loss, accuracy)를 반환합니다.
def train_epoch_sentiment(
    model: GPTForSequenceClassification,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.to(device)
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for input_ids, labels in train_loader:
        input_ids = input_ids.to(device)                    # (batch_size, seq_len)
        labels = labels.to(device).long()                   # (batch_size,)

        optimizer.zero_grad()
        loss, logits = model(input_ids, labels=labels)      # loss: scalar, logits: (batch_size, num_labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size

        preds = torch.argmax(logits, dim=-1)                # (batch_size,)
        total_correct += (preds == labels).sum().item()
        total_count += batch_size

    if total_count == 0:
        return float("nan"), 0.0

    avg_loss = total_loss / total_count                     # 샘플 단위 평균 loss
    accuracy = total_correct / total_count                  # 전체 샘플 중 맞춘 비율

    return avg_loss, accuracy

# 감성 분류 모델을 평가하고 (평균 loss, accuracy)를 반환합니다.
def evaluate_sentiment(
    model: GPTForSequenceClassification,
    data_loader,
    device: torch.device,
) -> tuple[float, float]:
    model.to(device)

    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for input_ids, labels in data_loader:
            input_ids = input_ids.to(device)                # (batch_size, seq_len)
            labels = labels.to(device).long()               # (batch_size,)

            loss, logits = model(input_ids, labels=labels)  # loss: scalar, logits: (batch_size, num_labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size

            preds = torch.argmax(logits, dim=-1)            # (batch_size,)
            total_correct += (preds == labels).sum().item()
            total_count += batch_size

    if was_training:
        model.train()

    if total_count == 0:
        return float("nan"), 0.0

    avg_loss = total_loss / total_count                     # 샘플 단위 평균 loss
    accuracy = total_correct / total_count                  # 전체 샘플 중 맞춘 비율

    return avg_loss, accuracy
