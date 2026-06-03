# -*- coding: utf-8 -*-
"""NSMC 감성 분류 미세 조정 과제 템플릿."""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import csv
import json
import random
from pathlib import Path

try:
    from .model import GPTModel
except ImportError:
    from model import GPTModel

def _read_nsmc_tsv(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            text = row["document"].strip()
            if not text:
                continue

            rows.append({
                "text": text,
                "label": int(row["label"]),
            })

    return rows


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
    train_val_data = _read_nsmc_tsv(train_tsv_path)

    rng = random.Random(seed)
    rng.shuffle(train_val_data)

    val_size = int(len(train_val_data) * val_ratio)

    val_data = train_val_data[:val_size]
    train_data = train_val_data[val_size:]

    if test_tsv_path is None:
        test_data = []
    else:
        test_data = _read_nsmc_tsv(test_tsv_path)
    
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

        padding_length = self.max_length - len(token_ids)
        token_ids = token_ids + [self.pad_id] * padding_length

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
        # TODO: dropout과 classifier를 정의하세요. classifier 입력 차원은 gpt_model.config["emb_dim"]입니다.
        self.dropout = nn.Dropout(drop_rate)
        self.classifier = nn.Linear(
            in_features=gpt_model.config["emb_dim"],
            out_features=num_labels
        )


    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: GPT hidden state에서 문장 대표 벡터를 뽑아 분류 logits를 만듭니다.

        labels가 있으면 (loss, logits), 없으면 logits를 반환합니다.
        """
        x = self.gpt.embedding(input_ids)
        x = self.gpt.blocks(x)
        x = self.gpt.final_norm(x)

        last_token_hidden = x [:, -1, :]

        logits = self.classifier(self.dropout(last_token_hidden))

        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
            return loss, logits
        
        return logits


def train_epoch_sentiment(
    model: GPTForSequenceClassification,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """TODO: 감성 분류 모델을 1 epoch 훈련하고 (평균 loss, accuracy)를 반환합니다."""
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for input_ids, labels in train_loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        loss, logits = model(input_ids, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size

        predictions = torch.argmax(logits, dim=-1)
        total_correct += (predictions == labels).sum().item()
        total_examples += batch_size
    
    avg_loss = total_loss / total_examples
    accuracy = total_correct / total_examples

    return avg_loss, accuracy


def evaluate_sentiment(
    model: GPTForSequenceClassification,
    data_loader,
    device: torch.device,
) -> tuple[float, float]:
    """TODO: 감성 분류 모델을 평가하고 (평균 loss, accuracy)를 반환합니다."""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for input_ids, labels in data_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            loss, logits = model(input_ids, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size

            predictions = torch.argmax(logits, dim=-1)
            total_correct += (predictions == labels).sum().item()
            total_examples += batch_size
    
    avg_loss = total_loss / total_examples
    accuracy = total_correct / total_examples

    return avg_loss, accuracy
