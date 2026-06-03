# -*- coding: utf-8 -*-
"""NSMC 감성 분류 미세 조정 과제 템플릿."""

from pathlib import Path
import csv
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset

try:
    from .model import GPTModel
except ImportError:
    from model import GPTModel


def make_sentiment_dataset(
    # 훈련용 TSV 파일 경로
    train_tsv_path: str | Path,
    # test용 파일 경로
    test_tsv_path: str | Path | None = None,
    # validation으로 가질 데이터 비율
    val_ratio: float = 0.08,
    # 데이터를 섞는 랜덤 seed
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    TODO: NSMC TSV를 읽어 train/validation/test 감성 분류 데이터를 만듭니다.

    반환 형식:
        [{"text": "리뷰", "label": 0 또는 1}, ...]
    """
    def read_tsv(path: str | Path) -> list[dict]:
        rows = []

        with open(path, "r", encoding="utf-8") as f:
            
            reader = csv.DictReader(f, delimiter="\t")

            for row in reader:
                text = row["document"]
                # 빈 리뷰 건너 뛰기
                if text is None or text == "":
                    continue
                
                rows.append({
                    "text": text,
                    "label": int(row["label"]),
                })

        return rows
    
    # train 파일 읽기
    train_all = read_tsv(train_tsv_path)

    # seed를 고정해서 랜덤으로 train 데이터 섞기
    rng = random.Random(seed)
    rng.shuffle(train_all)
    # validation으로 뺄 데이터 개수 계산
    val_size = int(len(train_all) * val_ratio)
    # 모델 검증용 data
    val_data = train_all[:val_size]
    train_data = train_all[val_size:]

    # test 파일이 있으면 따로읽음 (최종 성능 확인용)
    if test_tsv_path is None:
        test_data = []
    else:
        test_data = read_tsv(test_tsv_path)

    return train_data, val_data, test_data


class ReviewSentimentDataset(Dataset):
    """감성 분류용 Dataset. 리뷰 하나와 label 하나를 반환합니다."""

    def __init__(
        self,
        data: list[dict],
        # token to id
        tokenizer,
        max_length: int = 128,
        pad_id: int | None = None,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        # padding id가져오기
        self.pad_id = tokenizer.get_pad_id() if pad_id is None else pad_id

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """TODO: text를 encode하고 max_length까지 자르거나 padding한 뒤 label과 함께 반환합니다."""
        # data가져오기
        item = self.data[idx]
        # token to id
        token_ids = self.tokenizer.encode(item["text"])
        # max length까지 자르기
        token_ids = token_ids[: self.max_length]
        # max length에 맞게 padding 추가하기
        padding_length = self.max_length - len(token_ids)
        if padding_length > 0:
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
        # 분류 클래스 개수
        num_labels: int = 2,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.gpt = gpt_model
        self.num_labels = num_labels
        # TODO: dropout과 classifier를 정의하세요. classifier 입력 차원은 gpt_model.config["emb_dim"]입니다.
        self.dropout = nn.Dropout(drop_rate)
        self.classifier = nn.Linear(gpt_model.config["emb_dim"], num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: GPT hidden state에서 문장 대표 벡터를 뽑아 분류 logits를 만듭니다.

        labels가 있으면 (loss, logits), 없으면 logits를 반환합니다.
        """
        # GPT의 lm_head를 건너뛰고 hidden state까지만 얻기 위함
        # emb 결과
        embeddings = self.gpt.embedding(input_ids)
        # transformerBlock 결과
        hidden_states = self.gpt.blocks(embeddings)
        # 정규화된 hidden state
        hidden_states = self.gpt.result_norm(hidden_states)

        # [batch, seq_len, emb_dim]
        # [batch, emb_dim]
        pooled = hidden_states[:, -1, :]
        pooled = self.dropout(pooled)
        # 분류점수로 바꿈
        logits = self.classifier(pooled)

        if labels is None:
            return logits

        loss = nn.functional.cross_entropy(logits, labels)
        return loss, logits

# 1 epoch 훈련
def train_epoch_sentiment(
    model: GPTForSequenceClassification,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """TODO: 감성 분류 모델을 1 epoch 훈련하고 (평균 loss, accuracy)를 반환합니다."""
    model.to(device)
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for input_ids, labels in train_loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        loss, logits = model(input_ids, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total

    return avg_loss, accuracy

def evaluate_sentiment(
    model: GPTForSequenceClassification,
    data_loader,
    device: torch.device,
) -> tuple[float, float]:
    """TODO: 감성 분류 모델을 평가하고 (평균 loss, accuracy)를 반환합니다."""
    model.to(device)
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for input_ids, labels in data_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            loss, logits = model(input_ids, labels)

            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
    if len(data_loader) == 0:
        return float("nan"), float("nan")
    
    avg_loss = total_loss / len(data_loader)
    # 정확도
    accuracy = correct / total

    return avg_loss, accuracy