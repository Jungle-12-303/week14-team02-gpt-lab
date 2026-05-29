# -*- coding: utf-8 -*-
"""토큰 임베딩 + 위치 임베딩 과제 템플릿."""

import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    """
    token ID를 Transformer 입력 벡터로 바꿉니다.

    구현할 구조:
    - token embedding: nn.Embedding(vocab_size, emb_dim)
    - position embedding: nn.Embedding(context_length, emb_dim)
    - token embedding + position embedding
    - dropout
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        context_length: int,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.context_length = context_length
        # TODO: token_embedding, position_embedding, dropout을 정의하세요.
        # 랜덤 시드 고정
        # torch.manual_seed(123)
        self.token_embedding = nn.Embedding(vocab_size, self.emb_dim)
        self.position_embedding = nn.Embedding(self.context_length, self.emb_dim)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TODO: token embedding과 position embedding을 더한 뒤 dropout을 적용합니다.

        Args:
            x: (batch_size, seq_len) token IDs

        Returns:
            (batch_size, seq_len, emb_dim)
        """
        # 1. broadcasting
        seq_len = x.shape[1]
        # device=x.device 입력 x가 GPU에 있으면 positions도 GPU에서 만든다.
        # x랑 positions의 디바이스가 다르면 embedding 계산이나 덧셈에서 device mismatch 에러가 날 수 있다.
        positions = torch.arange(seq_len, device=x.device)

        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)

        return self.dropout(token_emb + pos_emb)

        # 2. position을 batch size로 명시적 확장
        batch_size, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)

        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)

        return self.dropout(token_emb + pos_emb)

        # 3. position id를 __init__ 에서 buffer로 미리 만들어두기
        # __init__에 추가해야 되는 코드
        self.register_buffer("position_ids", torch.arange(self.context_length).unsqueeze(0), persistent=False)

        # forward
        seq_len = x.shape[1]
        positions = self.position_ids[:, :seq_len]

        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)

        return self.dropout(token_emb + pos_emb)

        # 4. 생성용까지 고려해 position을 인자로 받기
        batch_size, seq_len = x.shape

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)

        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(position_ids)

        return self.dropout(token_emb + pos_emb)