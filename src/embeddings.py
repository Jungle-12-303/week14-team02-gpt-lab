# -*- coding: utf-8 -*-
"""토큰 임베딩 + 위치 정보 과제 템플릿."""

import math
import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    """
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
        position_mode: str = "embedding",
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.context_length = context_length
        self.position_mode = position_mode

        self.token_embedding = nn.Embedding(vocab_size, emb_dim)

        if position_mode == "embedding":
            self.position_embedding = nn.Embedding(context_length, emb_dim)
        elif position_mode == "encoding":
            position = torch.arange(context_length, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, emb_dim, 2, dtype=torch.float)
                * (-math.log(10000.0) / emb_dim)
            )
            position_encoding = torch.zeros(context_length, emb_dim)
            position_encoding[:, 0::2] = torch.sin(position * div_term)
            position_encoding[:, 1::2] = torch.cos(
                position * div_term[: position_encoding[:, 1::2].shape[1]]
            )
            self.register_buffer("position_encoding", position_encoding)
        else:
            raise ValueError(
                "position_mode must be either 'embedding' or 'encoding'"
            )

        self.dropout = nn.Dropout(drop_rate)

    # token embedding과 position embedding을 더한 뒤 dropout을 적용합니다.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len) token IDs
        Returns:
            (batch_size, seq_len, emb_dim)
        """
        _, seq_len = x.shape
        if seq_len > self.context_length:
            raise ValueError(
                f"seq_len ({seq_len}) must be <= context_length ({self.context_length})"
            )

        token_embeds = self.token_embedding(x)

        if self.position_mode == "embedding":
            positions = torch.arange(seq_len, device=x.device)
            position_signal = self.position_embedding(positions)
        else:
            position_signal = self.position_encoding[:seq_len].to(
                device=x.device,
                dtype=token_embeds.dtype,
            )
        embeddings = token_embeds + position_signal

        return self.dropout(embeddings)
