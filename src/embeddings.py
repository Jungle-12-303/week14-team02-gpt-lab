# -*- coding: utf-8 -*-
"""토큰 임베딩 + 위치 임베딩 과제 템플릿."""

import torch
import torch.nn as nn
import math


# token ID를 Transformer 입력 벡터로 바꿉니다.
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
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.context_length = context_length

        self.token_embedding = nn.Embedding(vocab_size, emb_dim)

        # [[ Embedding Training POS ]]>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        self.position_embedding = nn.Embedding(context_length, emb_dim)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<[[ Embedding Training POS ]]

        # [[ SIN/COS FIXED POS ]]=======================================================
        # PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
        # [[ SIN/COS FIXED POS ]]>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # position = torch.arange(context_length, dtype=torch.float).unsqueeze(1)
        # div_term = torch.exp(
        #     torch.arange(0, emb_dim, 2, dtype=torch.float)
        #     * (-math.log(10000.0) / emb_dim)
        # )
        # position_encoding = torch.zeros(context_length, emb_dim)
        # position_encoding[:, 0::2] = torch.sin(position * div_term)
        # position_encoding[:, 1::2] = torch.cos(
        #     position * div_term[: position_encoding[:, 1::2].shape[1]]
        # )

        # self.register_buffer("position_encoding", position_encoding)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<[[ SIN/COS FIXED POS ]]

        self.dropout = nn.Dropout(drop_rate)

    # token embedding과 position embedding을 더한 뒤 dropout을 적용합니다.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len) token IDs
        Returns:
            (batch_size, seq_len, emb_dim)
        """
        batch_size, seq_len = x.shape
        token_embeds = self.token_embedding(x)

        # [[ Embedding Training POS ]]>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        positions = torch.arange(seq_len, device=x.device)
        position_embeds = self.position_embedding(positions)
        embeddings = token_embeds + position_embeds
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<[[ Embedding Training POS ]]

        # [[ SIN/COS FIXED POS ]]>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # position_encodings = self.position_encoding[:seq_len].to(
        #     device=x.device,
        #     dtype=token_embeds.dtype,
        # )
        # embeddings = token_embeds + position_encodings
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<[[ SIN/COS FIXED POS ]]

        return self.dropout(embeddings)
