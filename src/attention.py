# -*- coding: utf-8 -*-
"""Multi-Head Self-Attention 과제 템플릿."""

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    GPT의 causal self-attention을 구현합니다.

    구현할 핵심:
    - Q/K/V projection
    - head 분리: (B, T, C) -> (B, n_heads, T, head_dim)
    - attention score = QK^T / sqrt(head_dim)
    - causal mask로 미래 토큰 가리기
    - attention weight와 V를 곱한 뒤 head를 다시 합치기
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        drop_rate: float = 0.1,
        qkv_bias: bool = False,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        # 토큰 벡터 하나의 크기
        self.d_model = d_model
        # 나눌 attention head
        self.n_heads = n_heads
        # head하나의 차원 (출력 차원에 맞도록)
        self.head_dim = d_model // n_heads
        # TODO: qkv projection, output projection, dropout을 정의하세요.
        
        self.W_query = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.W_key = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.W_value = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(drop_rate)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: bool = True,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: multi-head attention forward를 구현합니다.

        Args:
            x: (batch_size, seq_len, d_model)
            causal_mask: True이면 미래 위치를 볼 수 없게 mask 처리
            return_attention_weights: True이면 attention weight도 함께 반환
        """
        batch_size, sequence_length, d_model = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # tensor의 shape 바꾸기
        # (B, T, C) -> (B, T, n_heads, head_dim) -> (B, n_heads, T, head_dim)
        keys = keys.view(
            batch_size, sequence_length, self.n_heads, self.head_dim
        ).transpose(1, 2)

        queries = queries.view(
            batch_size, sequence_length, self.n_heads, self.head_dim
        ).transpose(1, 2)

        values = values.view(
            batch_size, sequence_length, self.n_heads, self.head_dim
        ).transpose(1, 2)

        # 각 query와 모든 key의 점곱으로 attention 원점수 구하기
        attention_scores = queries @ keys.transpose(-2, -1)
        attention_scores = attention_scores / (self.head_dim ** 0.5)

        # causal mask=True이면 미래 토큰을 볼 수 없도록 처리 
        if causal_mask:
            mask = torch.triu(
                torch.ones(sequence_length, sequence_length, device=x.device),
                diagonal=1,
            ).bool()
            attention_scores = attention_scores.masked_fill(mask, float("-inf"))

        # attention score를 softmax로 바꾼 뒤, attention weight에 비율을 곱해서 더함
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        context = attention_weights @ values

        # head별 context를 다시 하나의 d_model 벡터로 합침
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, sequence_length, d_model)

        output = self.out_proj(context)

        # attention weight 출력
        if return_attention_weights:
            return output, attention_weights

        return output

        
