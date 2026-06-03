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
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # TODO: qkv projection, output projection, dropout을 정의하세요.

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=qkv_bias)
        self.proj = nn.Linear(d_model, d_model)
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

        b, t, c = x.size()
        qkv = self.qkv(x)  # (b, t, 3 * d_model)
        qkv = qkv.view(b, t, self.n_heads, 3 * self.head_dim)  # (b, t, n_heads, 3 * head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # (b, n_heads, t, 3 * head_dim)
        q, k, v = qkv.chunk(3, dim=-1)  # q, k, v: (b, n_heads, t, head_dim)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (b, n_heads, t, t)
        if causal_mask:
            mask = torch.tril(torch.ones(t, t, device=x.device)).unsqueeze(0).unsqueeze(0)  # (1, 1, t, t)
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (b, n_heads, t, t)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)  # (b, n_heads, t, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(b, t, c)  # (b, t, d_model)
        
        output = self.proj(attn_output)  # (b, t, d_model)

        if return_attention_weights:
            return output, attn_weights
        return output
