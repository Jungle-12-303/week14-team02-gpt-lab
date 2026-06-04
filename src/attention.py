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

    입력은 기본적으로 (B, T, C)이지만, (B, H, W, C)나
    (B, D, H, W, C)처럼 마지막 차원이 d_model인 N차원 입력도 받을 수 있습니다.
    이 경우 attention 계산 동안 H*W 또는 D*H*W를 하나의 token 축 T로 펼칩니다.
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
            x: (batch_size, seq_len, d_model) 또는 (batch_size, *dims, d_model)
            causal_mask: True이면 flatten된 순서에서 미래 위치를 볼 수 없게 mask 처리
            return_attention_weights: True이면 attention weight도 함께 반환
        """
        if x.dim() < 3:
            raise ValueError("x must have shape (batch_size, *dims, d_model)")

        input_shape = x.shape
        batch_size = input_shape[0]
        spatial_shape = input_shape[1:-1]
        d_model = input_shape[-1]
        if d_model != self.d_model:
            raise ValueError(
                f"last dimension ({d_model}) must match d_model ({self.d_model})"
            )

        seq_len = 1
        for dim in spatial_shape:
            seq_len *= dim

        x = x.reshape(batch_size, seq_len, d_model)

        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        queries = queries.view(batch_size, seq_len, self.n_heads, self.head_dim)
        keys = keys.view(batch_size, seq_len, self.n_heads, self.head_dim)
        values = values.view(batch_size, seq_len, self.n_heads, self.head_dim)

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores / (self.head_dim ** 0.5)

        if causal_mask:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights_dropped = self.dropout(attn_weights)

        context = attn_weights_dropped @ values

        context = context.transpose(1, 2)
        context = context.contiguous().view(batch_size, seq_len, d_model)

        out = self.out_proj(context)
        out = out.reshape(input_shape)

        if return_attention_weights:
            return out, attn_weights
        return out


class TriangularRelationAttention(nn.Module):
    """
    [삼각 관계] attention.

    표준 attention이 score[i, j]로 두 위치의 관계를 보는 반면,
    이 모듈은 score[i, j, k]로 세 위치의 관계를 봅니다.

    - i: query token 또는 첫 번째 역할
    - j: 두 번째 역할 후보
    - k: 세 번째 역할 후보

    예를 들어 특정 head가 "give" 관계를 학습했다면,
    relation_weights[..., 철수, 책, 영희] 같은 칸이
    give(철수, 책, 영희)에 해당하는 ordered triple 신호가 될 수 있습니다.

    value_mode:
    - marginal: 기존 v0. score는 삼각 관계지만 value는 j/k marginal 평균으로 집계합니다.
    - pair: v1. score와 value 모두 (j, k) 조합을 직접 반영합니다.
    - pair_gated: v4. pair value 경로를 learned gate로 점진적으로 반영합니다.

    logit_scale:
    - fixed: score / sqrt(head_dim)
    - learned: score / sqrt(head_dim)에 exp(logit_scale_param)을 곱합니다.

    candidate_mode:
    - full: 모든 (j, k) pair를 봅니다. O(T^3)
    - topk: query별 후보 K개 안에서만 (j, k) pair를 봅니다. O(T*K^2)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        drop_rate: float = 0.1,
        qkv_bias: bool = False,
        value_mode: str = "marginal",
        logit_scale: str = "fixed",
        candidate_mode: str = "full",
        top_k: int = 16,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if value_mode not in {"marginal", "pair", "pair_gated"}:
            raise ValueError("value_mode must be 'marginal', 'pair', or 'pair_gated'")
        if logit_scale not in {"fixed", "learned"}:
            raise ValueError("logit_scale must be 'fixed' or 'learned'")
        if candidate_mode not in {"full", "topk"}:
            raise ValueError("candidate_mode must be 'full' or 'topk'")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.value_mode = value_mode
        self.logit_scale = logit_scale
        self.candidate_mode = candidate_mode
        self.top_k = top_k

        self.W_query = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.W_key_left = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.W_key_right = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.W_value_left = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.W_value_right = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.W_candidate = (
            nn.Linear(d_model, d_model, bias=qkv_bias)
            if candidate_mode == "topk"
            else None
        )

        if logit_scale == "learned":
            self.logit_scale_param = nn.Parameter(torch.zeros(()))
        else:
            self.register_buffer("logit_scale_param", torch.zeros(()), persistent=False)
        if value_mode == "pair_gated":
            self.pair_gate_param = nn.Parameter(torch.full((n_heads,), -2.0))
        else:
            self.register_buffer(
                "pair_gate_param",
                torch.zeros(n_heads),
                persistent=False,
            )

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(drop_rate)

    def _split_heads(self, x: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        x = x.view(batch_size, seq_len, self.n_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: bool = True,
        return_relation_weights: bool = False,
        return_relation_stats: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            x: (batch_size, seq_len, d_model) 또는 (batch_size, *dims, d_model)
            causal_mask: True이면 j/k가 query i보다 미래 위치일 때 mask 처리
            return_relation_weights: True이면 (B, n_heads, T, T, T) weight 반환
            return_relation_stats: True이면 entropy/logit 통계를 함께 반환
        """
        if x.dim() < 3:
            raise ValueError("x must have shape (batch_size, *dims, d_model)")

        input_shape = x.shape
        batch_size = input_shape[0]
        spatial_shape = input_shape[1:-1]
        d_model = input_shape[-1]
        if d_model != self.d_model:
            raise ValueError(
                f"last dimension ({d_model}) must match d_model ({self.d_model})"
            )

        seq_len = 1
        for dim in spatial_shape:
            seq_len *= dim

        x = x.reshape(batch_size, seq_len, d_model)

        queries = self._split_heads(self.W_query(x), batch_size, seq_len)
        keys_left = self._split_heads(self.W_key_left(x), batch_size, seq_len)
        keys_right = self._split_heads(self.W_key_right(x), batch_size, seq_len)
        values_left = self._split_heads(self.W_value_left(x), batch_size, seq_len)
        values_right = self._split_heads(self.W_value_right(x), batch_size, seq_len)
        candidate_keys = (
            self._split_heads(self.W_candidate(x), batch_size, seq_len)
            if self.W_candidate is not None
            else None
        )

        if self.candidate_mode == "topk":
            relation_weights, context, stats = self._topk_relation_context(
                queries=queries,
                keys_left=keys_left,
                keys_right=keys_right,
                candidate_keys=candidate_keys,
                values_left=values_left,
                values_right=values_right,
                causal_mask=causal_mask,
                return_relation_weights=return_relation_weights,
            )
        else:
            relation_weights, context, stats = self._full_relation_context(
                queries=queries,
                keys_left=keys_left,
                keys_right=keys_right,
                values_left=values_left,
                values_right=values_right,
                causal_mask=causal_mask,
            )
        self.last_relation_stats = stats

        context = context.transpose(1, 2)
        context = context.contiguous().view(batch_size, seq_len, d_model)

        out = self.out_proj(context)
        out = out.reshape(input_shape)

        if return_relation_weights and return_relation_stats:
            return out, relation_weights, stats
        if return_relation_weights:
            return out, relation_weights
        if return_relation_stats:
            return out, stats
        return out

    def _scale_scores(self, scores: torch.Tensor) -> torch.Tensor:
        scores = scores / (self.head_dim ** 0.5)
        if self.logit_scale == "learned":
            scores = scores * torch.exp(self.logit_scale_param)
        return scores

    def _relation_stats(
        self,
        relation_scores: torch.Tensor,
        relation_weights: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        weight_entropy = -(
            relation_weights * relation_weights.clamp_min(1e-12).log()
        ).sum(dim=(-1, -2))
        max_weight = relation_weights.amax(dim=(-1, -2))

        if valid_mask is None:
            finite_scores = relation_scores[torch.isfinite(relation_scores)]
        else:
            finite_scores = relation_scores.masked_select(valid_mask)
        if finite_scores.numel() == 0:
            finite_scores = relation_scores.new_zeros(1)

        return {
            "relation_entropy": weight_entropy.mean().detach(),
            "relation_max_weight": max_weight.mean().detach(),
            "relation_logit_mean": finite_scores.mean().detach(),
            "relation_logit_std": finite_scores.std(unbiased=False).detach(),
            "relation_score_elements": torch.tensor(
                relation_scores.numel(),
                device=relation_scores.device,
                dtype=torch.float,
            ),
            "logit_scale": torch.exp(self.logit_scale_param).detach(),
            "pair_gate": torch.sigmoid(self.pair_gate_param).mean().detach(),
        }

    def _full_relation_context(
        self,
        queries: torch.Tensor,
        keys_left: torch.Tensor,
        keys_right: torch.Tensor,
        values_left: torch.Tensor,
        values_right: torch.Tensor,
        causal_mask: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        batch_size, _, seq_len, _ = queries.shape
        relation_scores = torch.einsum(
            "bhid,bhjd,bhkd->bhijk",
            queries,
            keys_left,
            keys_right,
        )
        relation_scores = self._scale_scores(relation_scores)

        valid_mask = None
        if causal_mask:
            positions = torch.arange(seq_len, device=queries.device)
            query_pos = positions.view(seq_len, 1, 1)
            left_pos = positions.view(1, seq_len, 1)
            right_pos = positions.view(1, 1, seq_len)
            mask = (left_pos > query_pos) | (right_pos > query_pos)
            relation_scores = relation_scores.masked_fill(mask, float("-inf"))
            valid_mask = ~mask.view(1, 1, seq_len, seq_len, seq_len)

        relation_weights = torch.softmax(
            relation_scores.reshape(batch_size, self.n_heads, seq_len, seq_len * seq_len),
            dim=-1,
        )
        relation_weights = relation_weights.reshape(
            batch_size,
            self.n_heads,
            seq_len,
            seq_len,
            seq_len,
        )
        stats = self._relation_stats(relation_scores, relation_weights, valid_mask)
        relation_weights_dropped = self.dropout(relation_weights)
        context = self._weighted_context(
            relation_weights_dropped,
            values_left,
            values_right,
        )
        return relation_weights, context, stats

    def _topk_relation_context(
        self,
        queries: torch.Tensor,
        keys_left: torch.Tensor,
        keys_right: torch.Tensor,
        candidate_keys: torch.Tensor,
        values_left: torch.Tensor,
        values_right: torch.Tensor,
        causal_mask: bool,
        return_relation_weights: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor, dict[str, torch.Tensor]]:
        if candidate_keys is None:
            raise RuntimeError("candidate_keys must be defined for topk candidate mode")

        batch_size, _, seq_len, head_dim = queries.shape
        candidate_count = min(self.top_k, seq_len)

        pre_scores = queries @ candidate_keys.transpose(2, 3)
        pre_scores = pre_scores / (head_dim ** 0.5)
        if causal_mask:
            future_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=queries.device, dtype=torch.bool),
                diagonal=1,
            )
            pre_scores = pre_scores.masked_fill(future_mask, float("-inf"))

        top_values, top_indices = torch.topk(pre_scores, k=candidate_count, dim=-1)
        valid_candidates = torch.isfinite(top_values)

        keys_left_c = self._gather_candidates(keys_left, top_indices)
        keys_right_c = self._gather_candidates(keys_right, top_indices)
        values_left_c = self._gather_candidates(values_left, top_indices)
        values_right_c = self._gather_candidates(values_right, top_indices)

        relation_scores = torch.einsum(
            "bhid,bhikd,bhild->bhikl",
            queries,
            keys_left_c,
            keys_right_c,
        )
        relation_scores = self._scale_scores(relation_scores)
        invalid_pairs = (
            ~valid_candidates.unsqueeze(-1)
            | ~valid_candidates.unsqueeze(-2)
        )
        relation_scores = relation_scores.masked_fill(invalid_pairs, float("-inf"))
        valid_pairs = ~invalid_pairs

        relation_weights = torch.softmax(
            relation_scores.reshape(
                batch_size,
                self.n_heads,
                seq_len,
                candidate_count * candidate_count,
            ),
            dim=-1,
        )
        relation_weights = relation_weights.reshape(
            batch_size,
            self.n_heads,
            seq_len,
            candidate_count,
            candidate_count,
        )
        stats = self._relation_stats(relation_scores, relation_weights, valid_pairs)
        relation_weights_dropped = self.dropout(relation_weights)
        context = self._weighted_context(
            relation_weights_dropped,
            values_left_c,
            values_right_c,
        )

        dense_weights = None
        if return_relation_weights:
            pair_indices = (
                top_indices.unsqueeze(-1) * seq_len
                + top_indices.unsqueeze(-2)
            )
            dense_flat = relation_weights.new_zeros(
                batch_size,
                self.n_heads,
                seq_len,
                seq_len * seq_len,
            )
            dense_flat.scatter_(
                dim=3,
                index=pair_indices.reshape(
                    batch_size,
                    self.n_heads,
                    seq_len,
                    candidate_count * candidate_count,
                ),
                src=relation_weights.reshape(
                    batch_size,
                    self.n_heads,
                    seq_len,
                    candidate_count * candidate_count,
                ),
            )
            dense_weights = dense_flat.reshape(
                batch_size,
                self.n_heads,
                seq_len,
                seq_len,
                seq_len,
            )

        return dense_weights, context, stats

    def _gather_candidates(
        self,
        x: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, seq_len, head_dim = x.shape
        gather_index = indices.unsqueeze(-1).expand(-1, -1, -1, -1, head_dim)
        expanded = x.unsqueeze(2).expand(-1, -1, seq_len, -1, -1)
        return torch.gather(expanded, dim=3, index=gather_index)

    def _weighted_context(
        self,
        relation_weights: torch.Tensor,
        values_left: torch.Tensor,
        values_right: torch.Tensor,
    ) -> torch.Tensor:
        if values_left.dim() == 5:
            left_context = torch.einsum(
                "bhikl,bhikd->bhid",
                relation_weights,
                values_left,
            )
            right_context = torch.einsum(
                "bhikl,bhild->bhid",
                relation_weights,
                values_right,
            )
            marginal_context = 0.5 * (left_context + right_context)
            if self.value_mode in {"pair", "pair_gated"}:
                pair_values = values_left.unsqueeze(4) * values_right.unsqueeze(3)
                pair_context = torch.einsum(
                    "bhikl,bhikld->bhid",
                    relation_weights,
                    pair_values,
                )
                if self.value_mode == "pair_gated":
                    gate = torch.sigmoid(self.pair_gate_param).view(1, self.n_heads, 1, 1)
                    pair_context = pair_context * gate
                return marginal_context + pair_context

            return marginal_context

        left_context = torch.einsum("bhijk,bhjd->bhid", relation_weights, values_left)
        right_context = torch.einsum("bhijk,bhkd->bhid", relation_weights, values_right)
        marginal_context = 0.5 * (left_context + right_context)
        if self.value_mode in {"pair", "pair_gated"}:
            pair_values = values_left.unsqueeze(3) * values_right.unsqueeze(2)
            pair_context = torch.einsum(
                "bhijk,bhjkd->bhid",
                relation_weights,
                pair_values,
            )
            if self.value_mode == "pair_gated":
                gate = torch.sigmoid(self.pair_gate_param).view(1, self.n_heads, 1, 1)
                pair_context = pair_context * gate
            return marginal_context + pair_context

        return marginal_context


TernaryRelationAttention = TriangularRelationAttention
