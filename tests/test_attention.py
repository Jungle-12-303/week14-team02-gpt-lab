# -*- coding: utf-8 -*-
"""
MultiHeadAttention 단위 테스트.
실행: `pytest tests/test_attention.py -v`
"""

import sys
from pathlib import Path

import torch
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# =============================================================================
# MultiHeadAttention
# =============================================================================


class TestMultiHeadAttention:
    """MultiHeadAttention forward 구현 후 실행."""

    def test_mha_output_shape(self):
        """MultiHeadAttention 출력 shape가 입력과 같은 (batch, seq_len, d_model)인지 확인한다."""
        from attention import MultiHeadAttention

        batch_size, seq_len, d_model = 2, 10, 64
        n_heads = 4
        try:
            mha = MultiHeadAttention(d_model, n_heads, drop_rate=0.0)
            x = torch.randn(batch_size, seq_len, d_model)
            out = mha(x, causal_mask=True)
        except (NotImplementedError, TypeError):
            pytest.fail("MultiHeadAttention 미구현")
        if isinstance(out, tuple):
            out = out[0]
        assert out.shape == (batch_size, seq_len, d_model)

    def test_mha_2d_input_shape(self):
        """2D grid 입력도 flatten attention 후 원래 shape로 복원되는지 확인한다."""
        from attention import MultiHeadAttention

        batch_size, height, width, d_model = 2, 3, 4, 32
        n_heads = 4
        mha = MultiHeadAttention(d_model, n_heads, drop_rate=0.0)
        x = torch.randn(batch_size, height, width, d_model)
        out, attn_weights = mha(
            x,
            causal_mask=False,
            return_attention_weights=True,
        )

        assert out.shape == (batch_size, height, width, d_model)
        assert attn_weights.shape == (
            batch_size,
            n_heads,
            height * width,
            height * width,
        )

    def test_mha_3d_input_shape(self):
        """3D volume 입력도 flatten attention 후 원래 shape로 복원되는지 확인한다."""
        from attention import MultiHeadAttention

        batch_size, depth, height, width, d_model = 2, 2, 3, 4, 32
        n_heads = 4
        mha = MultiHeadAttention(d_model, n_heads, drop_rate=0.0)
        x = torch.randn(batch_size, depth, height, width, d_model)
        out, attn_weights = mha(
            x,
            causal_mask=False,
            return_attention_weights=True,
        )

        seq_len = depth * height * width
        assert out.shape == (batch_size, depth, height, width, d_model)
        assert attn_weights.shape == (batch_size, n_heads, seq_len, seq_len)

    def test_mha_causal_mask_future_zero(self):
        """Causal mask 적용 시 현재 토큰이 미래 위치에 attention을 주지 않는지 확인한다."""
        from attention import MultiHeadAttention

        batch_size, seq_len, d_model = 1, 4, 8
        n_heads = 2
        try:
            mha = MultiHeadAttention(d_model, n_heads, drop_rate=0.0)
            x = torch.randn(batch_size, seq_len, d_model)
            out, attn_weights = mha(x, causal_mask=True, return_attention_weights=True)
        except (NotImplementedError, TypeError):
            pytest.fail("MultiHeadAttention + return_attention_weights 미구현")
        # attn_weights: (B, n_heads, T, T). 상삼각(diagonal 제외)이 0이어야 함
        assert attn_weights.shape == (batch_size, n_heads, seq_len, seq_len)
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert torch.allclose(attn_weights[0, :, i, j], torch.zeros(n_heads))

    def test_mha_3d_causal_mask_future_zero(self):
        """3D 입력에서 causal mask가 flatten된 token 순서 기준으로 적용되는지 확인한다."""
        from attention import MultiHeadAttention

        batch_size, depth, height, width, d_model = 1, 2, 2, 2, 16
        n_heads = 4
        mha = MultiHeadAttention(d_model, n_heads, drop_rate=0.0)
        x = torch.randn(batch_size, depth, height, width, d_model)
        out, attn_weights = mha(
            x,
            causal_mask=True,
            return_attention_weights=True,
        )

        seq_len = depth * height * width
        assert out.shape == (batch_size, depth, height, width, d_model)
        assert attn_weights.shape == (batch_size, n_heads, seq_len, seq_len)
        future_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool),
            diagonal=1,
        )
        assert torch.all(attn_weights[0, :, future_mask] == 0)


class TestTernaryRelationAttention:
    """세 토큰 ordered triple 관계를 직접 다루는 attention 테스트."""

    def test_ternary_output_and_relation_weight_shape(self):
        """relation weight가 (B, heads, T, T, T)인지 확인한다."""
        from attention import TernaryRelationAttention

        batch_size, seq_len, d_model = 2, 5, 32
        n_heads = 4
        attention = TernaryRelationAttention(d_model, n_heads, drop_rate=0.0)
        x = torch.randn(batch_size, seq_len, d_model)
        out, relation_weights = attention(
            x,
            causal_mask=False,
            return_relation_weights=True,
        )

        assert out.shape == (batch_size, seq_len, d_model)
        assert relation_weights.shape == (
            batch_size,
            n_heads,
            seq_len,
            seq_len,
            seq_len,
        )

    def test_ternary_relation_weights_sum_over_token_pairs(self):
        """각 query i마다 모든 (j, k) pair weight 합이 1인지 확인한다."""
        from attention import TernaryRelationAttention

        batch_size, seq_len, d_model = 2, 4, 16
        n_heads = 4
        attention = TernaryRelationAttention(d_model, n_heads, drop_rate=0.0)
        x = torch.randn(batch_size, seq_len, d_model)
        _, relation_weights = attention(
            x,
            causal_mask=False,
            return_relation_weights=True,
        )

        weight_sums = relation_weights.sum(dim=(-1, -2))
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-6)

    def test_ternary_causal_mask_future_pairs_zero(self):
        """causal mask가 query보다 미래인 j 또는 k를 0으로 만드는지 확인한다."""
        from attention import TernaryRelationAttention

        batch_size, seq_len, d_model = 1, 4, 16
        n_heads = 4
        attention = TernaryRelationAttention(d_model, n_heads, drop_rate=0.0)
        x = torch.randn(batch_size, seq_len, d_model)
        _, relation_weights = attention(
            x,
            causal_mask=True,
            return_relation_weights=True,
        )

        positions = torch.arange(seq_len)
        query_pos = positions.view(seq_len, 1, 1)
        left_pos = positions.view(1, seq_len, 1)
        right_pos = positions.view(1, 1, seq_len)
        future_mask = (left_pos > query_pos) | (right_pos > query_pos)
        assert torch.all(relation_weights[0, :, future_mask] == 0)

    def test_ternary_can_prioritize_ordered_triple(self):
        """학습된 projection으로 특정 ordered triple (i, j, k)를 가장 크게 만들 수 있다."""
        from attention import TernaryRelationAttention

        seq_len, d_model = 3, 4
        attention = TernaryRelationAttention(
            d_model=d_model,
            n_heads=1,
            drop_rate=0.0,
            qkv_bias=False,
        )
        with torch.no_grad():
            for layer in (
                attention.W_query,
                attention.W_key_left,
                attention.W_key_right,
                attention.W_value_left,
                attention.W_value_right,
                attention.out_proj,
            ):
                layer.weight.zero_()
            attention.W_query.weight[0, 0] = 5.0
            attention.W_key_left.weight[0, 1] = 5.0
            attention.W_key_right.weight[0, 2] = 5.0

        x = torch.eye(d_model)[:seq_len].unsqueeze(0)
        _, relation_weights = attention(
            x,
            causal_mask=False,
            return_relation_weights=True,
        )

        query_zero_weights = relation_weights[0, 0, 0]
        max_pair = torch.nonzero(
            query_zero_weights == query_zero_weights.max(),
            as_tuple=False,
        )[0]
        assert max_pair.tolist() == [1, 2]
        assert query_zero_weights[1, 2] > query_zero_weights[2, 1]


class TestTriangularRelationAttention:
    """[삼각 관계] attention 고도화 옵션 테스트."""

    def test_triangular_alias_output_shape(self):
        """TriangularRelationAttention 이름으로도 기존 ternary 구조를 사용할 수 있어야 한다."""
        from attention import TriangularRelationAttention

        attention = TriangularRelationAttention(
            d_model=32,
            n_heads=4,
            drop_rate=0.0,
        )
        x = torch.randn(2, 5, 32)
        out, relation_weights = attention(
            x,
            causal_mask=False,
            return_relation_weights=True,
        )

        assert out.shape == x.shape
        assert relation_weights.shape == (2, 4, 5, 5, 5)

    def test_triangular_pair_value_mode_changes_context(self):
        """value_mode='pair'는 j/k marginal 평균이 아닌 pair interaction을 출력에 반영한다."""
        from attention import TriangularRelationAttention

        torch.manual_seed(123)
        marginal = TriangularRelationAttention(
            d_model=16,
            n_heads=4,
            drop_rate=0.0,
            value_mode="marginal",
        )
        pair = TriangularRelationAttention(
            d_model=16,
            n_heads=4,
            drop_rate=0.0,
            value_mode="pair",
        )
        pair.load_state_dict(marginal.state_dict(), strict=False)
        x = torch.randn(2, 4, 16)

        marginal_out = marginal(x, causal_mask=False)
        pair_out = pair(x, causal_mask=False)

        assert marginal_out.shape == pair_out.shape
        assert not torch.allclose(marginal_out, pair_out)

    def test_triangular_learned_scale_returns_relation_stats(self):
        """learned logit scale 설정에서 entropy/logit 통계를 반환한다."""
        from attention import TriangularRelationAttention

        attention = TriangularRelationAttention(
            d_model=16,
            n_heads=4,
            drop_rate=0.0,
            logit_scale="learned",
        )
        x = torch.randn(2, 4, 16)
        out, stats = attention(
            x,
            causal_mask=True,
            return_relation_stats=True,
        )

        assert out.shape == x.shape
        for key in (
            "relation_entropy",
            "relation_max_weight",
            "relation_logit_mean",
            "relation_logit_std",
            "relation_score_elements",
            "logit_scale",
            "pair_gate",
        ):
            assert key in stats
            assert torch.isfinite(stats[key])

    def test_triangular_pair_gated_mode_uses_learned_gate(self):
        """value_mode='pair_gated'는 pair 경로에 학습 가능한 gate를 둔다."""
        from attention import TriangularRelationAttention

        attention = TriangularRelationAttention(
            d_model=16,
            n_heads=4,
            drop_rate=0.0,
            value_mode="pair_gated",
            logit_scale="learned",
        )
        x = torch.randn(2, 4, 16)
        out, stats = attention(
            x,
            causal_mask=False,
            return_relation_stats=True,
        )

        assert out.shape == x.shape
        assert attention.pair_gate_param.requires_grad
        assert 0.0 < stats["pair_gate"].item() < 1.0

    def test_triangular_topk_returns_dense_relation_weights(self):
        """candidate_mode='topk'는 dense weight shape를 유지하되 후보 pair만 활성화한다."""
        from attention import TriangularRelationAttention

        batch_size, seq_len, d_model = 2, 6, 24
        n_heads = 4
        top_k = 2
        attention = TriangularRelationAttention(
            d_model=d_model,
            n_heads=n_heads,
            drop_rate=0.0,
            candidate_mode="topk",
            top_k=top_k,
        )
        x = torch.randn(batch_size, seq_len, d_model)
        out, relation_weights = attention(
            x,
            causal_mask=False,
            return_relation_weights=True,
        )

        assert out.shape == x.shape
        assert relation_weights.shape == (
            batch_size,
            n_heads,
            seq_len,
            seq_len,
            seq_len,
        )
        weight_sums = relation_weights.sum(dim=(-1, -2))
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-6)
        nonzero_pairs = (relation_weights > 0).sum(dim=(-1, -2))
        assert torch.all(nonzero_pairs <= top_k * top_k)
