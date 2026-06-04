# -*- coding: utf-8 -*-
"""GPT 모델 구성 요소 과제 템플릿."""

import torch
import torch.nn as nn
import math

try:
    from .attention import (
        MultiHeadAttention,
        TernaryRelationAttention,
        TriangularRelationAttention,
    )
    from .embeddings import InputEmbedding
except ImportError:
    from attention import MultiHeadAttention, TernaryRelationAttention, TriangularRelationAttention
    from embeddings import InputEmbedding


class LayerNorm(nn.Module):
    """마지막 차원 기준 Layer Normalization."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        # scale
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        # shift
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO: 마지막 차원의 평균과 분산으로 정규화한 뒤 gamma/beta를 적용합니다."""
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * norm_x + self.beta


class GELU(nn.Module):
    """GPT FeedForward에서 사용하는 GELU 활성화 함수."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO: tanh 근사식 또는 torch 연산으로 GELU를 구현합니다."""
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


class FeedForward(nn.Module):
    """Transformer FFN: Linear -> GELU -> Linear -> Dropout."""

    def __init__(self, d_model: int, dropout: float = 0.1, mult: int = 4):
        super().__init__()
        # TODO: d_model -> mult*d_model -> d_model 구조의 작은 MLP를 정의하세요.
        self.layers = nn.Sequential(nn.Linear(d_model, mult*d_model), GELU(), nn.Linear(mult*d_model, d_model), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO: FeedForward 네트워크를 통과시킵니다."""
        return self.layers(x)


class TransformerBlock(nn.Module):
    """
    GPT block: LayerNorm -> Causal Self-Attention -> residual,
    LayerNorm -> FeedForward -> residual.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        drop_rate: float = 0.1,
        qkv_bias: bool = False,
        attention_type: str = "standard",
        triangular_value_mode: str = "marginal",
        triangular_logit_scale: str = "fixed",
        triangular_candidate_mode: str = "full",
        triangular_top_k: int = 16,
    ):
        super().__init__()
        # TODO: attention, ffn, layernorm, dropout을 정의하세요.
        if attention_type == "standard":
            attention_cls = MultiHeadAttention
            attention_kwargs = {}
        elif attention_type in {"triangular", "ternary"}:
            attention_cls = TriangularRelationAttention
            attention_kwargs = {
                "value_mode": triangular_value_mode,
                "logit_scale": triangular_logit_scale,
                "candidate_mode": triangular_candidate_mode,
                "top_k": triangular_top_k,
            }
        else:
            raise ValueError(f"unknown attention_type: {attention_type}")

        self.attention = attention_cls(
            d_model=d_model,
            n_heads=n_heads,
            drop_rate=drop_rate,
            qkv_bias=qkv_bias,
            **attention_kwargs,
        )
        self.ffn = FeedForward(d_model=d_model, dropout=drop_rate)
        self.norm1 = LayerNorm(normalized_shape=d_model)
        self.norm2 = LayerNorm(normalized_shape=d_model)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor, causal_mask: bool = True) -> torch.Tensor:
        """TODO: attention과 ffn을 residual connection으로 연결합니다."""
        shortcut = x
        x = self.norm1(x)
        x = self.attention(x, causal_mask=causal_mask)
        x = self.dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + shortcut

        return x

class GPTModel(nn.Module):
    """InputEmbedding -> TransformerBlock N개 -> LayerNorm -> LM head."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        attention_type = config.get("attention_type", "standard")
        triangular_value_mode = config.get("triangular_value_mode", "marginal")
        triangular_logit_scale = config.get("triangular_logit_scale", "fixed")
        triangular_candidate_mode = config.get("triangular_candidate_mode", "full")
        triangular_top_k = config.get("triangular_top_k", 16)
        position_mode = config.get("position_mode", "embedding")
        if attention_type == "alternating" and config["n_layers"] < 2:
            raise ValueError("alternating attention requires at least 2 layers")
        # TODO: embedding, blocks, final layernorm, lm_head를 정의하세요.
        self.embedding = InputEmbedding(
            vocab_size=config["vocab_size"],
            emb_dim=config["emb_dim"],
            context_length=config["context_length"],
            drop_rate=config["drop_rate"],
            position_mode=position_mode,
        )

        self.blocks = nn.Sequential(
            *[TransformerBlock(
                d_model=config["emb_dim"],
                n_heads=config["n_heads"],
                drop_rate=config["drop_rate"],
                qkv_bias=config["qkv_bias"],
                attention_type=(
                    "standard"
                    if attention_type == "alternating" and layer_idx % 2 == 0
                    else "triangular"
                    if attention_type == "alternating"
                    else attention_type
                ),
                triangular_value_mode=triangular_value_mode,
                triangular_logit_scale=triangular_logit_scale,
                triangular_candidate_mode=triangular_candidate_mode,
                triangular_top_k=triangular_top_k,
            ) for layer_idx in range(config["n_layers"])]
        )

        # forward에서 causal_mask를 명시적으로 넘기기 위해 추천
        # self.blocks = nn.ModuleList([
        #     TransformerBlock(
        #         d_model=config["emb_dim"],
        #         n_heads=config["n_heads"],
        #         drop_rate=config["drop_rate"],
        #         qkv_bias=config["qkv_bias"],
        #     )
        #     for _ in range(config["n_layers"])
        # ])

        self.final_norm = LayerNorm(config["emb_dim"])
        self.lm_head = nn.Linear(config["emb_dim"], config["vocab_size"])

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: logits를 만들고, targets가 있으면 cross entropy loss도 함께 반환합니다.

        Returns:
            targets가 None이면 logits
            targets가 있으면 (loss, logits)
        """
        x = self.embedding(idx)
        # Sequential로 생성시 직접 호출
        x = self.blocks(x)
        # ModuleList로 생성시 직접 호출 불가
        # for block in self.blocks:
        #     x = block(x, causal_mask=True)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
            return (loss, logits)
        
        return logits

def generate_text_simple(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
) -> torch.Tensor:
    """TODO: greedy 방식으로 max_new_tokens만큼 다음 토큰을 이어 붙입니다."""
    for _ in range(max_new_tokens):
        # 마지막 context_size개만 사용
        # 시퀀스가 100이고 context size가 4이면 마지막 96~99번째 토큰으로 짜르겠다.
        idx_cond = idx[:, -context_size:]
        # no_grad: gradiant 계산하지 마라. 학습 x 예측만
        with torch.no_grad():
            # 모델에 넣어 logits 연산
            logits = model(idx_cond)
        
        # 마지막 위치의 예측 값만 봄
        logits = logits[:, -1, :]
        # probas는 이후 argmax를 하기 때문에 결과가 똑같아서 쓸 필요가 없음
        # probas = torch.softmax(logits, dim=-1)
        # 가장 점수가 높은 토큰 선택
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        # 선택한 토큰을 뒤에 붙임
        idx = torch.cat((idx, idx_next), dim=1)
        # 루프 반복
    
    return idx
