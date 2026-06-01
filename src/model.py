# -*- coding: utf-8 -*-
"""GPT 모델 구성 요소 과제 템플릿."""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .attention import MultiHeadAttention
    from .embeddings import InputEmbedding
except ImportError:
    from attention import MultiHeadAttention
    from embeddings import InputEmbedding


class LayerNorm(nn.Module):
    """마지막 차원 기준 Layer Normalization."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        # 학습되는 값들
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO: 마지막 차원의 평균과 분산으로 정규화한 뒤 gamma/beta를 적용합니다."""
        # token의 벡터 값 크기를 안정화(평균 0, 분산1 근처로 맞춘 뒤 gamma, beta로 조절)
        # 평균 구하기(마지막 차원)
        mean = x.mean(dim=-1, keepdim=True)
        # 분산 구하기
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # 정규화를 통해 0과 1에 가까운 형태로 만듦
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class GELU(nn.Module):
    """GPT FeedForward에서 사용하는 GELU 활성화 함수."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO: tanh 근사식 또는 torch 연산으로 GELU를 구현합니다."""
        return F.gelu(x)

    # 근사식으로 표현
    #   return 0.5 * x * (
    #     1.0 + torch.tanh(
    #         torch.sqrt(torch.tensor(2.0 / torch.pi, device=x.device))
    #         * (x + 0.044715 * torch.pow(x, 3))
    #     )
    # )

class FeedForward(nn.Module):
    """Transformer FFN: Linear -> GELU -> Linear -> Dropout."""

    def __init__(self, d_model: int, dropout: float = 0.1, mult: int = 4):
        super().__init__()
        # TODO: d_model -> mult*d_model -> d_model 구조의 작은 MLP를 정의하세요.
        # MLP : Linear - gelu - Linear
        # 구조 정의하기
        self.net = nn.Sequential(
            nn.Linear(d_model, mult * d_model),
            GELU(),
            nn.Linear(mult * d_model, d_model),
            nn.Dropout(dropout),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO: FeedForward 네트워크를 통과시킵니다."""
        # 하나의 layer는 같은 w를 사용
        return self.net(x)

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
    ):
        super().__init__()
        # TODO: attention, ffn, layernorm, dropout을 정의하세요.
        # attention 정의
        self.attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            drop_rate=drop_rate,
            qkv_bias=qkv_bias,
        )
        # feedforward 정의
        self.ffn = FeedForward(
            d_model=d_model,
            dropout=drop_rate
        )

        # layerNorm 정의
        self.layerNorm1 = LayerNorm(d_model)
        self.layerNorm2 = LayerNorm(d_model)

        # dropout 정의
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor, causal_mask: bool = True) -> torch.Tensor:
        """TODO: attention과 ffn을 residual connection으로 연결합니다."""
        attention_result = self.attention(
            self.layerNorm1(x),
            causal_mask=causal_mask,
        )
        # 일반 attention결과에 dropput 적용
        x = x + self.drop(attention_result)

        ffn_result = self.ffn(self.layerNorm2(x))
        x = x + ffn_result

        return x

class GPTModel(nn.Module):
    """InputEmbedding -> TransformerBlock N개 -> LayerNorm -> LM head."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        # TODO: embedding, blocks, final layernorm, lm_head를 정의하세요.
        self.embedding = InputEmbedding(
            # 토큰 vocab size
            vocab_size=config["vocab_size"],
            # embedding 차원
            emb_dim=config["emb_dim"],
            # 시퀀스 길이
            context_length=config["context_length"],
            # dropout rate
            drop_rate=config["drop_rate"],
        )

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    d_model=config["emb_dim"],
                    n_heads=config["n_heads"],
                    drop_rate=config["drop_rate"],
                    qkv_bias=config["qkv_bias"],
                )
                for _ in range(config["n_layers"])
            ]
        )

        # transformer을 거친 결과 정규화
        self.result_norm = LayerNorm(config["emb_dim"])
        # 후보 점수로 바꿔주는 선형 변환 (logits)
        self.lm_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)

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
        x = self.blocks(x)
        x = self.result_norm(x)
        logits = self.lm_head(x)

        # target 없으면 예측만하고 있으면 loss까지 계산
        if targets is None:
            return logits

        # 역전파에 사용할 정답 token loss계산
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
        )

        return loss, logits
 
def generate_text_simple(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
) -> torch.Tensor:
    """TODO: greedy 방식으로 max_new_tokens만큼 다음 토큰을 이어 붙입니다."""
    # 가장 큰 점수의 token 구하기
    for _ in range(max_new_tokens):
        # 최대 context길이만큼 자르기
        idx_cond = idx[:, -context_size:]
        # token점수 계산
        logits = model(idx_cond)
        # 마지막 token 위치 예측
        logits = logits[:, -1, :]
        # 다음 token id 고르기
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        # 기존토큰 뒤에 고른 토큰 붙이기
        idx = torch.cat((idx, idx_next), dim=1)

    return idx