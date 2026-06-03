# -*- coding: utf-8 -*-
"""GPT 모델 구성 요소 과제 템플릿."""

import math

import torch
import torch.nn as nn

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
        self.gamma = nn.Parameter(torch.ones(normalized_shape)) #scale
        self.beta = nn.Parameter(torch.zeros(normalized_shape)) #shift
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

        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class FeedForward(nn.Module):
    """Transformer FFN: Linear -> GELU -> Linear -> Dropout."""

    def __init__(self, d_model: int, dropout: float = 0.1, mult: int = 4):
        super().__init__()
        # TODO: d_model -> mult*d_model -> d_model 구조의 작은 MLP를 정의하세요.
       
        self.layers = nn.Sequential(
            nn.Linear(d_model, mult * d_model),
            GELU(),
            nn.Linear(mult * d_model, d_model),
            nn.Dropout(dropout),
        )

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
    ):
        super().__init__()
        # TODO: attention, ffn, layernorm, dropout을 정의하세요.

        self.att = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            drop_rate=drop_rate,
            qkv_bias=qkv_bias,
        )
        self.ff = FeedForward(d_model=d_model, dropout=drop_rate)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.drop_shortcut = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor, causal_mask: bool = True) -> torch.Tensor:
        """TODO: attention과 ffn을 residual connection으로 연결합니다."""

        shortcut = x
        x = self.norm1(x)
        x = self.att(x, causal_mask=causal_mask)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x

class GPTModel(nn.Module):
    """InputEmbedding -> TransformerBlock N개 -> LayerNorm -> LM head."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        # TODO: embedding, blocks, final layernorm, lm_head를 정의하세요.

        vocab_size = config["vocab_size"]
        emb_dim = config.get("emb_dim", config.get("d_model"))
        context_length = config.get("context_length", config.get("context_size"))
        if emb_dim is None:
            raise KeyError("config must contain 'emb_dim' or 'd_model'")
        if context_length is None:
            raise KeyError("config must contain 'context_length' or 'context_size'")

        self.tok_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(context_length, emb_dim)
        self.drop_emb = nn.Dropout(config.get("drop_rate", 0.1))

        self.trf_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    d_model=emb_dim,
                    n_heads=config["n_heads"],
                    drop_rate=config.get("drop_rate", 0.1),
                    qkv_bias=config.get("qkv_bias", False),
                )
                for _ in range(config["n_layers"])
            ]
        )
        self.final_norm = LayerNorm(emb_dim)
        self.lm_head = nn.Linear(emb_dim, vocab_size, bias=False)

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

        batch_size , seq_len = idx.size()
        tok_embeds = self.tok_emb(idx)

        pos_embeds = self.pos_emb(torch.arange(seq_len, device=idx.device))

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits if targets is None else (nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), targets.view(-1)), logits)
    
def generate_text_simple(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
) -> torch.Tensor:
    """TODO: greedy 방식으로 max_new_tokens만큼 다음 토큰을 이어 붙입니다."""

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
