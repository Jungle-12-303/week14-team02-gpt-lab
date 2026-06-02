# -*- coding: utf-8 -*-
"""GPT 사전 학습 유틸리티 과제 템플릿."""

import matplotlib.pyplot as plt
import torch

try:
    from .model import GPTModel
except ImportError:
    from model import GPTModel


def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: GPTModel,
    device: torch.device,
) -> torch.Tensor:
    # device 전송
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    # loss계산
    loss, logits = model(input_batch, target_batch)

    return loss

# 평가 및 출력용 loss계산 함수
def calc_loss_loader(
    data_loader,
    model: GPTModel,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    """TODO: data_loader의 평균 loss를 계산합니다. 검증에서는 torch.no_grad()를 사용하세요."""
    total_loss = 0

    if len(data_loader) == 0 :
        return float("nan")
    elif num_batches is None :
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    if num_batches == 0 :
        return float("nan")

    # 평가용으로 가중치 업데이트가 되지 않기 위해 gradient 추적 off
    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i >= num_batches:
                break

            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            # 각 배치의 손실 더하기
            total_loss += loss.item()
                
        # 모든 배치의 손실 평균
        return total_loss/num_batches 

def save_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    path: str,
) -> None:
    """TODO: model/optimizer 상태, epoch, global_step을 torch.save로 저장합니다."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }
    # 진행사항을 checkpoint로 저장
    torch.save(checkpoint, path)

def load_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer | None,
    path: str,
    device: torch.device,
) -> tuple[int, int]:
    """TODO: torch.load로 checkpoint를 읽어 model/optimizer 상태를 복원합니다."""
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint["epoch"], checkpoint["global_step"]    

def generate(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_id: int | None = None,
) -> torch.Tensor:
    for _ in range(max_new_tokens):
        # 마지막 context_size만 자름
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)
        
        # 마지막 위치 logits만 꺼냄
        logits = logits[:, -1, :]

        if top_k is not None:
            # top_k가 vocab_size보다 크지 않도록 설정
            top_k = min(top_k, logits.size(-1))
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1].unsqueeze(-1)
            logits = torch.where(
                logits < min_val,
                torch.tensor(float("-inf"), device=logits.device),
                logits,
            )

        if temperature > 0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # 다음 토큰이 모두 eos 토큰이면 생성 중단
        if eos_id is not None and torch.all(idx_next == eos_id):
            break

        # idx뒤에 다음 토큰 idx_next 붙임
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

def generate_and_print_sample(
    model: GPTModel,
    tokenizer,
    device: torch.device,
    start_context: str,
    max_new_tokens: int = 50,
    context_size: int = 256,
    temperature: float = 0.8,
    top_k: int | None = 40,
) -> None:
    """TODO: start_context를 encode하고 generate 후 decode하여 출력합니다."""
    raise NotImplementedError("generate_and_print_sample을 구현하세요.")


def train_model(
    model: GPTModel,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    start_context: str,
    tokenizer,
    ckpt_freq: int | None = None,
    start_epoch: int = 0,
    global_step: int = 0,
) -> list[float]:
    """TODO: 사전 학습 루프를 구현하고 epoch별 train loss 리스트를 반환합니다."""
    raise NotImplementedError("train_model을 구현하세요.")


def plot_losses(train_losses: list[float], val_losses: list[float] | None = None) -> None:
    """훈련/검증 손실 그래프를 그리는 제공 함수."""
    plt.plot(train_losses, label="Train")
    if val_losses is not None:
        plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training / Validation Loss")
    plt.show()
