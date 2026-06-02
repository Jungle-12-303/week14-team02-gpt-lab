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
    # 생성 시작 문자열
    start_context: str,
    # 새 토큰을 생성할 개수
    max_new_tokens: int = 50,
    context_size: int = 256,
    temperature: float = 0.8,
    top_k: int | None = 40,
) -> None:
    """TODO: start_context를 encode하고 generate 후 decode하여 출력합니다."""
    # 생성시에는 dropout 끄기
    model.eval()
    
    encoded = tokenizer.encode(start_context)
    idx = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)

    # generate 함수 호출해서 next_token 저장
    generated_ids = generate(
        model=model,
        idx=idx,
        max_new_tokens=max_new_tokens,
        context_size=context_size,
        temperature=temperature,
        top_k=top_k,
    )

    generated_text = tokenizer.decode(generated_ids[0].tolist())
    print(generated_text)

    # 학습시 dropout 켜기
    model.train()

def train_model(
    model: GPTModel,
    # 훈련 batch data
    train_loader,
    # 검증 batch data
    val_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    # 몇 step마다 train/val loss를 평가할지
    eval_freq: int,
    # 몇 batch만 볼지
    eval_iter: int,
    start_context: str,
    tokenizer,
    # checkpoint를 저장할 주기
    ckpt_freq: int | None = None,
    start_epoch: int = 0,
    global_step: int = 0,
) -> list[float]:
    """TODO: 사전 학습 루프를 구현하고 epoch별 train loss 리스트를 반환합니다."""
    # 평가한 손실값 기록 리스트
    train_losses = []
    val_losses = []

    model.to(device)
    model.train()

    # 전체 데이터셋 반복
    for epoch in range(start_epoch, num_epochs):
        for input_batch, target_batch in train_loader:
            # batch마다 gradient 초기화
            optimizer.zero_grad()

            # loss계산
            loss = calc_loss_batch(input_batch, target_batch, model, device)

            # gradient 계산
            loss.backward()
            # weight 업데이트
            optimizer.step()

            global_step += 1

            if global_step % eval_freq == 0:
                # 평균 loss계산
                train_loss = calc_loss_loader(
                    train_loader, model, device, num_batches=eval_iter
                )
                val_loss = calc_loss_loader(
                    val_loader, model, device, num_batches=eval_iter
                )
                # loss 결과 기록
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                print(
                    f"Epoch {epoch + 1}, Step {global_step}: "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )

                # 다음 예측 token 샘플 출력
                generate_and_print_sample(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    start_context=start_context,
                    context_size=model.config["context_length"],
                )
                
            # checkpoint 저장
            if ckpt_freq is not None and global_step % ckpt_freq == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    path=f"checkpoint_step_{global_step}.pt",
                )

    return train_losses, val_loss


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
