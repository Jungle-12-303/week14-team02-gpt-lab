# -*- coding: utf-8 -*-
"""GPT 사전 학습 유틸리티 과제 템플릿."""

import matplotlib.pyplot as plt
import torch

try:
    from .model import GPTModel
except ImportError:
    from model import GPTModel


# 한 배치를 device로 옮긴 뒤 다음 토큰 예측 cross entropy loss를 계산합니다.
def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: GPTModel,
    device: torch.device,
) -> torch.Tensor:
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)  # (B, T, vocab_size)

    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),    # (B, T, vocab_size) -> (B*T, vocab_size)
        target_batch.reshape(-1)                # (B, T) -> (B*T)
    )
    return loss                                 # 모든 위치의 평균 loss를 scalar Tensor로 반환.


# data_loader의 평균 loss를 계산합니다. 검증에서는 torch.no_grad()를 사용하세요.
def calc_loss_loader(
    data_loader,
    model: GPTModel,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    if len(data_loader) == 0:
        return float("nan")

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    if num_batches == 0:
        return float("nan")

    was_training = model.training
    model.eval()                                # dropout 등 해제

    total_loss = 0.0
    with torch.no_grad():
        for batch_idx, (input_batch, target_batch) in enumerate(data_loader):
            if batch_idx >= num_batches:
                break
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()

    if was_training:
        model.train()

    return total_loss / num_batches             # 평균 loss를 반환.


# model/optimizer 상태, epoch, global_step을 torch.save로 저장합니다.
def save_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    path: str,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),             # 모델 학습 파라미터
        "optimizer_state_dict": optimizer.state_dict(),     # optimizer 상태
        "epoch": epoch,                                     # 다음 학습 epoch 번호
        "global_step": global_step                          # 다음 학습 global step
    }
    torch.save(checkpoint, path)                            # checkpoint dict 저장.


# torch.load로 checkpoint를 읽어 model/optimizer 상태를 복원합니다.
def load_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer | None,
    path: str,
    device: torch.device,
) -> tuple[int, int]:
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint["epoch"], checkpoint["global_step"]   # 다음 학습 위치 정보 반환.


# temperature와 top-k 샘플링을 지원하는 생성 함수를 구현합니다.
def generate(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_id: int | None = None,
) -> torch.Tensor:
    was_training = model.training
    model.eval()

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)                        # (B, T, vocab_size)

        logits = logits[:, -1, :]                           # (B, vocab_size)

        if top_k is not None:                               # top-k 필터링
            top_k = min(top_k, logits.size(-1))
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1].unsqueeze(-1)
            logits = torch.where(
                logits < min_val,
                torch.full_like(logits, float("-inf")),
                logits
            )

        if temperature > 0.0:                               # temperature 확률 샘플링
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and (idx_next == eos_id).all():
            break

        idx = torch.cat((idx, idx_next), dim=1)             # 기존 토큰에 새 토큰 연장.

    if was_training:
        model.train()

    return idx                                              # 생성된 전체 토큰 반환.


# start_context를 encode하고 generate 후 decode하여 출력합니다.
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
    encoded = tokenizer.encode(start_context)                                       # (T,)
    idx = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)       # (1, T)

    token_ids = generate(
        model=model,
        idx=idx,
        max_new_tokens=max_new_tokens,
        context_size=context_size,
        temperature=temperature,
        top_k=top_k
    )                                                               # (1, T + max_new_tokens)

    decoded_text = tokenizer.decode(token_ids.squeeze(0).tolist())  # (T + max_new_tokens,)
    print(decoded_text.replace("\n", " "))


# 사전 학습 루프를 구현하고 epoch별 train loss 리스트를 반환합니다.
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
    model.to(device)

    train_losses = []

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            global_step += 1

            if eval_freq > 0 and global_step % eval_freq == 0:              # loss 결과 출력 주기
                train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
                val_loss = calc_loss_loader(val_loader, model, device, eval_iter)
                print(
                    f"Ep {epoch + 1} "
                    f"(Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}"
                )

            if ckpt_freq is not None and global_step % ckpt_freq == 0:      # checkpoint 저장 주기
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    path=f"checkpoint_step_{global_step}.pt"
                )

        epoch_train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
        train_losses.append(epoch_train_loss)                               # epoch별 train loss 저장.

        generate_and_print_sample(
            model=model,
            tokenizer=tokenizer,
            device=device,
            start_context=start_context,
            context_size=model.config["context_length"]
        )

    return train_losses                                             # epoch별 train loss 리스트를 반환.


# 훈련/검증 손실 그래프를 그리는 제공 함수.
def plot_losses(train_losses: list[float], val_losses: list[float] | None = None) -> None:
    plt.plot(train_losses, label="Train")
    if val_losses is not None:
        plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training / Validation Loss")
    plt.show()
