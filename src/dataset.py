# -*- coding: utf-8 -*-
"""GPT 사전 학습용 Dataset/DataLoader 과제 템플릿."""

import torch
from torch.utils.data import DataLoader, Dataset


class GPTDataset(Dataset):
    """
    token ID 리스트를 다음 토큰 예측용 input/target 쌍으로 자릅니다.

    예: token_ids=[10, 11, 12, 13], context_length=3
    - input:  [10, 11, 12]
    - target: [11, 12, 13]
    """

    def __init__(
        self,
        token_ids: list[int],
        context_length: int,
        stride: int | None = None,
    ):
        self.token_ids = token_ids
        self.context_length = context_length
        self.stride = stride if stride is not None else context_length
        # 만들 수 있는 학습 샘플 개수를 self._length에 저장하세요.
        if len(self.token_ids) <= self.context_length:
            self._length = 0
        else:
            self._length = (len(self.token_ids) - self.context_length - 1) // self.stride + 1

    # 전체 샘플 개수를 반환합니다.
    def __len__(self) -> int:
        return self._length

    # idx번째 input_ids와 target_ids를 LongTensor로 반환합니다.
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        end = start + self.context_length

        input_ids = self.token_ids[start:end]
        target_ids = self.token_ids[start + 1:end + 1]

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long)
        )


def create_dataloader(
    token_ids: list[int],
    context_length: int,
    batch_size: int = 8,
    stride: int | None = None,
    drop_last: bool = False,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    # GPTDataset을 만들고 torch.utils.data.DataLoader로 감싸 반환합니다.
    dataset = GPTDataset(
        token_ids = token_ids,
        context_length = context_length,
        stride = stride
    )
    return DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        drop_last = drop_last,
        num_workers = num_workers
    )
