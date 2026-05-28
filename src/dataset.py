# -*- coding: utf-8 -*-
"""GPT 사전 학습용 Dataset/DataLoader 과제 템플릿."""

import torch
from torch.utils.data import DataLoader, Dataset


class GPTDataset(Dataset):
    """
    token ID 리스트를 다음 토큰 예측용 input/target 쌍으로 자릅니다.
    
    tokenizer가 만든 숫자 리스트 (전체)
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
        # stride값이 들어왔는지 확인하고 없으면 context_length사용
        self.stride = stride if stride is not None else context_length
        # TODO: 만들 수 있는 학습 샘플 개수를 self._length에 저장하세요.

        #start의 최대 시작점
        max_start_index = len(self.token_ids) - self.context_length - 1

        # 최대 시작점이 0보다 작으면 0으로 설정
        if max_start_index < 0:
            self._length = 0

        # sample개수 구하기
        # dataset에 sample이 얼마나있는지 확인하고 epoch의 기준이 됨
        else:
            self._length = max_start_index // self.stride + 1

    def __len__(self) -> int:
        """TODO: 전체 샘플 개수를 반환합니다."""
        return self._length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: idx번째 input_ids와 target_ids를 LongTensor로 반환합니다.
        
        Returns:
            input_ids: (context_length,)
            target_ids: (context_length,)
        """
        # sample의 시작 index구하기
        start = idx * self.stride
        # 시작점에서 자를 길이를 더함
        end = start + self.context_length
        
        # 입력값 만들기
        input_ids = self.token_ids[start:end]
        # 입력값 보다 한 칸 뒤의 정답 target 만들기
        target_ids = self.token_ids[start + 1: end + 1]

        # tensor로 바꿔서 반환
        return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(target_ids, dtype=torch.long),
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
    """TODO: GPTDataset을 만들고 torch.utils.data.DataLoader로 감싸 반환합니다."""

