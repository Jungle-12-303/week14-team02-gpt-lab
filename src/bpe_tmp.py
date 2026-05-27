# -*- coding: utf-8 -*-
"""
UTF-8 byte-level BPE 토크나이저 과제 템플릿.

외부 tokenizer 라이브러리 없이 BPE(Byte Pair Encoding)를 직접 구현합니다.
한국어 NSMC 리뷰를 다루므로 문자열을 글자/공백 단위로 먼저 자르지 말고,
항상 `text.encode("utf-8")`로 byte ID 시퀀스를 만든 뒤 merge를 적용하세요.
"""

import json
from pathlib import Path


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
SPECIAL_IDS = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
BYTE_OFFSET = len(SPECIAL_TOKENS)
NUM_BYTES = 256


class BPETokenizer:
    """
    UTF-8 byte-level BPE 토크나이저.

    권장 ID 배치:
    - 0~3: <pad>, <unk>, <bos>, <eos>
    - 4~259: 원본 byte 0~255
    - 260 이상: BPE merge로 생성한 토큰
    """

    def __init__(self, vocab_size: int = 3000):
        self.vocab_size = vocab_size
        self.id_to_token = {}
        self.token_to_id = {}
        self.merges = []

    # 1. 특수 토큰 4개를 고정 ID 0~3에 등록합니다.
    # 2. byte 0~255를 ID 4~259에 bytes([byte_value]) 형태로 등록합니다.
    def _init_special_tokens(self):
        self.id_to_token = {}
        self.token_to_id = {}
        self.merges = []

        for idx, tkn in enumerate(SPECIAL_TOKENS):
            self.id_to_token[idx] = tkn
            self.token_to_id[tkn] = idx

        for byte_val in range(NUM_BYTES):
            token_id = BYTE_OFFSET + byte_val
            token = bytes([byte_val])
            self.id_to_token[token_id] = token
            self.token_to_id[token] = token_id

    def get_pad_id(self):
        """padding 토큰 ID."""
        return SPECIAL_IDS[PAD_TOKEN]

    def get_unk_id(self):
        """unknown 토큰 ID."""
        return SPECIAL_IDS[UNK_TOKEN]

    def get_bos_id(self):
        """문장 시작 토큰 ID."""
        return SPECIAL_IDS[BOS_TOKEN]

    def get_eos_id(self):
        """문장 끝 토큰 ID."""
        return SPECIAL_IDS[EOS_TOKEN]

    # 코퍼스에서 BPE merge rule과 vocabulary를 학습합니다.
    def train(self, corpus: str):
        self._init_special_tokens()
        ids = [BYTE_OFFSET + byte_val for byte_val in corpus.encode("utf-8")]

        while len(self.id_to_token) < self.vocab_size:
            pair_counts = {}

            for left, right in zip(ids, ids[1:]):
                pair = (left, right)
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

            if not pair_counts:
                break

            best_pair = max(pair_counts, key=pair_counts.get)
            # best_pair = max(pair_counts.items(), key=lambda item: (item[1], item[0]))[0]
            new_token_id = len(self.id_to_token)

            self.id_to_token[new_token_id] = best_pair
            self.token_to_id[best_pair] = new_token_id
            self.merges.append(best_pair)

            merged_ids = []
            i = 0
            while i < len(ids):
                if i + 1 < len(ids) and (ids[i], ids[i + 1]) == best_pair:
                    merged_ids.append(new_token_id)
                    i += 2
                else:
                    merged_ids.append(ids[i])
                    i += 1

            ids = merged_ids

    # vocabulary와 merge rule을 JSON 파일로 저장합니다.
    def save(self, path: str | Path):
        data = {
            "vocab_size": self.vocab_size,
            "merges": [[left, right] for left, right in self.merges]
        }
        Path(path).write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    # save()로 저장한 JSON 파일을 읽어 vocabulary와 merge rule을 복원합니다.
    def load(self, path: str | Path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.vocab_size = data["vocab_size"]

        self._init_special_tokens()

        for pair in data["merges"]:
            merge = tuple(pair)
            token_id = len(self.id_to_token)
            self.id_to_token[token_id] = merge
            self.token_to_id[merge] = token_id
            self.merges.append(merge)

    # 문자열을 token ID 리스트로 변환합니다.
    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        ids = [BYTE_OFFSET + byte_val for byte_val in text.encode("utf-8")]

        for pair in self.merges:
            new_token_id = self.token_to_id[pair]
            new_ids = []
            i = 0

            while i < len(ids):
                if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
                    new_ids.append(new_token_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1

            ids = new_ids
        
        if add_bos_eos:
            ids = [self.get_bos_id()] + ids + [self.get_eos_id()]

        return ids

    # token ID 리스트를 문자열로 복원합니다.
    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        special_ids = set(SPECIAL_IDS.values())

        def rcs_expand_token(token_id: int) -> list[int]:
            token = self.id_to_token[token_id]

            if isinstance(token, bytes):
                return list(token)

            if isinstance(token, tuple):
                left, right = token
                return rcs_expand_token(left) + rcs_expand_token(right)

            if isinstance(token, str):
                return list(token.encode("utf-8"))

            raise TypeError(f"Unsupported token type: {type(token)}")

        byte_values = []
        for tkn_id in ids:
            if skip_special and tkn_id in special_ids:
                continue
            byte_values.extend(rcs_expand_token(tkn_id))

        return bytes(byte_values).decode("utf-8")
