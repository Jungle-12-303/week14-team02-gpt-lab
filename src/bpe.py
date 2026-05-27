# -*- coding: utf-8 -*-
"""
UTF-8 byte-level BPE 토크나이저 과제 템플릿.

외부 tokenizer 라이브러리 없이 BPE(Byte Pair Encoding)를 직접 구현합니다.
한국어 NSMC 리뷰를 다루므로 문자열을 글자/공백 단위로 먼저 자르지 말고,
항상 `text.encode("utf-8")`로 byte ID 시퀀스를 만든 뒤 merge를 적용하세요.
"""

from pathlib import Path
import json

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

    def _init_special_tokens(self):
        """
        TODO:
        1. 특수 토큰 4개를 고정 ID 0~3에 등록합니다.
        2. byte 0~255를 ID 4~259에 bytes([byte_value]) 형태로 등록합니다.
        """
        # 1. 특수 토큰 등록
        self.id_to_token[self.get_pad_id()] = PAD_TOKEN
        self.id_to_token[self.get_unk_id()] = UNK_TOKEN
        self.id_to_token[self.get_bos_id()] = BOS_TOKEN
        self.id_to_token[self.get_eos_id()] = EOS_TOKEN

        self.token_to_id[PAD_TOKEN] = self.get_pad_id()
        self.token_to_id[UNK_TOKEN] = self.get_unk_id()
        self.token_to_id[BOS_TOKEN] = self.get_bos_id()
        self.token_to_id[EOS_TOKEN] = self.get_eos_id()

        # 2. byte 등록
        for i in range(256):
            self.id_to_token[i + BYTE_OFFSET] = bytes([i])
            self.token_to_id[bytes([i])] = i + BYTE_OFFSET

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

    def train(self, corpus: str):
        """
        TODO: 코퍼스에서 BPE merge rule과 vocabulary를 학습합니다.

        구현 힌트:
        - `corpus.encode("utf-8")`로 byte ID 시퀀스를 만듭니다.
        - 가장 자주 등장하는 이웃 token pair를 찾습니다.
        - 새 token ID를 만들고, 시퀀스의 해당 pair를 새 ID로 치환합니다.
        - `self.merges`, `self.id_to_token`, `self.token_to_id`를 갱신합니다.
        """

        self._init_special_tokens()
        byte_ids = [i + BYTE_OFFSET for i in corpus.encode("utf-8")]

        while len(self.id_to_token) < self.vocab_size:
            pair_freq = {}
            for i in range(len(byte_ids) - 1):
                pair = (byte_ids[i], byte_ids[i + 1])
                pair_freq[pair] = pair_freq.get(pair, 0) + 1

            #페어가 업서..?
            if not pair_freq:
                break
            
            # 가장 자주 등장하는 pair를 찾고 merge rule과 vocabulary에 추가
            most_freq_pair = max(pair_freq, key=pair_freq.get)
            self.merges.append(most_freq_pair)

            new_id = len(self.id_to_token)
            self.id_to_token[new_id] = most_freq_pair
            self.token_to_id[most_freq_pair] = new_id

            #bye_ids에서 most_freq_pair를 new_id로 치환
            new_byte_ids = []
            i = 0
            while i < len(byte_ids):
                if i < len(byte_ids) - 1 and (byte_ids[i], byte_ids[i + 1]) == most_freq_pair:
                    new_byte_ids.append(new_id)
                    i += 2
                else:
                    new_byte_ids.append(byte_ids[i])
                    i += 1
            byte_ids = new_byte_ids


    def save(self, path: str | Path):
        """
        TODO: vocabulary와 merge rule을 JSON 파일로 저장합니다.

        bytes와 tuple은 JSON에 바로 저장할 수 없으므로 type 정보를 함께 저장하세요.
        """
        data = {
            "vocab_size": self.vocab_size,
            "id_to_token": {},
            "merges": [list(pair) for pair in self.merges]
        }

        for id, token in self.id_to_token.items():
            if isinstance(token, str):
                data["id_to_token"][str(id)] = {"type": "str", "value": token}

            elif isinstance(token, bytes):
                data["id_to_token"][str(id)] = {"type": "bytes", "value": list(token)}

            elif isinstance(token, tuple):
                data["id_to_token"][str(id)] = {"type": "tuple", "value": list(token)}

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self, path: str | Path):
        """
        TODO: save()로 저장한 JSON 파일을 읽어 vocabulary와 merge rule을 복원합니다.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.vocab_size = data["vocab_size"]
        self.merges = [tuple(pair) for pair in data["merges"]]

        self.id_to_token = {}
        self.token_to_id = {}
        for id, token_data in data["id_to_token"].items():
            if token_data["type"] == "str":
                token = token_data["value"]

            elif token_data["type"] == "bytes":
                token = bytes(token_data["value"])

            elif token_data["type"] == "tuple":
                token = tuple(token_data["value"])

            self.id_to_token[int(id)] = token
            self.token_to_id[token] = int(id)

    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        """
        TODO: 문자열을 token ID 리스트로 변환합니다.

        구현 힌트:
        - 먼저 UTF-8 byte ID 리스트를 만듭니다.
        - train/load에서 얻은 merge rule을 학습 순서대로 적용합니다.
        - add_bos_eos=True이면 앞뒤에 bos/eos ID를 붙입니다.
        """

        ids = [i + BYTE_OFFSET for i in text.encode("utf-8")]

        for merge in self.merges:
            new_ids = []
            
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i+1]) == merge:
                    new_ids.append(self.token_to_id[merge])
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1

            ids = new_ids

        if add_bos_eos:
            ids = [self.get_bos_id()] + ids + [self.get_eos_id()]

        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """
        TODO: token ID 리스트를 문자열로 복원합니다.

        주의:
        - merge token은 원본 byte token까지 재귀적으로 펼칩니다.
        - byte를 하나씩 decode하지 말고, 마지막에 `bytes(...).decode("utf-8")`를 한 번만 호출합니다.
        """

        tokens = []
        for id in ids:
            if skip_special and id in SPECIAL_IDS.values():
                continue

            tokens.extend(self.expand(id))

        return bytes(tokens).decode("utf-8")
    
    def expand(self, id: int) -> list[int]:
        if id < BYTE_OFFSET + NUM_BYTES:
            return [id - BYTE_OFFSET]
        
        merge = self.id_to_token[id]
        return self.expand(merge[0]) + self.expand(merge[1])