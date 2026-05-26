# WEEK13-14 LLM mini GPT 학습/구현 계획

## 0. 전제

- 기간: **2026년 5월 26일 화요일 19:00 ~ 2026년 6월 2일 화요일 24:00**
- 기본 작업 시간: **매일 11:00 ~ 24:00**
- 예외:
  - **5월 26일 화요일**: 19:00 ~ 24:00
  - **5월 27일 수요일 저녁**: 다음날 오전 발표 준비 중심
  - **5월 28일 목요일 오전**: 발표, 이후 11:00부터 계획 진행
  - **5월 31일 일요일**: 휴식
- 진행 방식:
  - 구현 분담 없이 **각자 같은 과제를 구현**한다.
  - 각자 테스트를 통과시킨 뒤 코어타임에서 구현을 비교한다.
  - 최종 제출 코드는 여러 브랜치를 바로 main에 합치지 않고, **통합 브랜치에서 파일별로 좋은 구현을 선택/조합**한다.
- 학습 대상:
  - 지난주 MNIST를 처음 구현해본 사람이 있어도 따라올 수 있도록, 매일 코어타임에서 개념과 shape를 맞춘다.
- 구현 대상:
  - `src/bpe.py`
  - `src/dataset.py`
  - `src/embeddings.py`
  - `src/attention.py`
  - `src/model.py`
  - `src/train.py`
  - `src/finetune.py`
  - `REPORT.md`

---

## 1. 이번 주 최종 목표

### 최소 목표

- 단계별 테스트 통과
- 전체 테스트 `pytest tests/ -v` 통과
- `REPORT.md` 작성
- 짧은 생성 샘플 1개 이상 확보

### 표준 목표

- 전체 테스트 통과
- NSMC 기반 작은 GPT 사전학습 smoke run 완료
- loss 기록
- 간단한 감성 분류 fine-tuning 1회 실행
- 결과를 보고서에 정리

### 도전 목표

- train/validation loss 그래프 정리
- 같은 prompt에 대한 생성 샘플 비교
- fine-tuning validation/test accuracy 기록
- 하이퍼파라미터 변경 실험 1~2개 수행

---

## 2. 우선순위

시간이 부족하면 아래 순서대로 우선한다.

1. 단계별 테스트 통과
2. 전체 테스트 통과
3. 작은 데이터로 사전학습 1회 실행
4. 생성 샘플 확보
5. 감성 분류 fine-tuning 1회 실행
6. 보고서 완성
7. 성능 개선, 하이퍼파라미터 실험

추가 미션인 warmup, cosine decay, weight decay 실험, 하이퍼파라미터 탐색은 필수 구현이 끝난 뒤에만 진행한다.

---

## 3. 매일 기본 진행 방식

하루 전체를 강의식으로만 보내거나 각자 따로만 구현하면 이해가 갈라질 수 있다. 그래서 아래 루프를 반복한다.

| 단계 | 시간 | 내용 |
|---|---:|---|
| 개념 확인 | 40~50분 | 책, README, 테스트 코드, TODO 읽기 |
| 개인 시도 | 40~60분 | 각자 구현 방향 생각 또는 로컬 구현 |
| 코어타임 | 20~30분 | 이해한 것, shape, 테스트 기준 맞추기 |
| 구현 | 60~90분 | 각자 구현하거나 화면 공유로 대표 구현 확인 |
| 테스트 | 20~40분 | 해당 단계 pytest 실행, 실패 원인 정리 |
| 기록 | 10~20분 | 오늘 배운 것, 막힌 것, 다음 액션 기록 |

---

## 4. 매일 공통 체크 질문

매일 마지막 30~40분에는 아래 질문을 확인한다.

```text
1. 오늘 구현한 파일은 무엇인가?
2. 입력 shape와 출력 shape는 무엇인가?
3. 테스트가 검증하는 핵심은 무엇인가?
4. 처음 신경망을 공부하는 사람에게 5분 안에 설명할 수 있는가?
5. 내일 가장 먼저 확인해야 할 실패/질문은 무엇인가?
```

---

## 5. 주간 계획 요약

| 날짜 | 시간 | 핵심 목표 | 주요 산출물 |
|---|---|---|---|
| 5/26 화 | 19:00~24:00 | 구조 파악, 환경 세팅 | 실패 테스트 로그, 구현 순서 이해 |
| 5/27 수 | 11:00~24:00 | BPE 시작, Dataset/Embedding 이해, 발표 준비 | BPE 초안, 발표 준비 |
| 5/28 목 | 11:00~24:00 | BPE 마무리, Dataset/Embedding 통과, Attention 시작 | `test_bpe`, `test_dataset` 통과 목표 |
| 5/29 금 | 11:00~24:00 | Attention 통과, GPTModel 구현 | `test_attention`, `test_model` 통과 목표 |
| 5/30 토 | 11:00~24:00 | Train 구현, 작은 학습 실행 | `test_train` 통과, smoke training |
| 5/31 일 | 휴식 | 휴식 | 없음 |
| 6/1 월 | 11:00~24:00 | Fine-tune 구현, 전체 테스트, 보고서 | `test_finetune`, 전체 테스트 1차 |
| 6/2 화 | 11:00~24:00 | 최종 안정화, 재현성, 제출 준비 | 전체 테스트 최종, 보고서 완성 |

---

# 6. 하루별 상세 계획

## 5월 26일 화요일: 19:00 ~ 24:00

### 목표

- 프로젝트 전체 구조 파악
- 환경 세팅
- 실패 테스트가 정상이라는 것 이해
- 이번 주 구현 순서 합의

| 시간 | 내용 | 산출물 |
|---|---|---|
| 19:00~19:30 | 이번 주 목표 정리 | 최종 목표 공유 |
| 19:30~20:10 | README, REPORT, tests 구조 같이 읽기 | 구현 순서 이해 |
| 20:10~20:40 | 코어타임: MNIST와 GPT 학습 비교 | 공통 설명 정리 |
| 20:40~21:30 | clone, 가상환경, requirements 설치 | 실행 환경 준비 |
| 21:30~22:00 | `pytest tests/ -v` 실행 | 실패 로그 확보 |
| 22:00~22:30 | TODO / NotImplementedError 목록 확인 | 구현 대상 확인 |
| 22:30~23:20 | 책 1장 훑기: LLM, next token prediction | 5줄 요약 |
| 23:20~24:00 | 내일 할 일 정리 | BPE 진입 준비 |

### 오늘 코어타임 질문

```text
1. GPT도 MNIST처럼 입력 → 모델 → loss → backward → optimizer 구조인가?
2. MNIST의 label과 GPT의 target은 무엇이 다른가?
3. 지금 테스트가 실패하는 이유는 무엇인가?
4. 이번 주 최종 제출물은 무엇인가?
```

---

## 5월 27일 수요일: 11:00 ~ 24:00

### 목표

- BPE tokenizer 개념 이해
- BPE 구현 시작
- Dataset/Embedding의 역할 이해
- 저녁에는 다음날 발표 준비

| 시간 | 내용 | 산출물 |
|---|---|---|
| 11:00~11:30 | 전날 복습: mini GPT 전체 흐름 | 흐름 재정리 |
| 11:30~12:20 | 책 2장 읽기: tokenization, BPE, sliding window | 개념 메모 |
| 12:20~13:00 | 코어타임: 문장을 token ID로 바꾸는 이유 | BPE 설명 초안 |
| 13:00~14:00 | 점심 |  |
| 14:00~14:50 | `src/bpe.py` 읽기: special token, byte token, merge rule | TODO 목록 |
| 14:50~15:40 | 각자 BPE 구현 방향 생각 | 개인 구현 초안 |
| 15:40~16:20 | 코어타임: BPE 구현 순서 합의 | 구현 순서 정리 |
| 16:20~17:40 | `_init_special_tokens`, encode/decode 기본 구조 구현 | BPE 일부 구현 |
| 17:40~18:20 | `pytest tests/test_bpe.py -v` 일부 실행 | 실패 원인 확인 |
| 18:20~19:20 | 저녁 |  |
| 19:20~20:20 | `src/dataset.py`, `src/embeddings.py` 읽기 | input/target 이해 |
| 20:20~21:00 | 코어타임: input과 target이 왜 한 칸 차이나는지 | shape 정리 |
| 21:00~22:00 | 다음날 오전 발표 준비 | 발표 자료 |
| 22:00~23:00 | 발표 리허설 / 자료 보완 | 리허설 완료 |
| 23:00~24:00 | 오늘 정리, 내일 질문 목록 작성 | 질문 목록 |

### 오늘 코어타임 질문

```text
1. 문자열은 어떻게 숫자 token ID가 되는가?
2. 한국어를 byte-level로 다루는 이유는 무엇인가?
3. <pad>, <bos>, <eos>, <unk>는 각각 언제 쓰는가?
4. input과 target이 한 칸 차이나는 이유는 무엇인가?
```

---

## 5월 28일 목요일: 11:00 ~ 24:00

### 목표

- BPE 구현 마무리
- Dataset/Embedding 테스트 통과
- Attention 개념 진입

| 시간 | 내용 | 산출물 |
|---|---|---|
| 11:00~11:30 | 발표 후 정리, 오늘 목표 재설정 | 오늘 목표 합의 |
| 11:30~12:30 | BPE 구현 마무리: train, save, load | BPE 완성 후보 |
| 12:30~13:00 | `pytest tests/test_bpe.py -v` 실행 | BPE 테스트 결과 |
| 13:00~14:00 | 점심 |  |
| 14:00~15:00 | BPE 버그 수정, encode/decode 복원 확인 | BPE 통과 목표 |
| 15:00~15:30 | 코어타임: BPE를 5분 안에 설명하기 | 설명 통일 |
| 15:30~16:30 | `src/dataset.py` 구현: sliding window, input/target | dataset 구현 |
| 16:30~17:20 | `src/embeddings.py` 구현: token/position embedding, dropout | embedding 구현 |
| 17:20~18:00 | `pytest tests/test_dataset.py -v` 실행 및 수정 | dataset 테스트 결과 |
| 18:00~19:00 | 저녁 |  |
| 19:00~20:00 | 책 3장 읽기: self-attention 기본 | attention 개념 메모 |
| 20:00~20:40 | 코어타임: Q, K, V를 말로 설명하기 | QKV 설명 |
| 20:40~22:00 | `src/attention.py` 읽기, shape 표 만들기 | shape 표 |
| 22:00~23:20 | Attention 구현 시작: QKV projection, head split | attention 일부 구현 |
| 23:20~24:00 | 오늘 정리: attention shape 암기표 작성 | shape 암기표 |

### 오늘 코어타임 질문

```text
1. BPE encode/decode가 깨지면 어디를 먼저 봐야 하는가?
2. Dataset에서 input과 target의 길이는 왜 같은가?
3. Embedding output shape은 무엇인가?
4. Q, K, V는 각각 어떤 역할인가?
```

### Attention shape 표

```text
x:        (B, T, C)
q/k/v:    (B, T, C)
split:    (B, heads, T, head_dim)
score:    (B, heads, T, T)
weights:  (B, heads, T, T)
context:  (B, heads, T, head_dim)
merge:    (B, T, C)
```

---

## 5월 29일 금요일: 11:00 ~ 24:00

### 목표

- Attention 테스트 통과
- GPTModel 구현
- Model 테스트 통과 또는 최소한 주요 버그 정리

| 시간 | 내용 | 산출물 |
|---|---|---|
| 11:00~11:30 | 전날 attention shape 복습 | shape 재확인 |
| 11:30~12:30 | Attention 구현: scaled dot-product, causal mask | attention 구현 |
| 12:30~13:00 | Attention 구현: softmax, dropout, output projection | attention 완성 후보 |
| 13:00~14:00 | 점심 |  |
| 14:00~15:00 | `pytest tests/test_attention.py -v` 실행 및 수정 | attention 통과 목표 |
| 15:00~15:40 | 코어타임: causal mask가 왜 필요한지 설명 | mask 설명 |
| 15:40~16:40 | 책 4장 읽기: LayerNorm, GELU, FFN, residual | model 개념 메모 |
| 16:40~17:40 | `src/model.py` 구현 1: LayerNorm, GELU, FeedForward | model 일부 구현 |
| 17:40~18:20 | 코어타임: residual connection 설명 | residual 설명 |
| 18:20~19:20 | 저녁 |  |
| 19:20~20:40 | `TransformerBlock` 구현 | block 구현 |
| 20:40~21:40 | `GPTModel` 구현: embedding, blocks, final norm, lm_head | model 구현 |
| 21:40~22:30 | `generate_text_simple` 구현 | greedy generation |
| 22:30~23:30 | `pytest tests/test_model.py -v` 실행 및 수정 | model 테스트 결과 |
| 23:30~24:00 | 오늘 정리: 모델 전체 흐름 한 장으로 정리 | 구조도 |

### 오늘 코어타임 질문

```text
1. attention score shape은 왜 (B, heads, T, T)인가?
2. causal mask가 없으면 GPT 학습에서 어떤 문제가 생기는가?
3. residual connection은 왜 필요한가?
4. LM head의 출력 차원은 왜 vocab_size인가?
```

---

## 5월 30일 토요일: 11:00 ~ 24:00

### 목표

- Model 테스트 통과
- train.py 구현
- 작은 데이터로 학습 루프 실행
- 생성 샘플 확보

| 시간 | 내용 | 산출물 |
|---|---|---|
| 11:00~11:40 | 전날 model 버그 정리 | 버그 목록 |
| 11:40~13:00 | `pytest tests/test_model.py -v` 통과까지 수정 | model 통과 목표 |
| 13:00~14:00 | 점심 |  |
| 14:00~14:50 | 책 5장 읽기: pretraining, loss, generation | train 개념 메모 |
| 14:50~15:30 | 코어타임: logits와 targets shape 맞추기 | loss shape 정리 |
| 15:30~16:30 | `src/train.py` 구현: `calc_loss_batch`, `calc_loss_loader` | loss 구현 |
| 16:30~17:30 | checkpoint save/load 구현 | checkpoint 구현 |
| 17:30~18:20 | generate 함수 구현: temperature, top-k, eos | sampling 구현 |
| 18:20~19:20 | 저녁 |  |
| 19:20~20:40 | `train_model` 구현 | train loop 구현 |
| 20:40~21:30 | `pytest tests/test_train.py -v` 실행 및 수정 | train 테스트 결과 |
| 21:30~22:30 | 아주 작은 데이터로 smoke training | loss 기록 |
| 22:30~23:20 | 생성 샘플 출력 | sample text |
| 23:20~24:00 | REPORT 초안 작성: BPE, Dataset, Attention, Model, Train | 보고서 초안 |

### 오늘 코어타임 질문

```text
1. logits shape은 무엇인가?
2. targets shape은 무엇인가?
3. CrossEntropyLoss에 넣기 위해 왜 reshape이 필요한가?
4. validation loss를 계산할 때 왜 torch.no_grad()를 쓰는가?
5. temperature와 top-k sampling은 generation에 어떤 영향을 주는가?
```

### Loss shape 메모

```text
logits:  (B, T, vocab_size)
targets: (B, T)

CrossEntropyLoss 입력:
logits.reshape(B*T, vocab_size)
targets.reshape(B*T)
```

---

## 5월 31일 일요일

### 목표

- 휴식

팀 전체 공식 일정은 없다.

가능하면 작업하지 않는다. 월요일에 통합 디버깅과 보고서를 집중해서 진행하기 위해 휴식한다.

---

## 6월 1일 월요일: 11:00 ~ 24:00

### 목표

- Fine-tuning 구현
- 전체 테스트 1차 통과
- 보고서 80% 이상 완성

| 시간 | 내용 | 산출물 |
|---|---|---|
| 11:00~11:30 | 전체 진행 상황 확인 | 남은 작업 목록 |
| 11:30~12:20 | 책 6장 읽기: classification fine-tuning | fine-tune 개념 메모 |
| 12:20~13:00 | 코어타임: pretraining과 fine-tuning 차이 설명 | 차이점 정리 |
| 13:00~14:00 | 점심 |  |
| 14:00~15:00 | `src/finetune.py` 구현: sentiment dataset 생성 | dataset 생성 |
| 15:00~16:00 | `ReviewSentimentDataset` 구현: encode, padding, label | classification dataset |
| 16:00~16:40 | 코어타임: classifier head를 왜 붙이는지 설명 | classifier 설명 |
| 16:40~17:40 | `GPTForSequenceClassification` 구현 | 분류 모델 구현 |
| 17:40~18:20 | train/evaluate sentiment 구현 | fine-tune loop |
| 18:20~19:20 | 저녁 |  |
| 19:20~20:20 | `pytest tests/test_finetune.py -v` 실행 및 수정 | fine-tune 테스트 |
| 20:20~21:30 | 전체 테스트: `pytest tests/ -v` | 전체 실패 목록 |
| 21:30~22:30 | 실패 테스트 집중 수정 | 전체 통과 목표 |
| 22:30~23:30 | REPORT 작성: 테스트 결과, 학습 결과, fine-tuning 결과 | 보고서 80% |
| 23:30~24:00 | 6/2 최종 제출 체크리스트 작성 | 제출 체크리스트 |

### 오늘 코어타임 질문

```text
1. pretraining과 fine-tuning의 target은 어떻게 다른가?
2. LM head와 classifier head는 무엇이 다른가?
3. 문장 대표 벡터는 hidden state의 어느 위치에서 가져올 수 있는가?
4. classification loss도 CrossEntropyLoss인가?
5. accuracy가 낮으면 무엇부터 의심해야 하는가?
```

---

## 6월 2일 화요일: 11:00 ~ 24:00

### 목표

- 최종 안정화
- 재현성 확인
- 전체 테스트 최종 실행
- 보고서 마무리
- 발표/설명 준비

| 시간 | 내용 | 산출물 |
|---|---|---|
| 11:00~11:40 | fresh clone 또는 clean 환경 기준 실행 순서 확인 | 재현성 점검 |
| 11:40~12:30 | 데이터 준비, tokenizer, model, train 흐름 점검 | 실행 순서 정리 |
| 12:30~13:00 | 코어타임: 처음부터 끝까지 파이프라인 설명 | 전체 설명 |
| 13:00~14:00 | 점심 |  |
| 14:00~15:00 | 최종 전체 테스트: `pytest tests/ -v` | 최종 테스트 로그 |
| 15:00~16:00 | 실패가 있으면 최소 수정, 없으면 로그 정리 | 안정화 |
| 16:00~17:00 | notebook 실행 가능 여부 확인 | notebook 검증 |
| 17:00~18:00 | REPORT 최종 작성 | 제출용 보고서 |
| 18:00~19:00 | 저녁 |  |
| 19:00~20:00 | 생성 샘플, loss 그래프, fine-tuning 결과 정리 | 결과 정리 |
| 20:00~21:00 | 발표/설명용 흐름 정리 | 발표 흐름 |
| 21:00~22:00 | 각자 한 파트씩 설명 연습 | Q&A 대비 |
| 22:00~23:00 | 최종 제출물 확인 | 제출 준비 |
| 23:00~24:00 | README/REPORT 오탈자, git status, commit 정리 | 최종 정리 |

### 오늘 코어타임 질문

```text
1. 우리 모델은 데이터를 처음부터 끝까지 어떤 순서로 처리하는가?
2. tokenizer → dataset → embedding → attention → model → loss 흐름을 설명할 수 있는가?
3. 전체 테스트 결과는 보고서에 어떻게 적을 것인가?
4. 생성 샘플이 이상하면 어떤 이유를 설명할 수 있는가?
5. 최종 제출물에 data, checkpoint, token이 포함되어 있지 않은가?
```

---

# 7. 구현 단계별 체크리스트

## 1단계: BPE Tokenizer

관련 파일:

```text
src/bpe.py
```

테스트:

```bash
pytest tests/test_bpe.py -v
```

체크리스트:

```text
- special token 4개 고정 ID 등록
- byte 0~255를 token으로 등록
- corpus를 UTF-8 byte로 변환
- 가장 자주 등장하는 pair 찾기
- merge rule 학습
- encode에서 merge rule 적용
- decode에서 merge token을 byte로 복원
- save/load 구현
```

코어 질문:

```text
1. byte-level BPE가 한국어에서 필요한 이유는?
2. encode 후 decode했을 때 원문이 복원되어야 하는 이유는?
3. merge token은 decode 때 어떻게 다시 byte로 펼치는가?
```

---

## 2단계: Dataset / Embedding

관련 파일:

```text
src/dataset.py
src/embeddings.py
```

테스트:

```bash
pytest tests/test_dataset.py -v
```

체크리스트:

```text
- token_ids에서 input/target 쌍 생성
- context_length, stride 처리
- DataLoader 생성
- token embedding 구현
- position embedding 구현
- dropout 적용
```

예시:

```text
token_ids = [10, 11, 12, 13]
context_length = 3

input  = [10, 11, 12]
target = [11, 12, 13]
```

코어 질문:

```text
1. input과 target은 왜 한 칸 차이나는가?
2. position embedding이 없으면 어떤 문제가 생기는가?
3. embedding output shape은 무엇인가?
```

---

## 3단계: Multi-Head Attention

관련 파일:

```text
src/attention.py
```

테스트:

```bash
pytest tests/test_attention.py -v
```

체크리스트:

```text
- Q/K/V projection
- head split
- score = QK^T / sqrt(head_dim)
- causal mask
- softmax
- dropout
- attention weight @ V
- head merge
- output projection
```

코어 질문:

```text
1. Q, K, V 각각의 역할은?
2. score shape은 왜 (B, heads, T, T)인가?
3. causal mask가 없으면 왜 cheating이 되는가?
```

---

## 4단계: GPT Model

관련 파일:

```text
src/model.py
```

테스트:

```bash
pytest tests/test_model.py -v
```

체크리스트:

```text
- LayerNorm
- GELU
- FeedForward
- TransformerBlock
- GPTModel
- generate_text_simple
```

전체 구조:

```text
InputEmbedding
-> TransformerBlock x N
-> final LayerNorm
-> LM Head
-> logits
```

TransformerBlock 구조:

```text
x = x + attention(layernorm(x))
x = x + feedforward(layernorm(x))
```

코어 질문:

```text
1. residual connection은 왜 필요한가?
2. LayerNorm은 어떤 차원을 정규화하는가?
3. LM head의 출력 차원은 왜 vocab_size인가?
```

---

## 5단계: Training

관련 파일:

```text
src/train.py
```

테스트:

```bash
pytest tests/test_train.py -v
```

체크리스트:

```text
- calc_loss_batch
- calc_loss_loader
- save_checkpoint
- load_checkpoint
- generate
- generate_and_print_sample
- train_model
```

코어 질문:

```text
1. CrossEntropyLoss에 logits와 targets를 어떻게 넣는가?
2. validation loss 계산 시 왜 no_grad를 쓰는가?
3. checkpoint에는 무엇이 저장되어야 하는가?
4. temperature와 top-k는 generation에서 어떤 역할을 하는가?
```

---

## 6단계: Fine-tuning

관련 파일:

```text
src/finetune.py
```

테스트:

```bash
pytest tests/test_finetune.py -v
```

체크리스트:

```text
- make_sentiment_dataset
- ReviewSentimentDataset
- GPTForSequenceClassification
- train_epoch_sentiment
- evaluate_sentiment
```

구조:

```text
review text
-> tokenizer
-> token IDs
-> GPT backbone
-> sentence representation
-> classifier head
-> positive/negative logits
```

코어 질문:

```text
1. pretraining과 fine-tuning은 label이 어떻게 다른가?
2. LM head를 그대로 쓰지 않고 classifier head를 붙이는 이유는?
3. 문장 대표 벡터는 어디서 가져올 수 있는가?
```

---

# 8. GitHub 코드 통합 계획

## 기본 원칙

```text
1. main 직접 push 금지
2. 각자 feat/이름 브랜치에서 구현
3. 각자 테스트 통과 후 push
4. 개인 브랜치를 main에 바로 merge하지 않기
5. integration/week13-14 브랜치에서 최종본 조립
6. 파일 하나 합칠 때마다 해당 테스트 실행
7. 전체 테스트 통과 후 integration -> main PR 생성
8. 팀원 리뷰 후 main에 merge
```

## 각자 브랜치 생성

```bash
git checkout main
git pull origin main

git checkout -b feat/your-name
```

작업 후:

```bash
pytest tests/ -v

git add src/ REPORT.md
git commit -m "Implement mini GPT components"
git push -u origin feat/your-name
```

## 통합 브랜치 생성

```bash
git checkout main
git pull origin main

git checkout -b integration/week13-14
```

## 파일별로 좋은 구현 가져오기

예시:

```bash
git fetch origin

# BPE만 가져오기
git checkout origin/feat/minsu -- src/bpe.py
pytest tests/test_bpe.py -v
git add src/bpe.py
git commit -m "Integrate BPE tokenizer"

# Attention만 가져오기
git checkout origin/feat/jiwon -- src/attention.py
pytest tests/test_attention.py -v
git add src/attention.py
git commit -m "Integrate multi-head attention"
```

## 추천 통합 순서

```text
1. src/bpe.py
2. src/dataset.py, src/embeddings.py
3. src/attention.py
4. src/model.py
5. src/train.py
6. src/finetune.py
7. REPORT.md
8. pytest tests/ -v
```

## 최종 PR 본문 예시

```md
## 구현 내용

- UTF-8 byte-level BPE tokenizer 구현
- GPTDataset / InputEmbedding 구현
- Causal Multi-Head Self-Attention 구현
- GPTModel 및 generation 구현
- Pretraining utilities 구현
- NSMC sentiment fine-tuning 구현
- REPORT.md 작성

## 테스트

- [ ] pytest tests/test_bpe.py -v
- [ ] pytest tests/test_dataset.py -v
- [ ] pytest tests/test_attention.py -v
- [ ] pytest tests/test_model.py -v
- [ ] pytest tests/test_train.py -v
- [ ] pytest tests/test_finetune.py -v
- [ ] pytest tests/ -v

## 주의사항

- data/ 파일 commit하지 않음
- checkpoint 파일 commit하지 않음
- token/password 없음
```

---

# 9. 매일 기록 템플릿

매일 23:20~24:00 사이에 아래를 작성한다.

```md
## 날짜

## 오늘 구현한 파일

## 오늘 통과한 테스트

## 실패한 테스트

## 실패 원인 요약

## 오늘 이해한 핵심 개념

## 헷갈리는 shape

## 내일 가장 먼저 할 일
```

---

# 10. REPORT.md 작성 체크리스트

보고서에는 아래 항목을 채운다.

```text
- 반 / 팀명 / 팀원
- 구현 현황
- 테스트 통과 현황
- 실패 테스트와 해결 시도
- 데이터 전처리 방식
- BPE 구현 방식
- 특수 토큰 ID
- vocab_size
- 모델 구조
- context_length
- emb_dim
- n_heads
- n_layers
- 파라미터 수
- 사전학습 하이퍼파라미터
- train loss
- validation loss
- 생성 샘플
- checkpoint 경로
- fine-tuning 설정
- validation/test accuracy
- 실험 환경
- 어려웠던 점
- 개선하고 싶은 점
```

---

# 11. 최종 제출 전 체크리스트

6월 2일 최종 확인한다.

```text
[ ] pytest tests/test_bpe.py -v 통과
[ ] pytest tests/test_dataset.py -v 통과
[ ] pytest tests/test_attention.py -v 통과
[ ] pytest tests/test_model.py -v 통과
[ ] pytest tests/test_train.py -v 통과
[ ] pytest tests/test_finetune.py -v 통과
[ ] pytest tests/ -v 전체 통과
[ ] gpt-lab.ipynb 실행 가능
[ ] REPORT.md 작성 완료
[ ] 생성 샘플 포함
[ ] loss 기록 포함
[ ] fine-tuning 결과 포함 또는 미실행 사유 작성
[ ] data/ 파일 commit하지 않음
[ ] checkpoint 파일 commit하지 않음
[ ] token/password commit하지 않음
[ ] main PR 생성
[ ] 팀원 리뷰 후 merge
```

---

# 12. 이번 주 한 줄 전략

```text
화요일은 구조 파악,
수요일은 BPE와 발표 준비,
목요일은 BPE/Dataset/Embedding과 Attention 시작,
금요일은 Attention과 Model,
토요일은 Train과 작은 학습,
일요일은 휴식,
월요일은 Fine-tune과 전체 테스트,
화요일은 최종 안정화와 보고서 마무리.
```

가장 중요한 기준은 다음이다.

```text
코드가 돌아가는 것뿐 아니라,
처음 신경망을 공부하는 사람도 각 단계의 입력, 출력, loss를 설명할 수 있어야 한다.
```
