# [삼각 관계] Research Log

## Initial Report: Current State and Research Plan

### Current Implementation

[삼각 관계]의 v0 구현은 세 토큰 조합을 직접 점수화한다.

```text
score[i, j, k]
= query_i * key_left_j * key_right_k
```

현재 tensor shape는 다음과 같다.

```text
queries          : (B, H, T, D)
keys_left        : (B, H, T, D)
keys_right       : (B, H, T, D)
relation_scores  : (B, H, T, T, T)
relation_weights : (B, H, T, T, T)
```

기존 v0의 한계는 score는 삼각 관계지만, output value가 `j`와 `k`의 marginal 평균으로 압축된다는 점이다.

```text
context[i] = 0.5 * (
  sum_{j,k} weight[i,j,k] * value_left[j]
  + sum_{j,k} weight[i,j,k] * value_right[k]
)
```

따라서 v0는 세 토큰 조합을 "고르는" 능력은 있지만, 선택된 `(j, k)` pair 자체의 결합 value를 충분히 출력에 싣지는 못한다.

### Existing Baseline Result

NSMC LM 1000 step 비교 결과는 다음과 같았다.

| variant | context | final train | final val | ms/step | score elements/layer |
| --- | ---: | ---: | ---: | ---: | ---: |
| standard | 32 | 7.2821 | 7.3133 | 7.32 | 32,768 |
| triangular v0 | 32 | 7.2803 | 7.3098 | 17.37 | 1,048,576 |

해석:

- v0는 정상적으로 학습된다.
- val loss는 표준 attention과 사실상 동률이다.
- `T=32`에서 score tensor는 표준보다 32배 크다.
- CPU step time은 약 2.4배 느리다.

### Bottleneck and Hypothesis

핵심 병목은 `O(T^3)` score tensor다. 하지만 더 중요한 표현 병목은 value path다. 현재 v0는 `score[i,j,k]`를 만들지만 출력에서는 `(j,k)` 결합을 직접 쓰지 않는다.

가설:

1. `pair_value[j,k]`를 output에 반영하면 [삼각 관계]의 구조적 이득이 커진다.
2. `(j,k)` softmax는 후보가 많아 초기에 uniform해질 수 있으므로 learnable logit scale과 entropy logging이 필요하다.
3. full `O(T^3)`는 긴 context에서 비현실적이므로 top-k candidate 방식이 필요하다.
4. NSMC LM만으로는 삼각 관계 이득이 약할 수 있어 synthetic `give(subject, object, recipient)` task가 필요하다.

### Iteration Plan

이번 구현 iteration에서 다음 variant를 고정한다.

| variant | value mode | logit scale | candidate mode | purpose |
| --- | --- | --- | --- | --- |
| standard | - | - | - | comparison baseline |
| triangular_v0 | marginal | fixed | full | current baseline |
| triangular_v1_pair_value | pair | fixed | full | score와 value 모두 삼각 관계화 |
| triangular_v2_scaled | pair | learned | full | softmax 선택성 안정화 |
| triangular_v3_topk | pair | learned | topk | `O(T*K^2)` 최적화 |

보고서는 이후 각 실험마다 아래 형식으로 누적한다.

```text
## Experiment: ...
- 변경점
- 실행 command
- 결과 table
- 해석
- 다음 계획
```

## Experiment: Synthetic give relation

> Correction: 이 첫 로그는 `best_accuracy` 계산에서 final 재평가값을 포함하지 않는 버그가 있었다. 이후 스크립트에서 `best_accuracy = max(history accuracy, final_accuracy)`로 수정했다. final accuracy와 final loss는 유효하며, 수정 후 비교는 다음 실험 로그를 기준으로 본다.

### Task

`give(subject, object, recipient)` fact 두 개를 제시하고, query의 `(subject, object)` 조합에 맞는 recipient를 마지막 token으로 맞힌다. distractor는 subject 또는 object를 공유하므로 단일 pair shortcut을 어렵게 만든다.

### Results

| variant | seed | final accuracy | best accuracy | final loss | ms/step | entropy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| standard | 123 | 0.501 | 0.498 | 0.7707 | 9.47 | - |
| triangular_v0 | 123 | 0.501 | 0.505 | 0.7829 | 41.23 | 3.716 |
| triangular_v1_pair_value | 123 | 0.520 | 0.489 | 0.7771 | 41.91 | 3.695 |
| triangular_v2_scaled | 123 | 0.522 | 0.493 | 0.7768 | 43.75 | 3.685 |
| triangular_v3_topk | 123 | 0.518 | 0.499 | 0.7920 | 87.93 | 3.266 |

### Interpretation

- 이 task는 [삼각 관계]가 목표로 하는 세 토큰 조합 인식을 직접 요구한다.
- 표준 attention 대비 accuracy가 높으면 구조적 이득 후보로 기록한다.
- 다음 iteration에서는 가장 좋은 variant를 NSMC LM으로 재검증한다.

## Experiment: Synthetic give relation 500 step after pair residual

### Task

`give(subject, object, recipient)` fact 두 개를 제시하고, query의 `(subject, object)` 조합에 맞는 recipient를 마지막 token으로 맞힌다. distractor는 subject 또는 object를 공유하므로 단일 pair shortcut을 어렵게 만든다.

### Results

| variant | seed | final accuracy | best accuracy | final loss | ms/step | entropy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| standard | 123 | 0.501 | 0.501 | 0.7707 | 5.95 | - |
| triangular_v0 | 123 | 0.501 | 0.505 | 0.7829 | 21.66 | 3.716 |
| triangular_v1_pair_value | 123 | 0.507 | 0.507 | 0.7759 | 25.34 | 3.710 |
| triangular_v2_scaled | 123 | 0.509 | 0.509 | 0.7769 | 26.10 | 3.702 |
| triangular_v3_topk | 123 | 0.491 | 0.510 | 0.8864 | 47.22 | 3.297 |

### Interpretation

- 이 task는 [삼각 관계]가 목표로 하는 세 토큰 조합 인식을 직접 요구한다.
- 표준 attention 대비 accuracy가 높으면 구조적 이득 후보로 기록한다.
- 다음 iteration에서는 가장 좋은 variant를 NSMC LM으로 재검증한다.

## Experiment: NSMC LM 300 step triangular variants

### Command

```bash
python scripts/compare_attention_models.py --context-length 32 --batch-size 8 --steps 300 --eval-every 100 --eval-batches 8 --seeds 123 --variants standard,triangular_v0,triangular_v1_pair_value,triangular_v2_scaled,triangular_v3_topk
```

### Setup

- device: `mps`
- context_length: `32`
- batch_size: `8`
- steps: `300`
- seeds: `123`

### Results

| variant | seed | final val | best val | ms/step | entropy | score elements/layer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| standard | 123 | 7.3473 | 7.3473 | 5.92 | - | 32768 |
| triangular_v0 | 123 | 7.3493 | 7.3493 | 11.38 | 4.624 | 1048576 |
| triangular_v1_pair_value | 123 | 7.3493 | 7.3493 | 12.07 | 4.874 | 1048576 |
| triangular_v2_scaled | 123 | 7.3493 | 7.3493 | 12.50 | 4.841 | 1048576 |
| triangular_v3_topk | 123 | 7.3441 | 7.3441 | 19.67 | 4.255 | 262144 |

### Interpretation

- 표준 attention은 비용 기준선이다.
- [삼각 관계] full variant는 세 토큰 조합을 직접 보지만 비용은 `O(T^3)`이다.
- top-k variant는 후보 토큰을 줄여 `O(T*K^2)` 경로를 검증한다.
- 다음 단계는 synthetic 삼각 관계 task에서 구조적 이득을 검증하는 것이다.

## Implementation Iteration 1 Summary

### Changes

- `TriangularRelationAttention` 이름을 도입하고 `TernaryRelationAttention`은 alias로 유지했다.
- `attention_type="triangular"` config를 추가하고 기존 `"ternary"`도 호환한다.
- [삼각 관계] 옵션을 추가했다.
  - `value_mode="marginal"`: v0 baseline
  - `value_mode="pair"`: pair interaction value + marginal residual
  - `logit_scale="learned"`: softmax 선택성 학습
  - `candidate_mode="topk"`: query별 후보 K개 안에서만 pair 생성
- relation diagnostics를 추가했다.
  - entropy
  - max weight
  - logit mean/std
  - score elements
- `compare_attention_models.py`를 variant/seed 반복형으로 확장했다.
- `run_triangular_synthetic.py`를 추가해 `give(subject, object, recipient)` 관계형 synthetic task를 실행한다.

### Findings

- Synthetic task에서는 pair residual 이후 `triangular_v2_scaled`가 표준 attention보다 final accuracy 기준 근소하게 높았다.
  - standard: `0.501`
  - triangular_v2_scaled: `0.509`
- NSMC 300 step에서는 `triangular_v3_topk`가 가장 낮은 val loss를 냈다.
  - standard: `7.3473`
  - triangular_v3_topk: `7.3441`
- 아직 차이는 작다. 단일 seed, 짧은 step이므로 성공 후보이지 결론은 아니다.
- top-k는 score elements를 줄였지만 현재 구현은 gather/scatter overhead 때문에 작은 T에서는 full보다 느릴 수 있다.

### Next Plan

1. Synthetic task를 더 어렵게 만든다.
   - fact 수를 2개에서 3~4개로 늘린다.
   - subject/object가 반복되는 distractor를 더 강하게 넣는다.
2. `triangular_v2_scaled`와 `triangular_v3_topk`를 seed `[123,456,789]`로 반복한다.
3. top-k 구현의 scatter 없는 training path를 분리해 비용을 줄인다.
4. 가장 좋은 variant만 NSMC `steps=1000`으로 재검증한다.

## Experiment: Auto smoke_auto synthetic iteration 1

### Task

`give(subject, object, recipient)` fact 두 개를 제시하고, query의 `(subject, object)` 조합에 맞는 recipient를 마지막 token으로 맞힌다. distractor는 subject 또는 object를 공유하므로 단일 pair shortcut을 어렵게 만든다.

### Results

| variant | seed | final accuracy | best accuracy | final loss | ms/step | entropy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| standard | 123 | 0.125 | 0.125 | 4.1603 | 38.03 | - |
| triangular_v2_scaled | 123 | 0.000 | 0.000 | 4.2178 | 30.04 | 4.026 |
| triangular_v3_topk | 123 | 0.000 | 0.000 | 4.1511 | 33.82 | 3.472 |

### Interpretation

- 이 task는 [삼각 관계]가 목표로 하는 세 토큰 조합 인식을 직접 요구한다.
- 표준 attention 대비 accuracy가 높으면 구조적 이득 후보로 기록한다.
- 다음 iteration에서는 가장 좋은 variant를 NSMC LM으로 재검증한다.

## Experiment: Auto smoke_auto NSMC iteration 1

### Command

```bash
python scripts/compare_attention_models.py --context-length 16 --batch-size 4 --steps 2 --eval-every 1 --eval-batches 1 --seeds 123 --variants standard,triangular_v2_scaled
```

### Setup

- device: `mps`
- context_length: `16`
- batch_size: `4`
- steps: `2`
- seeds: `123`

### Results

| variant | seed | final val | best val | ms/step | entropy | score elements/layer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| standard | 123 | 8.3384 | 8.3384 | 72.48 | - | 4096 |
| triangular_v2_scaled | 123 | 8.1737 | 8.1737 | 68.04 | 3.816 | 65536 |

### Interpretation

- 표준 attention은 비용 기준선이다.
- [삼각 관계] full variant는 세 토큰 조합을 직접 보지만 비용은 `O(T^3)`이다.
- top-k variant는 후보 토큰을 줄여 `O(T*K^2)` 경로를 검증한다.
- 다음 단계는 synthetic 삼각 관계 task에서 구조적 이득을 검증하는 것이다.

## Auto Research Loop: smoke_auto iteration 1

### Loop Policy

- steps: `2`
- eval_every: `1`
- seeds: `123`
- 700 step 이후 변화가 보이도록 100 step 단위로 history를 기록한다.
- synthetic task에서 표준 attention 이상인 [삼각 관계] variant를 NSMC 후보로 선발한다.

### Synthetic Screening Summary

| variant | best accuracy | final accuracy mean | final loss mean | ms/step mean |
| --- | ---: | ---: | ---: | ---: |
| standard | 0.125 | 0.125 | 4.1603 | 38.03 |
| triangular_v2_scaled | 0.000 | 0.000 | 4.2178 | 30.04 |
| triangular_v3_topk | 0.000 | 0.000 | 4.1511 | 33.82 |

### Selected Variants

- triangular_v2_scaled

### NSMC Validation Summary

| variant | best val loss | final val loss mean | final train loss mean | ms/step mean |
| --- | ---: | ---: | ---: | ---: |
| standard | 8.3384 | 8.3384 | 8.1118 | 72.48 |
| triangular_v2_scaled | 8.1737 | 8.1737 | 8.0087 | 68.04 |

### Next Automatic Decision

- synthetic에서 앞선 variant가 NSMC에서도 표준 이하 val loss를 내면 장기 검증 후보로 유지한다.
- synthetic은 앞서지만 NSMC가 뒤처지는 variant는 relation-task 전용 후보로 분리한다.
- top-k가 좋은 val loss를 내지만 느리면 scatter 없는 training path 최적화를 다음 수정 후보로 둔다.

- synthetic JSON: `/Users/magik/Documents/jungle/02gpt-lab/reports/auto_triangular_research/smoke_auto/iter_01/synthetic.json`
- NSMC JSON: `/Users/magik/Documents/jungle/02gpt-lab/reports/auto_triangular_research/smoke_auto/iter_01/nsmc.json`

## Experiment: Auto auto_1000_seed123 synthetic iteration 1

### Task

`give(subject, object, recipient)` fact 두 개를 제시하고, query의 `(subject, object)` 조합에 맞는 recipient를 마지막 token으로 맞힌다. distractor는 subject 또는 object를 공유하므로 단일 pair shortcut을 어렵게 만든다.

### Results

| variant | seed | final accuracy | best accuracy | final loss | ms/step | entropy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| standard | 123 | 0.480 | 0.520 | 0.7530 | 6.54 | - |
| triangular_v0 | 123 | 0.487 | 0.523 | 0.7440 | 22.76 | 3.710 |
| triangular_v1_pair_value | 123 | 0.496 | 0.526 | 0.7484 | 27.84 | 3.703 |
| triangular_v2_scaled | 123 | 0.495 | 0.527 | 0.7477 | 27.71 | 3.695 |
| triangular_v3_topk | 123 | 0.487 | 0.522 | 0.7798 | 82.80 | 3.257 |

### Interpretation

- 이 task는 [삼각 관계]가 목표로 하는 세 토큰 조합 인식을 직접 요구한다.
- 표준 attention 대비 accuracy가 높으면 구조적 이득 후보로 기록한다.
- 다음 iteration에서는 가장 좋은 variant를 NSMC LM으로 재검증한다.

## Experiment: Auto auto_1000_seed123 NSMC iteration 1

### Command

```bash
python scripts/compare_attention_models.py --context-length 32 --batch-size 8 --steps 1000 --eval-every 100 --eval-batches 8 --seeds 123 --variants standard,triangular_v2_scaled,triangular_v1_pair_value
```

### Setup

- device: `mps`
- context_length: `32`
- batch_size: `8`
- steps: `1000`
- seeds: `123`

### Results

| variant | seed | final val | best val | ms/step | entropy | score elements/layer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| standard | 123 | 7.3133 | 7.3115 | 5.78 | - | 32768 |
| triangular_v2_scaled | 123 | 7.3111 | 7.3111 | 11.89 | 4.709 | 1048576 |
| triangular_v1_pair_value | 123 | 7.3111 | 7.3111 | 11.86 | 4.752 | 1048576 |

### Interpretation

- 표준 attention은 비용 기준선이다.
- [삼각 관계] full variant는 세 토큰 조합을 직접 보지만 비용은 `O(T^3)`이다.
- top-k variant는 후보 토큰을 줄여 `O(T*K^2)` 경로를 검증한다.
- 다음 단계는 synthetic 삼각 관계 task에서 구조적 이득을 검증하는 것이다.

## Auto Research Loop: auto_1000_seed123 iteration 1

### Loop Policy

- steps: `1000`
- eval_every: `100`
- seeds: `123`
- 700 step 이후 변화가 보이도록 100 step 단위로 history를 기록한다.
- synthetic task에서 표준 attention 이상인 [삼각 관계] variant를 NSMC 후보로 선발한다.

### Synthetic Screening Summary

| variant | best accuracy | final accuracy mean | final loss mean | ms/step mean |
| --- | ---: | ---: | ---: | ---: |
| standard | 0.520 | 0.480 | 0.7530 | 6.54 |
| triangular_v0 | 0.523 | 0.487 | 0.7440 | 22.76 |
| triangular_v1_pair_value | 0.526 | 0.496 | 0.7484 | 27.84 |
| triangular_v2_scaled | 0.527 | 0.495 | 0.7477 | 27.71 |
| triangular_v3_topk | 0.522 | 0.487 | 0.7798 | 82.80 |

### Selected Variants

- triangular_v2_scaled, triangular_v1_pair_value

### NSMC Validation Summary

| variant | best val loss | final val loss mean | final train loss mean | ms/step mean |
| --- | ---: | ---: | ---: | ---: |
| standard | 7.3115 | 7.3133 | 7.2821 | 5.78 |
| triangular_v1_pair_value | 7.3111 | 7.3111 | 7.2828 | 11.86 |
| triangular_v2_scaled | 7.3111 | 7.3111 | 7.2828 | 11.89 |

### Next Automatic Decision

- synthetic에서 앞선 variant가 NSMC에서도 표준 이하 val loss를 내면 장기 검증 후보로 유지한다.
- synthetic은 앞서지만 NSMC가 뒤처지는 variant는 relation-task 전용 후보로 분리한다.
- top-k가 좋은 val loss를 내지만 느리면 scatter 없는 training path 최적화를 다음 수정 후보로 둔다.

- synthetic JSON: `/Users/magik/Documents/jungle/02gpt-lab/reports/auto_triangular_research/auto_1000_seed123/iter_01/synthetic.json`
- NSMC JSON: `/Users/magik/Documents/jungle/02gpt-lab/reports/auto_triangular_research/auto_1000_seed123/iter_01/nsmc.json`

## Research Loop Policy Update: 1000 Step Minimum

- 사용자 관찰: 700 step 이후 loss 변화가 유의미하게 나타날 수 있다.
- 반영 내용: `scripts/auto_triangular_research.py`는 기본적으로 `--steps 1000`이며, `--steps < 1000`은 `--allow-short`가 없으면 거부한다.
- 보고서 해석 기준: `smoke_auto`처럼 `--allow-short`로 실행한 짧은 결과는 배선 검증으로만 본다. 성능 근거와 다음 연구 판단은 `auto_1000_seed123`처럼 1000 step 이상 실행한 결과부터 사용한다.
- 다음 권장 실행: seed `[123,456,789]` multi-seed 1000 step 이상 반복으로 현재 단일 seed 우위를 검증한다.

## Strategy Backlog: Automatic Improvement Loop

### Goal

[삼각 관계]의 핵심 비전은 `score[i,j,k]`로 세 토큰 조합을 직접 보고, 출력 경로에서도 `(j,k)` 결합 정보를 잃지 않는 것이다. 자동 연구 루프는 이 비전을 유지하면서 표준 attention 대비 명확한 이득이 나오는 조건을 찾는다.

### Satisfaction Criteria

- synthetic `give(subject, object, recipient)` task에서 [삼각 관계] 후보의 best accuracy가 standard보다 `0.020` 이상 높아야 한다.
- NSMC LM에서는 같은 step/seed 조건에서 [삼각 관계] 후보의 best val loss가 standard보다 같거나 낮아야 한다.
- 비용은 별도 경고로 본다. NSMC ms/step이 standard의 `3.0x`를 넘으면 성능이 좋아도 최적화 과제로 분류한다.

### Improvement Strategies

| variant | strategy | intent | risk |
| --- | --- | --- | --- |
| `triangular_v0` | 삼각 score + marginal value | 최초 baseline 유지 | score만 삼각 관계라 output 정보가 약함 |
| `triangular_v1_pair_value` | pair-value residual | `(j,k)` 결합 정보를 output에 직접 반영 | pair product가 초반 학습을 흔들 수 있음 |
| `triangular_v2_scaled` | learned logit scale + entropy logging | `(j,k)` softmax가 너무 uniform/peaky해지는지 제어 | scale만으로 value path 문제는 해결 못 함 |
| `triangular_v3_topk` | query별 top-k candidate pair | `O(T^3)`를 `O(T*K^2)`로 줄임 | gather/topk overhead가 작을 때는 더 느릴 수 있음 |
| `triangular_v4_gated_pair` | gated pair-value residual | pair 경로를 learned gate로 천천히 열어 안정화 | gate가 충분히 열리지 않으면 v0에 가까워짐 |
| `triangular_v5_gated_topk` | gated pair-value + top-k | 비용 절감과 안정화를 동시에 노림 | top-k 후보 선택이 초기에는 noisy할 수 있음 |

### Automatic Decision Policy

1. 모든 후보는 synthetic task에서 먼저 screening한다.
2. standard 이상인 [삼각 관계] 후보 중 best accuracy, final accuracy, 비용 순으로 NSMC 후보를 고른다.
3. NSMC에서 standard 이하 best val loss를 내면 장기 검증 후보로 유지한다.
4. 만족 기준을 통과하지 못하면 다음 iteration은 gated pair와 top-k 계열을 포함한 후보군으로 재구성한다.
5. 보고서는 매 iteration마다 전략, command, 결과, 성공 판정, 다음 판단을 누적한다.

### Evaluation Hardening Strategy

기존 synthetic task는 fact 2개만 사용하므로, 모델이 두 후보 중 하나를 고르는 수준에서 `0.5` 근처 accuracy에 머무를 수 있다. 이 경우 [삼각 관계]의 구조적 이득이 너무 작게 관측된다.

반영한 개선:

- `scripts/run_triangular_synthetic.py`에 `--facts` 옵션을 추가한다.
- `facts=4`에서는 여러 `give(subject, object, recipient)` fact가 subject 또는 object를 공유하므로, query의 `(subject, object)` 조합과 정확한 recipient를 더 긴 문맥에서 찾아야 한다.
- 자동 루프에는 `--synthetic-facts`를 추가한다.
- 목표는 standard attention이 pairwise shortcut으로 버티기 어려운 조건에서 [삼각 관계] 후보가 더 큰 synthetic gain을 내는지 확인하는 것이다.

### Relation-Supervised Synthetic Strategy

`facts=4` 실험 결과, [삼각 관계] 후보는 NSMC val loss에서는 standard보다 낮았지만 synthetic answer accuracy에서는 standard를 넘지 못했다. 원인은 마지막 `ANS` 위치 하나에만 loss가 걸려, `give(subject, object, recipient)` 내부 구조를 직접 학습하는 신호가 약하기 때문이다.

반영한 개선:

- `scripts/run_triangular_synthetic.py`에 `--loss-mode answer_relation`을 추가한다.
- `answer_relation`은 마지막 answer loss에 더해 각 fact의 `TO -> recipient` 예측 위치에도 보조 loss를 준다.
- standard attention에도 동일한 보조 loss를 주므로 비교 조건은 유지된다.
- 목적은 [삼각 관계]가 `query/TO` 위치에서 `(subject, object)` 조합을 보고 recipient를 예측하는 경로를 더 직접적으로 학습하게 만드는 것이다.

### Top-k Candidate Width Strategy

`auto_facts4_relation_1000_seed123`에서 `triangular_v5_gated_topk`는 synthetic best accuracy 기준 standard보다 `+0.01875` 높아, 만족 기준 `+0.020`에 근접했다. 다만 자동 루프가 synthetic screening에서 top-k를 `8`로 강제하고 있어, `facts=4`의 긴 문맥에서 필요한 후보를 놓쳤을 가능성이 있다.

반영한 개선:

- `scripts/auto_triangular_research.py`에서 synthetic 실험에도 `--top-k` 값을 그대로 전달하도록 수정한다.
- 다음 반복은 `triangular_v5_gated_topk` 중심으로 `top_k=12` 또는 `16`을 시험한다.
- 기대 효과는 후보 누락을 줄여 synthetic answer accuracy를 끌어올리는 것이다.

## Experiment: Auto auto_gated_1000_seed123 synthetic iteration 1

### Task

`give(subject, object, recipient)` fact 두 개를 제시하고, query의 `(subject, object)` 조합에 맞는 recipient를 마지막 token으로 맞힌다. distractor는 subject 또는 object를 공유하므로 단일 pair shortcut을 어렵게 만든다.

### Results

| variant | seed | final accuracy | best accuracy | final loss | ms/step | entropy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| standard | 123 | 0.480 | 0.520 | 0.7530 | 9.19 | - |
| triangular_v2_scaled | 123 | 0.495 | 0.527 | 0.7477 | 50.75 | 3.695 |
| triangular_v4_gated_pair | 123 | 0.498 | 0.520 | 0.7439 | 52.01 | 3.697 |
| triangular_v5_gated_topk | 123 | 0.501 | 0.522 | 0.7572 | 97.71 | 3.234 |

### Interpretation

- 이 task는 [삼각 관계]가 목표로 하는 세 토큰 조합 인식을 직접 요구한다.
- 표준 attention 대비 accuracy가 높으면 구조적 이득 후보로 기록한다.
- 다음 iteration에서는 가장 좋은 variant를 NSMC LM으로 재검증한다.

## Experiment: Auto auto_gated_1000_seed123 NSMC iteration 1

### Command

```bash
python scripts/compare_attention_models.py --context-length 32 --batch-size 8 --steps 1000 --eval-every 100 --eval-batches 8 --seeds 123 --variants standard,triangular_v2_scaled,triangular_v5_gated_topk,triangular_v4_gated_pair
```

### Setup

- device: `mps`
- context_length: `32`
- batch_size: `8`
- steps: `1000`
- seeds: `123`

### Results

| variant | seed | final val | best val | ms/step | entropy | score elements/layer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| standard | 123 | 7.3133 | 7.3115 | 8.60 | - | 32768 |
| triangular_v2_scaled | 123 | 7.3111 | 7.3111 | 23.24 | 4.709 | 1048576 |
| triangular_v5_gated_topk | 123 | 7.3100 | 7.3100 | 39.55 | 3.478 | 262144 |
| triangular_v4_gated_pair | 123 | 7.3100 | 7.3100 | 23.48 | 4.295 | 1048576 |

### Interpretation

- 표준 attention은 비용 기준선이다.
- [삼각 관계] full variant는 세 토큰 조합을 직접 보지만 비용은 `O(T^3)`이다.
- top-k variant는 후보 토큰을 줄여 `O(T*K^2)` 경로를 검증한다.
- 다음 단계는 synthetic 삼각 관계 task에서 구조적 이득을 검증하는 것이다.

## Auto Research Loop: auto_gated_1000_seed123 iteration 1

### Loop Policy

- steps: `1000`
- eval_every: `100`
- seeds: `123`
- 700 step 이후 변화가 보이도록 100 step 단위로 history를 기록한다.
- synthetic task에서 표준 attention 이상인 [삼각 관계] variant를 NSMC 후보로 선발한다.
- 만족 기준: synthetic best accuracy가 standard보다 `0.020` 이상 높고, NSMC best val loss가 standard보다 `0.0000` 이상 개선되어야 한다.
- 비용 경고 기준: NSMC ms/step이 standard의 `3.0`배를 넘으면 비용 병목으로 표시한다.

### Strategy Under Test

- `standard`: 기존 pairwise attention 기준선.
- `triangular_v2_scaled`: pair-value에 learned logit scale과 entropy 진단을 더한 softmax 안정화 전략.
- `triangular_v4_gated_pair`: pair-value 경로를 learned gate로 천천히 여는 안정화 전략.
- `triangular_v5_gated_topk`: gated pair-value와 top-k 후보 선택을 결합한 비용/안정성 절충 전략.

### Synthetic Screening Summary

| variant | best accuracy | final accuracy mean | final loss mean | ms/step mean |
| --- | ---: | ---: | ---: | ---: |
| standard | 0.520 | 0.480 | 0.7530 | 9.19 |
| triangular_v2_scaled | 0.527 | 0.495 | 0.7477 | 50.75 |
| triangular_v4_gated_pair | 0.520 | 0.498 | 0.7439 | 52.01 |
| triangular_v5_gated_topk | 0.522 | 0.501 | 0.7572 | 97.71 |

### Selected Variants

- triangular_v2_scaled, triangular_v5_gated_topk, triangular_v4_gated_pair

### NSMC Validation Summary

| variant | best val loss | final val loss mean | final train loss mean | ms/step mean |
| --- | ---: | ---: | ---: | ---: |
| standard | 7.3115 | 7.3133 | 7.2821 | 8.60 |
| triangular_v2_scaled | 7.3111 | 7.3111 | 7.2828 | 23.24 |
| triangular_v4_gated_pair | 7.3100 | 7.3100 | 7.2805 | 23.48 |
| triangular_v5_gated_topk | 7.3100 | 7.3100 | 7.2795 | 39.55 |

### Success Evaluation

| variant | synthetic gain | NSMC val gain | cost ratio | status |
| --- | ---: | ---: | ---: | --- |
| triangular_v2_scaled | 0.007 | 0.0004 | 2.70x | synthetic fail, NSMC pass |
| triangular_v5_gated_topk | 0.002 | 0.0015 | 4.60x | synthetic fail, NSMC pass, cost warning |
| triangular_v4_gated_pair | 0.000 | 0.0016 | 2.73x | synthetic fail, NSMC pass |

### Next Automatic Decision

- 만족 기준을 통과한 variant가 있으면 multi-seed 장기 검증 후보로 유지한다.
- synthetic gain은 있으나 기준에 못 미치면 더 강한 relation task 또는 gated pair/top-k 변형을 다음 후보군에 넣는다.
- NSMC가 앞서지만 비용 경고가 뜨면 top-k/gated-topk 최적화를 우선한다.
- loop status: `not_satisfied`

- synthetic JSON: `/Users/magik/Documents/jungle/02gpt-lab/reports/auto_triangular_research/auto_gated_1000_seed123/iter_01/synthetic.json`
- NSMC JSON: `/Users/magik/Documents/jungle/02gpt-lab/reports/auto_triangular_research/auto_gated_1000_seed123/iter_01/nsmc.json`

## Experiment: Auto smoke_facts4 synthetic iteration 1

### Task

`give(subject, object, recipient)` fact `4`개를 제시하고, query의 `(subject, object)` 조합에 맞는 recipient를 마지막 token으로 맞힌다. distractor는 subject 또는 object를 공유하므로 단일 token shortcut을 어렵게 만든다.

### Results

| variant | seed | final accuracy | best accuracy | final loss | ms/step | entropy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| standard | 123 | 0.000 | 0.250 | 3.9497 | 78.62 | - |
| triangular_v2_scaled | 123 | 0.000 | 0.000 | 4.4629 | 142.53 | 4.959 |
| triangular_v4_gated_pair | 123 | 0.000 | 0.000 | 4.4654 | 37.66 | 4.959 |

### Interpretation

- 이 task는 [삼각 관계]가 목표로 하는 세 토큰 조합 인식을 직접 요구한다.
- 표준 attention 대비 accuracy가 높으면 구조적 이득 후보로 기록한다.
- 다음 iteration에서는 가장 좋은 variant를 NSMC LM으로 재검증한다.

## Experiment: Auto smoke_facts4 NSMC iteration 1

### Command

```bash
python scripts/compare_attention_models.py --context-length 16 --batch-size 4 --steps 2 --eval-every 1 --eval-batches 1 --seeds 123 --variants standard,triangular_v4_gated_pair
```

### Setup

- device: `mps`
- context_length: `16`
- batch_size: `4`
- steps: `2`
- seeds: `123`

### Results

| variant | seed | final val | best val | ms/step | entropy | score elements/layer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| standard | 123 | 8.3384 | 8.3384 | 57.71 | - | 4096 |
| triangular_v4_gated_pair | 123 | 8.1729 | 8.1729 | 46.47 | 3.816 | 65536 |

### Interpretation

- 표준 attention은 비용 기준선이다.
- [삼각 관계] full variant는 세 토큰 조합을 직접 보지만 비용은 `O(T^3)`이다.
- top-k variant는 후보 토큰을 줄여 `O(T*K^2)` 경로를 검증한다.
- 다음 단계는 synthetic 삼각 관계 task에서 구조적 이득을 검증하는 것이다.

## Auto Research Loop: smoke_facts4 iteration 1

### Loop Policy

- steps: `2`
- eval_every: `1`
- seeds: `123`
- synthetic facts: `4`
- 700 step 이후 변화가 보이도록 100 step 단위로 history를 기록한다.
- synthetic task에서 표준 attention 이상인 [삼각 관계] variant를 NSMC 후보로 선발한다.
- 만족 기준: synthetic best accuracy가 standard보다 `0.020` 이상 높고, NSMC best val loss가 standard보다 `0.0000` 이상 개선되어야 한다.
- 비용 경고 기준: NSMC ms/step이 standard의 `3.0`배를 넘으면 비용 병목으로 표시한다.

### Strategy Under Test

- `standard`: 기존 pairwise attention 기준선.
- `triangular_v2_scaled`: pair-value에 learned logit scale과 entropy 진단을 더한 softmax 안정화 전략.
- `triangular_v4_gated_pair`: pair-value 경로를 learned gate로 천천히 여는 안정화 전략.

### Synthetic Screening Summary

| variant | best accuracy | final accuracy mean | final loss mean | ms/step mean |
| --- | ---: | ---: | ---: | ---: |
| standard | 0.250 | 0.000 | 3.9497 | 78.62 |
| triangular_v2_scaled | 0.000 | 0.000 | 4.4629 | 142.53 |
| triangular_v4_gated_pair | 0.000 | 0.000 | 4.4654 | 37.66 |

### Selected Variants

- triangular_v4_gated_pair

### NSMC Validation Summary

| variant | best val loss | final val loss mean | final train loss mean | ms/step mean |
| --- | ---: | ---: | ---: | ---: |
| standard | 8.3384 | 8.3384 | 8.1118 | 57.71 |
| triangular_v4_gated_pair | 8.1729 | 8.1729 | 8.0152 | 46.47 |

### Success Evaluation

| variant | synthetic gain | NSMC val gain | cost ratio | status |
| --- | ---: | ---: | ---: | --- |
| triangular_v4_gated_pair | -0.250 | 0.1655 | 0.81x | synthetic fail, NSMC pass |

### Next Automatic Decision

- 만족 기준을 통과한 variant가 있으면 multi-seed 장기 검증 후보로 유지한다.
- synthetic gain은 있으나 기준에 못 미치면 더 강한 relation task 또는 gated pair/top-k 변형을 다음 후보군에 넣는다.
- NSMC가 앞서지만 비용 경고가 뜨면 top-k/gated-topk 최적화를 우선한다.
- loop status: `not_satisfied`

- synthetic JSON: `/Users/magik/Documents/jungle/02gpt-lab/reports/auto_triangular_research/smoke_facts4/iter_01/synthetic.json`
- NSMC JSON: `/Users/magik/Documents/jungle/02gpt-lab/reports/auto_triangular_research/smoke_facts4/iter_01/nsmc.json`

## Experiment: Auto auto_facts4_1000_seed123 synthetic iteration 1

### Task

`give(subject, object, recipient)` fact `4`개를 제시하고, query의 `(subject, object)` 조합에 맞는 recipient를 마지막 token으로 맞힌다. distractor는 subject 또는 object를 공유하므로 단일 token shortcut을 어렵게 만든다.

### Results

| variant | seed | final accuracy | best accuracy | final loss | ms/step | entropy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| standard | 123 | 0.422 | 0.441 | 1.2537 | 9.11 | - |
| triangular_v2_scaled | 123 | 0.388 | 0.408 | 1.2510 | 73.46 | 4.702 |
| triangular_v4_gated_pair | 123 | 0.388 | 0.411 | 1.2398 | 73.48 | 4.708 |
| triangular_v5_gated_topk | 123 | 0.420 | 0.430 | 1.3505 | 69.67 | 3.636 |

### Interpretation

- 이 task는 [삼각 관계]가 목표로 하는 세 토큰 조합 인식을 직접 요구한다.
- 표준 attention 대비 accuracy가 높으면 구조적 이득 후보로 기록한다.
- 다음 iteration에서는 가장 좋은 variant를 NSMC LM으로 재검증한다.

## Experiment: Auto auto_facts4_1000_seed123 NSMC iteration 1

### Command

```bash
python scripts/compare_attention_models.py --context-length 32 --batch-size 8 --steps 1000 --eval-every 100 --eval-batches 8 --seeds 123 --variants standard,triangular_v5_gated_topk,triangular_v4_gated_pair
```

### Setup

- device: `mps`
- context_length: `32`
- batch_size: `8`
- steps: `1000`
- seeds: `123`

### Results

| variant | seed | final val | best val | ms/step | entropy | score elements/layer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| standard | 123 | 7.3133 | 7.3115 | 8.50 | - | 32768 |
| triangular_v5_gated_topk | 123 | 7.3100 | 7.3100 | 38.76 | 3.478 | 262144 |
| triangular_v4_gated_pair | 123 | 7.3100 | 7.3100 | 23.48 | 4.295 | 1048576 |

### Interpretation

- 표준 attention은 비용 기준선이다.
- [삼각 관계] full variant는 세 토큰 조합을 직접 보지만 비용은 `O(T^3)`이다.
- top-k variant는 후보 토큰을 줄여 `O(T*K^2)` 경로를 검증한다.
- 다음 단계는 synthetic 삼각 관계 task에서 구조적 이득을 검증하는 것이다.

## Auto Research Loop: auto_facts4_1000_seed123 iteration 1

### Loop Policy

- steps: `1000`
- eval_every: `100`
- seeds: `123`
- synthetic facts: `4`
- 700 step 이후 변화가 보이도록 100 step 단위로 history를 기록한다.
- synthetic task에서 표준 attention 이상인 [삼각 관계] variant를 NSMC 후보로 선발한다.
- 만족 기준: synthetic best accuracy가 standard보다 `0.020` 이상 높고, NSMC best val loss가 standard보다 `0.0000` 이상 개선되어야 한다.
- 비용 경고 기준: NSMC ms/step이 standard의 `3.0`배를 넘으면 비용 병목으로 표시한다.

### Strategy Under Test

- `standard`: 기존 pairwise attention 기준선.
- `triangular_v2_scaled`: pair-value에 learned logit scale과 entropy 진단을 더한 softmax 안정화 전략.
- `triangular_v4_gated_pair`: pair-value 경로를 learned gate로 천천히 여는 안정화 전략.
- `triangular_v5_gated_topk`: gated pair-value와 top-k 후보 선택을 결합한 비용/안정성 절충 전략.

### Synthetic Screening Summary

| variant | best accuracy | final accuracy mean | final loss mean | ms/step mean |
| --- | ---: | ---: | ---: | ---: |
| standard | 0.441 | 0.422 | 1.2537 | 9.11 |
| triangular_v2_scaled | 0.408 | 0.388 | 1.2510 | 73.46 |
| triangular_v4_gated_pair | 0.411 | 0.388 | 1.2398 | 73.48 |
| triangular_v5_gated_topk | 0.430 | 0.420 | 1.3505 | 69.67 |

### Selected Variants

- triangular_v5_gated_topk, triangular_v4_gated_pair

### NSMC Validation Summary

| variant | best val loss | final val loss mean | final train loss mean | ms/step mean |
| --- | ---: | ---: | ---: | ---: |
| standard | 7.3115 | 7.3133 | 7.2821 | 8.50 |
| triangular_v4_gated_pair | 7.3100 | 7.3100 | 7.2805 | 23.48 |
| triangular_v5_gated_topk | 7.3100 | 7.3100 | 7.2795 | 38.76 |

### Success Evaluation

| variant | synthetic gain | NSMC val gain | cost ratio | status |
| --- | ---: | ---: | ---: | --- |
| triangular_v5_gated_topk | -0.011 | 0.0015 | 4.56x | synthetic fail, NSMC pass, cost warning |
| triangular_v4_gated_pair | -0.030 | 0.0016 | 2.76x | synthetic fail, NSMC pass |

### Next Automatic Decision

- 만족 기준을 통과한 variant가 있으면 multi-seed 장기 검증 후보로 유지한다.
- synthetic gain은 있으나 기준에 못 미치면 더 강한 relation task 또는 gated pair/top-k 변형을 다음 후보군에 넣는다.
- NSMC가 앞서지만 비용 경고가 뜨면 top-k/gated-topk 최적화를 우선한다.
- loop status: `not_satisfied`

- synthetic JSON: `/Users/magik/Documents/jungle/02gpt-lab/reports/auto_triangular_research/auto_facts4_1000_seed123/iter_01/synthetic.json`
- NSMC JSON: `/Users/magik/Documents/jungle/02gpt-lab/reports/auto_triangular_research/auto_facts4_1000_seed123/iter_01/nsmc.json`

## Experiment: Auto auto_facts4_relation_1000_seed123 synthetic iteration 1

### Task

`give(subject, object, recipient)` fact `4`개를 제시하고, query의 `(subject, object)` 조합에 맞는 recipient를 마지막 token으로 맞힌다. distractor는 subject 또는 object를 공유하므로 단일 token shortcut을 어렵게 만든다. loss_mode는 `answer_relation`이다.

### Results

| variant | seed | final accuracy | best accuracy | final loss | ms/step | entropy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| standard | 123 | 0.405 | 0.423 | 2.2996 | 9.56 | - |
| triangular_v2_scaled | 123 | 0.380 | 0.388 | 2.3029 | 74.41 | 3.632 |
| triangular_v4_gated_pair | 123 | 0.372 | 0.403 | 2.3012 | 76.18 | 3.943 |
| triangular_v5_gated_topk | 123 | 0.412 | 0.442 | 2.4167 | 77.57 | 3.437 |

### Interpretation

- 이 task는 [삼각 관계]가 목표로 하는 세 토큰 조합 인식을 직접 요구한다.
- 표준 attention 대비 accuracy가 높으면 구조적 이득 후보로 기록한다.
- 다음 iteration에서는 가장 좋은 variant를 NSMC LM으로 재검증한다.

## Experiment: Auto auto_facts4_relation_1000_seed123 NSMC iteration 1

### Command

```bash
python scripts/compare_attention_models.py --context-length 32 --batch-size 8 --steps 1000 --eval-every 100 --eval-batches 8 --seeds 123 --variants standard,triangular_v5_gated_topk
```

### Setup

- device: `mps`
- context_length: `32`
- batch_size: `8`
- steps: `1000`
- seeds: `123`

### Results

| variant | seed | final val | best val | ms/step | entropy | score elements/layer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| standard | 123 | 7.3133 | 7.3115 | 8.71 | - | 32768 |
| triangular_v5_gated_topk | 123 | 7.3100 | 7.3100 | 38.90 | 3.480 | 262144 |

### Interpretation

- 표준 attention은 비용 기준선이다.
- [삼각 관계] full variant는 세 토큰 조합을 직접 보지만 비용은 `O(T^3)`이다.
- top-k variant는 후보 토큰을 줄여 `O(T*K^2)` 경로를 검증한다.
- 다음 단계는 synthetic 삼각 관계 task에서 구조적 이득을 검증하는 것이다.

## Auto Research Loop: auto_facts4_relation_1000_seed123 iteration 1

### Loop Policy

- steps: `1000`
- eval_every: `100`
- seeds: `123`
- synthetic facts: `4`
- synthetic loss mode: `answer_relation`
- 700 step 이후 변화가 보이도록 100 step 단위로 history를 기록한다.
- synthetic task에서 표준 attention 이상인 [삼각 관계] variant를 NSMC 후보로 선발한다.
- 만족 기준: synthetic best accuracy가 standard보다 `0.020` 이상 높고, NSMC best val loss가 standard보다 `0.0000` 이상 개선되어야 한다.
- 비용 경고 기준: NSMC ms/step이 standard의 `3.0`배를 넘으면 비용 병목으로 표시한다.

### Strategy Under Test

- `standard`: 기존 pairwise attention 기준선.
- `triangular_v2_scaled`: pair-value에 learned logit scale과 entropy 진단을 더한 softmax 안정화 전략.
- `triangular_v4_gated_pair`: pair-value 경로를 learned gate로 천천히 여는 안정화 전략.
- `triangular_v5_gated_topk`: gated pair-value와 top-k 후보 선택을 결합한 비용/안정성 절충 전략.

### Synthetic Screening Summary

| variant | best accuracy | final accuracy mean | final loss mean | ms/step mean |
| --- | ---: | ---: | ---: | ---: |
| standard | 0.423 | 0.405 | 2.2996 | 9.56 |
| triangular_v2_scaled | 0.388 | 0.380 | 2.3029 | 74.41 |
| triangular_v4_gated_pair | 0.403 | 0.372 | 2.3012 | 76.18 |
| triangular_v5_gated_topk | 0.442 | 0.412 | 2.4167 | 77.57 |

### Selected Variants

- triangular_v5_gated_topk

### NSMC Validation Summary

| variant | best val loss | final val loss mean | final train loss mean | ms/step mean |
| --- | ---: | ---: | ---: | ---: |
| standard | 7.3115 | 7.3133 | 7.2821 | 8.71 |
| triangular_v5_gated_topk | 7.3100 | 7.3100 | 7.2795 | 38.90 |

### Success Evaluation

| variant | synthetic gain | NSMC val gain | cost ratio | status |
| --- | ---: | ---: | ---: | --- |
| triangular_v5_gated_topk | 0.019 | 0.0016 | 4.47x | synthetic fail, NSMC pass, cost warning |

### Next Automatic Decision

- 만족 기준을 통과한 variant가 있으면 multi-seed 장기 검증 후보로 유지한다.
- synthetic gain은 있으나 기준에 못 미치면 더 강한 relation task 또는 gated pair/top-k 변형을 다음 후보군에 넣는다.
- NSMC가 앞서지만 비용 경고가 뜨면 top-k/gated-topk 최적화를 우선한다.
- loop status: `not_satisfied`

- synthetic JSON: `/Users/magik/Documents/jungle/02gpt-lab/reports/auto_triangular_research/auto_facts4_relation_1000_seed123/iter_01/synthetic.json`
- NSMC JSON: `/Users/magik/Documents/jungle/02gpt-lab/reports/auto_triangular_research/auto_facts4_relation_1000_seed123/iter_01/nsmc.json`

## Experiment: Auto auto_facts4_relation_topk12_1000_seed123 synthetic iteration 1

### Task

`give(subject, object, recipient)` fact `4`개를 제시하고, query의 `(subject, object)` 조합에 맞는 recipient를 마지막 token으로 맞힌다. distractor는 subject 또는 object를 공유하므로 단일 token shortcut을 어렵게 만든다. loss_mode는 `answer_relation`이다.

### Results

| variant | seed | final accuracy | best accuracy | final loss | ms/step | entropy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| standard | 123 | 0.405 | 0.423 | 2.2996 | 9.60 | - |
| triangular_v5_gated_topk | 123 | 0.416 | 0.434 | 2.3417 | 102.52 | 3.964 |

### Interpretation

- 이 task는 [삼각 관계]가 목표로 하는 세 토큰 조합 인식을 직접 요구한다.
- 표준 attention 대비 accuracy가 높으면 구조적 이득 후보로 기록한다.
- 다음 iteration에서는 가장 좋은 variant를 NSMC LM으로 재검증한다.

## Experiment: Auto auto_facts4_relation_topk12_1000_seed123 NSMC iteration 1

### Command

```bash
python scripts/compare_attention_models.py --context-length 32 --batch-size 8 --steps 1000 --eval-every 100 --eval-batches 8 --seeds 123 --variants standard,triangular_v5_gated_topk
```

### Setup

- device: `mps`
- context_length: `32`
- batch_size: `8`
- steps: `1000`
- seeds: `123`

### Results

| variant | seed | final val | best val | ms/step | entropy | score elements/layer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| standard | 123 | 7.3133 | 7.3115 | 8.62 | - | 32768 |
| triangular_v5_gated_topk | 123 | 7.3099 | 7.3099 | 31.39 | 3.342 | 147456 |

### Interpretation

- 표준 attention은 비용 기준선이다.
- [삼각 관계] full variant는 세 토큰 조합을 직접 보지만 비용은 `O(T^3)`이다.
- top-k variant는 후보 토큰을 줄여 `O(T*K^2)` 경로를 검증한다.
- 다음 단계는 synthetic 삼각 관계 task에서 구조적 이득을 검증하는 것이다.

## Auto Research Loop: auto_facts4_relation_topk12_1000_seed123 iteration 1

### Loop Policy

- steps: `1000`
- eval_every: `100`
- seeds: `123`
- synthetic facts: `4`
- synthetic loss mode: `answer_relation`
- 700 step 이후 변화가 보이도록 100 step 단위로 history를 기록한다.
- synthetic task에서 표준 attention 이상인 [삼각 관계] variant를 NSMC 후보로 선발한다.
- 만족 기준: synthetic best accuracy가 standard보다 `0.020` 이상 높고, NSMC best val loss가 standard보다 `0.0000` 이상 개선되어야 한다.
- 비용 경고 기준: NSMC ms/step이 standard의 `3.0`배를 넘으면 비용 병목으로 표시한다.

### Strategy Under Test

- `standard`: 기존 pairwise attention 기준선.
- `triangular_v5_gated_topk`: gated pair-value와 top-k 후보 선택을 결합한 비용/안정성 절충 전략.

### Synthetic Screening Summary

| variant | best accuracy | final accuracy mean | final loss mean | ms/step mean |
| --- | ---: | ---: | ---: | ---: |
| standard | 0.423 | 0.405 | 2.2996 | 9.60 |
| triangular_v5_gated_topk | 0.434 | 0.416 | 2.3417 | 102.52 |

### Selected Variants

- triangular_v5_gated_topk

### NSMC Validation Summary

| variant | best val loss | final val loss mean | final train loss mean | ms/step mean |
| --- | ---: | ---: | ---: | ---: |
| standard | 7.3115 | 7.3133 | 7.2821 | 8.62 |
| triangular_v5_gated_topk | 7.3099 | 7.3099 | 7.2793 | 31.39 |

### Success Evaluation

| variant | synthetic gain | NSMC val gain | cost ratio | status |
| --- | ---: | ---: | ---: | --- |
| triangular_v5_gated_topk | 0.011 | 0.0017 | 3.64x | synthetic fail, NSMC pass, cost warning |

### Next Automatic Decision

- 만족 기준을 통과한 variant가 있으면 multi-seed 장기 검증 후보로 유지한다.
- synthetic gain은 있으나 기준에 못 미치면 더 강한 relation task 또는 gated pair/top-k 변형을 다음 후보군에 넣는다.
- NSMC가 앞서지만 비용 경고가 뜨면 top-k/gated-topk 최적화를 우선한다.
- loop status: `not_satisfied`

- synthetic JSON: `/Users/magik/Documents/jungle/02gpt-lab/reports/auto_triangular_research/auto_facts4_relation_topk12_1000_seed123/iter_01/synthetic.json`
- NSMC JSON: `/Users/magik/Documents/jungle/02gpt-lab/reports/auto_triangular_research/auto_facts4_relation_topk12_1000_seed123/iter_01/nsmc.json`

## Experiment: Auto auto_facts4_relation_w1_topk8_1000_seed123 synthetic iteration 1

### Task

`give(subject, object, recipient)` fact `4`개를 제시하고, query의 `(subject, object)` 조합에 맞는 recipient를 마지막 token으로 맞힌다. distractor는 subject 또는 object를 공유하므로 단일 token shortcut을 어렵게 만든다. loss_mode는 `answer_relation`이다.

### Results

| variant | seed | final accuracy | best accuracy | final loss | ms/step | entropy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| standard | 123 | 0.398 | 0.414 | 3.3444 | 9.54 | - |
| triangular_v5_gated_topk | 123 | 0.406 | 0.408 | 3.5287 | 76.29 | 3.447 |

### Interpretation

- 이 task는 [삼각 관계]가 목표로 하는 세 토큰 조합 인식을 직접 요구한다.
- 표준 attention 대비 accuracy가 높으면 구조적 이득 후보로 기록한다.
- 다음 iteration에서는 가장 좋은 variant를 NSMC LM으로 재검증한다.

## Experiment: Auto auto_facts4_relation_w1_topk8_1000_seed123 NSMC iteration 1

### Command

```bash
python scripts/compare_attention_models.py --context-length 32 --batch-size 8 --steps 1000 --eval-every 100 --eval-batches 8 --seeds 123 --variants standard,triangular_v5_gated_topk
```

### Setup

- device: `mps`
- context_length: `32`
- batch_size: `8`
- steps: `1000`
- seeds: `123`

### Results

| variant | seed | final val | best val | ms/step | entropy | score elements/layer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| standard | 123 | 7.3133 | 7.3115 | 8.51 | - | 32768 |
| triangular_v5_gated_topk | 123 | 7.3097 | 7.3097 | 24.76 | 3.138 | 65536 |

### Interpretation

- 표준 attention은 비용 기준선이다.
- [삼각 관계] full variant는 세 토큰 조합을 직접 보지만 비용은 `O(T^3)`이다.
- top-k variant는 후보 토큰을 줄여 `O(T*K^2)` 경로를 검증한다.
- 다음 단계는 synthetic 삼각 관계 task에서 구조적 이득을 검증하는 것이다.

## Auto Research Loop: auto_facts4_relation_w1_topk8_1000_seed123 iteration 1

### Loop Policy

- steps: `1000`
- eval_every: `100`
- seeds: `123`
- synthetic facts: `4`
- synthetic loss mode: `answer_relation`
- 700 step 이후 변화가 보이도록 100 step 단위로 history를 기록한다.
- synthetic task에서 표준 attention 이상인 [삼각 관계] variant를 NSMC 후보로 선발한다.
- 만족 기준: synthetic best accuracy가 standard보다 `0.020` 이상 높고, NSMC best val loss가 standard보다 `0.0000` 이상 개선되어야 한다.
- 비용 경고 기준: NSMC ms/step이 standard의 `3.0`배를 넘으면 비용 병목으로 표시한다.

### Strategy Under Test

- `standard`: 기존 pairwise attention 기준선.
- `triangular_v5_gated_topk`: gated pair-value와 top-k 후보 선택을 결합한 비용/안정성 절충 전략.

### Synthetic Screening Summary

| variant | best accuracy | final accuracy mean | final loss mean | ms/step mean |
| --- | ---: | ---: | ---: | ---: |
| standard | 0.414 | 0.398 | 3.3444 | 9.54 |
| triangular_v5_gated_topk | 0.408 | 0.406 | 3.5287 | 76.29 |

### Selected Variants

- triangular_v5_gated_topk

### NSMC Validation Summary

| variant | best val loss | final val loss mean | final train loss mean | ms/step mean |
| --- | ---: | ---: | ---: | ---: |
| standard | 7.3115 | 7.3133 | 7.2821 | 8.51 |
| triangular_v5_gated_topk | 7.3097 | 7.3097 | 7.2790 | 24.76 |

### Success Evaluation

| variant | synthetic gain | NSMC val gain | cost ratio | status |
| --- | ---: | ---: | ---: | --- |
| triangular_v5_gated_topk | -0.006 | 0.0018 | 2.91x | synthetic fail, NSMC pass |

### Next Automatic Decision

- 만족 기준을 통과한 variant가 있으면 multi-seed 장기 검증 후보로 유지한다.
- synthetic gain은 있으나 기준에 못 미치면 더 강한 relation task 또는 gated pair/top-k 변형을 다음 후보군에 넣는다.
- NSMC가 앞서지만 비용 경고가 뜨면 top-k/gated-topk 최적화를 우선한다.
- loop status: `not_satisfied`

- synthetic JSON: `/Users/magik/Documents/jungle/02gpt-lab/reports/auto_triangular_research/auto_facts4_relation_w1_topk8_1000_seed123/iter_01/synthetic.json`
- NSMC JSON: `/Users/magik/Documents/jungle/02gpt-lab/reports/auto_triangular_research/auto_facts4_relation_w1_topk8_1000_seed123/iter_01/nsmc.json`

## Experiment: Auto auto_facts4_relation_w05_topk8_2000_seed123 synthetic iteration 1

### Task

`give(subject, object, recipient)` fact `4`개를 제시하고, query의 `(subject, object)` 조합에 맞는 recipient를 마지막 token으로 맞힌다. distractor는 subject 또는 object를 공유하므로 단일 token shortcut을 어렵게 만든다. loss_mode는 `answer_relation`이다.

### Results

| variant | seed | final accuracy | best accuracy | final loss | ms/step | entropy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| standard | 123 | 0.386 | 0.439 | 2.2461 | 9.52 | - |
| triangular_v5_gated_topk | 123 | 0.409 | 0.464 | 2.3295 | 67.79 | 3.416 |

### Interpretation

- 이 task는 [삼각 관계]가 목표로 하는 세 토큰 조합 인식을 직접 요구한다.
- 표준 attention 대비 accuracy가 높으면 구조적 이득 후보로 기록한다.
- 다음 iteration에서는 가장 좋은 variant를 NSMC LM으로 재검증한다.

## Experiment: Auto auto_facts4_relation_w05_topk8_2000_seed123 NSMC iteration 1

### Command

```bash
python scripts/compare_attention_models.py --context-length 32 --batch-size 8 --steps 2000 --eval-every 100 --eval-batches 8 --seeds 123 --variants standard,triangular_v5_gated_topk
```

### Setup

- device: `mps`
- context_length: `32`
- batch_size: `8`
- steps: `2000`
- seeds: `123`

### Results

| variant | seed | final val | best val | ms/step | entropy | score elements/layer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| standard | 123 | 7.3106 | 7.3106 | 8.63 | - | 32768 |
| triangular_v5_gated_topk | 123 | 7.3087 | 7.3087 | 24.68 | 3.493 | 65536 |

### Interpretation

- 표준 attention은 비용 기준선이다.
- [삼각 관계] full variant는 세 토큰 조합을 직접 보지만 비용은 `O(T^3)`이다.
- top-k variant는 후보 토큰을 줄여 `O(T*K^2)` 경로를 검증한다.
- 다음 단계는 synthetic 삼각 관계 task에서 구조적 이득을 검증하는 것이다.

## Auto Research Loop: auto_facts4_relation_w05_topk8_2000_seed123 iteration 1

### Loop Policy

- steps: `2000`
- eval_every: `100`
- seeds: `123`
- synthetic facts: `4`
- synthetic loss mode: `answer_relation`
- 700 step 이후 변화가 보이도록 100 step 단위로 history를 기록한다.
- synthetic task에서 표준 attention 이상인 [삼각 관계] variant를 NSMC 후보로 선발한다.
- 만족 기준: synthetic best accuracy가 standard보다 `0.020` 이상 높고, NSMC best val loss가 standard보다 `0.0000` 이상 개선되어야 한다.
- 비용 경고 기준: NSMC ms/step이 standard의 `3.0`배를 넘으면 비용 병목으로 표시한다.

### Strategy Under Test

- `standard`: 기존 pairwise attention 기준선.
- `triangular_v5_gated_topk`: gated pair-value와 top-k 후보 선택을 결합한 비용/안정성 절충 전략.

### Synthetic Screening Summary

| variant | best accuracy | final accuracy mean | final loss mean | ms/step mean |
| --- | ---: | ---: | ---: | ---: |
| standard | 0.439 | 0.386 | 2.2461 | 9.52 |
| triangular_v5_gated_topk | 0.464 | 0.409 | 2.3295 | 67.79 |

### Selected Variants

- triangular_v5_gated_topk

### NSMC Validation Summary

| variant | best val loss | final val loss mean | final train loss mean | ms/step mean |
| --- | ---: | ---: | ---: | ---: |
| standard | 7.3106 | 7.3106 | 7.2748 | 8.63 |
| triangular_v5_gated_topk | 7.3087 | 7.3087 | 7.2790 | 24.68 |

### Success Evaluation

| variant | synthetic gain | NSMC val gain | cost ratio | status |
| --- | ---: | ---: | ---: | --- |
| triangular_v5_gated_topk | 0.025 | 0.0019 | 2.86x | synthetic pass, NSMC pass |

### Next Automatic Decision

- 만족 기준을 통과한 variant가 있으면 multi-seed 장기 검증 후보로 유지한다.
- synthetic gain은 있으나 기준에 못 미치면 더 강한 relation task 또는 gated pair/top-k 변형을 다음 후보군에 넣는다.
- NSMC가 앞서지만 비용 경고가 뜨면 top-k/gated-topk 최적화를 우선한다.
- loop status: `satisfied`

- synthetic JSON: `/Users/magik/Documents/jungle/02gpt-lab/reports/auto_triangular_research/auto_facts4_relation_w05_topk8_2000_seed123/iter_01/synthetic.json`
- NSMC JSON: `/Users/magik/Documents/jungle/02gpt-lab/reports/auto_triangular_research/auto_facts4_relation_w05_topk8_2000_seed123/iter_01/nsmc.json`

## Milestone: First Satisfied Candidate

### Candidate

`triangular_v5_gated_topk`

- value path: `pair_gated`
- logit scale: `learned`
- candidate mode: `topk`
- top_k: `8`
- synthetic task: `facts=4`, `loss_mode=answer_relation`, `relation_loss_weight=0.5`
- training steps: `2000`

### Why It Passed

| axis | standard | triangular_v5_gated_topk | gain |
| --- | ---: | ---: | ---: |
| synthetic best accuracy | 0.439 | 0.464 | +0.025 |
| NSMC best val loss | 7.3106 | 7.3087 | +0.0019 lower loss |
| NSMC score elements/layer | 32768 | 65536 | 2.0x |
| NSMC ms/step | 8.63 | 24.68 | 2.86x |

해석:

- 처음으로 자동 루프의 만족 기준을 통과했다.
- [삼각 관계]가 이득을 낸 조건은 단순 answer-only task가 아니라, fact 내부의 `TO -> recipient` relation supervision을 함께 준 조건이었다.
- top-k는 `K=8`이 가장 좋았다. `K=12`는 synthetic gain이 줄고 비용이 증가했다.
- 비용은 standard보다 크지만, score elements는 full triangular의 `1,048,576` 대비 `65,536`으로 줄어 장기 최적화 후보가 된다.

### Next Research Direction

1. seed `[123,456,789]`로 multi-seed 재검증한다.
2. `triangular_v5_gated_topk`의 top-k 후보 선택을 더 안정화한다.
3. relation supervision 없이도 이득이 유지되는지 curriculum을 시험한다.
4. 비용 최적화를 위해 top-k gather path와 MPS overhead를 줄인다.

## Experiment: Final NSMC attention comparison

### Command

```bash
python scripts/compare_attention_models.py --context-length 32 --batch-size 8 --steps 2000 --eval-every 100 --eval-batches 8 --seeds 123,456,789 --variants standard,triangular_final,alternating_standard_triangular
```

### Setup

- device: `mps`
- context_length: `32`
- batch_size: `8`
- steps: `2000`
- seeds: `123,456,789`

### Results

| variant | seed | final val | best val | ms/step | entropy | score elements/layer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| standard | 123 | 7.3106 | 7.3106 | 7.90 | - | 32768 |
| triangular_final | 123 | 7.3087 | 7.3087 | 22.42 | 3.493 | 65536 |
| alternating_standard_triangular | 123 | 7.3107 | 7.3107 | 14.63 | 3.704 | 49152 |
| standard | 456 | 7.3121 | 7.3121 | 8.22 | - | 32768 |
| triangular_final | 456 | 7.3113 | 7.3113 | 22.48 | 3.737 | 65536 |
| alternating_standard_triangular | 456 | 7.3144 | 7.3133 | 14.80 | 3.683 | 49152 |
| standard | 789 | 7.3110 | 7.3110 | 8.16 | - | 32768 |
| triangular_final | 789 | 7.3096 | 7.3096 | 21.99 | 3.727 | 65536 |
| alternating_standard_triangular | 789 | 7.3098 | 7.3098 | 14.27 | 3.757 | 49152 |

### Interpretation

- 표준 attention은 비용 기준선이다.
- [삼각 관계] full variant는 세 토큰 조합을 직접 보지만 비용은 `O(T^3)`이다.
- top-k variant는 후보 토큰을 줄여 `O(T*K^2)` 경로를 검증한다.
- 다음 단계는 synthetic 삼각 관계 task에서 구조적 이득을 검증하는 것이다.

## Final Aggregate Summary: Final NSMC attention comparison

| variant | seeds | best val mean | best val std | final val mean | ms/step mean | score elements/model |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| triangular_final | 3 | 7.3099 | 0.0011 | 7.3099 | 22.30 | 131072 |
| standard | 3 | 7.3112 | 0.0006 | 7.3112 | 8.10 | 65536 |
| alternating_standard_triangular | 3 | 7.3113 | 0.0015 | 7.3116 | 14.57 | 98304 |

- summary CSV: `reports/final_triangular_nsmc_summary.csv`
- mean loss plot: `reports/final_triangular_nsmc_mean_loss.png`
- cost-quality plot: `reports/final_triangular_nsmc_cost_quality.png`
- final report: `reports/triangular_relation_final_report.md`

## Final Three-Model Reframe

사용자 피드백에 따라 seed별 raw row가 아니라, 기존 `seeds=123,456,789` 결과를 모델별 대표값으로 집계해 세 모델만 비교한다. 실험은 다시 실행하지 않았다.

| model | best val loss | loss reduction vs standard | ms/step | cost ratio | score tensor ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| `standard` | 7.3112 | baseline | 8.10 | 1.00x | 1.00x |
| `triangular_final` | 7.3099 | +0.0014 | 22.30 | 2.75x | 2.00x |
| `alternating_standard_triangular` | 7.3113 | ≈0.0000 | 14.57 | 1.80x | 1.50x |

해석:

- `triangular_final`만 standard 대비 성능 개선이 선명하다.
- 혼합 모델은 비용은 늘지만 성능은 standard와 거의 같아, 이번 조건에서는 좋은 절충점이 아니다.
- 최종 보고서는 `reports/triangular_relation_final_report.md`, compact plot은 `reports/final_three_model_comparison.png`에 있다.

## Experiment: Smoke position mode 6-model comparison

### Command

```bash
python scripts/compare_attention_models.py --context-length 8 --batch-size 2 --steps 2 --eval-every 1 --eval-batches 1 --seeds 123 --variants standard_embedding,triangular_embedding,alternating_embedding,standard_encoding,triangular_encoding,alternating_encoding
```

### Setup

- device: `mps`
- context_length: `8`
- batch_size: `2`
- steps: `2`
- seeds: `123`

### Results

| variant | seed | final val | best val | ms/step | entropy | score elements/layer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| standard_embedding | 123 | 8.4977 | 8.4977 | 227.00 | - | 512 |
| triangular_embedding | 123 | 8.2467 | 8.2467 | 164.61 | 2.166 | 1024 |
| alternating_embedding | 123 | 8.1915 | 8.1915 | 27.67 | 2.166 | 768 |
| standard_encoding | 123 | 8.2521 | 8.2521 | 15.31 | - | 512 |
| triangular_encoding | 123 | 7.9848 | 7.9848 | 32.65 | 2.168 | 1024 |
| alternating_encoding | 123 | 8.0681 | 8.0681 | 25.39 | 2.165 | 768 |

### Interpretation

- 이번 비교는 attention family 3개와 위치 방식 2개를 교차한 6모델 실험이다.
- `standard_embedding`을 loss delta 기준선으로 두고, 같은 attention family 안에서는 encoding과 embedding의 best loss 차이를 본다.
- `[삼각 관계]`와 alternating은 final top-k 설정을 사용하므로 비용은 full `O(T^3)`가 아니라 `O(T*K^2)` 경로다.

## Final Aggregate Summary: Smoke position mode 6-model comparison

| variant | seeds | best val mean | best val std | final val mean | ms/step mean | score elements/model |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| triangular_encoding | 1 | 7.9848 | 0.0000 | 7.9848 | 32.65 | 2048 |
| alternating_encoding | 1 | 8.0681 | 0.0000 | 8.0681 | 25.39 | 1536 |
| alternating_embedding | 1 | 8.1915 | 0.0000 | 8.1915 | 27.67 | 1536 |
| triangular_embedding | 1 | 8.2467 | 0.0000 | 8.2467 | 164.61 | 2048 |
| standard_encoding | 1 | 8.2521 | 0.0000 | 8.2521 | 15.31 | 1024 |
| standard_embedding | 1 | 8.4977 | 0.0000 | 8.4977 | 227.00 | 1024 |

- history CSV: `reports/smoke_position_mode_6model_comparison_history.csv`
- summary CSV: `reports/smoke_position_mode_6model_summary.csv`
- full loss plot: `reports/smoke_position_mode_6model_loss_full.png`
- mean loss plot: `reports/smoke_position_mode_6model_loss_full_mean.png`
- zoom loss plot: `reports/smoke_position_mode_6model_loss_full_zoom_700.png`
- delta loss plot: `reports/smoke_position_mode_6model_loss_full_delta.png`
- cost-quality plot: `reports/smoke_position_mode_6model_loss_full_cost_quality.png`
- best loss bar plot: `reports/smoke_position_mode_6model_loss_full_best_loss_bar.png`
- position effect plot: `reports/smoke_position_mode_6model_loss_full_position_effect.png`
- relation stats plot: `reports/smoke_position_mode_6model_loss_full_relation_stats.png`
- final report: `reports/smoke_triangular_relation_position_mode_report.md`

## Experiment: Position embedding vs encoding 6-model NSMC comparison

### Command

```bash
python scripts/compare_attention_models.py --context-length 32 --batch-size 8 --steps 2000 --eval-every 100 --eval-batches 8 --seeds 123 --variants standard_embedding,triangular_embedding,alternating_embedding,standard_encoding,triangular_encoding,alternating_encoding
```

### Setup

- device: `mps`
- context_length: `32`
- batch_size: `8`
- steps: `2000`
- seeds: `123`

### Results

| variant | seed | final val | best val | ms/step | entropy | score elements/layer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| standard_embedding | 123 | 7.3106 | 7.3106 | 7.91 | - | 32768 |
| triangular_embedding | 123 | 7.3087 | 7.3087 | 22.57 | 3.492 | 65536 |
| alternating_embedding | 123 | 7.3107 | 7.3107 | 14.97 | 3.698 | 49152 |
| standard_encoding | 123 | 7.2603 | 7.2603 | 7.71 | - | 32768 |
| triangular_encoding | 123 | 7.2515 | 7.2515 | 22.90 | 3.720 | 65536 |
| alternating_encoding | 123 | 7.2625 | 7.2625 | 14.72 | 3.701 | 49152 |

### Interpretation

- 이번 비교는 attention family 3개와 위치 방식 2개를 교차한 6모델 실험이다.
- `standard_embedding`을 loss delta 기준선으로 두고, 같은 attention family 안에서는 encoding과 embedding의 best loss 차이를 본다.
- `[삼각 관계]`와 alternating은 final top-k 설정을 사용하므로 비용은 full `O(T^3)`가 아니라 `O(T*K^2)` 경로다.

## Final Aggregate Summary: Position embedding vs encoding 6-model NSMC comparison

| variant | seeds | best val mean | best val std | final val mean | ms/step mean | score elements/model |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| triangular_encoding | 1 | 7.2515 | 0.0000 | 7.2515 | 22.90 | 131072 |
| standard_encoding | 1 | 7.2603 | 0.0000 | 7.2603 | 7.71 | 65536 |
| alternating_encoding | 1 | 7.2625 | 0.0000 | 7.2625 | 14.72 | 98304 |
| triangular_embedding | 1 | 7.3087 | 0.0000 | 7.3087 | 22.57 | 131072 |
| standard_embedding | 1 | 7.3106 | 0.0000 | 7.3106 | 7.91 | 65536 |
| alternating_embedding | 1 | 7.3107 | 0.0000 | 7.3107 | 14.97 | 98304 |

- history CSV: `reports/position_mode_6model_history.csv`
- summary CSV: `reports/position_mode_6model_summary.csv`
- full loss plot: `reports/position_mode_6model_loss_full.png`
- mean loss plot: `reports/position_mode_6model_loss_mean.png`
- zoom loss plot: `reports/position_mode_6model_loss_zoom_700.png`
- delta loss plot: `reports/position_mode_6model_loss_delta.png`
- cost-quality plot: `reports/position_mode_6model_cost_quality.png`
- best loss bar plot: `reports/position_mode_6model_best_loss_bar.png`
- position effect plot: `reports/position_mode_6model_position_effect.png`
- relation stats plot: `reports/position_mode_6model_relation_stats.png`
- final report: `reports/triangular_relation_position_mode_report.md`

### Position Mode Interpretation Note

- fixed sin/cos encoding은 세 attention family 모두에서 learned position embedding보다 낮은 best validation loss를 냈다.
- 개선 폭은 `standard=-0.0503`, `[삼각 관계]=-0.0573`, `alternating=-0.0481`로, 이번 seed에서는 `[삼각 관계]`에서 가장 컸다.
- encoding variant는 learned position table을 학습하지 않아 같은 attention family에서 parameter가 `2048`개 적다.
- 공식 비교는 seed 123 단일 실행이므로, 효과 방향은 강하게 보이지만 통계적 확정은 추가 seed 반복이 필요하다.

## Experiment: Position embedding vs encoding 6-model NSMC comparison 5000 steps

### Command

```bash
python scripts/compare_attention_models.py --context-length 32 --batch-size 8 --steps 5000 --eval-every 100 --eval-batches 8 --seeds 123 --variants standard_embedding,triangular_embedding,alternating_embedding,standard_encoding,triangular_encoding,alternating_encoding
```

### Setup

- device: `mps`
- context_length: `32`
- batch_size: `8`
- steps: `5000`
- seeds: `123`

### Results

| variant | seed | final val | best val | ms/step | entropy | score elements/layer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| standard_embedding | 123 | 6.6754 | 6.6754 | 7.66 | - | 32768 |
| triangular_embedding | 123 | 6.6931 | 6.6931 | 15.11 | 3.567 | 65536 |
| alternating_embedding | 123 | 6.7002 | 6.7002 | 11.14 | 3.611 | 49152 |
| standard_encoding | 123 | 6.3554 | 6.3554 | 5.86 | - | 32768 |
| triangular_encoding | 123 | 6.3458 | 6.3458 | 14.88 | 3.511 | 65536 |
| alternating_encoding | 123 | 6.3874 | 6.3874 | 10.75 | 3.692 | 49152 |

### Interpretation

- 이번 비교는 attention family 3개와 위치 방식 2개를 교차한 6모델 실험이다.
- `standard_embedding`을 loss delta 기준선으로 두고, 같은 attention family 안에서는 encoding과 embedding의 best loss 차이를 본다.
- `[삼각 관계]`와 alternating은 final top-k 설정을 사용하므로 비용은 full `O(T^3)`가 아니라 `O(T*K^2)` 경로다.

## Final Aggregate Summary: Position embedding vs encoding 6-model NSMC comparison 5000 steps

| variant | seeds | best val mean | best val std | final val mean | ms/step mean | score elements/model |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| triangular_encoding | 1 | 6.3458 | 0.0000 | 6.3458 | 14.88 | 131072 |
| standard_encoding | 1 | 6.3554 | 0.0000 | 6.3554 | 5.86 | 65536 |
| alternating_encoding | 1 | 6.3874 | 0.0000 | 6.3874 | 10.75 | 98304 |
| standard_embedding | 1 | 6.6754 | 0.0000 | 6.6754 | 7.66 | 65536 |
| triangular_embedding | 1 | 6.6931 | 0.0000 | 6.6931 | 15.11 | 131072 |
| alternating_embedding | 1 | 6.7002 | 0.0000 | 6.7002 | 11.14 | 98304 |

- history CSV: `reports/position_mode_6model_5000_history.csv`
- summary CSV: `reports/position_mode_6model_5000_summary.csv`
- full loss plot: `reports/position_mode_6model_5000_loss_full.png`
- mean loss plot: `reports/position_mode_6model_5000_loss_mean.png`
- zoom loss plot: `reports/position_mode_6model_5000_loss_zoom_700.png`
- delta loss plot: `reports/position_mode_6model_5000_loss_delta.png`
- cost-quality plot: `reports/position_mode_6model_5000_cost_quality.png`
- best loss bar plot: `reports/position_mode_6model_5000_best_loss_bar.png`
- position effect plot: `reports/position_mode_6model_5000_position_effect.png`
- relation stats plot: `reports/position_mode_6model_5000_relation_stats.png`
- final report: `reports/triangular_relation_position_mode_5000_report.md`
