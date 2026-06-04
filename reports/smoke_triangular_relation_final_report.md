# [삼각 관계] Final NSMC Report

## Background

[삼각 관계]는 기존 attention의 `score[i,j]`를 `score[i,j,k]`로 확장해 세 토큰 조합을 직접 보려는 실험이다. 최종 후보는 pair-value 경로를 learned gate로 조절하고, top-k 후보 안에서만 삼각 관계를 계산해 full `O(T^3)` 비용을 줄인다.

기대는 두 가지였다. 첫째, 세 토큰 관계를 직접 보는 inductive bias가 NSMC LM에서도 작은 loss 개선을 만들 수 있는지 확인한다. 둘째, 표준 attention과 [삼각 관계]를 섞은 모델이 성능과 비용 사이의 절충점을 만들 수 있는지 본다.

## Setup

- device: `mps`
- variants: `standard,triangular_final,alternating_standard_triangular`
- seeds: `123`
- steps: `2`
- context_length: `16`
- batch_size: `4`
- top_k: `8`
- alternating order: `standard -> triangular`

## Raw Results

| variant | seed | final val | best val | ms/step | score elements/model |
| --- | ---: | ---: | ---: | ---: | ---: |
| standard | 123 | 8.3384 | 8.3384 | 84.40 | 8192 |
| triangular_final | 123 | 8.0955 | 8.0955 | 75.37 | 32768 |
| alternating_standard_triangular | 123 | 8.2991 | 8.2991 | 27.06 | 20480 |

## Aggregate Results

| variant | seeds | best val mean | best val std | final val mean | ms/step mean | score elements/model | best-val gain vs standard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| triangular_final | 1 | 8.0955 | 0.0000 | 8.0955 | 75.37 | 32768 | 0.2429 |
| alternating_standard_triangular | 1 | 8.2991 | 0.0000 | 8.2991 | 27.06 | 20480 | 0.0393 |
| standard | 1 | 8.3384 | 0.0000 | 8.3384 | 84.40 | 8192 | - |

## Figures

- Seed loss curves: `reports/smoke_final_triangular_nsmc_loss.png`
- Mean loss curves: `reports/smoke_final_triangular_nsmc_mean_loss.png`
- Cost-quality scatter: `reports/smoke_final_triangular_nsmc_cost_quality.png`
- Summary CSV: `reports/smoke_final_triangular_nsmc_summary.csv`

## Interpretation

- 평균 best val loss 기준 최상위 모델은 `triangular_final`이다.
- `triangular_final`은 standard 대비 평균 best val loss를 `0.2429`만큼 낮췄다.
- `alternating_standard_triangular`은 standard 대비 평균 best val loss를 `0.0393`만큼 바꿨다.
- 혼합 모델은 순수 [삼각 관계] 대비 비용과 성능의 절충점인지 확인하는 기준점이다. 평균 ms/step은 `27.06`이고 순수 [삼각 관계]는 `75.37`이다.

## Next Direction

- multi-seed 결과에서 [삼각 관계] 또는 혼합 모델의 개선이 유지되면 top-k gather path 최적화를 우선한다.
- 혼합 모델이 순수 [삼각 관계]보다 비슷한 성능에 낮은 비용을 보이면, 층별 배치 비율을 `standard, triangular, standard, triangular` 식으로 더 깊은 모델에서 탐색한다.
- 개선이 seed 평균에서 사라지면 synthetic 성공은 relation supervision 조건부 효과로 제한해 해석한다.
