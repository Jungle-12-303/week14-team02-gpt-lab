
## Experiment: Smoke final NSMC attention comparison

### Command

```bash
python scripts/compare_attention_models.py --context-length 16 --batch-size 4 --steps 2 --eval-every 1 --eval-batches 1 --seeds 123 --variants standard,triangular_final,alternating_standard_triangular
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
| standard | 123 | 8.3384 | 8.3384 | 84.40 | - | 4096 |
| triangular_final | 123 | 8.0955 | 8.0955 | 75.37 | 3.388 | 16384 |
| alternating_standard_triangular | 123 | 8.2991 | 8.2991 | 27.06 | 3.389 | 10240 |

### Interpretation

- 표준 attention은 비용 기준선이다.
- [삼각 관계] full variant는 세 토큰 조합을 직접 보지만 비용은 `O(T^3)`이다.
- top-k variant는 후보 토큰을 줄여 `O(T*K^2)` 경로를 검증한다.
- 다음 단계는 synthetic 삼각 관계 task에서 구조적 이득을 검증하는 것이다.

## Final Aggregate Summary: Smoke final NSMC attention comparison

| variant | seeds | best val mean | best val std | final val mean | ms/step mean | score elements/model |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| triangular_final | 1 | 8.0955 | 0.0000 | 8.0955 | 75.37 | 32768 |
| alternating_standard_triangular | 1 | 8.2991 | 0.0000 | 8.2991 | 27.06 | 20480 |
| standard | 1 | 8.3384 | 0.0000 | 8.3384 | 84.40 | 8192 |

- summary CSV: `reports/smoke_final_triangular_nsmc_summary.csv`
- mean loss plot: `reports/smoke_final_triangular_nsmc_mean_loss.png`
- cost-quality plot: `reports/smoke_final_triangular_nsmc_cost_quality.png`
- final report: `reports/smoke_triangular_relation_final_report.md`
