# POGO: Policy Optimization by Gradient Flow on Offline RL

POGO는 gradient flow 기반의 오프라인 강화학습 정책 최적화 프레임워크로, diagonal Gaussian policy를 넘어선 일반화된 접근법을 제공합니다.

> **연구 계보:** POGO는 [MPI](https://github.com/SChoish/MPI)의 전신 연구입니다. 이 저장소는 transport-map actor와 JKO/Sinkhorn 기반 multi-step policy flow를 탐색한 초기 구현이며, 후속 연구인 MPI는 이를 표준 behavior-regularized actor update에 적용되는 re-centered proximal framework로 정식화하고 여러 policy geometry와 base algorithm으로 확장합니다.

**연구 기여자:** [Soohyun Choi](https://github.com/SChoish), [Seonvin Cho](https://github.com/seonvin0319)

## 주요 특징

- **Transport Map Architecture**: Actor가 `z ~ N(0,I)`에서 행동 공간으로의 transport map을 근사
- **JKO Flow**: Jordan-Kinderlehrer-Otto 흐름을 통한 multi-step 최적화
- **Flexible Distance Metrics**: L2 거리와 Sinkhorn 거리를 상황에 맞게 사용
- **Adaptive Regularization**: Q값 크기에 따른 적응적 정규화
- **Flexible Training**: 원스텝 체크포인트에서 투스텝만 실행 가능

## 파일 구조

```
POGO_Sto/
├── main.py                 # 메인 실행 스크립트
├── agent.py               # POGO 및 POGO_Refine 에이전트 구현
├── utils.py               # 유틸리티 함수들 (ReplayBuffer 등)
├── run_comparison.py      # 병렬 실험 실행기
├── config.yaml            # 실험 설정
└── README.md              # 이 파일
```

## 사용법

### 1. 기본 POGO 학습 (Single-step)
```bash
conda activate offrl
python main.py --env hopper-medium-v2 --seed 0
```

### 2. 전체 학습 (Single-step + Two-step)
```bash
python main.py --env hopper-medium-v2 --seed 0 --two_step
```

### 3. Two-step만 실행 (원스텝 체크포인트에서 시작)
```bash
python main.py --env hopper-medium-v2 --seed 0 --two_step \
    --start_mode two_step_only \
    --load_prefix ./logs/checkpoints/POGO_hopper-medium-v2_0_mid_50000
```

### 4. 체크포인트에서 계속 학습
```bash
python main.py --env hopper-medium-v2 --seed 0 --two_step \
    --start_mode load \
    --load_prefix ./logs/checkpoints/POGO_hopper-medium-v2_0_mid_50000
```

## 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--env` | hopper-medium-v2 | D4RL 환경 |
| `--seed` | 0 | 랜덤 시드 |
| `--two_step` | False | Two-step 학습 활성화 |
| `--split_ratio` | 0.5 | Single-step과 Two-step 분할 비율 |
| `--alpha` | 1.0 | 정규화 강도 |
| `--w2_weight` | 0.5 | Wasserstein 거리 페널티 가중치 |
| `--freeze_critic_mode` | True | Two-step에서 critic 고정 여부 |
| `--start_mode` | scratch | 시작 모드 (scratch/load/two_step_only) |
| `--load_prefix` | "" | 체크포인트 파일 경로 |

## 알고리즘 개요

### POGO (Single-step)
- TD3+BC를 gradient flow 관점으로 확장
- Transport map `T_s: z ~ N(0,I) → action space` 학습
- Behavior policy와의 W2 거리 페널티
- 적응적 정규화: `λ = α / |Q|_mean`

### POGO_Refine (Multi-step JKO)
- JKO (Jordan-Kinderlehrer-Otto) 흐름의 2단계 확장
- Sinkhorn 거리를 통한 복잡한 분포 처리
- Critic 고정 옵션으로 실험적 유연성 제공
- Diagonal Gaussian policy를 넘어선 일반화

## 출력 파일

```
POGO_Sto/
├── results/              # 평가 결과 (.npy)
├── models/              # 학습된 모델 (.pth)
├── logs/               # 로그 및 체크포인트
└── comparison_results_*.csv  # 비교 실험 결과
```

## 의존성

- Python 3.8+
- PyTorch
- D4RL
- NumPy
- Geomloss (Sinkhorn 거리 계산)

## 설치

```bash
conda activate offrl
pip install -e .
```

## 라이선스

MIT License
