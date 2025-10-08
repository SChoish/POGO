# POGO: Policy Optimization with GFlow Networks

POGO는 GFlow Networks를 활용한 오프라인 강화학습 정책 최적화 프레임워크입니다.

## 주요 특징

- **2단계 학습**: Single-step → Two-step 순차 학습
- **Natural Gradient**: NatGrad 지원으로 안정적인 학습
- **병렬 실험**: 다중 환경, 다중 시드 병렬 실행
- **Resume 기능**: 중단된 실험 자동 재개
- **체계적 로깅**: 타임스탬프 기반 로그 관리

## 파일 구조

```
POGO/
├── agent_gflow.py          # GFlow 에이전트 구현
├── main_two.py             # 2단계 실험 메인 스크립트
├── run_comparison.py       # 병렬 실험 실행기
├── config.yaml             # 실험 설정 (모든 환경)
├── config_test.yaml        # 테스트 설정
├── utils.py                # 유틸리티 함수들
└── README.md               # 이 파일
```

## 실험 설정

### 환경별 파라미터 (config.yaml)

| 환경 | w2_weight | entropy_weight | learning_rate |
|------|-----------|----------------|---------------|
| hopper-medium | 0.2 | 1e-5 | 3e-4 |
| hopper-medium-replay | 0.1 | 1e-5 | 3e-4 |
| hopper-medium-expert | 0.1 | 1e-5 | 3e-4 |
| halfcheetah-medium | 0.05 | 5e-3 | 3e-4 |
| halfcheetah-medium-replay | 0.1 | 1e-5 | 3e-4 |
| walker2d-medium | 0.1 | 1e-5 | 3e-4 |
| walker2d-medium-replay | 0.1 | 1e-5 | 3e-4 |
| walker2d-medium-expert | 0.3 | 1e-2 | 3e-4 |
| antmaze-umaze-v2 | 0.9 | 0.0 | 1e-4 |
| antmaze-umaze-diverse-v2 | 0.5 | 1e-3 | 1e-4 |
| antmaze-medium-play-v2 | 0.2 | 3e-3 | 1e-4 |
| antmaze-medium-diverse-v2 | 0.2 | 3e-3 | 1e-4 |
| antmaze-large-play-v2 | 0.2 | 3e-3 | 1e-4 |
| antmaze-large-diverse-v2 | 0.2 | 3e-3 | 1e-4 |

## 사용법

### 1. 전체 실험 실행
```bash
conda activate offrl
python run_comparison.py --parallel 2 --resume
```

### 2. 단일 실험 실행
```bash
python main_two.py --env hopper-medium --seed 0 --w2_weight 0.2 --entropy_weight 1e-5 --lr 3e-4 --use_natgrad
```

### 3. 2단계 실험 실행
```bash
python main_two.py --env hopper-medium --seed 0 --two_step --split_ratio 0.5 --use_natgrad
```

## 실험 결과

### Hopper-Medium 결과
- **Single-Step 평균**: 68.882 ± 6.133
- **Two-Step 평균**: 85.060 ± 5.024
- **개선도**: +16.177 (+23.5%)

### AntMaze-Umaze-Diverse-v2 결과
- **Single-Step 평균**: 32.800 ± 31.705
- **Two-Step 평균**: 40.800 ± 26.781
- **개선도**: +8.000 (+24.4%)

### AntMaze-Medium-Play-v2 결과
- **Single-Step 평균**: 2.400 ± 5.367
- **Two-Step 평균**: 32.000 ± 22.361
- **개선도**: +29.600 (+1233.3%)

## 로그 구조

```
logs/
├── {env}/
│   └── w2_{w2_weight}_ent_{entropy_weight}/
│       └── seed_{seed}_natgrad/
│           ├── phase1_single_step/
│           │   └── POGO_single_step_{env}_{seed}_{timestamp}.log
│           └── phase2_two_step/
│               └── POGO_two_step_{env}_{seed}_{timestamp}.log
```

## 의존성

- Python 3.10+
- PyTorch
- D4RL
- NumPy
- YAML

## 라이선스

MIT License
