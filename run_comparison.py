#!/usr/bin/env python3
"""
POGO 두 단계(Phase-1 / Phase-2) 실험 런처 (순차 실행 버전)
- config.yaml에 정의된 환경/하이퍼를 순차 수행 (병렬처리 없음)
- 필요한 경우에만 Phase-1을 돌려 split 체크포인트 생성
- 항상 체크포인트에서 Phase-2(two_step)만 실행
- Phase-1: CPU 사용, Phase-2: GPU 사용
"""

import os
import sys
import time
import json
import yaml
import subprocess
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser

# ----------------------------
# 유틸
# ----------------------------
def now_str():
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

def safe(s: str) -> str:
    return s.replace('/', '_').replace('-', '_')

def load_yaml(path: Path):
    with path.open('r') as f:
        return yaml.safe_load(f)

def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2))

def tail(path: Path, n=50):
    if not path.exists():
        return ""
    with path.open('r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    return ''.join(lines[-n:])

# ----------------------------
# 체크 / 실행 함수들
# ----------------------------
def find_split_checkpoint(ckpt_dir: Path, split_step: int) -> Path | None:
    """정확 매치 우선, 아니면 가장 가까운 _mid_<t>_*_actor 파일을 찾음(프리픽스 반환용)."""
    # 정확 매치
    for f in ckpt_dir.glob(f"*_mid_{split_step}_*_actor"):
        return f
    # 근접 매치
    best = None
    best_diff = 1e18
    for f in ckpt_dir.glob("*_mid_*_actor"):
        parts = f.stem.split('_')
        for i, p in enumerate(parts):
            if p == 'mid' and i+1 < len(parts):
                try:
                    t = int(parts[i+1])
                except:
                    continue
                diff = abs(t - split_step)
                if diff < best_diff:
                    best_diff = diff
                    best = f
    return best

def phase2_done(log_file: Path) -> bool:
    """2단계가 완료되었는지 로그로 판정(최종 표기 문자열 기반)."""
    if not log_file.exists():
        return False
    txt = log_file.read_text(errors='replace')
    return ('======== Final Evaluation' in txt
            and '[FINAL] Deterministic:' in txt
            and '[FINAL] Stochastic:' in txt)

def copy_phase1_log_prefix(phase1_log: Path, phase2_log: Path, split_step: int):
    """1단계 로그에서 split 지점까지를 2단계 로그 머리말로 복사(있으면)."""
    if not phase1_log.exists():
        phase2_log.touch()
        return
    try:
        with phase1_log.open('r', encoding='utf-8', errors='replace') as src:
            lines = src.readlines()
        with phase2_log.open('w', encoding='utf-8') as dst:
            for line in lines:
                dst.write(line)
                if f'Time steps: {split_step}' in line or f'timesteps: {split_step}' in line:
                    break
    except Exception:
        phase2_log.touch()

def run_phase(
    pyexec: Path, root_dir: Path, args: list[str], log_path: Path, env: dict | None = None
) -> tuple[int, str | None]:
    """main.py 한 번 실행. rc, 예외메시지 반환."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rc, err = 0, None
    try:
        with log_path.open('w', encoding='utf-8') as logf:
            proc = subprocess.Popen(
                [str(pyexec), '-u', 'main.py'] + args,
                cwd=str(root_dir),
                env=env or os.environ.copy(),
                stdout=logf,
                stderr=subprocess.STDOUT,
                text=True
            )
            proc.wait()
            rc = proc.returncode
    except Exception as e:
        rc, err = -999, f"{type(e).__name__}: {e}"
    return rc, err

def strip_suffix(load_prefix: str) -> str:
    """..._actor / _critic 등의 접미를 제거한 프리픽스 반환."""
    for suf in ['_actor', '_critic', '_behavior', '_actor_optimizer', '_critic_optimizer', '_behavior_optimizer']:
        if load_prefix.endswith(suf):
            return load_prefix[:-len(suf)]
    return load_prefix

# ----------------------------
# 메인 파이프라인
# ----------------------------
def run_one_env(env_id: str, seed: int, w2_weight: float, lr: float,
                max_steps: int, eval_freq: int, split_ratio: float,
                root_dir: Path, pyexec: Path, resume: bool) -> dict:
    start = time.time()
    split_step = int(round(max_steps * split_ratio))

    # 로그/체크포인트
    logs_root = Path('logs')
    base = logs_root / safe(env_id) / f"w2_{w2_weight}" / f"seed_{seed}"
    ckpt_dir = base / "checkpoints"
    p1_dir = base / "phase1_single_step"
    p2_dir = base / "phase2_two_step"
    p1_dir.mkdir(parents=True, exist_ok=True)
    p2_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 기존 2단계 로그들 중 완료된 것이 있는지 확인 — 완료면 스킵
    if resume:
        existing_p2_logs = list(p2_dir.glob("POGO_two_step_*_*.log"))
        for log_file in existing_p2_logs:
            if phase2_done(log_file):
                print(f"⏭️  RESUME: {env_id} seed={seed} — 완료된 로그 발견 ({log_file.name}), 전체 스킵")
                return {'env': env_id, 'seed': seed, 'overall_status': 'skipped_resume', 'overall_duration_min': 0.0}

    # 1) split 체크포인트 찾기 (없으면 Phase-1로 생성)
    print(f"🔍 DEBUG: max_steps={max_steps}, split_ratio={split_ratio}, split_step={split_step}")
    cp_actor = find_split_checkpoint(ckpt_dir, split_step)
    if not cp_actor:
        print(f"🔧 {env_id} seed={seed}: split({split_step}) 체크포인트 없음 → Phase-1 수행(0→{split_step})")
        p1_log = p2_dir / f"POGO_two_step_{safe(env_id)}_{seed}_{now_str().replace(':','-')}.log"
        # Phase-1: CPU 사용(원하시면 빈 문자열로 두되, GPU가 낫다면 '0')
        env_p1 = os.environ.copy()
        env_p1['CUDA_VISIBLE_DEVICES'] = ''
        rc, err = run_phase(
            pyexec, root_dir,
            args=[
                '--env', env_id,
                '--seed', str(seed),
                '--max_timesteps', str(max_steps),  # 전체 1M으로 설정
                '--eval_freq', str(eval_freq),
                '--w2_weight', str(w2_weight),
                '--lr', str(lr),
                '--split_ratio', str(split_ratio),
                '--checkpoint_dir', str(ckpt_dir),
                '--save_model',
                '--two_step'  # split_step 계산을 위해 필요하면 유지
            ],
            log_path=p1_log,
            env=env_p1
        )
        if rc != 0 or err:
            print(f"❌ Phase-1 실패: rc={rc}, err={err}\n{tail(p1_log, 30)}")
            return {
                'env': env_id, 'seed': seed, 'overall_status': 'failed_phase1',
                'phase1_rc': rc, 'phase1_err': err, 'phase1_log': str(p1_log.resolve())
            }
        # 다시 검색
        cp_actor = find_split_checkpoint(ckpt_dir, split_step)
        if not cp_actor:
            print(f"❌ Phase-1 완료 후에도 체크포인트 없음")
            return {
                'env': env_id, 'seed': seed, 'overall_status': 'failed_no_ckpt',
                'phase1_log': str(p1_log.resolve())
            }
        print(f"✅ 체크포인트 확보: {cp_actor.name}")
    else:
        print(f"✅ 기존 체크포인트 발견: {cp_actor.name}")

    # 2) Phase-2: 체크포인트에서 2단계 (GPU 사용)
    load_prefix = strip_suffix(str(cp_actor))
    p2_log = p2_dir / f"POGO_two_step_{safe(env_id)}_{seed}_{now_str().replace(':','-')}.log"

    # (옵션) 1단계 로그 머리말 복사: 가장 최근 1개만 참조
    latest_p1 = sorted(p1_dir.glob("POGO_single_step_*_*.log"))[-1] if list(p1_dir.glob("POGO_single_step_*_*.log")) else None
    copy_phase1_log_prefix(latest_p1 if latest_p1 else Path(''), p2_log, split_step)

    print(f"🔄 Phase-2 시작: {env_id} seed={seed} — {split_step}→{max_steps}")
    env_p2 = os.environ.copy()
    env_p2['CUDA_VISIBLE_DEVICES'] = '0'
    rc2, err2 = run_phase(
        pyexec, root_dir,
        args=[
            '--env', env_id,
            '--seed', str(seed),
            '--max_timesteps', str(max_steps),
            '--eval_freq', str(eval_freq),
            '--w2_weight', str(w2_weight),
            '--lr', str(lr),
            '--split_ratio', str(split_ratio),
            '--checkpoint_dir', str(ckpt_dir),
            '--save_model',
            '--two_step',
            '--start_mode', 'two_step_only',
            '--load_prefix', load_prefix
        ],
        log_path=p2_log,
        env=env_p2
    )
    if rc2 != 0 or err2:
        print(f"❌ Phase-2 실패: rc={rc2}, err={err2}\n{tail(p2_log, 30)}")
        return {
            'env': env_id, 'seed': seed, 'overall_status': 'failed_phase2',
            'phase2_rc': rc2, 'phase2_err': err2, 'phase2_log': str(p2_log.resolve()),
            'load_prefix': load_prefix
        }

    # 완료 확인(선택)
    if not phase2_done(p2_log):
        print(f"⚠️ Phase-2 완료 마커 미검출(로그 포맷 확인 요망). 그래도 성공으로 표기합니다.")
    dur_min = (time.time() - start) / 60.0
    return {
        'env': env_id,
        'seed': seed,
        'overall_status': 'success',
        'overall_duration_min': round(dur_min, 3),
        'phase2_log': str(p2_log.resolve()),
        'checkpoint_dir': str(ckpt_dir.resolve()),
        'load_prefix': load_prefix
    }

def main():
    ap = ArgumentParser()
    ap.add_argument('--resume', action='store_true', help='완료된 2단계 실험 스킵')
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--root_dir', default='/home/offrl/POGO_Sto')
    ap.add_argument('--pyexec', default='/home/offrl/miniconda3/envs/offrl/bin/python')
    args = ap.parse_args()

    root_dir = Path(args.root_dir)
    pyexec = Path(args.pyexec)
    cfg = load_yaml(Path(args.config))

    common = cfg['common']
    max_steps  = common['max_timesteps']
    eval_freq  = common['eval_freq']
    seeds      = common['seeds']
    split_ratio= common.get('split_ratio', 0.5)

    all_runs = []
    for env_key, datasets in cfg['environments'].items():
        for dataset_key, env_cfg in datasets.items():
            env_id = f"{env_key}-{dataset_key}"
            all_runs.append({
                'env_id': env_id,
                'w2_weight': env_cfg['w2_weight'],
                'lr': env_cfg['learning_rate'],
            })

    print(f"🔬 총 {len(all_runs)*len(seeds)}개 러닝 예정 | resume={args.resume}")
    print(f"📋 순차 실행 모드 (병렬처리 없음)")
    print(f"🔄 Phase-1: CPU 사용 → Phase-2: GPU 사용")
    results = []
    t0 = time.time()

    for seed in seeds:
        print(f"\n🎲 SEED {seed} 시작")
        for e in all_runs:
            print(f"— {e['env_id']} | w2={e['w2_weight']} lr={e['lr']}")
            r = run_one_env(
                env_id=e['env_id'], seed=seed,
                w2_weight=e['w2_weight'], lr=e['lr'],
                max_steps=max_steps, eval_freq=eval_freq,
                split_ratio=split_ratio,
                root_dir=root_dir, pyexec=pyexec,
                resume=args.resume
            )
            results.append(r)
            print(f"✅ {e['env_id']} seed={seed} 완료")

    # 요약 저장
    ts = now_str().replace(':','-')
    out_dir = Path(f"results_{ts}")
    out_dir.mkdir(exist_ok=True)
    write_json(out_dir / "summary.json", results)

    # CSV 저장
    csv = out_dir / "summary.csv"
    with csv.open('w', encoding='utf-8') as f:
        f.write("env,seed,overall_status,overall_duration_min,phase2_log,checkpoint_dir,load_prefix\n")
        for r in results:
            f.write(f"{r.get('env','')},{r.get('seed','')},{r.get('overall_status','')},"
                    f"{r.get('overall_duration_min','')},{r.get('phase2_log','')},"
                    f"{r.get('checkpoint_dir','')},{r.get('load_prefix','')}\n")

    print(f"\n🏁 완료 | 총 소요 { (time.time()-t0)/60:.1f }분 | 결과: {out_dir}")

if __name__ == "__main__":
    main()
