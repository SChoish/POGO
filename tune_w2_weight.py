#!/usr/bin/env python3
"""POGO 원스텝 w2_weight 튜닝 실험"""

import os
import sys
import time
import json
import signal
import yaml
from datetime import datetime
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

def now_str():
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

def safe_str(s: str) -> str:
    """파일명으로 사용할 수 있도록 문자열 정리"""
    return s.replace('/', '_').replace('-', '_')

def run_single_experiment(env_id, seed, w2_weight, learning_rate, max_timesteps, eval_freq, root_dir, pyexec, resume=False):
    """단일 원스텝 실험 실행"""
    overall_start = time.time()
    
    exp_name = f"{env_id}, w2={w2_weight}, lr={learning_rate}, seed={seed}"
    print(f"\n{'='*80}")
    print(f"🚀 [{now_str()}] 원스텝 실험 시작: {exp_name}")
    print(f"{'='*80}")
    
    # 로그 디렉토리 생성
    logs_root = Path('logs')
    base_log_dir = logs_root / safe_str(env_id) / f"w2_{w2_weight}" / f"seed_{seed}"
    base_log_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Resume 체크 (타임스탬프 있는 파일들도 확인)
    if resume:
        # 기존 로그 파일들 찾기 (타임스탬프 있거나 없는 것 모두)
        existing_logs = list(base_log_dir.glob(f"POGO_{safe_str(env_id)}_{seed}*.log"))
        for log_file in existing_logs:
            try:
                with log_file.open('r', encoding='utf-8', errors='replace') as f:
                    if '[FINAL]' in f.read():
                        print(f"⏭️  RESUME: 이미 완료된 실험 스킵")
                        return {
                            'env': env_id,
                            'seed': seed,
                            'w2_weight': w2_weight,
                            'status': 'skipped_resume',
                            'duration_min': 0.0,
                        }
            except Exception:
                pass
    
    # 실험 실행
    print(f"\n📍 [원스텝] {env_id} w2={w2_weight} seed={seed} 시작...")
    exp_start = time.time()
    
    # 타임스탬프 추가
    timestamp = now_str().replace(":", "-")
    log_file = base_log_dir / f"POGO_{safe_str(env_id)}_{seed}_{timestamp}.log"
    
    cmd = [str(pyexec), '-u', 'main.py',
           '--env', env_id,
           '--seed', str(seed),
           '--max_timesteps', str(max_timesteps),
           '--eval_freq', str(eval_freq),
           '--w2_weight', str(w2_weight),
           '--lr', str(learning_rate),
           '--save_model']
    
    # 환경 변수 설정 (CPU만 사용)
    env_vars = os.environ.copy()
    env_vars['CUDA_VISIBLE_DEVICES'] = ''
    
    # 실행
    returncode = None
    exc = None
    
    try:
        with log_file.open('w') as logf:
            proc = subprocess.Popen(
                cmd,
                cwd=str(root_dir),
                env=env_vars,
                stdout=logf,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True
            )
            proc.wait()
            returncode = proc.returncode
    except Exception as e:
        exc = f"{type(e).__name__}: {e}"
        returncode = -999
    
    duration = (time.time() - exp_start) / 60.0
    status = 'success' if (returncode == 0 and exc is None) else 'failed'
    
    status_tag = '✅' if status == 'success' else '❌'
    print(f"{status_tag} [원스텝 완료] {status} ({duration:.1f}분), rc={returncode}")
    
    results = {
        'env': env_id,
        'seed': seed,
        'w2_weight': w2_weight,
        'learning_rate': learning_rate,
        'status': status,
        'duration_min': round(duration, 3),
        'returncode': returncode,
        'log_path': str(log_file.resolve()),
        'error': exc,
        'started_at': overall_start,
        'finished_at': time.time(),
    }
    
    # 요약 저장
    try:
        (base_log_dir / 'experiment_summary.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
    except Exception:
        pass
    
    overall_tag = '✅' if status == 'success' else '❌'
    print(f"\n{overall_tag} [{now_str()}] 원스텝 실험 완료: {exp_name} → {status} (총 {duration:.1f}분)")
    print(f"{'='*80}\n")
    
    return results

def main():
    """메인 실행 함수"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='완료된 실험 스킵')
    parser.add_argument('--parallel', type=int, default=2, help='병렬 실행 개수')
    parser.add_argument('--env', type=str, default=None, help='특정 환경만 실행 (예: hopper-medium-v2)')
    parser.add_argument('--w2_weights', nargs='+', type=float, default=None, help='w2_weight 값들 (예: 0.1 0.2 0.5)')
    args = parser.parse_args()
    
    print("🔬 POGO 원스텝 w2_weight 튜닝 실험 시작")
    print("   - 원스텝만 실행 (0→max_timesteps)")
    print("   - w2_weight 값들 튜닝")
    print("   - config_tuning.yaml에서 파라미터 로드")
    if args.resume:
        print("   - Resume 모드: 완료된 실험 스킵")
    
    # config_tuning.yaml 로드
    config_path = Path('config_tuning.yaml')
    if not config_path.exists():
        print(f"❌ {config_path} 파일을 찾을 수 없습니다")
        return
    
    with config_path.open('r') as f:
        config = yaml.safe_load(f)
    
    # 공통 설정
    common_config = config['common']
    max_timesteps = common_config['max_timesteps']
    eval_freq = common_config['eval_freq']
    seeds = common_config['seeds']
    default_lr = common_config['learning_rate']
    
    # 실험 목록 생성
    all_experiments = []
    
    # w2_weight 값들 설정
    if args.w2_weights:
        w2_weights = args.w2_weights
    else:
        # config에서 기본 w2_weight 값들 가져오기
        w2_weights = config.get('default_w2_weights', [0.01, 0.05, 0.1, 0.2, 0.5, 1.0])
    
    print(f"📊 w2_weight 값들: {w2_weights}")
    
    # 환경별 실험 생성
    for env_id, env_config in config['environments'].items():
        # 특정 환경만 실행하는 경우
        if args.env and args.env != env_id:
            continue
        
        # config에서 learning_rate 가져오기
        lr = env_config.get('learning_rate', default_lr)
        
        print(f"✅ {env_id}: lr={lr}")
        
        # w2_weight별, 시드별 실험 추가
        for w2_weight in w2_weights:
            for seed in seeds:
                all_experiments.append({
                    'env_id': env_id,
                    'seed': seed,
                    'w2_weight': w2_weight,
                    'learning_rate': lr,
                    'max_timesteps': max_timesteps,
                    'eval_freq': eval_freq,
                    'root_dir': Path('/home/offrl/POGO_Sto'),
                    'pyexec': Path('/home/offrl/miniconda3/envs/offrl/bin/python'),
                    'resume': args.resume
                })
    
    # 환경 설정
    root_dir = Path('/home/offrl/POGO_Sto')
    pyexec = Path('/home/offrl/miniconda3/envs/offrl/bin/python')
    
    if not pyexec.exists():
        print(f"❌ Python 실행파일을 찾을 수 없습니다: {pyexec}")
        return
    
    print(f"\n📁 작업 디렉토리: {root_dir}")
    print(f"🐍 Python 실행파일: {pyexec}")
    
    experiments = all_experiments
    
    print(f"📊 총 {len(experiments)}개 실험 예정")
    print(f"⚙️  병렬 실행 (max_workers={args.parallel})")
    
    # 병렬 실행
    results = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        # 모든 작업 제출
        future_to_exp = {
            executor.submit(run_single_experiment, **exp): exp for exp in experiments
        }
        
        # 완료된 작업 처리
        for future in as_completed(future_to_exp):
            exp = future_to_exp[future]
            try:
                result = future.result()
                results.append(result)
                print(f"\n✅ 실험 완료: {exp['env_id']} w2={exp['w2_weight']} seed={exp['seed']}")
            except Exception as exc:
                print(f"❌ 실험 전체 실패: {exp['env_id']} w2={exp['w2_weight']} seed={exp['seed']} → {exc}")
                results.append({
                    'env': exp['env_id'],
                    'seed': exp['seed'],
                    'w2_weight': exp['w2_weight'],
                    'status': 'exception',
                    'error': str(exc)
                })
    
    # 최종 요약
    total_time = time.time() - start_time
    
    # 결과 CSV 저장
    timestamp = now_str().replace(":", "-")
    results_dir = Path(f'w2_tuning_results_{timestamp}')
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / f'w2_tuning_results_{timestamp}.csv'
    
    with csv_path.open('w') as f:
        f.write('env,seed,w2_weight,learning_rate,status,duration_min,log_path\n')
        for result in results:
            f.write(f"{result.get('env', 'unknown')},"
                   f"{result.get('seed', 'unknown')},"
                   f"{result.get('w2_weight', 'unknown')},"
                   f"{result.get('learning_rate', 'unknown')},"
                   f"{result.get('status', 'unknown')},"
                   f"{result.get('duration_min', 'unknown')},"
                   f"{result.get('log_path', 'unknown')}\n")
    
    print(f"\n🏁 w2_weight 튜닝 완료!")
    print(f"⏱️  총 소요시간: {total_time/3600:.1f}시간")
    print(f"📁 결과 디렉토리: {results_dir}")
    print(f"📄 결과 CSV: {csv_path}")
    
    # 성공/실패 통계
    success_count = sum(1 for r in results if r.get('status') == 'success')
    print(f"✅ 성공: {success_count}/{len(results)}")
    
    # w2_weight별 성공률
    w2_stats = {}
    for result in results:
        w2 = result.get('w2_weight', 'unknown')
        if w2 not in w2_stats:
            w2_stats[w2] = {'total': 0, 'success': 0}
        w2_stats[w2]['total'] += 1
        if result.get('status') == 'success':
            w2_stats[w2]['success'] += 1
    
    print(f"\n📊 w2_weight별 성공률:")
    for w2, stats in sorted(w2_stats.items()):
        success_rate = stats['success'] / stats['total'] * 100
        print(f"  w2={w2}: {stats['success']}/{stats['total']} ({success_rate:.1f}%)")

if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))
    main()
