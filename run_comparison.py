#!/usr/bin/env python3
"""POGO 2단계 NatGrad 실험: config.yaml 기반 (모든 환경)"""

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

def load_last_lines(log_file: Path, n: int = 3) -> str:
    """로그 파일의 마지막 n줄 반환"""
    try:
        if log_file.exists() and log_file.stat().st_size > 0:
            with log_file.open('r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
                return ''.join(lines[-n:]).strip()
        return ""
    except Exception:
        return ""

def run_two_phase_experiment(env_id, seed, w2_weight, entropy_weight, learning_rate, use_natgrad, max_timesteps, eval_freq, split_ratio, root_dir, pyexec, resume=False):
    """2단계 실험 실행: 1) single_step 0→max, 2) split_point 체크포인트 로드 후 two_step split_point→max"""
    overall_start = time.time()
    
    exp_name = f"{env_id}, w2={w2_weight}, ent={entropy_weight}, lr={learning_rate}, seed={seed}, natgrad={use_natgrad}"
    print(f"\n{'='*80}")
    print(f"🚀 [{now_str()}] 2단계 실험 시작: {exp_name}")
    print(f"{'='*80}")
    
    # 로그 디렉토리 생성
    logs_root = Path('logs')
    natgrad_suffix = "_natgrad" if use_natgrad else ""
    base_log_dir = logs_root / safe_str(env_id) / f"w2_{w2_weight}_ent_{entropy_weight}" / f"seed_{seed}{natgrad_suffix}"
    base_log_dir.mkdir(parents=True, exist_ok=True)
    
    # 체크포인트 디렉토리
    checkpoint_dir = base_log_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    results = {}
    skip_phase1 = False
    
    # Resume 체크
    if resume:
        phase1_log_dir = base_log_dir / "phase1_single_step"
        # 기존 로그 파일 찾기 (타임스탬프 있거나 없는 것 모두)
        phase1_log = None
        for log_file in phase1_log_dir.glob(f"POGO_single_step_{safe_str(env_id)}_{seed}*.log"):
            phase1_log = log_file
            break
        
        phase2_log_dir = base_log_dir / "phase2_two_step"
        # 기존 로그 파일 찾기 (타임스탬프 있거나 없는 것 모두)
        phase2_log = None
        for log_file in phase2_log_dir.glob(f"POGO_two_step_{safe_str(env_id)}_{seed}*.log"):
            phase2_log = log_file
            break
        
        # 2단계 완료 체크
        if phase2_log and phase2_log.exists():
            try:
                with phase2_log.open('r', encoding='utf-8', errors='replace') as f:
                    if '[FINAL]' in f.read():
                        print(f"⏭️  RESUME: 이미 완료된 실험 스킵 (2단계 완료)")
                        return {
                            'env': env_id,
                            'seed': seed,
                            'natgrad': use_natgrad,
                            'overall_status': 'skipped_resume',
                            'overall_duration_min': 0.0,
                        }
            except Exception:
                pass
        
        # 1단계 완료 체크
        if phase1_log and phase1_log.exists():
            try:
                with phase1_log.open('r', encoding='utf-8', errors='replace') as f:
                    if '[FINAL]' in f.read():
                        print(f"⏭️  RESUME: 1단계 완료됨, 2단계부터 시작")
                        skip_phase1 = True
            except Exception:
                pass
    
    # =============================================================================
    # 1단계: single_step으로 1M까지 학습 (중간에 500k 체크포인트 저장)
    # =============================================================================
    if not skip_phase1:
        print(f"\n📍 [1단계] single_step 0→1M 시작...")
        phase1_start = time.time()
        
        phase1_log_dir = base_log_dir / "phase1_single_step"
        phase1_log_dir.mkdir(exist_ok=True)
        timestamp = now_str().replace(":", "-")
        phase1_log = phase1_log_dir / f"POGO_single_step_{safe_str(env_id)}_{seed}_{timestamp}.log"
        
        cmd_phase1 = [str(pyexec), '-u', 'main_two.py',
                  '--env', env_id,
                  '--seed', str(seed),
                  '--max_timesteps', str(max_timesteps),  # 1M
                  '--eval_freq', str(eval_freq),
                  '--w2_weight', str(w2_weight),
                  '--entropy_weight', str(entropy_weight),
                  '--lr', str(learning_rate),
                  '--split_ratio', str(split_ratio),
                  '--checkpoint_dir', str(checkpoint_dir),
                  '--save_model']
    
        if use_natgrad:
            cmd_phase1.append('--use_natgrad')
        
        # 환경 변수 설정 (CPU만 사용)
        env_vars = os.environ.copy()
        env_vars['CUDA_VISIBLE_DEVICES'] = ''
        
        # 1단계 실행
        phase1_returncode = None
        phase1_exc = None
        
        try:
            with phase1_log.open('w') as logf:
                proc = subprocess.Popen(
                    cmd_phase1,
                    cwd=str(root_dir),
                    env=env_vars,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True
                )
                proc.wait()
                phase1_returncode = proc.returncode
        except Exception as e:
            phase1_exc = f"{type(e).__name__}: {e}"
            phase1_returncode = -999
        
        phase1_duration = (time.time() - phase1_start) / 60.0
        phase1_status = 'success' if (phase1_returncode == 0 and phase1_exc is None) else 'failed'
        
        phase1_tag = '✅' if phase1_status == 'success' else '❌'
        print(f"{phase1_tag} [1단계 완료] {phase1_status} ({phase1_duration:.1f}분), rc={phase1_returncode}")
        
        results['phase1'] = {
            'status': phase1_status,
            'duration_min': round(phase1_duration, 3),
            'returncode': phase1_returncode,
            'log_path': str(phase1_log.resolve()),
            'error': phase1_exc,
        }
        
        # 1단계 실패 시 2단계 스킵
        if phase1_status != 'success':
            print(f"❌ 1단계 실패로 2단계 스킵")
            results['phase2'] = {'status': 'skipped', 'reason': 'phase1_failed'}
            results['overall_status'] = 'failed'
            results['overall_duration_min'] = round(phase1_duration, 3)
            return results
    else:
        print(f"⏭️  1단계 스킵 (이미 완료)")
        # 1단계 로그에서 정보 읽기
        phase1_log_dir = base_log_dir / "phase1_single_step"
        phase1_log = phase1_log_dir / f"POGO_single_step_{safe_str(env_id)}_{seed}.log"
        results['phase1'] = {
            'status': 'skipped_resume',
            'duration_min': 0.0,
            'log_path': str(phase1_log.resolve()),
        }
        
        # env_vars 정의 (Phase-2에서 사용하기 위해)
        env_vars = os.environ.copy()
        env_vars['CUDA_VISIBLE_DEVICES'] = ''
    
    # 중간 체크포인트 찾기 (split_ratio 기준)
    split_timestep = int(round(max_timesteps * split_ratio))
    checkpoint_split = None
    
    # 더 유연한 체크포인트 검색
    # 1) 정확한 split_timestep 체크포인트 찾기
    for cp_file in checkpoint_dir.glob(f"*_mid_{split_timestep}_*"):
        checkpoint_split = cp_file
        break
    
    # 2) 정확한 것이 없으면 가장 가까운 체크포인트 찾기
    if not checkpoint_split or not checkpoint_split.exists():
        print(f"⚠️  정확한 {split_timestep} 체크포인트를 찾을 수 없음, 가장 가까운 것 검색...")
        closest_timestep = None
        min_diff = float('inf')
        
        for cp_file in checkpoint_dir.glob("*_mid_*_actor"):
            try:
                # 파일명에서 timestep 추출
                parts = cp_file.stem.split('_')
                for i, part in enumerate(parts):
                    if part == 'mid' and i + 1 < len(parts):
                        timestep = int(parts[i + 1])
                        diff = abs(timestep - split_timestep)
                        if diff < min_diff:
                            min_diff = diff
                            closest_timestep = timestep
                            checkpoint_split = cp_file
                        break
            except (ValueError, IndexError):
                continue
        
        if checkpoint_split and checkpoint_split.exists():
            print(f"✅ 가장 가까운 체크포인트 발견: {checkpoint_split.name} (차이: {min_diff})")
        else:
            print(f"❌ 체크포인트를 전혀 찾을 수 없음: {checkpoint_dir}")
            results['phase2'] = {'status': 'skipped', 'reason': 'checkpoint_not_found'}
            results['overall_status'] = 'failed'
            results['overall_duration_min'] = round(phase1_duration, 3)
            return results
    
    print(f"✅ {split_timestep} 체크포인트 발견: {checkpoint_split.name}")
    
    # =============================================================================
    # 2단계: two_step으로 split→max 학습
    # =============================================================================
    print(f"\n📍 [2단계] two_step {split_timestep}→{max_timesteps} 시작...")
    time.sleep(5)  # 잠시 대기
    phase2_start = time.time()
    
    phase2_log_dir = base_log_dir / "phase2_two_step"
    phase2_log_dir.mkdir(exist_ok=True)
    timestamp = now_str().replace(":", "-")
    phase2_log = phase2_log_dir / f"POGO_two_step_{safe_str(env_id)}_{seed}_{timestamp}.log"
    
    # 1단계 로그의 split_timestep까지 부분을 2단계 로그로 복사
    print(f"📋 1단계 로그 {split_timestep}까지 복사 중...")
    try:
        if phase1_log.exists():
            with phase1_log.open('r', encoding='utf-8', errors='replace') as src:
                lines = src.readlines()
                # split_timestep까지만 복사
                with phase2_log.open('w', encoding='utf-8') as dst:
                    for line in lines:
                        dst.write(line)
                        # "Time steps: {split_timestep}" 또는 유사한 패턴을 찾으면 중단
                        if f'Time steps: {split_timestep}' in line or f'timesteps: {split_timestep}' in line:
                            break
            print(f"✅ {split_timestep}까지 로그 복사 완료")
        else:
            print(f"⚠️  1단계 로그를 찾을 수 없음, 2단계 로그를 새로 시작")
            phase2_log.touch()
    except Exception as e:
        print(f"⚠️  로그 복사 실패: {e}, 2단계 로그를 새로 시작")
        phase2_log.touch()
    
    # load_prefix 생성 (확장자 제거)
    load_prefix = str(checkpoint_split)
    for suffix in ['_actor', '_critic', '_behavior', '_actor_optimizer', '_critic_optimizer', '_behavior_optimizer']:
        if load_prefix.endswith(suffix):
            load_prefix = load_prefix[:-len(suffix)]
            break
    
    cmd_phase2 = [str(pyexec), '-u', 'main_two.py',
                  '--env', env_id,
                  '--seed', str(seed),
                  '--max_timesteps', str(max_timesteps),  # 1M
                  '--eval_freq', str(eval_freq),
                  '--w2_weight', str(w2_weight),
                  '--entropy_weight', str(entropy_weight),
                  '--lr', str(learning_rate),
                  '--split_ratio', str(split_ratio),
                  '--checkpoint_dir', str(checkpoint_dir),
                  '--save_model',
                  '--two_step',
                  '--start_mode', 'load',
                  '--load_prefix', load_prefix]
    
    if use_natgrad:
        cmd_phase2.append('--use_natgrad')
    
    # 2단계 실행 (로그는 append 모드로)
    phase2_returncode = None
    phase2_exc = None
    
    try:
        with phase2_log.open('a') as logf:  # append 모드로 변경
            proc = subprocess.Popen(
                cmd_phase2,
                cwd=str(root_dir),
                env=env_vars,
                stdout=logf,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True
            )
            proc.wait()
            phase2_returncode = proc.returncode
    except Exception as e:
        phase2_exc = f"{type(e).__name__}: {e}"
        phase2_returncode = -999
    
    phase2_duration = (time.time() - phase2_start) / 60.0
    phase2_status = 'success' if (phase2_returncode == 0 and phase2_exc is None) else 'failed'
    
    phase2_tag = '✅' if phase2_status == 'success' else '❌'
    print(f"{phase2_tag} [2단계 완료] {phase2_status} ({phase2_duration:.1f}분), rc={phase2_returncode}")
    
    results['phase2'] = {
        'status': phase2_status,
        'duration_min': round(phase2_duration, 3),
        'returncode': phase2_returncode,
        'log_path': str(phase2_log.resolve()),
        'load_prefix': load_prefix,
        'error': phase2_exc,
    }
    
    # 전체 결과
    overall_duration = (time.time() - overall_start) / 60.0
    overall_status = 'success' if (phase1_status == 'success' and phase2_status == 'success') else 'failed'
    
    results.update({
        'env': env_id,
        'seed': seed,
        'w2_weight': w2_weight,
        'entropy_weight': entropy_weight,
        'natgrad': use_natgrad,
        'overall_status': overall_status,
        'overall_duration_min': round(overall_duration, 3),
        'checkpoint_dir': str(checkpoint_dir.resolve()),
        'started_at': overall_start,
        'finished_at': time.time(),
    })
    
    
    # 요약 저장
    try:
        (base_log_dir / 'experiment_summary.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
    except Exception:
        pass
    
    overall_tag = '✅' if overall_status == 'success' else '❌'
    print(f"\n{overall_tag} [{now_str()}] 2단계 실험 완료: {exp_name} → {overall_status} (총 {overall_duration:.1f}분)")
    print(f"{'='*80}\n")
    
    return results

def main():
    """메인 실행 함수"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='완료된 실험 스킵')
    parser.add_argument('--parallel', type=int, default=2, help='병렬 실행 개수')
    args = parser.parse_args()
    
    print("🔬 POGO 2단계 NatGrad 실험 시작 (모든 환경)")
    print("   - 1단계: single_step 0→max_timesteps")
    print("   - 2단계: split_point 체크포인트 로드 후 two_step split_point→max")
    print("   - split_point는 split_ratio에 의해 결정")
    print("   - config.yaml에서 파라미터 로드")
    if args.resume:
        print("   - Resume 모드: 완료된 실험 스킵")
    
    # config.yaml 로드
    config_path = Path('config.yaml')
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
    use_natgrad = common_config['use_natgrad']
    split_ratio = common_config.get('split_ratio', 0.5)
    
    # 모든 환경에 대한 실험 목록 생성
    all_experiments = []
    exclude_env = "halfcheetah-medium-expert"
    
    for env_key, datasets in config['environments'].items():
        for dataset_key, env_config in datasets.items():
            env_id = f"{env_key}-{dataset_key}"
            
            # halfcheetah-medium-expert 제외
            if env_id == exclude_env:
                print(f"⏭️  {env_id} 제외")
                continue
            
            w2_weight = env_config['w2_weight']
            entropy_weight = env_config['entropy_weight']
            learning_rate = env_config['learning_rate']
            
            print(f"✅ {env_id}: w2={w2_weight}, ent={entropy_weight}, lr={learning_rate}")
            
            # 시드별 실험 추가
            for seed in seeds:
                all_experiments.append({
                    'env_id': env_id,
                    'seed': seed,
                    'w2_weight': w2_weight,
                    'entropy_weight': entropy_weight,
                    'learning_rate': learning_rate,
                    'use_natgrad': use_natgrad,
                    'max_timesteps': max_timesteps,
                    'eval_freq': eval_freq,
                    'split_ratio': split_ratio,
                    'root_dir': Path('/home/offrl/POGO'),
                    'pyexec': Path('/home/offrl/miniconda3/envs/offrl/bin/python'),
                    'resume': args.resume
                })
    
    # 환경 설정
    root_dir = Path('/home/offrl/POGO')
    pyexec = Path('/home/offrl/miniconda3/envs/offrl/bin/python')
    
    if not pyexec.exists():
        print(f"❌ Python 실행파일을 찾을 수 없습니다: {pyexec}")
        return
    
    print(f"\n📁 작업 디렉토리: {root_dir}")
    print(f"🐍 Python 실행파일: {pyexec}")
    
    experiments = all_experiments
    
    print(f"📊 총 {len(experiments)}개 실험 예정 (각 실험은 2단계로 구성)")
    print(f"⚙️  병렬 실행 (max_workers={args.parallel})")
    
    # 병렬 실행
    results = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        # 모든 작업 제출
        future_to_exp = {
            executor.submit(run_two_phase_experiment, **exp): exp for exp in experiments
        }
        
        # 완료된 작업 처리
        for future in as_completed(future_to_exp):
            exp = future_to_exp[future]
            try:
                result = future.result()
                results.append(result)
                print(f"\n✅ 실험 완료: seed={exp['seed']}")
            except Exception as exc:
                print(f"❌ 실험 전체 실패: seed={exp['seed']} → {exc}")
                results.append({
                    'env': exp['env_id'],
                    'seed': exp['seed'],
                    'natgrad': exp['use_natgrad'],
                    'overall_status': 'exception',
                    'error': str(exc)
                })
    
    # 최종 요약
    total_time = time.time() - start_time
    
    # 결과 CSV 저장 (시간 포함 디렉토리)
    timestamp = now_str().replace(":", "-")
    results_dir = Path(f'results_{timestamp}')
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / f'comparison_results_{timestamp}.csv'
    
    with csv_path.open('a') as f:
        f.write('env,seed,natgrad,w2_weight,entropy_weight,overall_status,overall_duration_min,phase1_status,phase1_duration_min,phase2_status,phase2_duration_min,phase1_log,phase2_log,checkpoint_dir\n')
        for result in results:
            phase1 = result.get('phase1', {})
            phase2 = result.get('phase2', {})
            f.write(f"{result.get('env', 'unknown')},"
                   f"{result.get('seed', 'unknown')},"
                   f"{result.get('natgrad', 'unknown')},"
                   f"{result.get('w2_weight', 'unknown')},"
                   f"{result.get('entropy_weight', 'unknown')},"
                   f"{result.get('overall_status', 'unknown')},"
                   f"{result.get('overall_duration_min', 'unknown')},"
                   f"{phase1.get('status', 'unknown')},"
                   f"{phase1.get('duration_min', 'unknown')},"
                   f"{phase2.get('status', 'unknown')},"
                   f"{phase2.get('duration_min', 'unknown')},"
                   f"{phase1.get('log_path', 'unknown')},"
                   f"{phase2.get('log_path', 'unknown')},"
                   f"{result.get('checkpoint_dir', 'unknown')}\n")
    
    print(f"\n🏁 실험 완료!")
    print(f"⏱️  총 소요시간: {total_time/3600:.1f}시간")
    print(f"📁 결과 디렉토리: {results_dir}")
    print(f"📄 결과 CSV: {csv_path}")
    
    # 성공/실패 통계
    success_count = sum(1 for r in results if r.get('overall_status') == 'success')
    print(f"✅ 성공: {success_count}/{len(results)}")

if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))
    main()

