#!/usr/bin/env python3
"""시드 0 결과 직접 계산"""

import re
from pathlib import Path

def extract_final_evaluation(log_file):
    """로그 파일에서 Final Evaluation 결과 추출"""
    det_scores = []
    stoch_scores = []
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
            
        # Final Evaluation 섹션 찾기
        in_final_section = False
        for line in lines:
            if '======== Final Evaluation' in line:
                in_final_section = True
                continue
            elif '[FINAL]' in line:
                break
                
            if in_final_section and 'Deterministic:' in line and 'D4RL score:' in line:
                # Deterministic: 2405.806, D4RL score: 74.544 형태에서 점수 추출
                det_match = re.search(r'D4RL score: ([\d.-]+)', line)
                if det_match:
                    det_scores.append(float(det_match.group(1)))
                    
            if in_final_section and 'Stochastic:' in line and 'D4RL score:' in line:
                # Stochastic: 2366.277, D4RL score: 73.329 형태에서 점수 추출
                stoch_match = re.search(r'D4RL score: ([\d.-]+)', line)
                if stoch_match:
                    stoch_scores.append(float(stoch_match.group(1)))
        
        return det_scores, stoch_scores
        
    except Exception as e:
        print(f"Error reading {log_file}: {e}")
        return None, None

def main():
    """메인 함수"""
    logs_dir = Path('/home/offrl/POGO_Sto/logs')
    
    # 시드 0 환경들 (모든 환경 포함)
    environments = [
        'hopper_medium',
        'hopper_medium_replay', 
        'hopper_medium_expert',
        'halfcheetah_medium',
        'halfcheetah_medium_replay',
        'halfcheetah_medium_expert',
        'walker2d_medium',
        'walker2d_medium_replay',
        'walker2d_medium_expert',
        'antmaze_umaze_v2',
        'antmaze_umaze_diverse_v2',
        'antmaze_medium_play_v2',
        'antmaze_medium_diverse_v2',
        'antmaze_large_play_v2',
        'antmaze_large_diverse_v2'
    ]
    
    print("📊 시드 0 최종 결과 직접 계산")
    print("=" * 80)
    print(f"{'환경':<25} {'Deterministic':<15} {'Stochastic':<15} {'차이':<10} {'차이율':<10}")
    print("-" * 80)
    
    for env in environments:
        log_path = logs_dir / env / 'w2_0.2' / 'seed_0' / 'phase1_single_step'
        
        # 로그 파일 찾기
        log_files = list(log_path.glob('*.log'))
        if not log_files:
            print(f"{env:<25} {'로그 없음':<15} {'로그 없음':<15} {'-':<10} {'-':<10}")
            continue
            
        # 가장 최근 로그 파일 사용
        log_file = max(log_files, key=lambda x: x.stat().st_mtime)
        
        det_scores, stoch_scores = extract_final_evaluation(log_file)
        
        if det_scores is None or stoch_scores is None or len(det_scores) == 0:
            print(f"{env:<25} {'파싱 실패':<15} {'파싱 실패':<15} {'-':<10} {'-':<10}")
            continue
            
        # 평균 계산
        det_mean = sum(det_scores) / len(det_scores)
        stoch_mean = sum(stoch_scores) / len(stoch_scores)
        
        diff = det_mean - stoch_mean
        diff_pct = (diff / stoch_mean * 100) if stoch_mean != 0 else 0
        
        print(f"{env:<25} {det_mean:<15.3f} {stoch_mean:<15.3f} {diff:<10.3f} {diff_pct:<10.1f}%")
        
        # 개별 점수도 출력
        print(f"  Det scores: {[f'{s:.1f}' for s in det_scores]}")
        print(f"  Stoch scores: {[f'{s:.1f}' for s in stoch_scores]}")
        print()

if __name__ == '__main__':
    main()
