#!/usr/bin/env python3
"""
POGO: Single-step / Two-step 실행 메인 (단순화)
- 사용법은 질문의 1~4번 시나리오와 완전히 호환됩니다.
"""

import os
import json
import argparse
import numpy as np
import torch
import gym
import d4rl

import utils
from agent import POGO, POGO_Refine


# ---------------------------
# 유틸
# ---------------------------
def set_global_seed(env, seed: int):
    """Gym 버전 호환 시드 설정."""
    try:
        env.reset(seed=seed)
    except TypeError:
        # old gym
        env.seed(seed)
    try:
        env.action_space.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    np.random.seed(seed)


def make_env(env_name: str, seed: int):
    env = gym.make(env_name)
    set_global_seed(env, seed)
    return env


def normalize(replay_buffer, enable: bool):
    if enable:
        mean, std = replay_buffer.normalize_states()
    else:
        sdim = replay_buffer.state.shape[1]
        mean = np.zeros((1, sdim), dtype=np.float32)
        std  = np.ones((1, sdim), dtype=np.float32)
    return mean, std


# ---------------------------
# 평가 루틴 (환경 1개 재사용)
# ---------------------------
@torch.no_grad()
def eval_policy(policy, eval_env, mean, std, base_seed, eval_episodes=10, deterministic=True):
    """정책 평가: deterministic과 stochastic 모두 평가하고 기존 로그 형식으로 출력"""
    
    # Deterministic 평가
    det_total = 0.0
    for ep in range(eval_episodes):
        try:
            eval_env.reset(seed=base_seed + ep)
            state = eval_env.state if hasattr(eval_env, "state") else eval_env.reset()[0]
        except Exception:
            state = eval_env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            nstate = (np.asarray(state).reshape(1, -1) - mean) / std
            action = policy.select_action(nstate, deterministic=True)
            step_out = eval_env.step(action)
            if len(step_out) == 5:
                state, reward, terminated, truncated, _ = step_out
                done = terminated or truncated
            else:
                state, reward, done, _ = step_out
            ep_ret += reward
        det_total += ep_ret
    
    # Stochastic 평가
    stoch_total = 0.0
    for ep in range(eval_episodes):
        try:
            eval_env.reset(seed=base_seed + ep + 1000)  # 다른 시드 사용
            state = eval_env.state if hasattr(eval_env, "state") else eval_env.reset()[0]
        except Exception:
            state = eval_env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            nstate = (np.asarray(state).reshape(1, -1) - mean) / std
            action = policy.select_action(nstate, deterministic=False)
            step_out = eval_env.step(action)
            if len(step_out) == 5:
                state, reward, terminated, truncated, _ = step_out
                done = terminated or truncated
            else:
                state, reward, done, _ = step_out
            ep_ret += reward
        stoch_total += ep_ret
    
    # 결과 계산 및 출력
    det_avg = det_total / eval_episodes
    stoch_avg = stoch_total / eval_episodes
    det_score = eval_env.get_normalized_score(det_avg) * 100.0
    stoch_score = eval_env.get_normalized_score(stoch_avg) * 100.0
    
    # 기존 로그 형식으로 출력
    print("---------------------------------------")
    print("Evaluation over 10 episodes:")
    print(f"  Deterministic: {det_avg:.3f}, D4RL score: {det_score:.3f}")
    print(f"  Stochastic: {stoch_avg:.3f}, D4RL score: {stoch_score:.3f}")
    print("---------------------------------------")
    
    # deterministic 점수를 메인으로 반환 (기존 호환성)
    return det_score


def final_evaluation(policy, env_name, seed, mean, std, runs=5, episodes=10):
    eval_env = make_env(env_name, seed + 10_000)  # 훈련 시드와 멀리 떨어뜨림

    det_scores, stoch_scores = [], []
    for r in range(runs):
        det = eval_policy(
            policy, eval_env, mean, std,
            base_seed=1000 + 100 * r, eval_episodes=episodes, deterministic=True
        )
        st = eval_policy(
            policy, eval_env, mean, std,
            base_seed=2000 + 100 * r, eval_episodes=episodes, deterministic=False
        )
        det_scores.append(det)
        stoch_scores.append(st)

    det_scores = np.array(det_scores, dtype=np.float32)
    stoch_scores = np.array(stoch_scores, dtype=np.float32)

    print("======== Final Evaluation (trained weights) ========")
    print(f"[FINAL] Deterministic: mean={det_scores.mean():.3f}, std={det_scores.std():.3f} over {runs}x{episodes}")
    print(f"[FINAL] Stochastic:   mean={stoch_scores.mean():.3f}, std={stoch_scores.std():.3f} over {runs}x{episodes}")
    return det_scores, stoch_scores


# ---------------------------
# 체크포인트 유틸
# ---------------------------
def save_checkpoint(agent, ckpt_dir: str, prefix: str, step: int, phase: str, extra_meta=None):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, prefix)
    agent.save(path)
    meta = {
        "step": int(step),
        "phase": phase,
        "total_it": int(getattr(agent, "total_it", step)),
        "checkpoint_name": prefix,
    }
    if extra_meta:
        meta.update(extra_meta)
    with open(path + "_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[CKPT] Saved: {path}_* (step={step}, phase={phase})")


def load_checkpoint_into_AgentA(agentA, load_prefix: str):
    print(f"[LOAD] Loading checkpoint from: {load_prefix}")
    agentA.load(load_prefix)
    meta_path = load_prefix + "_meta.json"
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        total_it = metadata.get("total_it")
        if total_it is None:
            total_it = metadata.get("step", 0)
        if total_it is not None:
            agentA.total_it = int(total_it)
        print(f"[LOAD] Done. (step={metadata.get('step', 'unknown')}, phase={metadata.get('phase', 'unknown')})")
    else:
        print("[LOAD] Metadata not found alongside checkpoint. Assuming fresh start from weights only.")
    return metadata


# ---------------------------
# Phase-1 (POGO)
# ---------------------------
def train_phase1(agentA, env_name, seed, replay_buffer, mean, std,
                 max_steps, eval_freq, save_model, file_name, ckpt_dir,
                 split_ratio, start_step=0):
    eval_env = make_env(env_name, seed + 1234)
    best_eval = -np.inf
    eval_file = f"./results/{file_name}.npy"
    if start_step > 0 and os.path.exists(eval_file):
        evaluations = list(np.load(eval_file))
    else:
        evaluations = []

    split_step = int(round(max_steps * split_ratio))
    midpoint_saved = start_step >= split_step

    print(f"🚀 Phase-1 시작: {start_step} ~ {split_step-1} steps (POGO)")
    for global_step in range(start_step, split_step):
        metrics = agentA.train(replay_buffer, batch_size=256)

        if (global_step + 1) % eval_freq == 0:
            print(f"[Phase-1] Time steps: {global_step + 1}")
            d4rl_score = eval_policy(agentA, eval_env, mean, std, base_seed=100, eval_episodes=10, deterministic=True)
            evaluations.append(d4rl_score)
            np.save(eval_file, evaluations)
            if save_model:
                agentA.save(f"./models/{file_name}")
                if d4rl_score > best_eval:
                    best_eval = d4rl_score
                    agentA.save(f"./models/{file_name}_best")

        if not midpoint_saved and (global_step + 1) == split_step:
            mid_name = f"{file_name}_mid_{global_step + 1}"
            save_checkpoint(
                agentA,
                ckpt_dir,
                mid_name,
                step=global_step + 1,
                phase="phase1",
                extra_meta={
                    "file_name": file_name,
                    "max_timesteps": max_steps,
                    "split_ratio": split_ratio,
                    "env": env_name,
                    "seed": seed,
                },
            )
            midpoint_saved = True

    return agentA  # 최종 상태 반환


# ---------------------------
# Phase-2 (POGO_Refine)
# ---------------------------
def init_agentB_from_A(agentA, args, state_dim, action_dim, max_action):
    agentB = POGO_Refine(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        discount=args.discount,
        tau=args.tau,
        policy_noise=args.policy_noise * max_action,
        noise_clip=args.noise_clip * max_action,
        policy_freq=args.policy_freq,
        alpha=args.alpha,
        w2_weight=args.w2_weight,
        lr=args.lr,
        freeze_critic=args.freeze_critic_mode,
        copy_actor=True,
    )
    # A → B 초기화
    agentB.actor.load_state_dict(agentA.actor.state_dict())
    agentB.actor_target.load_state_dict(agentA.actor_target.state_dict())
    agentB.critic.load_state_dict(agentA.critic.state_dict())
    agentB.critic_target.load_state_dict(agentA.critic_target.state_dict())
    agentB.behavior_policy.load_state_dict(agentA.actor.state_dict())
    return agentB


def train_phase2(agentB, env_name, seed, replay_buffer, mean, std,
                 start_step, max_steps, eval_freq, save_model, file_name):
    eval_env = make_env(env_name, seed + 2345)
    best_eval = -np.inf
    eval_file = f"./results/{file_name}.npy"
    evaluations = list(np.load(eval_file)) if os.path.exists(eval_file) else []

    remaining = max_steps - start_step
    if remaining <= 0:
        print("ℹ️  Phase-2 스킵: 남은 스텝 없음")
        return agentB

    print(f"🔄 Phase-2 시작: {start_step} ~ {max_steps-1} steps (POGO_Refine)")
    for t2 in range(remaining):
        global_step = start_step + t2
        metrics = agentB.train(replay_buffer, batch_size=256)

        if (global_step + 1) % eval_freq == 0:
            print(f"[Phase-2] Time steps: {global_step + 1}")
            d4rl_score = eval_policy(agentB, eval_env, mean, std, base_seed=200, eval_episodes=10, deterministic=True)
            evaluations.append(d4rl_score)
            np.save(eval_file, evaluations)
            if save_model:
                agentB.save(f"./models/{file_name}")
                if d4rl_score > best_eval:
                    best_eval = d4rl_score
                    agentB.save(f"./models/{file_name}_best")

    return agentB


# ---------------------------
# main
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    # Experiment
    p.add_argument("--env", default="hopper-medium-v2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval_freq", type=int, default=5000)
    p.add_argument("--max_timesteps", type=int, default=1_000_000)
    p.add_argument("--save_model", action="store_true")
    p.add_argument("--load_model", default="")           # "./models/<name>"에서 로드
    p.add_argument("--normalize", default=True, type=bool)

    # Shared/TD3
    p.add_argument("--batch_size", type=int, default=256)  # agent 내부에서 사용
    p.add_argument("--discount", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--policy_noise", type=float, default=0.2)
    p.add_argument("--noise_clip", type=float, default=0.5)
    p.add_argument("--policy_freq", type=int, default=2)

    # POGO
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--w2_weight", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=3e-4)

    # Final eval
    p.add_argument("--final_eval_runs", type=int, default=5)
    p.add_argument("--final_eval_episodes", type=int, default=10)

    # Two-step control
    step_mode = p.add_mutually_exclusive_group()
    step_mode.add_argument(
        "--two_step",
        dest="two_step",
        action="store_true",
        help="Run Phase-1 + Phase-2 sequentially (default).",
    )
    step_mode.add_argument(
        "--one_step_only",
        dest="two_step",
        action="store_false",
        help="Run only Phase-1 without launching Phase-2.",
    )
    p.set_defaults(two_step=True)
    p.add_argument("--split_ratio", type=float, default=0.5)
    p.add_argument("--freeze_critic_mode", default=True, type=bool)
    p.add_argument("--checkpoint_dir", type=str, default="./logs/checkpoints")

    p.add_argument("--start_mode", choices=["scratch", "load", "two_step_only"], default="scratch",
                   help="scratch: 0→max 전부 진행 / load: 체크포인트에서 이어서 계속 / two_step_only: 원스텝 체크포인트에서 Phase-2만")
    p.add_argument("--load_prefix", type=str, default="",
                   help="체크포인트 prefix (확장자 없이). 예: ./logs/checkpoints/POGO_hopper-medium-v2_0_mid_500000")

    return p.parse_args()


def main():
    args = parse_args()

    # 파일 이름 기본값 (resume 시 덮어쓰기)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_file_name = f"POGO_{args.env}_{args.seed}_{timestamp}"
    file_name = default_file_name
    os.makedirs("./results", exist_ok=True)
    if args.save_model:
        os.makedirs("./models", exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 환경 및 데이터셋
    env = make_env(args.env, args.seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # 데이터 적재
    rb = utils.ReplayBuffer(state_dim, action_dim)
    rb.convert_D4RL(d4rl.qlearning_dataset(env))
    mean, std = normalize(rb, args.normalize)

    # Agent A (Phase-1)
    agentA = POGO(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        discount=args.discount,
        tau=args.tau,
        policy_noise=args.policy_noise * max_action,
        noise_clip=args.noise_clip * max_action,
        policy_freq=args.policy_freq,
        alpha=args.alpha,
        w2_weight=args.w2_weight,
        lr=args.lr,
    )

    # 로드 옵션
    resume_metadata = None
    if args.start_mode in ["load", "two_step_only"] and args.load_prefix:
        resume_metadata = load_checkpoint_into_AgentA(agentA, args.load_prefix)
    elif args.load_model:
        policy_file = file_name if args.load_model == "default" else args.load_model
        agentA.load(f"./models/{policy_file}")
    else:
        resume_metadata = {}

    if resume_metadata is None:
        resume_metadata = {}

    if resume_metadata:
        loaded_name = resume_metadata.get("file_name")
        if loaded_name:
            file_name = loaded_name

    resume_step = int(resume_metadata.get("step", 0))
    resume_phase = resume_metadata.get("phase")

    # 스케줄
    split_step = int(round(args.max_timesteps * args.split_ratio)) if args.two_step else args.max_timesteps

    # Phase-1
    phase1_ratio = args.split_ratio if args.two_step else 1.0

    if args.start_mode == "two_step_only":
        print("⏭️  Phase-1 스킵 (two_step_only 모드)")
        phase1_end = resume_step if resume_step > 0 else split_step
    elif args.start_mode == "load":
        if resume_step <= 0 and not resume_phase:
            print("⚠️  체크포인트 메타데이터가 없어 scratch와 동일하게 진행합니다.")
            agentA = train_phase1(
                agentA, args.env, args.seed, rb, mean, std,
                max_steps=args.max_timesteps, eval_freq=args.eval_freq,
                save_model=args.save_model, file_name=file_name,
                ckpt_dir=args.checkpoint_dir, split_ratio=phase1_ratio,
            )
            phase1_end = split_step
        elif resume_phase in (None, "phase1") and resume_step < split_step:
            print(f"🔁 Phase-1 재개: step {resume_step} → {split_step}")
            agentA = train_phase1(
                agentA, args.env, args.seed, rb, mean, std,
                max_steps=args.max_timesteps, eval_freq=args.eval_freq,
                save_model=args.save_model, file_name=file_name,
                ckpt_dir=args.checkpoint_dir, split_ratio=phase1_ratio,
                start_step=resume_step,
            )
            phase1_end = split_step
        else:
            print("⏭️  Phase-1 스킵 (체크포인트에서 이미 완료)")
            phase1_end = max(resume_step, split_step)
    else:
        agentA = train_phase1(
            agentA, args.env, args.seed, rb, mean, std,
            max_steps=args.max_timesteps, eval_freq=args.eval_freq,
            save_model=args.save_model, file_name=file_name,
            ckpt_dir=args.checkpoint_dir, split_ratio=phase1_ratio,
        )
        phase1_end = split_step

    # Phase-2
    if args.two_step and phase1_end < args.max_timesteps:
        agentB = init_agentB_from_A(agentA, args, state_dim, action_dim, max_action)
        agentB = train_phase2(
            agentB, args.env, args.seed, rb, mean, std,
            start_step=phase1_end, max_steps=args.max_timesteps,
            eval_freq=args.eval_freq, save_model=args.save_model,
            file_name=file_name
        )
        active_policy = agentB
    else:
        active_policy = agentA

    # 최종 평가
    final_evaluation(
        active_policy, args.env, args.seed, mean, std,
        runs=args.final_eval_runs, episodes=args.final_eval_episodes
    )


if __name__ == "__main__":
    main()
