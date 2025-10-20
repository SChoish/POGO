# File: POGO/agent.py

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from geomloss import SamplesLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def per_state_sinkhorn(actor, ref_policy, states, K=4, blur=0.05, p=2,
    use_ref_grad=False, backend="tensorized"
):
    """
    배치 Sinkhorn (GeomLoss 구버전 호환)
    reduction 인자 제거 — 항상 batch mean을 반환
    """
    B = states.size(0)
    a_dim = actor.head.out_features
    device = states.device

    # [B,K,d]
    z_a = torch.randn(B, K, a_dim, device=device)
    z_b = torch.randn(B, K, a_dim, device=device)

    states_tiled = states.unsqueeze(1).expand(B, K, states.size(1))
    a = actor(states_tiled.reshape(B*K, -1), z_a.reshape(B*K, a_dim)).reshape(B, K, a_dim)

    with torch.set_grad_enabled(use_ref_grad):
        b = ref_policy(states_tiled.reshape(B*K, -1), z_b.reshape(B*K, a_dim)).reshape(B, K, a_dim)

    # GeomLoss batch call (no reduction arg)
    sinkhorn = SamplesLoss(loss="sinkhorn", p=p, blur=blur, backend=backend)
    # [B,K,d] 입력 시 자동으로 batch 평균을 계산
    loss = sinkhorn(a, b)
    if loss.dim() >0:
        loss = loss.mean()
    return loss

class Actor(nn.Module):
    """
    Transport Map Actor Network
    Implements T_s: z ~ N(0,I) → action space
    """
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.max_action = max_action
        # Transport map: (state, z) → action
        # Input: state_dim + action_dim (z is action_dim dimensional)
        self.l1 = nn.Linear(action_dim + state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.head = nn.Linear(256, action_dim)

    def forward(self, state, z):
        # Transport map: T_s(state, z) where z ~ N(0,I) in action space
        x = F.relu(self.l1(torch.cat([state, z], 1))) # (batch_size, state_dim + action_dim)
        x = F.relu(self.l2(x))
        return torch.tanh(self.head(x)) * self.max_action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa)); q1 = F.relu(self.l2(q1)); q1 = self.l3(q1)
        q2 = F.relu(self.l4(sa)); q2 = F.relu(self.l5(q2)); q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa)); q1 = F.relu(self.l2(q1)); q1 = self.l3(q1)
        return q1

class POGO:
    """
    POGO: Policy Optimization by Gradient Flow on Offline RL
    
    Implements JKO (Jordan-Kinderlehrer-Otto) proximal energy:
    L_actor = -λ * E[Q(s,a)] + w2_weight * W2(π_actor, π_behavior)
    
    Key components:
    - Transport Map: Actor approximates T_s: z ~ N(0,I) → action space
    - Adaptive Regularization: λ = α / |Q|_mean
    - W2 Distance: L2 distance for deterministic behavior policies
    - JKO Scheme: π_{k+1} = argmin_π [E_π[V] + (1/2τ) * W2(π, π_k)]
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=1,
        alpha=1.0,
        w2_weight=1.0,
        lr=3e-4,
    ):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha
        self.w2_weight = w2_weight
        self.total_it = 0

    def select_action(self, state, deterministic = True):
        """
        Select action using transport map
        - deterministic=True: z = 0 (deterministic policy)
        - deterministic=False: z ~ N(0,I) (stochastic policy)
        """
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        if deterministic:
            z = torch.zeros(state.shape[0], self.actor.head.out_features).to(device)
        else:
            z = torch.randn(state.shape[0], self.actor.head.out_features).to(device)
        return self.actor(state, z).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        """
        JKO Proximal Energy Training
        Implements: L_actor = -λ * E[Q(s,a)] + w2_weight * W2(π_actor, π_behavior)
        """
        self.total_it += 1
        
        # ----- Critic Training (TD3) -----
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        
        with torch.no_grad():
            # Target policy smoothing (TD3)
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            z_target = torch.zeros_like(action, device=device)
            target_act = self.actor_target(next_state, z_target)
            next_action = (target_act + noise).clamp(-self.max_action, self.max_action)
            
            # Target Q-values (TD3)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = reward + not_done * self.discount * torch.min(target_Q1, target_Q2)
        
        # Current Q-values
        current_Q1, current_Q2 = self.critic(state, action)
        
        # Critic loss (TD3)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ----- Actor Training (JKO Proximal Energy) -----
        if self.total_it % self.policy_freq == 0:
            # Independent z sampling for actor training (different from critic target)
            z_actor = torch.randn_like(action).to(device)
            
            # Transport map: π_actor = T_s(state, z_actor)
            pi = self.actor(state, z_actor)
            Q = self.critic.Q1(state, pi)
            
            # Adaptive regularization: λ = α / |Q|_mean
            lmbda = self.alpha / Q.abs().mean().detach()
            
            # W2 distance: For deterministic behavior policy, L2 = W2
            w2 = ((pi - action) ** 2).mean()
            
            # JKO Proximal Energy: L = -λ * E[Q] + w2_weight * W2(π, π_behavior)
            actor_loss = -lmbda * Q.mean() + self.w2_weight * w2

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Target network soft-update
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

            metrics = {
                "critic_loss": float(critic_loss.item()),
                "actor_loss": float(actor_loss.item()),
                "lambda": float(lmbda.item()),
                "Q_mean": float(Q.mean().item()),
                "w2_distance": float(w2.item()),
            }
        else:
            metrics = {
                "critic_loss": float(critic_loss.item()),
                "actor_loss": 0.0,
                "lambda": 0.0,
                "Q_mean": 0.0,
                "w2_distance": 0.0,
            }

        return metrics

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        map_location = device
        self.critic.load_state_dict(torch.load(filename + "_critic", map_location=map_location))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer", map_location=map_location))
        self.critic_target = copy.deepcopy(self.critic)
        self.actor.load_state_dict(torch.load(filename + "_actor", map_location=map_location))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer", map_location=map_location))
        self.actor_target = copy.deepcopy(self.actor)

class POGO_Refine:
    """
    POGO Multi-step JKO Extension
    
    Implements JKO 2nd step: π_1 = argmin_π [E_π[V] + (1/2τ) * W2(π, π_0)]
    where π_0 is the behavior policy (from Phase 1) and π_1 is the refined policy.
    
    Key features:
    - Sinkhorn Distance: For complex policy distributions
    - Behavior Policy: π_0 (reference from Phase 1)
    - Actor Policy: π_1 (optimized in Phase 2)
    - Critic Freezing: Optional for experimental purposes
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=1,
        alpha=1.0,
        w2_weight=1.0,
        lr=3e-4,
        freeze_critic=True,
        copy_actor=True,
    ):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # Behavior policy: π_0 (reference from Phase 1)
        self.behavior_policy = Actor(state_dim, action_dim, max_action).to(device)
        self.behavior_optimizer = torch.optim.Adam(self.behavior_policy.parameters(), lr=lr)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha
        self.w2_weight = w2_weight
        self.freeze_critic = freeze_critic
        self.copy_actor = copy_actor
        self.total_it = 0

    def select_action(self, state, deterministic = True):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        if deterministic:
            z = torch.zeros(state.shape[0], self.actor.head.out_features).to(device)
        else:
            z = torch.randn(state.shape[0], self.actor.head.out_features).to(device)
        return self.actor(state, z).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        """
        JKO 2nd Step Training
        Implements: π_1 = argmin_π [E_π[V] + (1/2τ) * W2(π, π_0)]
        """
        self.total_it += 1
        self.behavior_policy.eval()  # Fix behavior policy (π_0)
        # ----- Critic Training (Optional) -----
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        if not self.freeze_critic:
            with torch.no_grad():
                # Target policy smoothing (TD3)
                noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                z_target = torch.zeros_like(action, device=device)
                target_act = self.actor_target(next_state, z_target)
                next_action = (target_act + noise).clamp(-self.max_action, self.max_action)
                
                # Target Q-values (TD3)
                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                target_Q = reward + not_done * self.discount * torch.min(target_Q1, target_Q2)
            
            # Current Q-values
            current_Q1, current_Q2 = self.critic(state, action)
            
            # Critic loss (TD3)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
        else:
            critic_loss = torch.tensor(0.0).to(device)
        metrics = {"critic_loss": float(critic_loss.item())}

        # ----- Actor Training (JKO 2nd Step) -----
        # Independent z sampling for actor training (different from critic target)
        z_actor = torch.randn_like(action).to(device)
        
        # Transport map: π_1 = T_s(state, z_actor)
        mean_a = self.actor(state, z_actor)
        Q = self.critic.Q1(state, mean_a)
        
        # Adaptive regularization: λ = α / |Q|_mean
        lmbda = self.alpha / Q.abs().mean().detach()

        # Per-state Sinkhorn distance: W2(π_1, π_0) for complex distributions
        # Compute Sinkhorn for each state separately to avoid mixing different states
        w2 = per_state_sinkhorn(self.actor, self.behavior_policy, state, K=4, blur=0.05, p=2, use_ref_grad=False)

        # JKO 2nd Step: L = -λ * E[Q] + w2_weight * W2(π_1, π_0)
        actor_loss = -lmbda * Q.mean() + self.w2_weight * w2

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # target soft-update
        if not self.freeze_critic:
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        metrics.update({
            "actor_loss": float(actor_loss.item()),
            "lambda": float(lmbda.item()),
            "Q_mean": float(Q.mean().item()),
            "w2_distance": float(w2.item()),
        })

        return metrics

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        map_location = device
        self.critic.load_state_dict(torch.load(filename + "_critic", map_location=map_location))
        if not self.freeze_critic:
            # If not freeze_critic, load the networks used for critic training
            self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer", map_location=map_location))
            if self.copy_actor:
                self.actor_target.load_state_dict(torch.load(filename + "_actor", map_location=map_location))
                self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer", map_location=map_location))
        self.critic_target = copy.deepcopy(self.critic)
        self.behavior_policy.load_state_dict(torch.load(filename + "_actor", map_location=map_location))
        if self.copy_actor:
            self.actor.load_state_dict(torch.load(filename + "_actor", map_location=map_location))
            self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer", map_location=map_location))
