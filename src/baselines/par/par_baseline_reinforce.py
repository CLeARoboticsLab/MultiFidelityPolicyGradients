from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import copy

import torch as th
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from torch.nn import functional as F
from scipy import stats
import numpy as np
import random

from encoder import Encoder
from baselines.common.baseline_reinforce_rollout_buffer import BaselineReinforceRolloutBuffer
from baselines.common.baseline_reinforce_on_policy_algorithms import BaselineReinforceOnPolicyAlgorithm
from baselines.common.replay_buffer import ReplayBuffer
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.env_util import make_vec_env

import wandb
import copy


SelfBaselineREINFORCE = TypeVar("SelfBaselineREINFORCE", bound="BaselineREINFORCE")


class PARBaselineREINFORCE(BaselineReinforceOnPolicyAlgorithm):
    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 7.0e-4,
        n_steps: int = 100,
        gamma: float = 0.97,
        gae_lambda: float = 1.0, # not used
        ent_coef: float = 0.0, # not used
        vf_coef: float = 1.0,
        max_grad_norm: float = 1.0,
        rms_prop_eps: float = 1e-5, # not used
        use_rms_prop: bool = False,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        normalize_advantage: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        env_low_multiple = None,
        num_low_fidelity_envs: int = 10,
        shift_level = None,
        env_name = None,
        encoder_learning_rate = 3e-4,
        tau = 0.005,
        reward_augmentation_beta = 0.1,
        replay_buffer_size = int(1e6),
        encoder_batch_size = 128,
        eval_freq: int = 2000,
        large_n_steps: int = 1000,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
            env_low_multiple=env_low_multiple,
            num_low_fidelity_envs = num_low_fidelity_envs,  
            large_n_steps = large_n_steps,
        )

        self.normalize_advantage = normalize_advantage
        self.num_low_fidelity_envs = num_low_fidelity_envs
        self.shift_level = shift_level
        self.env_name = env_name
        self.eval_freq = eval_freq
        self.large_n_steps = large_n_steps
        # Get state and action dimensions from environment
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]

        self.tau = tau
        self.reward_augmentation_beta = reward_augmentation_beta
        # encoder
        self.encoder_learning_rate = encoder_learning_rate
        self.encoder = Encoder(state_dim, action_dim).to(self.device)
        self.encoder_target = copy.deepcopy(self.encoder)
        self.encoder_target.eval()
        self.encoder_optimizer = th.optim.Adam(self.encoder.parameters(), lr=self.encoder_learning_rate)

        # replay bufferAdd commentMore actions
        self.replay_buffer_size = replay_buffer_size
        self.encoder_batch_size = encoder_batch_size
        self.high_fidelity_replay_buffer = ReplayBuffer(replay_buffer_size, self.encoder_batch_size)

        # Update optimizer inside the policy if we want to use RMSProp
        # (original implementation) rather than Adam
        if use_rms_prop and "optimizer_class" not in self.policy_kwargs:
            self.policy_kwargs["optimizer_class"] = th.optim.RMSprop
            self.policy_kwargs["optimizer_kwargs"] = dict(alpha=0.99, eps=rms_prop_eps, weight_decay=0)

        if _init_setup_model:
            self._setup_model()
        self.wandb_init(stats_window_size)

    def update_encoder(self, state_batch, action_batch, nextstate_batch, writer=None):
        with th.no_grad():
            next_zs = self.encoder.zs(nextstate_batch)

        zs = self.encoder.zs(state_batch)
        pred_zs = self.encoder.zsa(zs, action_batch)
        encoder_loss = F.mse_loss(pred_zs, next_zs)

        self.encoder_optimizer.zero_grad()
        encoder_loss.backward()
        self.encoder_optimizer.step()

        return encoder_loss.item()

    def update_target(self):
        """moving average update of target networks"""
        with th.no_grad():
            # update encoder
            for target_q_param, q_param in zip(self.encoder_target.parameters(), self.encoder.parameters()):
                target_q_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_q_param.data)

    def reward_augmentation(self):
        next_zs_low_fidelity = self.encoder_target.zs(th.tensor(self.rollout_buffer_low_fidelity_unconstrained.next_observations, device=self.device).float())
        zs_low_fidelity = self.encoder_target.zs(th.tensor(self.rollout_buffer_low_fidelity_unconstrained.observations, device=self.device).float())
        pred_zs_low_fidelity = self.encoder_target.zsa(zs_low_fidelity, th.tensor(self.rollout_buffer_low_fidelity_unconstrained.actions, device=self.device).float())

        distance = ((pred_zs_low_fidelity - next_zs_low_fidelity)**2).mean(dim=-1, keepdim=True).squeeze(-1)

        original_rewards = copy.deepcopy(self.rollout_buffer_low_fidelity_unconstrained.rewards)
        original_returns = copy.deepcopy(self.rollout_buffer_low_fidelity_unconstrained.returns)
        original_advantages = copy.deepcopy(self.rollout_buffer_low_fidelity_unconstrained.advantages)
        self.rollout_buffer_low_fidelity_unconstrained.rewards = original_rewards - (self.reward_augmentation_beta * distance).cpu().numpy()

        # update returns and advantages upon augmenting the rewards
        self.rollout_buffer_low_fidelity_unconstrained.compute_returns_and_advantage(last_values=self._last_values, dones=self._last_dones)

        wandb.log({"reward_augmentation/distance": distance.mean().item()}, step=self.num_timesteps)
        wandb.log({"reward_augmentation/original_rewards": original_rewards.mean()}, step=self.num_timesteps)
        wandb.log({"reward_augmentation/augmented_rewards": np.mean(self.rollout_buffer_low_fidelity_unconstrained.rewards)}, step=self.num_timesteps)

    def train(self) -> None:
        """
        Update policy using the currently gathered
        rollout buffer (one gradient step over whole data).
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        with th.no_grad():
            self.reward_augmentation()

        encoder_losses = []

        # get all the samples in one go
        indices = np.random.permutation(len(self.rollout_buffer_high_fidelity.observations) * self.n_envs)
        indices_low_unconstrained = np.random.permutation(self.rollout_buffer_low_fidelity_unconstrained.buffer_size * self.num_low_fidelity_envs)
        rollout_data_high_fidelity = list(self.rollout_buffer_high_fidelity.get(indices, batch_size=None))[0]
        rollout_data_low_fidelity_unconstrained = list(self.rollout_buffer_low_fidelity_unconstrained.get(indices_low_unconstrained, batch_size=None))[0]

        # update encoder
        tar_state, tar_action, t_r, tar_next_state, t_d = self.high_fidelity_replay_buffer.sample()
        if not th.is_tensor(tar_state):
            tar_state = th.as_tensor(tar_state, dtype=th.float32).to(self.device)
            tar_action = th.as_tensor(tar_action, dtype=th.float32).to(self.device)
            tar_next_state = th.as_tensor(tar_next_state, dtype=th.float32).to(self.device)
        if (len(tar_state.shape)==3) and (len(tar_action.shape)==3) and (len(tar_next_state.shape)==3):
            # Reshape 3D tensors to 2D by merging first two dimensions
            tar_state = tar_state.reshape(-1, tar_state.shape[-1])
            tar_action = tar_action.reshape(-1, tar_action.shape[-1])
            tar_next_state = tar_next_state.reshape(-1, tar_next_state.shape[-1])
            
            # # Print shapes after reshaping
            # print(f"Reshaped tar_state shape: {tar_state.shape}")
            # print(f"Reshaped tar_action shape: {tar_action.shape}")
            # print(f"Reshaped tar_next_state shape: {tar_next_state.shape}")
            
        encoder_loss = self.update_encoder(tar_state, tar_action, tar_next_state)
        encoder_losses.append(encoder_loss)
                
        actions_high = rollout_data_high_fidelity.actions
        returns_high = rollout_data_high_fidelity.returns
        actions_low_unconstrained = rollout_data_low_fidelity_unconstrained.actions
        returns_low_unconstrained = rollout_data_low_fidelity_unconstrained.returns
        
        if isinstance(self.action_space, spaces.Discrete):
            # Convert discrete action from float to long
            actions_high = actions_high.long().flatten()
            actions_low_unconstrained = actions_low_unconstrained.long().flatten()

        values_high, log_prob_high, entropy_high = self.policy.evaluate_actions(rollout_data_high_fidelity.observations, actions_high)
        values_high = values_high.flatten()
        values_low_unconstrained, log_prob_low_unconstrained, entropy_low_unconstrained = self.policy.evaluate_actions(rollout_data_low_fidelity_unconstrained.observations, actions_low_unconstrained)
        values_low_unconstrained = values_low_unconstrained.flatten()

        advantages_high = (returns_high - values_high).detach()
        advantages_low_unconstrained = (returns_low_unconstrained - values_low_unconstrained).detach()
        if self.normalize_advantage:
            advantages_high = (advantages_high - advantages_high.mean()) / (advantages_high.std() + 1e-8)
            advantages_low_unconstrained = (advantages_low_unconstrained - advantages_low_unconstrained.mean()) / (advantages_low_unconstrained.std() + 1e-8)
        
        policy_loss_high = -(advantages_high.detach() * log_prob_high)
        policy_loss_low_unconstrained = -(advantages_low_unconstrained.detach() * log_prob_low_unconstrained)
        policy_loss = th.cat([policy_loss_high, policy_loss_low_unconstrained], 0)
        
        mixed_values_pred = th.cat([values_high, values_low_unconstrained], 0)
        mixed_rollout_data_returns = th.cat([rollout_data_high_fidelity.returns.float(), rollout_data_low_fidelity_unconstrained.returns.float()], 0)
        value_loss = F.mse_loss(mixed_rollout_data_returns, mixed_values_pred)
        
        loss = policy_loss.mean() + self.vf_coef * value_loss

        # Optimization step
        self.policy.optimizer.zero_grad()
        loss.backward()

        # Clip grad norm
        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()

        self._n_updates += 1
        self.logger.record("train/policy_loss", loss.item())
        wandb.log({"train/policy_loss": loss.item()}, step=self.num_timesteps)
        wandb.log({"train/encoder_loss": np.mean(encoder_losses)}, step=self.num_timesteps)
        wandb.log({"train/policy_loss_std": policy_loss.std().item()}, step=self.num_timesteps)
        wandb.log({"train/value_loss": value_loss.item()}, step=self.num_timesteps)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
            wandb.log({"train/std": th.exp(self.policy.log_std).mean().item()}, step=self.num_timesteps)

        self.update_target()

        # recover the buffer sizes
        self.rollout_buffer_high_fidelity.buffer_size = self.n_steps

    def learn(
        self: SelfBaselineREINFORCE,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "REINFORCE",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfBaselineREINFORCE:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def wandb_init(self, stats_window_size):
        config = {
            "algorithm": "PAR-BaselineREINFORCE",
            "buffer_size": self.n_steps,
            "num_low_fidelity_envs": self.num_low_fidelity_envs,
            "seed": self.seed,
            "stats_window_size": stats_window_size,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
            "normalize_advantage": self.normalize_advantage,
            "device": self.device,
            "shift_level": self.shift_level,
            "env_name": self.env_name,
            "encoder_learning_rate": self.encoder_learning_rate,
            "tau": self.tau,
            "reward_augmentation_beta": self.reward_augmentation_beta,
            "replay_buffer_size": self.replay_buffer_size,
            "encoder_batch_size": self.encoder_batch_size,
            "large_n_steps": self.large_n_steps,
        }
        run = wandb.init(project="MultiFidelityPolicyGradients", notes="mujoco", tags = ["par-reinforce"], config=config)
