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


from common.baseline_reinforce_rollout_buffer import BaselineReinforceRolloutBuffer
from common.baseline_reinforce_on_policy_algorithms import BaselineReinforceOnPolicyAlgorithm
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.env_util import make_vec_env

import wandb
import copy


SelfBaselineREINFORCE = TypeVar("SelfBaselineREINFORCE", bound="LowFidelityOnlyBaselineREINFORCE")


class LowFidelityOnlyBaselineREINFORCE(BaselineReinforceOnPolicyAlgorithm):

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
        n_steps: int = 100, # not used
        gamma: float = 0.97,
        gae_lambda: float = 1.0, # not used
        ent_coef: float = 0.0, # not used
        vf_coef: float = 1.0,
        max_grad_norm: float = 1.0,
        rms_prop_eps: float = 1e-5, # not used
        use_rms_prop: bool = False,
        use_sde: bool = False,
        sde_sample_freq: int = -1, # not used
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

        # Update optimizer inside the policy if we want to use RMSProp
        # (original implementation) rather than Adam
        if use_rms_prop and "optimizer_class" not in self.policy_kwargs:
            self.policy_kwargs["optimizer_class"] = th.optim.RMSprop
            self.policy_kwargs["optimizer_kwargs"] = dict(alpha=0.99, eps=rms_prop_eps, weight_decay=0)

        if _init_setup_model:
            self._setup_model()
        self.wandb_init(stats_window_size)

    def train(self) -> None:
        """
        Update policy using the currently gathered
        rollout buffer (one gradient step over whole data).
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        # get all the samples in one go
        indices_low_unconstrained = np.random.permutation(self.rollout_buffer_low_fidelity_unconstrained.buffer_size * self.num_low_fidelity_envs)
        rollout_data_low_fidelity_unconstrained = list(self.rollout_buffer_low_fidelity_unconstrained.get(indices_low_unconstrained, batch_size=None))[0]
                
        actions_low_unconstrained = rollout_data_low_fidelity_unconstrained.actions
        returns_low_unconstrained = rollout_data_low_fidelity_unconstrained.returns
        
        if isinstance(self.action_space, spaces.Discrete):
            # Convert discrete action from float to long
            actions_low_unconstrained = actions_low_unconstrained.long().flatten()

        values_low_unconstrained, log_prob_low_unconstrained, entropy_low_unconstrained = self.policy.evaluate_actions(rollout_data_low_fidelity_unconstrained.observations, actions_low_unconstrained)
        values_low_unconstrained = values_low_unconstrained.flatten()

        advantages_low_unconstrained = (returns_low_unconstrained - values_low_unconstrained).detach()
        if self.normalize_advantage:
            advantages_low_unconstrained = (advantages_low_unconstrained - advantages_low_unconstrained.mean()) / (advantages_low_unconstrained.std() + 1e-8)
        
        policy_loss_low_unconstrained = -(advantages_low_unconstrained.detach() * log_prob_low_unconstrained)
        policy_loss = policy_loss_low_unconstrained
        
        value_loss = F.mse_loss(rollout_data_low_fidelity_unconstrained.returns.float(), values_low_unconstrained)
        
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
        wandb.log({"train/policy_loss_std": policy_loss.std().item()}, step=self.num_timesteps)
        wandb.log({"train/value_loss": value_loss.item()}, step=self.num_timesteps)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
            wandb.log({"train/std": th.exp(self.policy.log_std).mean().item()}, step=self.num_timesteps)

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
            "algorithm": "LowFidelityOnlyBaselineREINFORCE",
            "buffer_size": self.n_steps, # not used
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
            "large_n_steps": self.large_n_steps,
        }
        run = wandb.init(project="MultiFidelityPolicyGradients", notes="mujoco", tags = ["baseline-reinforce"], config=config)
