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

from baselines.darc.classifier import Classifier
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


SelfBaselineREINFORCE = TypeVar("SelfBaselineREINFORCE", bound="DARC_BaselineREINFORCE")


class DARC_BaselineREINFORCE(BaselineReinforceOnPolicyAlgorithm):
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
        classifier_learning_rate = 3e-4,
        reward_augmentation_beta = 1.0, # DARC default: no reweighting
        classifier_hidden_sizes = 256,
        classifier_gaussian_noise_std = 1.0,
        replay_buffer_size = int(1e6),
        classifier_batch_size = 128,
        eval_freq: int = 2000,
        large_n_steps: int = 1000,
        warmup_steps: int = 2000,
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
        self.warmup_steps = warmup_steps
        # Get state and action dimensions from environment
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]

        self.reward_augmentation_beta = reward_augmentation_beta
        # classifier
        self.classifier_learning_rate = classifier_learning_rate
        self.classifier_hidden_sizes = classifier_hidden_sizes
        self.classifier_gaussian_noise_std = classifier_gaussian_noise_std
        self.classifier = Classifier(state_dim, action_dim, classifier_hidden_sizes, classifier_gaussian_noise_std).to(self.device)
        self.classifier_optimizer = th.optim.Adam(self.classifier.parameters(), lr=classifier_learning_rate)

        # replay buffer
        self.replay_buffer_size = replay_buffer_size
        self.classifier_batch_size = classifier_batch_size
        self.high_fidelity_replay_buffer = ReplayBuffer(replay_buffer_size, self.classifier_batch_size)
        self.low_fidelity_replay_buffer = ReplayBuffer(replay_buffer_size, self.classifier_batch_size)

        # Update optimizer inside the policy if we want to use RMSProp
        # (original implementation) rather than Adam
        if use_rms_prop and "optimizer_class" not in self.policy_kwargs:
            self.policy_kwargs["optimizer_class"] = th.optim.RMSprop
            self.policy_kwargs["optimizer_kwargs"] = dict(alpha=0.99, eps=rms_prop_eps, weight_decay=0)

        if _init_setup_model:
            self._setup_model()
        self.wandb_init(stats_window_size)

    def update_classifier(self):
        src_state, src_action, s_rewards, src_next_state, s_done_masks = self.low_fidelity_replay_buffer.sample()
        tar_state, tar_action, t_r, tar_next_state, t_d = self.high_fidelity_replay_buffer.sample()
        if not th.is_tensor(src_state):
            src_state = th.as_tensor(src_state, dtype=th.float32).to(self.device)
            src_action = th.as_tensor(src_action, dtype=th.float32).to(self.device)
            src_next_state = th.as_tensor(src_next_state, dtype=th.float32).to(self.device)

            tar_state = th.as_tensor(tar_state, dtype=th.float32).to(self.device)
            tar_action = th.as_tensor(tar_action, dtype=th.float32).to(self.device)
            tar_next_state = th.as_tensor(tar_next_state, dtype=th.float32).to(self.device)
        
        if (len(src_state.shape) == 3) and (len(tar_state.shape) == 3) and (len(src_action.shape) == 3) and (len(tar_action.shape) == 3):
            # Reshape 3D tensors to 2D by merging first two dimensionsAdd commentMore actions
            src_state = src_state.reshape(-1, src_state.shape[-1])
            src_action = src_action.reshape(-1, src_action.shape[-1])
            src_next_state = src_next_state.reshape(-1, src_next_state.shape[-1])
            
            tar_state = tar_state.reshape(-1, tar_state.shape[-1])
            tar_action = tar_action.reshape(-1, tar_action.shape[-1])
            tar_next_state = tar_next_state.reshape(-1, tar_next_state.shape[-1])
            

        if src_state.shape[0] != tar_state.shape[0]:
            # Generate random indices to subsample source quantities to match target batch size
            random_indices = th.randperm(src_state.shape[0])[:tar_state.shape[0]]
            src_state = src_state[random_indices]
            src_action = src_action[random_indices]
            src_next_state = src_next_state[random_indices]

        state = th.cat([src_state, tar_state], 0)
        action = th.cat([src_action, tar_action], 0)
        next_state = th.cat([src_next_state, tar_next_state], 0)

        # set labels for different domains
        label = th.cat([th.zeros(size=(src_state.shape[0],)), th.ones(size=(tar_state.shape[0],))], dim=0).long().to(self.device)

        indexs = th.randperm(label.shape[0])
        state_batch, action_batch, nextstate_batch = state[indexs], action[indexs], next_state[indexs]
        label = label[indexs]

        sas_logits, sa_logits = self.classifier(state_batch, action_batch, nextstate_batch, with_noise=True)
        loss_sas = F.cross_entropy(sas_logits, label)
        loss_sa =  F.cross_entropy(sa_logits, label)
        classifier_loss = loss_sas + loss_sa
        self.classifier_optimizer.zero_grad()
        classifier_loss.backward()
        self.classifier_optimizer.step()

        return loss_sas.item(), loss_sa.item()

    def reward_augmentation(self):
        sas_logits, sa_logits = self.classifier(th.tensor(self.rollout_buffer_low_fidelity_unconstrained.observations, device=self.device).float(), \
                                                th.tensor(self.rollout_buffer_low_fidelity_unconstrained.actions, device=self.device).float(), \
                                                th.tensor(self.rollout_buffer_low_fidelity_unconstrained.next_observations, device=self.device).float(), with_noise=False)
        sas_probs, sa_probs = F.softmax(sas_logits, -1), F.softmax(sa_logits, -1)
        sas_log_probs, sa_log_probs = th.log(sas_probs + 1e-10), th.log(sa_probs + 1e-10)
        reward_penalty = sas_log_probs[:, :, 1:] - sa_log_probs[:, :, 1:] - sas_log_probs[:, :, :1] + sa_log_probs[:, :, :1]
        reward_penalty = reward_penalty.squeeze(-1)

        original_rewards = copy.deepcopy(self.rollout_buffer_low_fidelity_unconstrained.rewards)
        original_returns = copy.deepcopy(self.rollout_buffer_low_fidelity_unconstrained.returns)
        original_advantages = copy.deepcopy(self.rollout_buffer_low_fidelity_unconstrained.advantages)
        self.rollout_buffer_low_fidelity_unconstrained.rewards = original_rewards + (self.reward_augmentation_beta * reward_penalty).cpu().numpy()

        # update returns and advantages upon augmenting the rewards
        self.rollout_buffer_low_fidelity_unconstrained.compute_returns_and_advantage(last_values=self._last_values, dones=self._last_dones)

        wandb.log({"reward_augmentation/distance": reward_penalty.mean().item()}, step=self.num_timesteps)
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

        # follow the original paper, DARC has a warm-start phase that does not involve reward augmentation
        if self.num_timesteps > self.warmup_steps:
            with th.no_grad():
                self.reward_augmentation()

        classifier_losses_sas = []
        classifier_losses_sa = []

        # get all the samples in one go
        indices = np.random.permutation(len(self.rollout_buffer_high_fidelity.observations) * self.n_envs)
        indices_low_unconstrained = np.random.permutation(self.rollout_buffer_low_fidelity_unconstrained.buffer_size * self.num_low_fidelity_envs)
        rollout_data_high_fidelity = list(self.rollout_buffer_high_fidelity.get(indices, batch_size=None))[0]
        rollout_data_low_fidelity_unconstrained = list(self.rollout_buffer_low_fidelity_unconstrained.get(indices_low_unconstrained, batch_size=None))[0]

        # update classifier
        # follow the original paper, DARC has a warm-start phase that does not involve reward augmentation
        if self.num_timesteps > self.warmup_steps:
            classifier_loss_sas, classifier_loss_sa = self.update_classifier()
            classifier_losses_sas.append(classifier_loss_sas)
            classifier_losses_sa.append(classifier_loss_sa)
                
        # actions_high = rollout_data_high_fidelity.actions
        # returns_high = rollout_data_high_fidelity.returns
        actions_low_unconstrained = rollout_data_low_fidelity_unconstrained.actions
        returns_low_unconstrained = rollout_data_low_fidelity_unconstrained.returns
        
        if isinstance(self.action_space, spaces.Discrete):
            # Convert discrete action from float to long
            # actions_high = actions_high.long().flatten()
            actions_low_unconstrained = actions_low_unconstrained.long().flatten()

        # values_high, log_prob_high, entropy_high = self.policy.evaluate_actions(rollout_data_high_fidelity.observations, actions_high)
        # values_high = values_high.flatten()
        values_low_unconstrained, log_prob_low_unconstrained, entropy_low_unconstrained = self.policy.evaluate_actions(rollout_data_low_fidelity_unconstrained.observations, actions_low_unconstrained)
        values_low_unconstrained = values_low_unconstrained.flatten()

        # advantages_high = (returns_high - values_high).detach()
        advantages_low_unconstrained = (returns_low_unconstrained - values_low_unconstrained).detach()
        if self.normalize_advantage:
            # advantages_high = (advantages_high - advantages_high.mean()) / (advantages_high.std() + 1e-8)
            advantages_low_unconstrained = (advantages_low_unconstrained - advantages_low_unconstrained.mean()) / (advantages_low_unconstrained.std() + 1e-8)
        
        # policy_loss_high = -(advantages_high.detach() * log_prob_high)
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
        if self.num_timesteps > self.warmup_steps:
            wandb.log({"train/classifier_loss_sas": np.mean(classifier_losses_sas)}, step=self.num_timesteps)
            wandb.log({"train/classifier_loss_sa": np.mean(classifier_losses_sa)}, step=self.num_timesteps)
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
            "algorithm": "DARC-BaselineREINFORCE",
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
            "classifier_learning_rate": self.classifier_learning_rate,
            "classifier_hidden_sizes": self.classifier_hidden_sizes,
            "classifier_gaussian_noise_std": self.classifier_gaussian_noise_std,
            "reward_augmentation_beta": self.reward_augmentation_beta,
            "replay_buffer_size": self.replay_buffer_size,
            "classifier_batch_size": self.classifier_batch_size,
            "large_n_steps": self.large_n_steps,
            "warmup_steps": self.warmup_steps,
        }
        run = wandb.init(project="MultiFidelityPolicyGradients", notes="mujoco", tags = ["darc-baseline-reinforce"], config=config)
