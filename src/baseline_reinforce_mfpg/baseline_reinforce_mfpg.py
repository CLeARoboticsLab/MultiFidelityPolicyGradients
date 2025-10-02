from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union

import sys, os
from on_policy_algorithms import MyOnPolicyAlgorithm

import argparse
import copy

import torch as th
import gymnasium as gym
from gymnasium import spaces
from torch.nn import functional as F
from scipy import stats
import numpy as np
import random

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.env_util import make_vec_env

import wandb


SelfBaselineREINFORCE = TypeVar("SelfBaselineREINFORCE", bound="BaselineREINFORCE")


class BaselineREINFORCE(MyOnPolicyAlgorithm):

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 7e-4,
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
        env_low = None,
        env_low_multiple = None,
        useMultiFidelity = True,
        num_env_mean_calculation: int = 9,
        shift_level = None,
        env_name = None,
        low_env_recons_freq = 5000, # not used if the number is larger than the batch size n_steps
        large_n_steps = 1000,
        c_ema_alpha = 0.95,
        use_ema_c = True,
        algorithm = "BaselineREINFORCE",
        wandb_project = "MFPG",
        eval_freq = 10000
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
            env_low=env_low,
            env_low_multiple=env_low_multiple,
            useMultiFidelity = useMultiFidelity,
            num_env_mean_calculation = num_env_mean_calculation,
            large_n_steps = large_n_steps,
            low_env_recons_freq = low_env_recons_freq
        )

        self.normalize_advantage = normalize_advantage
        self.useMultiFidelity = useMultiFidelity
        self.num_env_mean_calculation = num_env_mean_calculation
        self.used_samples = 0
        self.large_n_steps = large_n_steps
        self.shift_level = shift_level
        self.env_name = env_name
        self.low_env_recons_freq = low_env_recons_freq
        self.algorithm = algorithm
        self.wandb_project = wandb_project
        # parameters for computing moving averages for c 
        self.use_ema_c = use_ema_c
        if self.use_ema_c:
            self.c_ema_alpha = c_ema_alpha
            self.rho_ema = 0.0
            self.high_std_ema = 0.0
            self.low_std_ema = 0.0
        self.eval_freq = eval_freq
        self.eval_returns = []
        self.eval_steps = []

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

        indices = np.random.permutation(len(self.rollout_buffer_high_fidelity.observations) * self.n_envs)
        indices_low_unconstrained = np.random.permutation(self.rollout_buffer_low_fidelity_unconstrained.buffer_size * self.num_env_mean_calculation)
        rollout_data_low_fidelity_unconstrained = list(self.rollout_buffer_low_fidelity_unconstrained.get(indices_low_unconstrained, batch_size=None))
        assert len(rollout_data_low_fidelity_unconstrained) == 1
        rollout_data_low_fidelity_unconstrained = rollout_data_low_fidelity_unconstrained[0]
        
        # This will only loop once (get all data in one go)
        for rollout_data_high_fidelity, rollout_data_low_fidelity_constrained,  in \
            zip(self.rollout_buffer_high_fidelity.get(indices, batch_size=None), 
                self.rollout_buffer_low_fidelity_constrained.get(indices, batch_size=None)):
            print("Training")
            actions_high = rollout_data_high_fidelity.actions
            returns_high = rollout_data_high_fidelity.returns
            if self.useMultiFidelity:
                actions_low_constrained = rollout_data_low_fidelity_constrained.actions
                returns_low_constrained = rollout_data_low_fidelity_constrained.returns
                actions_low_unconstrained = rollout_data_low_fidelity_unconstrained.actions
                returns_low_unconstrained = rollout_data_low_fidelity_unconstrained.returns
            
            if isinstance(self.action_space, spaces.Discrete):
                # Convert discrete action from float to long
                actions_high = actions_high.long().flatten()
                if self.useMultiFidelity:
                    actions_low_constrained = actions_low_constrained.long().flatten()
                    actions_low_unconstrained = actions_low_unconstrained.long().flatten()

            values_high, log_prob_high, entropy_high = self.policy.evaluate_actions(rollout_data_high_fidelity.observations, actions_high)
            values_high = values_high.flatten()
            if self.useMultiFidelity:
                values_low_constrained, log_prob_low_constrained, entropy_low_constrained = self.policy.evaluate_actions(rollout_data_low_fidelity_constrained.observations, actions_low_constrained)
                values_low_constrained = values_low_constrained.flatten()
                values_low_unconstrained, log_prob_low_unconstrained, entropy_low_unconstrained = self.policy.evaluate_actions(rollout_data_low_fidelity_unconstrained.observations, actions_low_unconstrained)
                values_low_unconstrained = values_low_unconstrained.flatten()

            # Policy gradient loss
            if self.useMultiFidelity: # MFPG
                # compute advantage and normalize
                advantages_high = (returns_high - values_high).detach()
                advantages_low_constrained = (returns_low_constrained - values_low_constrained).detach()
                advantages_low_unconstrained = (returns_low_unconstrained - values_low_unconstrained).detach()
                if self.normalize_advantage and not self.useMultiFidelity: 
                    advantages_high = (advantages_high - advantages_high.mean()) / (advantages_high.std() + 1e-8)
                elif self.normalize_advantage and self.useMultiFidelity:
                    raise NotImplementedError("Not implemented for MFPG yet.")

                with th.no_grad():
                    if len(advantages_high) > 1:
                        rho_A_log_prob, rho_A, rho_log_prob, closed_form_c = self.evaluate_correlation_and_c(
                            advantages_high,
                            advantages_low_constrained,
                            self.rollout_buffer_high_fidelity.log_probs.flatten(),
                            self.rollout_buffer_low_fidelity_constrained.log_probs.flatten()
                        )
                    else: # a safe guard
                        rho_A_log_prob, rho_A, rho_log_prob, closed_form_c = 0.0, 0.0, 0.0, 0.0
                    

                    c = closed_form_c
                    if rho_A_log_prob < 0.0:
                        c = 0.0

                total_low_samples = advantages_low_constrained.shape[0] + advantages_low_unconstrained.shape[0]
                mean_CV = (th.sum(advantages_low_constrained.detach() * log_prob_low_constrained) + th.sum((advantages_low_unconstrained.detach() * log_prob_low_unconstrained))) / total_low_samples
                policy_loss = -(advantages_high.detach() * log_prob_high + c*(advantages_low_constrained.detach() * log_prob_low_constrained - mean_CV)).mean()
                var_without_CV = (advantages_high.detach() * log_prob_high.detach()).var()
                var_with_CV = (advantages_high.detach() * log_prob_high.detach() + c*(advantages_low_constrained.detach() * log_prob_low_constrained.detach() - mean_CV.detach())).var()
                var_ratio = var_with_CV / var_without_CV
            else: # High-Fidelity Only
                advantages_high = (returns_high - values_high).detach()
                if self.normalize_advantage:
                    advantages_high = (advantages_high - advantages_high.mean()) / (advantages_high.std() + 1e-8)
                policy_loss = -(advantages_high.detach() * log_prob_high).mean() 

            value_loss = F.mse_loss(rollout_data_high_fidelity.returns.float(), values_high)

            loss = policy_loss + self.vf_coef * value_loss

            # Optimization step
            if (not self.useMultiFidelity) or (self.useMultiFidelity and len(returns_high) > 20): # safe guard
                self.policy.optimizer.zero_grad()
                loss.backward()

                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()


        num_samples = len(self.rollout_buffer_high_fidelity.observations)
        self.used_samples += num_samples

        self._n_updates += 1
        self.logger.record("train/policy_loss", policy_loss.item())
        wandb.log({"train/policy_loss": policy_loss.item()}, step=self.num_timesteps)
        wandb.log({"train/value_loss": value_loss.item()}, step=self.num_timesteps)
        if self.useMultiFidelity:
            self.logger.record("train/rho_A_log_prob", rho_A_log_prob)
            self.logger.record("train/rho_A", rho_A)
            self.logger.record("train/rho_log_prob", rho_log_prob)
            self.logger.record("train/var_without_CV", var_without_CV.item())
            self.logger.record("train/var_with_CV", var_with_CV.item())
            self.logger.record("train/var_ratio", var_ratio.item())
            self.logger.record("train/num_samples", num_samples)
            self.logger.record("train/total_samples", self.used_samples)
            wandb.log({"train/rho_A_log_prob": rho_A_log_prob}, step=self.num_timesteps)
            wandb.log({"train/rho_A": rho_A}, step=self.num_timesteps)
            wandb.log({"train/rho_log_prob": rho_log_prob}, step=self.num_timesteps)
            wandb.log({"train/var_without_CV": var_without_CV.item()}, step=self.num_timesteps)
            wandb.log({"train/var_with_CV": var_with_CV.item()}, step=self.num_timesteps)
            wandb.log({"train/var_ratio": var_ratio.item()}, step=self.num_timesteps)
            wandb.log({"train/num_samples": num_samples}, step=self.num_timesteps)
            wandb.log({"train/total_samples": self.used_samples},  step=self.num_timesteps)
            wandb.log({"train/c": c}, step=self.num_timesteps)
            if self.use_ema_c:
                wandb.log({"train/rho_ema": self.rho_ema}, step=self.num_timesteps)
                wandb.log({"train/high_std_ema": self.high_std_ema}, step=self.num_timesteps)
                wandb.log({"train/low_std_ema": self.low_std_ema}, step=self.num_timesteps)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
            wandb.log({"train/std": th.exp(self.policy.log_std).mean().item()}, step=self.num_timesteps)

        # recover the buffer sizes
        self.rollout_buffer_high_fidelity.buffer_size = self.n_steps
        self.rollout_buffer_low_fidelity_constrained.buffer_size = self.n_steps

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

    def evaluate_correlation_and_c(self, Ghigh, Glow, log_prob_high, log_prob_low):
        if type(Ghigh) == th.Tensor:
            Ghigh = Ghigh.cpu().numpy()
            Glow = Glow.cpu().numpy()
        if type(log_prob_high) == th.Tensor:
            log_prob_high = log_prob_high.cpu().numpy()
            log_prob_low = log_prob_low.cpu().numpy()
        
        rho_G = stats.pearsonr(Ghigh, Glow).statistic
        rho_log_prob = stats.pearsonr(log_prob_high, log_prob_low).statistic
        rho_G_log_prob = stats.pearsonr(Ghigh * log_prob_high, Glow * log_prob_low).statistic

        if self.use_ema_c:
            if self.rho_ema == 0.0:
                self.rho_ema = rho_G_log_prob
                self.high_std_ema = (Ghigh * log_prob_high).std()
                self.low_std_ema = (Glow * log_prob_low).std()
            else:
                self.rho_ema = self.c_ema_alpha * self.rho_ema + (1 - self.c_ema_alpha) * rho_G_log_prob
                self.high_std_ema = self.c_ema_alpha * self.high_std_ema + (1 - self.c_ema_alpha) * ((Ghigh * log_prob_high).std())
                self.low_std_ema = self.c_ema_alpha * self.low_std_ema + (1 - self.c_ema_alpha) * ((Glow * log_prob_low).std())
            closed_form_c = -self.rho_ema * (self.high_std_ema / self.low_std_ema)
        else:
            closed_form_c = -rho_G_log_prob * (((Ghigh * log_prob_high).std()) / ((Glow * log_prob_low).std()))

        return rho_G_log_prob, rho_G, rho_log_prob, closed_form_c

    def wandb_init(self, stats_window_size):
        config = {
            "algorithm": self.algorithm,
            "multi_fidelity": self.useMultiFidelity,
            "buffer_size": self.n_steps,
            "large_buffer_size": self.large_n_steps,
            "num_low_unconstrained_envs": self.num_env_mean_calculation,
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
            "low_env_recons_freq": self.low_env_recons_freq,
            "use_ema_c": self.use_ema_c,
        }
        if self.use_ema_c:
            config["c_ema_alpha"] = self.c_ema_alpha
        run = wandb.init(project=self.wandb_project, notes="mujoco", tags = ["reinforce", "multi-fidelility"+str(self.useMultiFidelity)], config=config)
