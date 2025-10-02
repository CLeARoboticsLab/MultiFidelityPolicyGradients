import sys, os
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from torch.distributions.utils import _standard_normal
from scipy import stats
from gymnasium import spaces
from gymnasium.wrappers.common import TimeLimit
import copy
from copy import deepcopy
import warnings

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.evaluation import evaluate_policy

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils import compute_buffer_episode_slices

import wandb

SelfOnPolicyAlgorithm = TypeVar("SelfOnPolicyAlgorithm", bound="OnPolicyAlgorithm")

class MyOnPolicyAlgorithm(OnPolicyAlgorithm):
    rollout_buffer: RolloutBuffer
    policy: ActorCriticPolicy

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        num_env_mean_calculation: int,
        rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        stats_window_size: int = 50,
        tensorboard_log: Optional[str] = None,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[Tuple[Type[spaces.Space], ...]] = None,
        env_low = None,
        env_low_multiple=None,
        useMultiFidelity = True,
        large_n_steps = None,
        low_env_recons_freq = 50,
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
        )
        self.env_low = env_low
        self.env_low_multiple=env_low_multiple
        self.useMultiFidelity = useMultiFidelity
        self.high_fidelity_env_last_termination_step = 0
        self.high_fidelity_env_last_batch_step_id = 0
        self.num_env_mean_calculation = num_env_mean_calculation
        self.large_n_steps = large_n_steps
        self.qpos_ids = np.arange(len(env_low.unwrapped.envs[0].unwrapped.data.qpos))
        self.qvel_ids = \
            np.arange(len(env_low.unwrapped.envs[0].unwrapped.data.qpos),(len(env_low.unwrapped.envs[0].unwrapped.data.qpos)+len(env_low.unwrapped.envs[0].unwrapped.data.qvel)))
        self.low_env_recons_freq = low_env_recons_freq

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        if self.rollout_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                self.rollout_buffer_class = DictRolloutBuffer
            else:
                self.rollout_buffer_class = RolloutBuffer

        self.rollout_buffer_high_fidelity = self.rollout_buffer_class(
            self.n_steps,
            self.observation_space,  # type: ignore[arg-type]
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            **self.rollout_buffer_kwargs,
        )
        self.rollout_buffer_low_fidelity_constrained = self.rollout_buffer_class(
            self.n_steps,
            self.observation_space,  # type: ignore[arg-type]
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            **self.rollout_buffer_kwargs,
        )
        self.rollout_buffer_low_fidelity_unconstrained = self.rollout_buffer_class(
            self.large_n_steps,
            self.observation_space,  # type: ignore[arg-type]
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs * self.num_env_mean_calculation,
            **self.rollout_buffer_kwargs,
        )

        self.policy = self.policy_class(  # type: ignore[assignment]
            self.observation_space, self.action_space, self.lr_schedule, use_sde=self.use_sde, **self.policy_kwargs
        )
        self.policy = self.policy.to(self.device)
        # Warn when not using CPU with MlpPolicy
        self._maybe_recommend_cpu()

    def _maybe_recommend_cpu(self, mlp_class_name: str = "ActorCriticPolicy") -> None:
        """
        Recommend to use CPU only when using A2C/PPO with MlpPolicy.

        :param: The name of the class for the default MlpPolicy.
        """
        policy_class_name = self.policy_class.__name__
        if self.device != th.device("cpu") and policy_class_name == mlp_class_name:
            warnings.warn(
                f"You are trying to run {self.__class__.__name__} on the GPU, "
                "but it is primarily intended to run on the CPU when not using a CNN policy "
                f"(you are using {policy_class_name} which should be a MlpPolicy). "
                "See https://github.com/DLR-RM/stable-baselines3/issues/1245 "
                "for more info. "
                "You can pass `device='cpu'` or `export CUDA_VISIBLE_DEVICES=` to force using the CPU."
                "Note: The model will train, but the GPU utilization will be poor and "
                "the training might take longer than on CPU.",
                UserWarning,
            ) 
    
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs, eps = self.rsample_action(obs_tensor)
                # actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            # new_obs = np.expand_dims(np.array(env.envs[0].unwrapped.state), axis = 0)
            self.num_timesteps += env.num_envs
            if dones:
                self.high_fidelity_env_last_termination_step = self.num_timesteps

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1
            if self.num_timesteps % self.eval_freq == 0:
                eval_results = evaluate_policy(self.policy, env, n_eval_episodes=10, deterministic=True)
                wandb.log({"rollout/eval_returns": eval_results[0]}, step=self.num_timesteps)
                self.eval_returns.append(eval_results[0])
                self.eval_steps.append(self.num_timesteps)

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
                eps,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def learn(
        self: SelfOnPolicyAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfOnPolicyAlgorithm:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None
        assert self.env_low is not None

        while self.num_timesteps < total_timesteps:
            # make sure two-fidelity buffers have the same start
            last_obs = copy.deepcopy(self._last_obs)
            last_episode_starts = copy.deepcopy(self._last_episode_starts)
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer_high_fidelity, n_rollout_steps=self.n_steps)
            if self.useMultiFidelity:
                self.collect_rollouts_constrained_s0(self.env_low, callback, self.rollout_buffer_low_fidelity_constrained, \
                                                    n_rollout_steps=self.n_steps, last_obs=last_obs, last_episode_starts=last_episode_starts, \
                                                        current_step_id = self.high_fidelity_env_last_batch_step_id)
                self.collect_rollouts_unconstrained_low_fidelity(self.env_low_multiple, callback, self.rollout_buffer_low_fidelity_unconstrained, \
                                                    n_rollout_steps=self.large_n_steps, last_obs=last_obs, last_episode_starts=last_episode_starts)
            else:
                pass
            self.high_fidelity_env_last_batch_step_id = self.num_timesteps - self.high_fidelity_env_last_termination_step
            if not continue_training:
                break

            if self.useMultiFidelity:
                with th.no_grad():
                    self.get_truncated_aligned_buffers()
                    assert th.all(self.rollout_buffer_high_fidelity.actions_eps == self.rollout_buffer_low_fidelity_constrained.actions_eps), "Action noise not constrained"

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                self._dump_logs(iteration)

            self.train()

        callback.on_training_end()

        return self

    def get_truncated_aligned_buffers(self):
        high_s0_ids = (np.where(self.rollout_buffer_high_fidelity.episode_starts)[0])
        low_s0_ids = (np.where(self.rollout_buffer_low_fidelity_constrained.episode_starts)[0]) 
        high_s0s = self.rollout_buffer_high_fidelity.observations[high_s0_ids]

        if high_s0_ids.size == low_s0_ids.size and np.all(high_s0_ids == low_s0_ids):
            assert np.all(self.rollout_buffer_high_fidelity.observations[high_s0_ids] == self.rollout_buffer_low_fidelity_constrained.observations[low_s0_ids])
            self.rollout_buffer_high_fidelity.buffer_size = self.rollout_buffer_high_fidelity.observations.shape[0]
            self.rollout_buffer_low_fidelity_constrained.buffer_size = self.rollout_buffer_low_fidelity_constrained.observations.shape[0]
            return

        high_time_indices = np.arange(self.n_steps)
        low_time_indices = np.arange(self.n_steps)

        high_indices_slices = np.split(high_time_indices, high_s0_ids)
        low_indices_slices = np.split(low_time_indices, low_s0_ids)

        # TODO: optimize sampling process for low uncosntrained 
        num_traj_high = len(high_indices_slices)
        num_traj_low = len(low_indices_slices)
        num_traj = min(num_traj_high, num_traj_low)

        trimmed_high_indices_slices = []
        trimmed_low_indices_slices = []

        new_s0_ids = [0]
        for traj_id in np.arange(num_traj):
            if high_indices_slices[traj_id].size != 0 and low_indices_slices[traj_id].size != 0: 
                len_high = high_indices_slices[traj_id].size
                len_low = low_indices_slices[traj_id].size
                min_len = min(len_high, len_low)
                new_s0_ids.append(min_len + new_s0_ids[-1])
                trimmed_high_indices_slices.append(high_indices_slices[traj_id][:min_len])
                trimmed_low_indices_slices.append(low_indices_slices[traj_id][:min_len])
            else:
                continue

        trimmed_high_indices = np.concatenate(trimmed_high_indices_slices)
        trimmed_low_indices = np.concatenate(trimmed_low_indices_slices)

        original_high_buffer = deepcopy(self.rollout_buffer_high_fidelity)
        original_low_buffer_constrained = deepcopy(self.rollout_buffer_low_fidelity_constrained)

        self.rollout_buffer_high_fidelity.observations = self.rollout_buffer_high_fidelity.observations[trimmed_high_indices]
        self.rollout_buffer_high_fidelity.actions = self.rollout_buffer_high_fidelity.actions[trimmed_high_indices]
        self.rollout_buffer_high_fidelity.rewards = self.rollout_buffer_high_fidelity.rewards[trimmed_high_indices]
        self.rollout_buffer_high_fidelity.advantages = self.rollout_buffer_high_fidelity.advantages[trimmed_high_indices]
        self.rollout_buffer_high_fidelity.returns = self.rollout_buffer_high_fidelity.returns[trimmed_high_indices]
        self.rollout_buffer_high_fidelity.episode_starts = self.rollout_buffer_high_fidelity.episode_starts[trimmed_high_indices]
        self.rollout_buffer_high_fidelity.log_probs = self.rollout_buffer_high_fidelity.log_probs[trimmed_high_indices]
        self.rollout_buffer_high_fidelity.values = self.rollout_buffer_high_fidelity.values[trimmed_high_indices]
        self.rollout_buffer_high_fidelity.actions_eps = self.rollout_buffer_high_fidelity.actions_eps[trimmed_high_indices]

        self.rollout_buffer_low_fidelity_constrained.observations = self.rollout_buffer_low_fidelity_constrained.observations[trimmed_low_indices]
        self.rollout_buffer_low_fidelity_constrained.actions = self.rollout_buffer_low_fidelity_constrained.actions[trimmed_low_indices]
        self.rollout_buffer_low_fidelity_constrained.rewards = self.rollout_buffer_low_fidelity_constrained.rewards[trimmed_low_indices]
        self.rollout_buffer_low_fidelity_constrained.advantages = self.rollout_buffer_low_fidelity_constrained.advantages[trimmed_low_indices]
        self.rollout_buffer_low_fidelity_constrained.returns = self.rollout_buffer_low_fidelity_constrained.returns[trimmed_low_indices]
        self.rollout_buffer_low_fidelity_constrained.episode_starts = self.rollout_buffer_low_fidelity_constrained.episode_starts[trimmed_low_indices]
        self.rollout_buffer_low_fidelity_constrained.log_probs = self.rollout_buffer_low_fidelity_constrained.log_probs[trimmed_low_indices]
        self.rollout_buffer_low_fidelity_constrained.values = self.rollout_buffer_low_fidelity_constrained.values[trimmed_low_indices]
        self.rollout_buffer_low_fidelity_constrained.actions_eps = self.rollout_buffer_low_fidelity_constrained.actions_eps[trimmed_low_indices]

        self.rollout_buffer_high_fidelity.buffer_size = len(trimmed_high_indices)
        self.rollout_buffer_low_fidelity_constrained.buffer_size = len(trimmed_low_indices)

        # some unit tests
        assert np.all(self.rollout_buffer_high_fidelity.observations[new_s0_ids[:-1]] == self.rollout_buffer_low_fidelity_constrained.observations[new_s0_ids[:-1]])

        self.truncated_high_returns = self.rollout_buffer_high_fidelity.returns.squeeze()
        self.truncated_low_returns = self.rollout_buffer_low_fidelity_constrained.returns.squeeze()
        self.truncated_high_log_prob = self.rollout_buffer_high_fidelity.log_probs.squeeze()
        self.truncated_low_log_prob = self.rollout_buffer_low_fidelity_constrained.log_probs.squeeze()

            
    def collect_rollouts_constrained_s0(
        self,
        env_low,
        callback: BaseCallback,
        rollout_buffer_low_fidelity_constrained: RolloutBuffer,
        n_rollout_steps: int,
        last_obs,
        last_episode_starts,
        current_step_id,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer_low_fidelity_constrained.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env_low.num_envs)

        high_buffer_id_slices = compute_buffer_episode_slices(self.rollout_buffer_high_fidelity)
        
        ids = self.rollout_buffer_high_fidelity.episode_starts.flatten().astype(bool)
        s0_constraints = self.rollout_buffer_high_fidelity.observations[ids]
        episode_counter = 0
        episode_step_pointer = [0, 0] # (current_episode, current_step)
        if not last_episode_starts:
            self.reset_mujoco_env_with_constraint(env_low, seed=self.seed, initial_state_constraint=last_obs)
            assert np.all(env_low.unwrapped.envs[0].unwrapped._get_obs() == last_obs)
        else:
            self.reset_mujoco_env_with_constraint(env_low, seed=self.seed, initial_state_constraint=s0_constraints[episode_counter])
            assert np.all(env_low.unwrapped.envs[0].unwrapped._get_obs() == s0_constraints[episode_counter])
            episode_counter += 1
            last_obs = np.array([subenv.unwrapped._get_obs() for subenv in env_low.unwrapped.envs])

        initial_last_obs = copy.deepcopy(last_obs)
        for subenv in env_low.unwrapped.envs: # set the current step number to be the same as high-fidelity environment
            assert type(subenv) == TimeLimit
            subenv._elapsed_steps = current_step_id

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env_low.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(last_obs, self.device)
                
                # if current step id is within the high buffer range, sample with constrained action noise
                if episode_step_pointer[1] >= high_buffer_id_slices[episode_step_pointer[0]].size:
                    actions, values, log_probs, eps = self.rsample_action(obs_tensor)
                else:
                    actions, values, log_probs, eps = self.rsample_action(obs_tensor, deterministic=True, \
                                                                      eps = self.rollout_buffer_high_fidelity.actions_eps[high_buffer_id_slices[episode_step_pointer[0]][episode_step_pointer[1]]])
                # actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)


            new_obs, rewards, terminated, truncated, infos = env_low.step(clipped_actions)
            dones = np.logical_or(terminated, truncated)
            episode_step_pointer[1] += 1

            # reconstrain the low fidelity environment (unused by default)
            if episode_step_pointer[1] < high_buffer_id_slices[episode_step_pointer[0]].size and episode_step_pointer[1] % self.low_env_recons_freq == 0:
                step_id = deepcopy(env_low.unwrapped.envs[0]._elapsed_steps)
                self.reset_mujoco_env_with_constraint(env_low, seed=self.seed, initial_state_constraint=self.rollout_buffer_high_fidelity.observations[high_buffer_id_slices[episode_step_pointer[0]][episode_step_pointer[1]]])
                assert np.all(env_low.unwrapped.envs[0].unwrapped._get_obs() == self.rollout_buffer_high_fidelity.observations[high_buffer_id_slices[episode_step_pointer[0]][episode_step_pointer[1]]])
                print("reconstrain low env at step: ", high_buffer_id_slices[episode_step_pointer[0]][episode_step_pointer[1]])
                new_obs = np.array([subenv.unwrapped._get_obs() for subenv in env_low.unwrapped.envs])
                for subenv in env_low.unwrapped.envs: # set the current step number to be the same as high-fidelity environment
                    assert type(subenv) == TimeLimit
                    subenv._elapsed_steps = step_id

            self._update_info_buffer(infos, dones)
            n_steps += 1 
            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            rollout_buffer_low_fidelity_constrained.add(
                last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
                eps,
            )
            if dones:
                if episode_counter >= sum(ids):
                    episode_counter = 0
                if sum(ids) > 0:
                    # env_low.reset(seed = self.seed, options = {"initial_state_constraint" : s0_constraints[episode_counter]})
                    self.reset_mujoco_env_with_constraint(env_low, seed=self.seed, initial_state_constraint=s0_constraints[episode_counter])
                    assert np.all(env_low.unwrapped.envs[0].unwrapped._get_obs() == s0_constraints[episode_counter])
                    episode_counter += 1
                    last_obs = np.array([subenv.unwrapped._get_obs() for subenv in env_low.unwrapped.envs])
                    last_episode_starts = dones
                else:
                    # env_low.reset(seed = self.seed, options = {"initial_state_constraint" : initial_last_obs})
                    self.reset_mujoco_env_with_constraint(env_low, seed=self.seed, initial_state_constraint=initial_last_obs)
                    assert np.all(env_low.unwrapped.envs[0].unwrapped._get_obs() == initial_last_obs)
                    last_obs = np.array([subenv.unwrapped._get_obs() for subenv in env_low.unwrapped.envs])
                    last_episode_starts = dones
                
                episode_step_pointer[0] += 1
                episode_step_pointer[1] = 0
                if episode_step_pointer[0] >= len(high_buffer_id_slices):
                    episode_step_pointer[0] = 0
            else:
                last_obs = new_obs  # type: ignore[assignment]
                last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]
            # TODO: make a decision: which network to use to compute this

        rollout_buffer_low_fidelity_constrained.compute_returns_and_advantage(last_values=values, dones=dones)

        return True

    def collect_rollouts_unconstrained_low_fidelity(
        self,
        env_low_multiple,
        callback: BaseCallback,
        rollout_buffer_low_fidelity_unconstrained: RolloutBuffer,
        n_rollout_steps: int,
        last_obs,
        last_episode_starts,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer_low_fidelity_unconstrained.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env_low_multiple.num_envs)

        # env_low_multiple.reset(seed = self.seed)
        # last_obs = np.array([subenv.unwrapped._get_obs() for subenv in env_low_multiple.unwrapped.envs])
        last_obs = env_low_multiple.unwrapped._observations
        last_episode_starts = True

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env_low_multiple.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, terminated, truncated, infos = env_low_multiple.step(clipped_actions)
            dones = np.logical_or(terminated, truncated)

            if np.any(dones):
                for idx, done in enumerate(dones):
                    if done:
                        env_low_multiple.unwrapped.envs[idx].reset()
                        env_low_multiple.unwrapped._autoreset_envs[idx] = False # a temporary fix, use autoreset mode when it's released
                new_obs= np.array([subenv.unwrapped._get_obs() for subenv in env_low_multiple.unwrapped.envs])

            self._update_info_buffer(infos, dones)
            n_steps += 1 
            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            rollout_buffer_low_fidelity_unconstrained.add(
                last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
                th.zeros(actions.shape, device=self.device), # place holder for action noise
            )
            last_obs = new_obs  # type: ignore[assignment]
            last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer_low_fidelity_unconstrained.compute_returns_and_advantage(last_values=values, dones=dones)


        return True

    def _dump_logs(self, iteration: int) -> None:
        """
        Write log.

        :param iteration: Current logging iteration
        """
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.logger.record("time/iterations", iteration, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
            wandb.log({"rollout/ep_rew_mean": safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]),
                       "rollout/ep_len_mean": safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer])}, step=self.num_timesteps)
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        wandb.log({"time/fps": fps, "time/time_elapsed": int(time_elapsed), "time/total_timesteps": self.num_timesteps}, step=self.num_timesteps)
        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
            wandb.log({"rollout/success_rate": safe_mean(self.ep_success_buffer)}, step=self.num_timesteps)
        self.logger.dump(step=self.num_timesteps)

    def reset_mujoco_env_with_constraint(self, env, seed = None, initial_state_constraint=None):
        if seed is not None:
            env.reset(seed=seed)
        else:
            env.reset()
        if initial_state_constraint is not None:
            env.unwrapped.envs[0].unwrapped.set_state(initial_state_constraint[0,self.qpos_ids], initial_state_constraint[0,self.qvel_ids])
        assert np.all(env.unwrapped.envs[0].unwrapped._get_obs() == initial_state_constraint)
    
    def rsample_action(self, obs: th.Tensor, deterministic: bool = False, eps = None) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Sample an action from the policy and also return the noise vector used for reparametrization.

        :param obs: Observation
        :param deterministic: Whether to sample deterministically
        :return: action, value, log_prob
        """
        # Preprocess the observation if needed
        features = self.policy.extract_features(obs)
        if self.policy.share_features_extractor:
            latent_pi, latent_vf = self.policy.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.policy.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.policy.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.policy.value_net(latent_vf)
        distribution = self.policy._get_action_dist_from_latent(latent_pi)
        
        assert type(distribution) == DiagGaussianDistribution, "This reparametrization trick has only been implemented for DiagGaussianDistribution"
        
        # TODO: check: if we need to set random seed separately here
        if deterministic and eps is None: # determistic action
            actions = distribution.mode()
        elif deterministic and eps is not None: # compute action with constrained noise vector
            # print("Deterministic action with constrained noise vector")
            actions = distribution.distribution.loc + eps * distribution.distribution.scale
        elif not deterministic:
            shape = distribution.distribution._extended_shape(th.Size())
            eps = _standard_normal(shape, dtype=th.float64, \
                                    device=distribution.distribution.loc.device)
            actions = distribution.distribution.loc + eps * distribution.distribution.scale
        
        # actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.policy.action_space.shape))
        
        return actions, values, log_prob, eps