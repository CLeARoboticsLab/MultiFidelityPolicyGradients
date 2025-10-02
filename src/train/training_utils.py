import sys
import os
import gymnasium as gym
from typing import Dict
from pathlib import Path
import copy
import wandb

def build_model(args, env_high, env_low_multiple, env_low):
    """
    Build and return the appropriate model based on the algorithm specified in args.
    
    Args:
        args: Argument object containing algorithm name and configuration parameters
        env_high: High fidelity environment
        env_low_multiple: Multiple low fidelity environments for mean calculation
        env_low: Single low fidelity environment
        
    Returns:
        model: The initialized model based on the specified algorithm
    """
    if args.algorithm == "baseline_reinforce_mfpg":
        sys.path.append(os.path.join(os.path.dirname(__file__), '../baseline_reinforce_mfpg/'))
        from baseline_reinforce_mfpg import BaselineREINFORCE
        from buffers import MyRolloutBuffer
        if args.multi_fidelity: # MFPG
            model = BaselineREINFORCE("MlpPolicy", env_high, verbose=1, n_steps = args.buffer_size, num_env_mean_calculation=args.num_env_mean_calculation, gamma=args.gamma, vf_coef=args.vf_coef, max_grad_norm=args.max_grad_norm,
                            normalize_advantage=args.normalize_advantage, rollout_buffer_class=MyRolloutBuffer, env_low_multiple = env_low_multiple, env_low=env_low, seed = args.random_seed, 
                            useMultiFidelity = args.multi_fidelity, learning_rate = args.lr, 
                            shift_level = args.shift_level, 
                            env_name = args.env,
                            use_ema_c=bool(args.use_ema_c), c_ema_alpha=args.c_ema_alpha, eval_freq=args.eval_freq)
        else: # High-Fidelity Only
            model = BaselineREINFORCE("MlpPolicy", env_high, verbose=1, n_steps = args.buffer_size, gamma=args.gamma, vf_coef=args.vf_coef, max_grad_norm=args.max_grad_norm,
                            normalize_advantage=args.normalize_advantage, rollout_buffer_class=MyRolloutBuffer, env_low_multiple = env_low_multiple, env_low=env_low, seed = args.random_seed, 
                            useMultiFidelity = args.multi_fidelity, learning_rate = args.lr, 
                            shift_level = args.shift_level, 
                            env_name = args.env,
                            eval_freq=args.eval_freq)
    elif args.algorithm == "par_baseline_reinforce":
        sys.path.append(os.path.join(os.path.dirname(__file__), '../baselines/par/'))
        sys.path.append(os.path.join(os.path.dirname(__file__), '../baselines/common/'))
        from par_baseline_reinforce import PARBaselineREINFORCE
        from baseline_reinforce_rollout_buffer import BaselineReinforceRolloutBuffer
        model = PARBaselineREINFORCE("MlpPolicy", env_high, verbose=1, n_steps = args.buffer_size, num_low_fidelity_envs=args.num_env_mean_calculation,
                        rollout_buffer_class=BaselineReinforceRolloutBuffer, env_low_multiple = env_low_multiple, seed = args.random_seed, 
                        learning_rate = args.lr, shift_level = args.shift_level, env_name = args.env, encoder_learning_rate = args.encoder_learning_rate,
                        reward_augmentation_beta = args.reward_augmentation_beta, tau = args.tau)
    elif args.algorithm == "darc_baseline_reinforce":
        sys.path.append(os.path.join(os.path.dirname(__file__), '../baselines/darc/'))
        sys.path.append(os.path.join(os.path.dirname(__file__), '../baselines/common/'))
        from darc_baseline_reinforce import DARC_BaselineREINFORCE
        from baseline_reinforce_rollout_buffer import BaselineReinforceRolloutBuffer
        model = DARC_BaselineREINFORCE("MlpPolicy", env_high, verbose=1, n_steps = args.buffer_size, num_low_fidelity_envs=args.num_env_mean_calculation,
                        rollout_buffer_class=BaselineReinforceRolloutBuffer, env_low_multiple = env_low_multiple, seed = args.random_seed, 
                        learning_rate = args.lr, shift_level = args.shift_level, env_name = args.env, classifier_learning_rate = args.classifier_learning_rate,
                        classifier_hidden_sizes = args.classifier_hidden_sizes, classifier_gaussian_noise_std = args.classifier_gaussian_noise_std, reward_augmentation_beta = args.reward_augmentation_beta,
                        classifier_batch_size = args.classifier_batch_size, warmup_steps = args.warmup_steps)
    elif args.algorithm == "low_fidelity_only_baseline_reinforce":
        sys.path.append(os.path.join(os.path.dirname(__file__), '../baselines/low_fidelity_only/'))
        sys.path.append(os.path.join(os.path.dirname(__file__), '../baselines/common/'))
        from low_fidelity_only_baseline_reinforce import LowFidelityOnlyBaselineREINFORCE
        from baseline_reinforce_rollout_buffer import BaselineReinforceRolloutBuffer
        model = LowFidelityOnlyBaselineREINFORCE("MlpPolicy", env_high, verbose=1, n_steps = args.buffer_size, num_low_fidelity_envs=args.num_env_mean_calculation,
                        rollout_buffer_class=BaselineReinforceRolloutBuffer, env_low_multiple = env_low_multiple, seed = args.random_seed, 
                        learning_rate = args.lr, shift_level = args.shift_level, env_name = args.env)
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")
        
    return model

def call_multi_fidelity_mujoco_env(env_config: Dict, env_type: str = "high", num_envs: int = 1) -> gym.Env:
    env_name     =   env_config['env_name'].lower()     #   eg. "hopper_friction" or "hopper_morph_foot", body_pard is required only in "morph" or 'kinematic" mode
    shift_level  =   env_config['shift_level']          #   either float(0.1/0.5/...) or level(easy/medium/hard)

    if '-' in env_name:
        env_name = env_name.replace('-', '_')

    # assert the shift level legal
    if 'morph' in env_name or 'kinematic' in env_name:
        assert shift_level in ['easy', 'medium', 'hard'], 'The required shift is not available yet, please consider modify the xml file on your own or use the shift scale among easy, medium, hard, easier'
    if 'friction' in env_name or 'gravity' in env_name:
        assert float(shift_level) in [0.1, 0.5, 0.8, 1.0, 1.2, 2.0, 5.0], 'The required shift is not available yet, please consider modify the xml file on your own or use the shift scale among 0.1, 0.5, 2.0, 5.0'
    assert env_type in ['high', 'low', 'low_multiple'], 'available env_type are high, low, low_multiple'

    if 'hopper' in env_name:
        if env_name == 'hopper' and env_type == 'low':
            return gym.make_vec('Hopper-v3', exclude_current_positions_from_observation=False, max_episode_steps=1000)
        elif env_name == 'hopper' and env_type == 'low_multiple':
            return gym.make_vec('Hopper-v3', num_envs=num_envs, exclude_current_positions_from_observation=False, max_episode_steps=1000)
        elif ('friction' in env_name or 'gravity' in env_name) and env_type == 'high':
            return gym.make('Hopper-v3', exclude_current_positions_from_observation=False, max_episode_steps=1000, 
                        xml_file=f"{str(Path(__file__).parent.parent.parent.absolute())}/odrl_benchmark/envs/mujoco/assets/{env_name}_{float(shift_level)}.xml")
        elif 'noise' in env_name and env_type == 'high':
            # todo: the modification is directly applied on the executed action, no need to modify the xml file itself
            return gym.make('Hopper-v3', exclude_current_positions_from_observation=False, max_episode_steps=1000)
        elif ('morph' in env_name or 'kinematic' in env_name) and env_type == 'high':
            return gym.make('Hopper-v3', exclude_current_positions_from_observation=False, max_episode_steps=1000, 
                        xml_file=f"{str(Path(__file__).parent.parent.parent.absolute())}/odrl_benchmark/envs/mujoco/assets/{env_name}_{shift_level}.xml")
        else:
            print("env_name {env_name} is illegal or not implemented")
            raise NotImplementedError
    elif "halfcheetah" in env_name:
        if env_name == 'halfcheetah' and env_type == 'low':
            return gym.make_vec('HalfCheetah-v3', exclude_current_positions_from_observation=False, max_episode_steps=1000)
        elif env_name == 'halfcheetah' and env_type == 'low_multiple':
            return gym.make_vec('HalfCheetah-v3', num_envs=num_envs, exclude_current_positions_from_observation=False, max_episode_steps=1000)
        elif ('friction' in env_name or 'gravity' in env_name) and env_type == 'high':
            return gym.make('HalfCheetah-v3', exclude_current_positions_from_observation=False, max_episode_steps=1000, 
                        xml_file=f"{str(Path(__file__).parent.parent.parent.absolute())}/odrl_benchmark/envs/mujoco/assets/{env_name}_{float(shift_level)}.xml")
        elif 'noise' in env_name and env_type == 'high':
            # todo: the modification is directly applied on the executed action, no need to modify the xml file itself
            return gym.make('HalfCheetah-v3', exclude_current_positions_from_observation=False, max_episode_steps=1000)
        elif ('morph' in env_name or 'kinematic' in env_name) and env_type == 'high':
            return gym.make('HalfCheetah-v3', exclude_current_positions_from_observation=False, max_episode_steps=1000, 
                        xml_file=f"{str(Path(__file__).parent.parent.parent.absolute())}/odrl_benchmark/envs/mujoco/assets/{env_name}_{shift_level}.xml")
        else:
            print("env_name {env_name} is illegal or not implemented")
            raise NotImplementedError
    elif "walker2d" in env_name:
        if env_name == 'walker2d' and env_type == 'low':
            return gym.make_vec('Walker2d-v3', exclude_current_positions_from_observation=False, max_episode_steps=1000)
        elif env_name == 'walker2d' and env_type == 'low_multiple':
            return gym.make_vec('Walker2d-v3', num_envs=num_envs, exclude_current_positions_from_observation=False, max_episode_steps=1000)
        elif ('friction' in env_name or 'gravity' in env_name) and env_type == 'high':
            return gym.make('Walker2d-v3', exclude_current_positions_from_observation=False, max_episode_steps=1000, 
                        xml_file=f"{str(Path(__file__).parent.parent.parent.absolute())}/odrl_benchmark/envs/mujoco/assets/{env_name}_{float(shift_level)}.xml")
        elif 'noise' in env_name and env_type == 'high':
            # todo: the modification is directly applied on the executed action, no need to modify the xml file itself
            return gym.make('Walker2d-v3', exclude_current_positions_from_observation=False, max_episode_steps=1000)
        elif ('morph' in env_name or 'kinematic' in env_name) and env_type == 'high':
            return gym.make('Walker2d-v3', exclude_current_positions_from_observation=False, max_episode_steps=1000, 
                        xml_file=f"{str(Path(__file__).parent.parent.parent.absolute())}/odrl_benchmark/envs/mujoco/assets/{env_name}_{shift_level}.xml")
        else:
            print("env_name {env_name} is illegal or not implemented")
            raise NotImplementedError
    else:
        print("env_name {env_name} is illegal or not implemented")
        raise NotImplementedError