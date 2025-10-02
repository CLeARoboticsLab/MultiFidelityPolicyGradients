import gymnasium as gym
import argparse
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../odrl_benchmark/'))
import wandb
import copy

import numpy as np
import torch as th
from gymnasium.wrappers.vector.dict_info_to_list import DictInfoToList
import random
import yaml
import json
from pathlib import Path
from training_utils import call_multi_fidelity_mujoco_env, build_model
from stable_baselines3.common.evaluation import evaluate_policy
from wandb.integration.sb3 import WandbCallback

# Create a simple args object from the config instead of parsing command line
class Args:
    def __init__(self):
        self.save_models = 1  # default value
    

def objective(config=None):

    args = Args()

    if not config is None:
        # Add config to args
        for key, value in config.items():
            setattr(args, key, value)

    # we support different ways of specifying tasks, e.g., hopper-friction, hopper_friction, hopper_morph_torso_easy, hopper-morph-torso-easy
    if '_' in args.env:
        args.env = args.env.replace('_', '-')

    if 'halfcheetah' in args.env or 'hopper' in args.env or 'walker2d' in args.env or args.env.split('-')[0] == 'ant':
        domain = 'mujoco'
    else:
        raise NotImplementedError
    print(domain)

    call_env = {
        'mujoco': call_multi_fidelity_mujoco_env,
    }

    # determine referenced environment name
    ref_env_name = args.env + '-' + str(args.shift_level)
    
    src_env_name = args.env.split('-')[0]
    src_env_name_config = src_env_name
    tar_env_name = args.env

    src_env_config = {
        'env_name': src_env_name,
        'shift_level': args.shift_level,
    }
    env_low = call_env[domain](src_env_config, env_type = 'low')
    env_low = DictInfoToList(env_low)
    env_low.reset(seed=args.random_seed)
    env_low_eval = copy.deepcopy(env_low.unwrapped.envs[0])
    env_low_eval.reset(seed=args.random_seed + 100)


    env_low_multiple = call_env[domain](src_env_config, env_type = 'low_multiple', num_envs = args.num_env_mean_calculation)
    env_low_multiple = DictInfoToList(env_low_multiple)
    env_low_multiple.reset(seed=args.random_seed)

    tar_env_config = {
        'env_name': tar_env_name,
        'shift_level': args.shift_level,
    }
    env_high = call_env[domain](tar_env_config, env_type = 'high')
    env_high.reset(seed=args.random_seed)
    env_high_eval = copy.deepcopy(env_high)
    env_high_eval.reset(seed=args.random_seed + 100)


    # Seed python RNG
    random.seed(args.random_seed)
    # Seed numpy RNG
    np.random.seed(args.random_seed)
    # seed the RNG for all devices (both CPU and CUDA)
    th.manual_seed(args.random_seed)

    # Build the model using the extracted function
    model = build_model(args, env_high, env_low_multiple, env_low)
    

    model.learn(total_timesteps=args.max_step)

    eval_results = evaluate_policy(model, env_high, n_eval_episodes=10, deterministic=True)
    eval_score = eval_results[0]

    return eval_score
    

def main():
    wandb.init(project=args_wandb_project_name)
    eval_score = objective(wandb.config)
    wandb.log({"eval_score": eval_score})


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    # general parameters
    parser.add_argument("--algorithm", help="algorithm to use",
                        type=str, default="baseline_reinforce_mfpg")
    parser.add_argument("--sweep_id", help="sweep id",
                        type=str, default=None)
    parser.add_argument("--config_file_name", help="config file name",
                        type=str, default="mfpg_gravity_friction")
    parser.add_argument("--wandb_project_name", help="wandb project name",
                        type=str, default="MultiFidelityPolicyGradients")
    args = parser.parse_args()

    # Load config from yaml file
    config_path = f"{str(Path(__file__).parent.parent.parent.absolute())}/config/odrl_mujoco_sweep/{args.algorithm}/{args.config_file_name}.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r", encoding='utf-8') as f:
            config = yaml.safe_load(f)

    assert args.algorithm == config["parameters"]["algorithm"]["values"][0], "The algorithm name in the config file and the command line argument must be the same"

    # Declare args.algorithm as global variable
    global args_algorithm
    global args_wandb_project_name
    args_algorithm = args.algorithm
    args_wandb_project_name = args.wandb_project_name

    sweep_id = wandb.sweep(sweep=config, project=args_wandb_project_name)

    if not args.sweep_id is None:
        sweep_id = args.sweep_id

    wandb.agent(sweep_id, function=main, count=150)




