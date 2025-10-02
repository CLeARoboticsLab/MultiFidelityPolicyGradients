import numpy as np
import gymnasium as gym
from typing import Dict
from pathlib import Path

def compute_buffer_episode_slices(buffer):
    """
    Compute the slices for the buffer episodes.
    :param buffer: (RolloutBuffer)
    :return: (np.ndarray)
    """
    s0_ids = (np.where(buffer.episode_starts)[0])
    time_indices = np.arange(buffer.returns.shape[0]) 
    time_indices_slices = np.split(time_indices, s0_ids)
    if time_indices_slices[0].size == 0:
        time_indices_slices = time_indices_slices[1:]
    return time_indices_slices