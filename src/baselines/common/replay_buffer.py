import numpy as np
import random
from collections import deque


class ReplayBuffer:
    """
    Flexible replay buffer that can store transitions with variable numbers of additional fields.
    Supports circular buffer behavior and random sampling.
    """
    
    def __init__(self, memory_size, batch_size):
        """
        Initialize replay buffer
        
        Args:
            memory_size: Maximum number of transitions to store
            batch_size: Default batch size for sampling
        """
        self.memory_size = int(memory_size)
        self.batch_size = batch_size
        self.buffer = deque(maxlen=self.memory_size)
        self.position = 0
        
    def add(self, state, action, reward, next_state, done_mask, *additional_fields):
        """
        Add a transition to the buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done_mask: Done flag (1.0 if episode ended, 0.0 otherwise)
            *additional_fields: Additional data (e.g., importance weights, discriminator outputs)
        """
        transition = (state, action, reward, next_state, done_mask) + additional_fields
        self.buffer.append(transition)
        
    def sample(self, batch_size=None):
        """
        Sample a batch of transitions
        
        Args:
            batch_size: Optional batch size (uses default if None)
            
        Returns:
            Tuple of (states, actions, rewards, next_states, done_masks, *additional_fields)
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        if len(self.buffer) < batch_size:
            # If not enough samples, return all available
            batch_size = len(self.buffer)
            
        # Randomly sample batch_size transitions
        batch = random.sample(self.buffer, batch_size)
        
        # Unzip the batch into separate arrays
        # Each transition is a tuple: (state, action, reward, next_state, done_mask, *additional_fields)
        unzipped = list(zip(*batch))
        
        # Convert to numpy arrays
        result = []
        for field in unzipped:
            if isinstance(field[0], np.ndarray):
                result.append(np.array(field))
            else:
                result.append(np.array(field))
                
        return tuple(result)
    
    def __len__(self):
        """Return the current number of transitions in the buffer"""
        return len(self.buffer)
    
    def clear(self):
        """Clear all transitions from the buffer"""
        self.buffer.clear()
        
    def is_full(self):
        """Check if the buffer is full"""
        return len(self.buffer) >= self.memory_size 