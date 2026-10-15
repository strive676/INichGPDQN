"""Replay buffer for storing and sampling reinforcement learning transitions."""
import random
from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity=5000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done, discount=None):
        # `discount=None` retains the legacy one-step replay contract.
        self.buffer.append((state, action, reward, next_state, done,
                            float(discount) if discount is not None else None))

    def sample(self, batch_size, include_discount=False):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, discounts = zip(*batch)
        result = (torch.FloatTensor(np.array(states)),
                  torch.LongTensor(np.array(actions)).unsqueeze(1),
                  torch.FloatTensor(np.array(rewards)).unsqueeze(1),
                  torch.FloatTensor(np.array(next_states)),
                  torch.FloatTensor(np.array(dones)).unsqueeze(1))
        if not include_discount:
            return result
        if any(x is None for x in discounts):
            raise ValueError("discount requested for replay item without a discount")
        return result + (torch.FloatTensor(np.array(discounts)).unsqueeze(1),)

    def __len__(self):
        return len(self.buffer)
