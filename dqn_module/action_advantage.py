"""Optional pairwise action-advantage supervision utilities for DQN training."""
import numpy as np
import torch


def pairwise_loss(prediction, target, weights):
    """Weighted same-state action differences, invariant to a common value offset."""
    error = ((prediction[:, :, None]-prediction[:, None, :])
             -(target[:, :, None]-target[:, None, :]))
    pair_weights = weights[:, :, None]*weights[:, None, :]
    return ((error.square()*pair_weights).sum((1, 2))/pair_weights.sum((1, 2))).mean()


def targets_and_weights(costs, choices, reference, scale):
    costs = np.asarray(costs, dtype=float)
    choices = np.asarray(choices)
    if not np.isfinite(costs).all() or scale <= 0:
        raise ValueError("invalid costs/scale")
    labels = (costs[:, reference, None]-costs)/scale
    weights = np.ones_like(costs)
    for i, row in enumerate(choices):
        for choice in np.unique(row):
            same = row == choice
            if not np.allclose(costs[i, same], costs[i, same][0], atol=1e-8):
                raise ValueError("equivalent actions have different labels")
            weights[i, same] = 1./same.sum()
    return labels, weights


class AuxiliaryAgent:
    def __init__(self, agent, norm, raw_states, targets, weights, enabled,
                 supervised_steps=3000, seed=0):
        self.agent = agent
        self.enabled = enabled
        self.x = torch.tensor(norm.transform(raw_states), dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32)
        self.w = torch.tensor(weights, dtype=torch.float32)
        self.rng = np.random.default_rng(seed)
        self.optimizer = torch.optim.Adam(agent.q_network.parameters(), lr=.001)
        self.aux_updates = 0; self.aux_loss_sum = 0.; self.td_updates = 0; self.td_loss_sum = 0.
        self.pretrain_first = None; self.pretrain_last = None
        for _ in range(supervised_steps):
            value = self.supervised_update()
            if self.pretrain_first is None:
                self.pretrain_first = value
            self.pretrain_last = value
        self.agent.target_network.load_state_dict(self.agent.q_network.state_dict())

    def __getattr__(self, key):
        return getattr(self.agent, key)

    def supervised_update(self):
        ix = self.rng.integers(len(self.x), size=128)
        loss = pairwise_loss(self.agent.q_network(self.x[ix]), self.y[ix], self.w[ix])
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite pairwise loss")
        self.optimizer.zero_grad()
        (loss * float(self.enabled)).backward()
        torch.nn.utils.clip_grad_norm_(self.agent.q_network.parameters(), 5.)
        self.optimizer.step()
        value = float(loss.detach())
        self.aux_updates += 1; self.aux_loss_sum += value
        return value

    def update(self):
        loss = self.agent.update()
        if loss is not None:
            if not np.isfinite(loss):
                raise RuntimeError("nonfinite TD loss")
            self.td_updates += 1; self.td_loss_sum += loss
            self.supervised_update()
        return loss
