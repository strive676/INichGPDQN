"""Double DQN agent implementation for state-dependent action selection."""
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dqn_module.q_network import QNetwork
from dqn_module.replay_buffer import ReplayBuffer


class DoubleDQN:
    """Double DQN（在线网络 + 目标网络 + 经验回放）。

    `hidden_dims`（**新增，默认 None**）：隐藏层每层宽度列表，支持任意层数；为 None 时
    走 `hidden_dim` 的**历史两层**路径 ⇒ 与改动前逐位一致（证据见
    `tools/verify_paper_params_defaults.py` 的 [B] 节）。
    """

    def __init__(self, state_dim=22, action_dim=10, hidden_dim=128, lr=1e-4,
                 gamma=0.95, batch_size=64, replay_capacity=5000,
                 target_update_steps=500, grad_clip=5.0, hidden_dims=None, n_step=1):
        self.q_network = QNetwork(state_dim, action_dim, hidden_dim, hidden_dims=hidden_dims)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dim, hidden_dims=hidden_dims)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(replay_capacity)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_steps = target_update_steps
        self.grad_clip = grad_clip
        self.action_dim = action_dim
        self.step_count = 0
        self.n_step = int(n_step)
        if self.n_step < 1:
            raise ValueError("n_step must be >= 1")
        self._n_step_queue = deque()

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            q = self.q_network(torch.FloatTensor(np.asarray(state, dtype=float)))
            return int(q.argmax().item())

    def action_values(self, state) -> np.ndarray:
        """返回**全部**动作的 Q 值向量（供 `DSH_SWITCH_MODE=q_gate` 的闸门使用）。

        `select_action` 只给 argmax，闸门需要"和默认动作比大小"⇒ 必须拿到整条 Q。
        新增方法而非改 `select_action` ⇒ 既有路径逐字不变。
        """
        with torch.no_grad():
            q = self.q_network(torch.FloatTensor(np.asarray(state, dtype=float)))
            return q.detach().cpu().numpy().astype(float).ravel()

    def target_values(self, state) -> np.ndarray:
        """返回**目标网络**对全部动作的 Q 值向量（`Q_target(s,·)`）。

        仅供 **P8-1 Bellman/Return 审计**（`DSH_RETURN_AUDIT`）**只读**测量使用：严格
        Double DQN 的一阶目标 `Y_1(a) = r_a + γ·Q_target(s'_a, argmax_b Q_online(s'_a,b))`
        需要能读到目标网的值，而 `action_values` 只读在线网。

        实现与 `action_values` **逐字同构**（同一 `torch.no_grad()` + 同一 `.ravel()`
        出口口径），只是把网络换成 `self.target_network`；**不改 `update()`/`select_action`
        的任何一步**（新增方法 ⇒ 训练路径逐字节不变：目标网的值只在 `update()` 里被
        `gather` 取一个动作位，本方法不参与任何梯度与状态更新）。

        ⚠ 严格只读：`torch.no_grad()` + 不触碰 `step_count`/`replay_buffer`/优化器。
        """
        with torch.no_grad():
            q = self.target_network(torch.FloatTensor(np.asarray(state, dtype=float)))
            return q.detach().cpu().numpy().astype(float).ravel()

    def _emit_n_step(self, length):
        items = list(self._n_step_queue)[:length]
        first, last = items[0], items[-1]
        ret = sum((self.gamma ** i) * float(item[2]) for i, item in enumerate(items))
        discount = self.gamma ** len(items)
        self.replay_buffer.add(first[0], first[1], ret, last[3], last[4], discount=discount)
        self._n_step_queue.popleft()

    def store(self, state, action, reward, next_state, done):
        if self.n_step == 1:
            self.replay_buffer.add(state, action, reward, next_state, done,
                                  discount=self.gamma)
            return
        self._n_step_queue.append((np.asarray(state, dtype=float).copy(), int(action),
                                   float(reward), np.asarray(next_state, dtype=float).copy(), bool(done)))
        if len(self._n_step_queue) >= self.n_step:
            self._emit_n_step(self.n_step)
        if done:
            while self._n_step_queue:
                self._emit_n_step(len(self._n_step_queue))

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None
        states, actions, rewards, next_states, dones, discounts = self.replay_buffer.sample(
            self.batch_size, include_discount=True)
        q_values = self.q_network(states).gather(1, actions)
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)
            next_q = self.target_network(next_states).gather(1, next_actions)
            target = rewards + discounts * next_q * (1 - dones)
        loss = nn.MSELoss()(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), self.grad_clip)
        self.optimizer.step()
        self.step_count += 1
        if self.step_count % self.target_update_steps == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        return float(loss.item())
