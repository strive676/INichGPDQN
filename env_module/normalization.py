"""State normalization utilities for reinforcement learning observations."""
import numpy as np


class Normalizer:
    """状态归一化：z-score 标准化，并对**近零方差维**做保护。

    为什么需要保护（实测 bug）：
        原实现为 (x − mean) / (std + eps)，eps=1e-8。而调用方只用少量**初始状态**
        拟合统计量，而初始状态下 BF / NIQ / MRT / n_urgent 等维度恒为 0（std 恰为 0）。
        这些维在 episode 中一旦变为非零（如 n_urgent=2），就会被放大到 2e8 量级。
        输入一个 1e8 量级的特征（Linear 默认初始化 ±0.21）⇒ 预激活 ~1e7、
        初始 MSE 损失达 1e10、Q 尺度失控（|Q|~520）、贪心策略塌缩为单一动作（94%），
        RL 因此退化成"固定使用某一条规则"。

    修复策略：std 小于 tol 的维度不做缩放（缩放因子取 1），即保留其原始量级，
    既避免 1/(std+eps) 的爆炸，也不丢信息；正常维仍严格按 z-score 变换（语义不变）。
    """

    def __init__(self, eps: float = 1e-8, tol: float = 1e-6, rel_tol: float = 1e-3):
        self.eps = eps
        self.tol = tol




        self.rel_tol = rel_tol
        self.mean = None
        self.std = None
        self.scale = None
        self.risky_dims = []

    def fit(self, samples: np.ndarray, fixed_dims=None):
        """拟合 mean/std/scale。

        `fixed_dims`（**可选，默认 None = 逐位等于改动前**）：这些维**不做 z-score**
        （mean 置 0、scale 置 1，即原样通过）。

        为什么需要它（2026-10-03，P2b-2 实测）：
            P2b-2 的 lookahead 特征块在**构造时就已经是 O(1)**（逐点按 `max|·|` 归一）。
            而 `collect_states_dual` 是**稀疏采样**（每 3 个排序点采 1 个 + 全部路由点），
            其中"分歧点且窗口完整"的比例本来就低 ⇒ 这些维在拟合样本里**绝大多数是 0**，
            std 会落在**危险带**：实测 `risky_dims = [23,24,25,26,31,32,33,34,35,36]`
            （std ∈ (1e-6, 1e-3·max_std)，而 max_std 由基础状态维给出、可达数千）
            ⇒ 常规 z-score 会把这些维**放大 1e2~1e5 倍**后送进网络，
            正是本文件开头那个「1e8 放大」bug 的同族事故（只是量级小一些）。
            而这里 z-score **没有任何收益**：块本身就是尺度自由、以 0 为中心的
            ⇒ 直接原样通过（scale=1、mean=0）才是正确的归一化口径。
        """
        samples = np.asarray(samples, dtype=float)
        if samples.ndim != 2:
            raise ValueError(f"fit 需要 2-D 样本 (n_samples, n_dims)，收到 shape={samples.shape}"
                             "（rank-1 输入会让 mean/std 塌缩为标量并静默给出错误缩放）")
        self.mean = samples.mean(axis=0)
        self.std = samples.std(axis=0)

        self.scale = np.where(self.std < self.tol, 1.0, self.std + self.eps)
        if fixed_dims is not None:
            idx = np.asarray(list(fixed_dims), dtype=int)
            if idx.size:
                self.mean[idx] = 0.0
                self.scale[idx] = 1.0
        mx = float(self.std.max()) if self.std.size else 0.0
        _fixed_set = set() if fixed_dims is None else set(int(i) for i in fixed_dims)
        self.risky_dims = [int(i) for i in np.flatnonzero(
            (self.std >= self.tol) & (self.std < self.rel_tol * mx))
            if int(i) not in _fixed_set]
        if self.risky_dims:
            import warnings
            warnings.warn(
                f"维度 {self.risky_dims} 的 std 落在 ({self.tol}, {self.rel_tol}·max_std) 危险带："
                f"这些维会被放大 1e2~1e6 倍；请检查特征定义或放宽 tol", stacklevel=2)
        return self

    def fixed_scale_dims(self):
        """返回被 `fixed_dims` 固定为"原样通过"（mean=0、scale=1）的维度索引，便于诊断。"""
        if self.mean is None or self.scale is None:
            return []
        return [int(i) for i in np.flatnonzero((self.mean == 0.0) & (self.scale == 1.0))]

    def transform(self, x) -> np.ndarray:
        if self.mean is None or self.scale is None:
            raise RuntimeError("Normalizer 未拟合：请先 fit(...)（原先的兜底分支用 std+eps ⇒ "
                              "缺少死维保护，会让 1e8 放大 bug 复活，已移除）")
        x = np.asarray(x, dtype=float)
        return (x - self.mean) / self.scale

    def dead_dims(self):
        """返回被识别为死维（未缩放）的维度索引，便于诊断。"""
        return np.flatnonzero(self.std < self.tol).tolist()


def collect_states(env, n_episodes: int = 6, seed0: int = 90000, rules=None):
    """在**真实 episode 过程中**采样状态（而不是只用初始状态拟合）。

    仅用初始状态拟合是本 bug 的源头：初始状态下 BF / NIQ / MRT / n_urgent 恒为 0，
    统计量严重失真（std=0）⇒ 后续 episode 中这些维被放大到 1e8。
    在 episode 内采样可得到有代表性的均值/方差，使各维都落在 O(1)。
    """
    import numpy as _np
    from env_module.fjsp_env import make_route_policy
    out = []
    for k in range(n_episodes):
        env.reset(seed0 + k)
        ea = make_route_policy("EA")
        i = 0
        while env.pending is not None:
            if env.pending == "route":
                env.step_route(ea(env))
            else:
                if i % 3 == 0:
                    out.append(env.state())
                i += 1
                rule = rules[(seed0 + k + i) % len(rules)] if rules else None
                if rule is not None:


                    env.step_sequence(rule, speed=1.0)
                else:
                    env.step_sequence_index(0)
    return _np.array(out)


def collect_states_dual(env, n_episodes: int = 6, seed0: int = 90000, rules=None,
                        ref_idx=None, la_feat_H=0, la_mag_H=0, la_mem=0, la_rich=None):
    """**单智能体双树**口径的状态采样：在**两类决策点上都采样**，末尾追加决策类型标志。

    与 `collect_states` 的差别（本轮改动的核心之一）：
        旧版只在 sequence 点采样、且状态是 22 维；而单智能体在 route 点也会被调用，
        route 点的 `state()` 里 `decision_machine_id = -1` ⇒ 第 5~21 维恒为 0，
        与「真的为 0」不可区分。故状态 = `env.state()`（22 维）+ 末尾 1 维标志
        （route=1 / sequence=0）= **23 维**，标志必须加在**末尾**（不打断既有维度语义）。

    采样点也覆盖 route 点，否则 Normalizer 从未见过带 route=1 的样本
    ⇒ 标志维 std=0 被当死维，route 点状态归一化后与其他维尺度不一致。

    ⚠ 2026-10-02（P1）：状态构造**不再在本函数内复写**，改为调用
      `experiments.run_fjsp_final.decision_state(env, rules=rules, ...)` —— 它是状态定义的
      **唯一实现**（`rollout` 的训练与评估路径也走它）。原因：P1 的两个开关
      （`DSH_STATE_ROUTE` 改 22 维内容、`DSH_STATE_ACTREW` 追加 K+2 维）都必须同时作用于
      "Normalizer 拟合用的样本"和"实际喂给网络的状态"，两处各写一份必然漂移
      （本仓库已踩过"训练/部署不一致"的坑）。
      · 两个开关都关（默认）⇒ 逐位等于改动前（仍是 `env.state()` + 1 维标志）。
      · `DSH_STATE_ACTREW=1` ⇒ 样本自动变成 23+K+2 维，`rules` 必须非空
        （缺参时 `decision_state` 直接报错，而不是静默给出错误维度）。
      · 延迟导入以避免 `experiments.run_fjsp_final` ↔ `env_module.normalization` 的循环导入。

    ⚠ 2026-10-03（P2b-2）：新增 `ref_idx` / `la_feat_H` 两个形参，把 **`DSH_LOOKAHEAD_FEAT`
      的特征块**也纳入拟合样本 —— 否则新维在拟合样本里**全是 0**（std=0 ⇒ 被当死维、缩放因子
      取 1），而部署时它们是真值 ⇒ 归一化口径在训练与部署之间就不一致了。
      · `la_feat_H = 0`（**默认**）⇒ 与改动前**逐位相同**（不起用任何新分支）；
      · `la_feat_H > 0` ⇒ 样本 = 23 (+K+2) (+2K+8) 维，此时 `ref_idx` **必须**给出
        （= `rules[default_idx]`，参考策略 = ④ 的规则；缺参时 `decision_state` 直接报错）。
      ⚠ 由**调用方**（`run_xu_tableV.train_sa`）显式传入，本函数**不读环境变量**
        ⇒ 不 opt-in 的调用点（探针/测试/其它脚本）逐位不变。

    ⚠ 2026-10-04（P2b-3 / 路径 1）：再新增 `la_mag_H` / `la_mem` 两个形参（同一套纪律）：
      · `la_mag_H > 0`（`DSH_LA_MAG`）⇒ 与 `la_feat_H` **同一个块（2K+8）、同一个位置**，
        只是归一化口径是"全局固定尺度"（保留绝对量级）。两者互斥（`decision_state` 报错）。
        样本里这个块的值由 `decision_state` 与**同一次运行**的标定尺度算出 ⇒ 与部署一致。
      · `la_mem > 0`（`DSH_LA_MEM`）⇒ 末尾追加 `K+1` 维动作记忆块。**本函数里的取值只是
        "走过这条采样轨迹时最近两次库内动作"**（route 点用的是 EA 而不是库内动作 ⇒ 那一点
        没有库内动作可记）—— 它**不会被用来训练**：`train_sa` 把该块交给
        `Normalizer.fit(fixed_dims=...)` **原样通过**（见 `state_fixed_dims`）。
        之所以仍然填真值而不是全 0：一旦将来有人漏传 `fixed_dims`，`std` 也不至于退化成
        "恒 0 死维"（那会把整块静默变成常数）。**真正的训练/部署取值来自 `rollout`。**

    ⚠ 2026-10-04（P2b-6 / 路径 3）：再新增 `la_rich`（`DSH_LA_RICH`：`"0"`|`"new"`|`"diff"`），
      同一套纪律 —— 由调用方显式传入、本函数不读环境变量、默认 `None` = 关 ⇒ 逐位等于改动前。
      · `la_rich != "0"` ⇒ 样本在 lookahead 块之后**再追加**一个子块（`4K+4` / `4K+2` 维），
        取值同样来自 `decision_state`（**唯一实现**）⇒ 拟合样本与部署状态必然同口径；
      · 该子块由 `state_fixed_dims` **原样通过**（构造时已 O(1)；理由同 lookahead 块：
        稀疏采样下这些维在样本里多为 0，std 会落进危险带被放大 1e2~1e6 倍）。
    """
    import numpy as _np
    from env_module.fjsp_env import make_route_policy
    from experiments.run_fjsp_final import (decision_state as _decision_state,
                                            la_mag_scale_value as _la_mag_scale_value)


    _mag_scale = _la_mag_scale_value() if int(la_mag_H or 0) > 0 else None
    out = []
    for k in range(n_episodes):
        env.reset(seed0 + k)
        ea = make_route_policy("EA")
        i = 0

        prev_a = prev2_a = None
        while env.pending is not None:
            if env.pending == "route":
                out.append(_decision_state(env, rules=rules, la_feat_H=la_feat_H,
                                           la_ref_idx=ref_idx, la_mag_H=la_mag_H,
                                           la_mag_scale=_mag_scale,
                                           la_mem=la_mem, la_prev=prev_a, la_prev2=prev2_a,
                                           la_rich=la_rich, action_ref_idx=ref_idx))
                env.step_route(ea(env))
            else:
                if i % 3 == 0:
                    out.append(_decision_state(env, rules=rules, la_feat_H=la_feat_H,
                                               la_ref_idx=ref_idx, la_mag_H=la_mag_H,
                                               la_mag_scale=_mag_scale,
                                               la_mem=la_mem, la_prev=prev_a, la_prev2=prev2_a,
                                               la_rich=la_rich, action_ref_idx=ref_idx))
                i += 1
                rule = rules[(seed0 + k + i) % len(rules)] if rules else None
                if rule is not None:
                    prev2_a, prev_a = prev_a, int((seed0 + k + i) % len(rules))
                    env.step_sequence(rule, speed=1.0)
                else:
                    env.step_sequence_index(0)
    return _np.array(out)
