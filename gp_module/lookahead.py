"""Offline lookahead utilities for generating decision labels and rollout targets."""
import copy
from typing import Callable, Dict, List, Optional

from config import SPEED_M
from env_module.fjsp_env import FJSPEnv, make_route_policy




EST_DECISIONS = 300
EST_CHOICE_POINTS = 36


def terminal_cost(env: FJSPEnv, lamb_e: float = 0.0) -> float:
    """终端代价：拖期 + λ_E·总能（到件口径，与主指标一致）。"""
    return float(env.tardiness_arrival()) + float(lamb_e) * float(env.arrival_energy())


def finish_with(env: FJSPEnv, rule: Optional[Callable], route_policy: str = "EA") -> None:
    """用基准策略把当前局面跑到终局（rule=None ⇒ FIFO）。"""
    pol = make_route_policy(route_policy)
    while env.pending is not None:
        if env.pending == "route":
            env.step_route(pol(env))
        elif rule is None:
            env.step_sequence_index(0)
        else:
            env.step_sequence(rule, speed=SPEED_M)


def _walk(env: FJSPEnv, seed: int, policy: Optional[Callable], route_policy: str,
          on_point=None) -> Dict[str, int]:
    """走一局，在**每个**决策点回调 `on_point(kind, i, n_cands, env)`（i = 该类决策的序号）。

    必须单遍实现：前瞻要对**当前局面**做 `deepcopy`，只有走到那个点时才拿得到。
    """
    pol = make_route_policy(route_policy)
    env.reset(seed)
    idx = {"route": 0, "seq": 0}
    while env.pending is not None:

        kind = "route" if env.pending == "route" else "seq"
        i = idx[kind]
        idx[kind] = i + 1
        if kind == "route":
            op = env.route_op
            if on_point is not None:
                on_point("route", i, len(op.candidates), env)
            env.step_route(pol(env))
            continue
        cands = env._get_candidates()
        if on_point is not None:
            on_point("seq", i, len(cands), env)
        if policy is None:
            env.step_sequence_index(0)
        else:
            env.step_sequence(policy, speed=SPEED_M)
    return idx


def plan_sample_indices(env: FJSPEnv, seed: int, policy: Optional[Callable],
                        n_route: int = 20, n_seq: int = 20, route_policy: str = "EA",
                        mode: str = "fixed_budget") -> Dict[str, set]:
    """第一遍：数出两类决策点数量，返回**要抽样的决策点序号集合**。

    `fixed_budget`（默认，用户指定）：对**全部**路由点/排序点各等距抽 n 个（与候选数无关）。
    `choice_only`（旧口径，保留对照）：只在候选数 >1 的点里等距抽。
    """
    counts = {"route": [], "seq": []}
    is_choice = {"route": [], "seq": []}

    def cb(kind, i, nc, _env):
        counts[kind].append(i)
        if nc > 1:
            is_choice[kind].append(i)

    _walk(env, seed, policy, route_policy, on_point=cb)
    out: Dict[str, set] = {}
    for kind, quota in (("route", n_route), ("seq", n_seq)):
        pool = counts[kind] if mode == "fixed_budget" else is_choice[kind]
        if not pool or quota <= 0:
            out[kind] = set()
            continue
        stride = max(1, len(pool) // quota)
        out[kind] = set(pool[::stride][:quota])
    return out


def sample_points(env: FJSPEnv, seed: int, policy: Optional[Callable],
                  n_route: int = 20, n_seq: int = 20, route_policy: str = "EA",
                  mode: str = "fixed_budget") -> List[Dict]:
    """抽样决策点**快照**（不做前瞻）：每局 `n_route` 个路由 + `n_seq` 个排序。

    返回 [{'kind', 't', 'n_cands', 'feats'}, ...]（按时间顺序）。
    用于口径核对：被抽到的点里有多少真的"有选择余地"。
    """
    plan = plan_sample_indices(env, seed, policy, n_route, n_seq, route_policy, mode)
    out: List[Dict] = []

    def cb(kind, i, nc, e):
        if i not in plan[kind]:
            return
        if kind == "route":
            feats = e.route_candidate_feats()
        else:
            ctx = e._machine_context(e.machines[e.decision_machine_id])
            feats = [e._candidate_features(j, o, ctx) for (j, o) in e._get_candidates()]
        out.append({"kind": kind, "t": float(e.time), "n_cands": int(nc), "feats": feats})

    _walk(env, seed, policy, route_policy, on_point=cb)
    return out


def lookahead_labels(env: FJSPEnv, seed: int, base_rule: Optional[Callable],
                     n_route: int = 20, n_seq: int = 20, route_policy: str = "EA",
                     lamb_e: float = 0.0, mode: str = "fixed_budget") -> List[Dict]:
    """按固定预算（默认每局 20 路由 + 20 排序）抽样并取前瞻标签。

    每条标签：
        {'kind': 'route'|'seq', 't', 'n_cands',
         'feats': [CandidateFeatures, ...],   # 供各规则复算"它会选哪个"
         'regret': [float, ...],              # 与 feats 同序的真实终局遗憾（最优为 0）
         'best': int}
    ⚠ 轨迹本身**不被干预**（抽样点上仍走基准策略）⇒ 标签是"基准策略所到状态"的离线测量，
      与度量尺子的状态分布一致。候选数 ≤1 的点无法产生偏好信息，故不生成标签（但仍占抽样名额，
      以保证口径无偏、可复现）。
    """
    plan = plan_sample_indices(env, seed, base_rule, n_route, n_seq, route_policy, mode)
    labels: List[Dict] = []

    def cb(kind, i, nc, e):
        if i not in plan[kind] or nc <= 1:
            return
        if kind == "route":
            feats = e.route_candidate_feats()
        else:
            ctx = e._machine_context(e.machines[e.decision_machine_id])
            feats = [e._candidate_features(j, o, ctx) for (j, o) in e._get_candidates()]
        J = []
        for k in range(len(feats)):
            sub = copy.deepcopy(e)
            if kind == "route":
                sub.step_route(k)
            else:
                jid, oi = sub._get_candidates()[k]
                sub._schedule((jid, oi), speed=SPEED_M)
            sub._advance()
            finish_with(sub, base_rule, route_policy)
            J.append(terminal_cost(sub, lamb_e))
        best = int(min(range(len(J)), key=lambda k: J[k]))
        j_best = J[best]
        labels.append({"kind": kind, "t": float(e.time), "n_cands": int(nc), "feats": feats,
                       "regret": [float(x - j_best) for x in J], "best": best})

    _walk(env, seed, base_rule, route_policy, on_point=cb)
    return labels


def reference_slack_q(env: FJSPEnv, rule: Optional[Callable], seed: int = 0, q: float = 0.25,
                      max_points: int = 500, route_policy: str = "EA") -> float:
    """情境第二维的阈值：min-slack 的下四分位。

    **必须与度量尺子用同一个定义**：在一条基准轨迹上遍历全部排序决策点（含被迫决策点），
    取其 min-slack 的分位数。若适应度用一套分位数、尺子用另一套，格子边界就不一致，
    gate 会退化成"优化了一个与度量不同的目标"。
    """
    from gp_module.context import slack_quantile
    pol = make_route_policy(route_policy)
    env.reset(seed)
    xs: List[float] = []
    while env.pending is not None and len(xs) < max_points:
        if env.pending == "route":
            env.step_route(pol(env))
            continue
        cands = env._get_candidates()
        ctx = env._machine_context(env.machines[env.decision_machine_id])
        feats = [env._candidate_features(j, o, ctx) for (j, o) in cands]
        xs.append(float(min(f.slack for f in feats)))
        if rule is None:
            env.step_sequence_index(0)
        else:
            env.step_sequence(rule, speed=SPEED_M)
    return slack_quantile(xs, q)


def chosen_index(rule: Callable, feats) -> int:
    """规则在该点会选哪个候选（与 rollout 里的 `int(np.argmax(pr))` 口径一致）。"""
    best_i, best_v = 0, None
    for i, f in enumerate(feats):
        v = rule(f)
        if best_v is None or v > best_v:
            best_v, best_i = v, i
    return best_i
