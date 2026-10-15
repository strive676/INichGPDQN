"""Decision-point sampling utilities for preference and behavioral evaluation."""
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import SPEED_M
from env_module.fjsp_env import FJSPEnv, make_route_policy

N_PC_SEQ = 20
N_PC_ROUTE = 20
N_RHO = 100


def collect_decisions(env: FJSPEnv, instance_seed: int, route_policy: str = "EA",
                      seq_policy: str = "fifo") -> Tuple[List[List], List[List]]:
    """走**一个未见实例**，收集其**全部**排序/路由决策点的候选特征。

    返回 (seq_points, route_points)：每个元素是"该决策点的候选特征列表"
    （可直接喂给 `compute_reference_pc` / `compute_normalized_pc` / `behaviour_difference`）。
    """
    env.reset(instance_seed)
    pol = make_route_policy(route_policy)
    seq_pts: List[List] = []
    route_pts: List[List] = []
    while env.pending is not None:
        if env.pending == "route":
            feats = env.route_candidate_feats()
            if feats:
                route_pts.append(feats)
            env.step_route(pol(env))
            continue
        ctx = env._machine_context(env.machines[env.decision_machine_id])
        feats = [env._candidate_features(j, o, ctx) for (j, o) in env._get_candidates()]
        seq_pts.append(feats)
        if seq_policy == "fifo":
            env.step_sequence_index(0)
        else:
            raise ValueError(f"未知 seq_policy: {seq_policy}")
    return seq_pts, route_pts


def _shuffle_take(n_avail: int, k: int, rng: np.random.Generator) -> List[int]:
    """shuffle 后随机取 k 个（返回**升序**下标，保证坐标顺序稳定、个体间可比）。"""
    if n_avail <= 0 or k <= 0:
        return []
    idx = rng.permutation(n_avail)[:min(k, n_avail)]
    return sorted(int(i) for i in idx)


def _min_cand() -> int:
    """`DSH_PC_MIN_CAND`：只保留候选数 ≥ 该阈值的决策点（**运行时读取**，避免 import 期常量陷阱）。

    动机（实测，见 `docs/方法方案_审计基线.md` 与 NichGP 审计）：
      本环境排序决策点的候选数分布为 `{1:211, 2:72, 3:21, 4:1}`（共 305 个），
      而 **候选数为 1 时 `top_choice` 恒返回 0** ⇒ 这类坐标对任何规则都相同、零信息。
      默认抽 20 个排序点时有 **17 个**落在单候选点上（PC 恒为 1.0），
      而同一份 PC 的 20 个**路由维全部有信息**（路由点候选数 2~3）
      ⇒ **PC 的欧氏距离实际由「路由行为」主导**，而输出只消费排序树
      ⇒ δ=0 的 clearing 去重的是路由重复，**排序行为没有被维持**（库塌缩的代码层根因）。

    设 `DSH_PC_MIN_CAND=2` 后，PC/ρ 的抽样只从"有信息"的决策点里取，
    使 20 个排序维**全部**携带排序行为信息。
    ⚠ 默认 1 = 与既有实现逐字相同（不改变任何既有结果）。
    """
    try:
        v = int(os.environ.get("DSH_PC_MIN_CAND", "1") or 1)
    except ValueError:
        v = 1
    return max(1, v)


def _filter_informative(pts: List[List]) -> List[List]:
    """只保留候选数 ≥ `DSH_PC_MIN_CAND` 的决策点（默认阈值 1 ⇒ 原样返回）。"""
    m = _min_cand()
    if m <= 1:
        return pts
    return [p for p in pts if len(p) >= m]


def _high_stakes() -> bool:
    """`DSH_PC_HIGH_STAKES=1`：PC 抽样改为**按"参考规则是否在该点分歧"加权**。

    动机（2026-09-28 实测，见 `docs/四项研究汇总报告.md` 研究三）：
      · 逐决策 oracle 增益 **+33.9%**，但**全部集中在 1.47% 的决策点上**；
      · 而当前 PC 是「全部决策点 shuffle 后**均匀随机**取 20+20」⇒ **几乎必然采不到那 1.5%**；
      · 于是 clearing 维持的是**与性能无关的多样性**，K 个个体在实际路由/排序点上
        行为全同（实测路由点全同率 **97.7%**、排序点 **71.7%**）。

    改法：一个决策点若让参考规则们**给出不同选择**，它才具备区分个体的能力；
    让所有规则选同一个候选的点，对任何个体都给出相同坐标 ⇒ 零信息。
    故按「不同选择数 − 1」加权抽样（权重 0 的点仍可能被抽到，只是概率低）。

    ⚠ **这偏离论文口径**（论文是 shuffle 后均匀随机）。默认 **0 = 关闭**，保持论文口径；
    仅作为对照实验开启。写论文时必须声明该差异。
    """
    return os.environ.get("DSH_PC_HIGH_STAKES", "0").strip() == "1"




_SEQ_SCORERS = (
    lambda f: -f.pt,
    lambda f: -(f.slack + f.wkr),
    lambda f: -f.wkr,
    lambda f: f.owt,
)
_ROUTE_SCORERS = (
    lambda f: -(f.mrt + f.pt),
    lambda f: -f.pt,
    lambda f: -f.mrt,
    lambda f: f.niq,
)


def _disagreement_weight(feats: List, kind: str) -> float:
    """该决策点上参考规则给出多少个**不同**的选择；1 = 全同（零信息）⇒ 权重 0。"""
    if not feats or len(feats) < 2:
        return 0.0
    scorers = _SEQ_SCORERS if kind == "seq" else _ROUTE_SCORERS
    choices = set()
    for sc in scorers:
        try:
            vals = [sc(f) for f in feats]
        except AttributeError:
            return 0.0
        choices.add(int(np.argmax(vals)))
    return float(len(choices) - 1)


def _shuffle_take_weighted(pts: List[List], k: int, rng: np.random.Generator,
                           kind: str) -> List[int]:
    """按 `_disagreement_weight` 加权、不放回地抽 k 个（返回**升序**下标）。"""
    if not pts or k <= 0:
        return []
    w = np.asarray([_disagreement_weight(p, kind) for p in pts], dtype=float)
    k = min(k, len(pts))
    if float(w.sum()) <= 0:
        return _shuffle_take(len(pts), k, rng)
    p = w / w.sum()
    idx = rng.choice(len(pts), size=k, replace=False, p=p)
    return sorted(int(i) for i in idx)


def xu_pc_points(env: FJSPEnv, instance_seed: int, n_seq: int = N_PC_SEQ,
                 n_route: int = N_PC_ROUTE, shuffle_seed: int = 0,
                 include_routing: bool = True, route_policy: str = "EA",
                 seq_policy: str = "fifo") -> Tuple[List[List], Dict]:
    """论文口径的 PC 决策点：**20 排序 + 20 路由**（`include_routing=False` 时只取排序部分）。

    坐标顺序固定为 [排序 20 个, 路由 20 个]（论文的 PC 含两类，顺序不影响欧氏距离，
    只要同一代内所有个体共用同一份抽样即可 —— 本函数每代调用一次、全体共用）。
    """
    seq_pts, route_pts = collect_decisions(env, instance_seed, route_policy, seq_policy)
    n_seq_all, n_route_all = len(seq_pts), len(route_pts)
    seq_pts = _filter_informative(seq_pts)
    route_pts = _filter_informative(route_pts)
    rng = np.random.default_rng(shuffle_seed)
    if _high_stakes():

        si = _shuffle_take_weighted(seq_pts, n_seq, rng, "seq")
        ri = (_shuffle_take_weighted(route_pts, n_route, rng, "route")
              if include_routing else [])
        _mode = "high_stakes"
    else:
        si = _shuffle_take(len(seq_pts), n_seq, rng)
        ri = _shuffle_take(len(route_pts), n_route, rng) if include_routing else []
        _mode = "uniform"
    pts = [seq_pts[i] for i in si] + [route_pts[i] for i in ri]
    _w_seq = [_disagreement_weight(seq_pts[i], "seq") for i in si]
    _w_route = [_disagreement_weight(route_pts[i], "route") for i in ri]
    meta = dict(n_seq=len(si), n_route=len(ri), dim=len(pts),
                n_seq_avail=len(seq_pts), n_route_avail=len(route_pts),
                n_seq_total=n_seq_all, n_route_total=n_route_all,
                min_cand=_min_cand(), pc_sample_mode=_mode,

                seq_disc_frac=(round(float(np.mean([w > 0 for w in _w_seq])), 4)
                               if _w_seq else None),
                route_disc_frac=(round(float(np.mean([w > 0 for w in _w_route])), 4)
                                 if _w_route else None),
                instance_seed=instance_seed, shuffle_seed=shuffle_seed)
    return pts, meta


def xu_rho_points(env: FJSPEnv, instance_seed: int, n: int = N_RHO, shuffle_seed: int = 0,
                  route_policy: str = "EA", seq_policy: str = "fifo") -> Tuple[List[List], Dict]:
    """论文口径的 ρ 决策点：**100 个排序决策**（同一未见实例）。"""
    seq_pts, _ = collect_decisions(env, instance_seed, route_policy, seq_policy)
    n_total = len(seq_pts)
    seq_pts = _filter_informative(seq_pts)
    rng = np.random.default_rng(shuffle_seed)
    idx = _shuffle_take(len(seq_pts), n, rng)
    meta = dict(n=len(idx), n_avail=len(seq_pts), n_total=n_total, min_cand=_min_cand(),
                instance_seed=instance_seed, shuffle_seed=shuffle_seed)
    return [seq_pts[i] for i in idx], meta
