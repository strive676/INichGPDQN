# -*- coding: utf-8 -*-
"""S1 reward shaping based on workload and queue congestion."""
from __future__ import annotations

import math

__all__ = ["CID", "shape_s1", "charge_vector", "queue_load", "install", "uninstall",
           "is_installed"]

CID = "s1"


def _machine_of(env, m):
    """取机器对象（只读；下标越界 ⇒ 报错，不静默）。"""
    machines = getattr(env, "machines", None)
    if machines is None:
        raise ValueError("S1: env 没有 `machines` ⇒ 无法读队列拥塞；拒绝静默返回 0。")
    try:
        return machines[int(m)]
    except (IndexError, TypeError, ValueError):
        raise ValueError("S1: 机器下标 %r 越界（env.machines 长度 %d）。"
                         % (m, len(machines) if hasattr(machines, "__len__") else -1))


def queue_load(env, m):
    """`(CW, nq)` = 机器 `m` 队列的（排队 WIP 工时, 排队工序条数）—— 纯读，绝不改队列。"""
    mc = _machine_of(env, m)
    q = getattr(mc, "queue", None)
    if q is None:
        raise ValueError("S1: env.machines[%r] 没有 `.queue` ⇒ 无法读拥塞；拒绝静默 0。" % (m,))
    jobs = getattr(env, "jobs", None)
    if jobs is None:
        raise ValueError("S1: env 没有 `jobs` ⇒ 无法把队列条目换成工时；拒绝静默 0。")
    cw = 0.0
    nq = 0
    for item in q:
        try:
            jid, oi = item
        except (TypeError, ValueError):
            raise ValueError("S1: 队列条目 %r 不是 (jid, op_idx) 二元组。" % (item,))
        try:
            op = jobs[int(jid)].ops[int(oi)]
        except (IndexError, TypeError, ValueError, AttributeError):
            raise ValueError("S1: 队列条目 (jid=%r, op_idx=%r) 在 env.jobs 里取不到工序。"
                             % (jid, oi))
        rem = getattr(op, "remaining", None)
        if rem is None:
            raise ValueError("S1: jobs[%r].ops[%r] 没有 `.remaining` ⇒ 拒绝静默 0。" % (jid, oi))
        cw += float(rem)
        nq += 1
    return cw, nq


def charge_vector(env, is_route, feats, op):
    """本点**每个候选**的拥塞负担向量（时间单位，越小越好）；公开给单测/审计工具独立复算。

    route 点：`[CW(m_k) + pt_k + p̄·nq(m_k)]`（`p̄` = 本点候选加工时间均值）
    sequence 点：`[pt_k · max(0, nq(m) − 1)]`（`m = env.decision_machine_id`）
    """
    if is_route:
        if op is None:
            raise ValueError("S1: route 点必须给 `op`（env.route_op）—— shape_term 的签名契约；"
                             "拒绝静默返回 0。")
        cands = list(getattr(op, "candidates", ()) or ())
        if not cands:
            return []
        procs = getattr(op, "procs", None)
        if procs is None:
            raise ValueError("S1: route 点的 op 缺 `.procs` ⇒ 算不出候选加工时间；拒绝静默 0。")
        pts = []
        for m in cands:
            if m not in procs:
                raise ValueError("S1: op.procs 里没有候选机器 %r（keys=%r）。"
                                 % (m, list(procs)[:8]))
            pts.append(float(procs[m]))
        p_bar = math.fsum(pts) / float(len(pts))
        out = []
        for m, pt in zip(cands, pts):
            cw, nq = queue_load(env, m)
            out.append(cw + pt + p_bar * float(nq))
        return out
    if feats is None:
        raise ValueError("S1: sequence 点必须给 `feats`（候选特征列表）—— 签名契约；"
                         "拒绝静默返回 0。")
    k_n = len(feats)
    if k_n <= 0:
        return []
    mid_raw = getattr(env, "decision_machine_id", None)
    if mid_raw is None:
        raise ValueError("S1: sequence 点 env 缺 `decision_machine_id` ⇒ 不知道在哪个队列里挑；"
                         "拒绝静默 0。")
    mid = int(mid_raw)
    if mid < 0:
        raise ValueError("S1: sequence 点 env.decision_machine_id=%r < 0 ⇒ 不知道在哪个队列里挑；"
                         "拒绝静默 0。" % (mid,))
    _, nq = queue_load(env, mid)
    rest = float(max(0, nq - 1))
    out = []
    for f in feats:
        pt = getattr(f, "pt", None)
        if pt is None:
            raise ValueError("S1: sequence 候选特征缺 `.pt` ⇒ 算不出等待负担；拒绝静默 0。")
        out.append(float(pt) * rest)
    return out



def shape_s1(env, rules, is_route, sel, feats, op) -> float:
    """**S1 = WIP / 队列拥塞 shaping**：`mean_j charge(j) − charge(sel)`（越大越好）。

    · 语义：本步造成的队列拥塞负担（时间单位）相对本点候选均值的偏离；小拥塞 ⇒ 正项。
    · 量级：与 base 奖励同量级（base 候选间跨度 1e0~1e2）⇒ λ 可解释为"每单位拥塞的奖励权重"。
    · `rules` 不使用（P8-2 要求候选不得依赖动作集存在）；`K<=1` ⇒ `0.0`；
      全部 charge 相等 ⇒ 恰好 `0.0`（无信息）；结构缺件 ⇒ `ValueError`（绝不静默 0）。
    · 纯读 / 逐点 / 零前滚 / 不用拖期（见模块 docstring §二）。
    """
    ch = charge_vector(env, is_route, feats, op)
    k_n = len(ch)
    if k_n <= 0:
        return 0.0
    try:
        s = int(sel)
    except (TypeError, ValueError):
        raise ValueError("S1: sel=%r 不是整数下标。" % (sel,))
    if not (0 <= s < k_n):
        raise ValueError("S1: sel=%d 越界（本点 K=%d）。" % (s, k_n))
    if k_n == 1:
        return 0.0
    mean = math.fsum(ch) / float(k_n)
    return float(mean - ch[s])



def install(dispatch=None):
    """把 `_DISPATCH["s1"]` 换成 `shape_s1`，返回可还原的 token。

    `dispatch=None` ⇒ 用 `gp_module.reward_shaping._DISPATCH`（T0 的唯一分派表）。
    **必须显式调用**：import 本模块不改任何全局状态（否则会破 T0 的桩契约单测）。
    """
    import gp_module.reward_shaping as RS
    d = RS._DISPATCH if dispatch is None else dispatch
    token = (d, CID, d.get(CID))
    d[CID] = shape_s1
    return token


def uninstall(token):
    """还原 `install()` 的改动（幂等；反复调用安全）。"""
    if not token:
        return
    d, cid, prev = token
    if prev is None:
        d.pop(cid, None)
    else:
        d[cid] = prev


def is_installed(dispatch=None):
    """当前 `_DISPATCH["s1"]` 是否就是本模块的 `shape_s1`。"""
    import gp_module.reward_shaping as RS
    d = RS._DISPATCH if dispatch is None else dispatch
    return d.get(CID) is shape_s1


def _main():
    import gp_module.reward_shaping as RS
    cur = RS._DISPATCH.get(CID)
    print("[P8-2 S1] cid=%s installed=%s dispatch=%r module=%r"
          % (CID, is_installed(), getattr(cur, "__name__", cur),
             getattr(cur, "__module__", None)))
    print("[P8-2 S1] route:    charge(k) = CW(m_k) + pt_k + mean_pt·nq(m_k); "
          "term(k) = mean_j charge(j) − charge(k)")
    print("[P8-2 S1] sequence: charge(k) = pt_k · max(0, nq(m) − 1); "
          "term(k) = mean_j charge(j) − charge(k)")


if __name__ == "__main__":
    _main()
