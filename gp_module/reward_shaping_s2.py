# -*- coding: utf-8 -*-
"""S2 reward shaping based on changes in remaining job slack."""

from __future__ import annotations

import math

CID = "s2"
SCHEMA = "p8-2.s2.1"

__all__ = ["CID", "SCHEMA", "n_unfinished", "weight", "tau_of", "tau_list",
           "shape_s2", "install_s2", "uninstall_s2", "installed"]



def n_unfinished(env) -> int:
    """未完工工件数 N_unf（纯读：只看 ``job.completed``）。"""
    jobs = getattr(env, "jobs", None)
    if jobs is None:
        raise ValueError("S2 需要 env.jobs（全局剩余松弛的工件集合）")
    return int(sum(1 for j in jobs if not j.completed))


def weight(env) -> float:
    """耦合权重 ``max(0, N_unf − 1)``：会被本决策侵蚀松弛的"其它工件"数。"""
    n = n_unfinished(env)
    return float(n - 1) if n > 1 else 0.0


def _route_proc(op, sel) -> float:
    cands = list(getattr(op, "candidates", []) or [])
    i = int(sel)
    if i < 0 or i >= len(cands):
        raise ValueError("S2 路由点 sel=%d 越界（候选机器数=%d）" % (i, len(cands)))
    m = cands[i]
    procs = getattr(op, "procs", None)
    if not procs or m not in procs:
        raise ValueError("S2 路由点：候选机器 %s 缺加工时间（op.procs 缺键）" % (m,))
    return float(procs[m])


def _seq_proc(feats, sel) -> float:
    if feats is None:
        raise ValueError("S2 在排序点需要 feats（候选特征列表）")
    i = int(sel)
    if i < 0 or i >= len(feats):
        raise ValueError("S2 排序点 sel=%d 越界（候选数=%d）" % (i, len(feats)))
    return float(feats[i].pt)


def tau_of(env, is_route, sel, feats, op) -> float:
    """本决策提交的机器占用时长 τ（时间单位，纯读）。"""
    if is_route:
        if op is None:
            raise ValueError("S2 在路由点需要 op=env.route_op")
        return _route_proc(op, sel)
    return _seq_proc(feats, sel)


def tau_list(env, is_route, feats, op) -> list:
    """该决策点**全部候选**的 τ（纯读）。审计/探针/单测共用，保证与 shape_s2 同源。"""
    if is_route:
        if op is None:
            raise ValueError("S2 在路由点需要 op=env.route_op")
        cands = list(getattr(op, "candidates", []) or [])
        return [float(op.procs[m]) for m in cands]
    if feats is None:
        raise ValueError("S2 在排序点需要 feats")
    return [float(f.pt) for f in feats]



def shape_s2(env, rules, is_route, sel, feats, op) -> float:
    """S2 成形项：``−(N_unf − 1)·τ_sel``（越大越好；纯读 / 逐点 / 零前滚 / 不含拖期）。

    参数与骨架契约一致（``代码/gp_module/reward_shaping.py`` §3）：
    route 点 ``op=env.route_op``、``feats=None``、``sel`` = 候选机器下标；
    sequence 点 ``op=None``、``feats`` = 候选特征列表、``sel`` = 候选工序下标。
    ``rules`` 在本项中**不使用**（本项不依赖任何规则/策略）。
    """
    w = weight(env)
    if w <= 0.0:

        return 0.0
    tau = tau_of(env, is_route, sel, feats, op)
    v = -w * tau
    if not math.isfinite(v):
        raise ValueError("S2 成形项非有限：tau=%r" % (tau,))
    return float(v)



_PREV = []


def _rs(mod=None):
    if mod is not None:
        return mod
    import gp_module.reward_shaping as _m
    return _m


def installed(mod=None) -> bool:
    """``_DISPATCH["s2"]`` 当前是否指向本模块的 :func:`shape_s2`。"""
    rs = _rs(mod)
    return rs._DISPATCH.get(CID) is shape_s2


def install_s2(mod=None) -> bool:
    """把 ``gp_module.reward_shaping._DISPATCH["s2"]`` 指向本模块的 :func:`shape_s2`。

    返回 True = 本次真的换了（幂等：已安装再调返回 False）。只改分派表一项，
    不改 `reward_shaping.py` 的任何源码（该文件不在 S2 的 inScope 内）。
    """
    rs = _rs(mod)
    disp = getattr(rs, "_DISPATCH", None)
    if not isinstance(disp, dict):
        raise ValueError("reward_shaping._DISPATCH 缺失/类型异常：%r" % (disp,))
    if disp.get(CID) is shape_s2:
        return False
    _PREV.append((rs, disp.get(CID)))
    disp[CID] = shape_s2
    return True


def uninstall_s2(mod=None) -> bool:
    """回滚 :func:`install_s2`（恢复原桩；未安装则返回 False）。"""
    rs = _rs(mod)
    disp = getattr(rs, "_DISPATCH", None)
    if not isinstance(disp, dict):
        raise ValueError("reward_shaping._DISPATCH 缺失/类型异常：%r" % (disp,))
    if disp.get(CID) is not shape_s2:
        return False
    prev = _PREV.pop()[1] if _PREV else None
    if prev is None:
        import gp_module.reward_shaping as _rsmod
        prev = getattr(_rsmod, "shape_" + CID)
    disp[CID] = prev
    return True
