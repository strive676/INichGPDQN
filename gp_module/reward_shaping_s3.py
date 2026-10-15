# -*- coding: utf-8 -*-
"""S3 reward shaping based on completion-time changes."""
from __future__ import annotations

import hashlib
import math
import os

import numpy as np

CID = "s3"

__all__ = [
    "CID", "shape_s3", "install", "uninstall", "is_installed", "assert_installed",
    "wiring_line", "stats_reset", "stats_snapshot", "stats_line", "describe",
]


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
_STATS = None


def stats_reset():
    """清零本模块的调用统计（启动器在开跑前调用）。"""
    global _STATS
    _STATS = {
        "n": 0, "n_route": 0, "n_seq": 0,
        "n_sel_min": 0, "n_zero": 0,
        "nz": 0, "s": 0.0, "sabs": 0.0,
        "vmin": float("inf"), "vmax": float("-inf"),
        "k_sum": 0, "k_min": 0, "k_max": 0,
        "skip": {},
    }
    return dict(_STATS)


def _skip(reason, is_route=False, k=None):
    """未定义情形的**显式**通道：计数 + 返回 0.0（调用方不静默丢掉原因）。

    注意：`n`/`n_route`/`n_seq` **含**未定义调用 —— 骨架的 `shaping_delta()` 对每个注入点都会
    记一次账（含返回 0.0 的），两本账要能逐项对上 ⇒ 这里必须同样按"调用次数"计数；
    `n_zero`/`skip` 才专门记未定义情形。
    """
    if _STATS is not None:
        st = _STATS
        st["n"] += 1
        st["n_route" if is_route else "n_seq"] += 1
        if k:
            st["k_sum"] += int(k)
            st["k_min"] = int(k) if st["k_min"] == 0 else min(st["k_min"], int(k))
            st["k_max"] = max(st["k_max"], int(k))
        st["skip"][reason] = st["skip"].get(reason, 0) + 1
        st["n_zero"] += 1
    return 0.0


def _tally(v, k, is_route):
    """记一次有效取值（值本身已在调用方算好）。"""
    if _STATS is None:
        return v
    st = _STATS
    st["n"] += 1
    st["n_route" if is_route else "n_seq"] += 1
    st["k_sum"] += k
    st["k_min"] = k if st["k_min"] == 0 else min(st["k_min"], k)
    st["k_max"] = max(st["k_max"], k)
    st["s"] += v
    st["sabs"] += abs(v)
    st["nz"] += int(v != 0.0)
    st["n_sel_min"] += int(v == 0.0)
    if v < st["vmin"]:
        st["vmin"] = v
    if v > st["vmax"]:
        st["vmax"] = v
    return v


def stats_snapshot():
    """当前统计（未初始化则 None）。"""
    return None if _STATS is None else dict(_STATS, skip=dict(_STATS["skip"]))


def stats_line(prefix="reward_shaping_s3[stats]"):
    """一行机器可读统计（启动器 atexit 打；证明"项真的被调用过"）。

    `n` **含**未定义调用（与骨架 `shaping_delta` 的记账口径一致，便于两本账逐项对账）；
    `n_zero`/`skip` 专门记未定义（返回 0.0）的情形 ⇒ 从不静默。
    """
    st = stats_snapshot()
    if st is None:
        return f"{prefix} n=0 (stats_reset() 未被调用)"
    n = st["n"]
    mean = (st["s"] / n) if n else 0.0
    sabs = (st["sabs"] / n) if n else 0.0
    ks = (st["k_sum"] / n) if n else 0.0
    skip = ",".join(f"{k}={v}" for k, v in sorted(st["skip"].items())) or "-"
    vmin = st["vmin"] if math.isfinite(st["vmin"]) else float("nan")
    vmax = st["vmax"] if math.isfinite(st["vmax"]) else float("nan")
    return (f"{prefix} cid=s3 n={n} n_route={st['n_route']} n_seq={st['n_seq']} "
            f"nz={st['nz']} n_sel_min={st['n_sel_min']} n_zero={st['n_zero']} "
            f"mean={mean:.6f} mean_abs={sabs:.6f} vmin={vmin:.6f} vmax={vmax:.6f} "
            f"K_mean={ks:.3f} K_min={st['k_min']} K_max={st['k_max']} skip={skip}")


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def _sel_ok(sel, k):
    """决策下标合法化：越界 = 编程错误，**抛**（绝不静默取错候选）。"""
    try:
        s = int(sel)
    except (TypeError, ValueError):
        raise ValueError(f"shape_s3: sel 不是整数：{sel!r}（K={k}）")
    if not 0 <= s < k:
        raise ValueError(f"shape_s3: sel={s} 越界（K={k}）")
    return s


def shape_s3(env, rules, is_route, sel, feats, op):
    """S3：`-(ct_sel - min_k ct_k) / T_ref`（未乘 λ；越大越好；恒有限浮点）。

    参数（infra 契约，`P8_2_T0_INFRA.md` §3）:
      · route 点：`is_route=True`、`op=env.route_op`、`sel=k_m`（`op.candidates` 下标）、`feats=None`
      · 排序点：`is_route=False`、`op=None`、`feats=[env._candidate_features(...)]`、`sel=argmax`
      · `rules` 可能为 None（本项不用规则）；K==0 由 infra 提前 return 0.0。

    未定义 ⇒ 返回 0.0 并计入 `skip`（原因见 `stats_line()`，不静默）：K<2、
    排序点机器 id 非法（`env.decision_machine_id` 不在 `[0, len(env.machines))`）、
    加工时长非有限值。
    """
    if is_route:
        if op is None:
            return _skip("route_op_none", True)
        cand = list(getattr(op, "candidates", ()) or ())
        k = len(cand)
        if k < 2:
            return _skip("k_lt_2_route", True, k)
        procs = getattr(op, "procs", None)
        if procs is None:
            return _skip("route_procs_none", True, k)

        try:
            pts = [float(procs[m]) for m in cand]
        except (KeyError, TypeError, ValueError):
            return _skip("route_pt_bad", True, k)
        if not all(math.isfinite(v) for v in pts):
            return _skip("route_pt_not_finite", True, k)
        cts = [float(env.machine_avail(m)) + pts[i] for i, m in enumerate(cand)]
        a = _sel_ok(sel, k)
    else:
        mid = int(getattr(env, "decision_machine_id", -1))
        try:
            n_mach = len(env.machines)
        except (AttributeError, TypeError):
            return _skip("seq_no_machines", False)
        if not 0 <= mid < n_mach:

            return _skip("seq_bad_machine_id", False)
        fs = list(feats or ())
        k = len(fs)
        if k < 2:
            return _skip("k_lt_2_seq", False, k)
        try:
            pts = [float(getattr(f, "pt")) for f in fs]
        except (AttributeError, TypeError, ValueError):
            return _skip("seq_pt_bad", False, k)
        if not all(math.isfinite(v) for v in pts):
            return _skip("seq_pt_not_finite", False, k)

        av = float(env.machine_avail(mid))
        cts = [av + p for p in pts]
        a = _sel_ok(sel, k)

    t_ref = max(1e-9, float(np.mean(pts)))
    mn = min(cts)


    v = -((cts[a] - mn) / t_ref)
    return _tally(v, k, bool(is_route))


def describe():
    """人读单行：本项的口径（进报告用）。"""
    return ("S3 completion-time: term = -(ct_a - min_k ct_k)/T_ref, "
            "ct_k = machine_avail(m_k) + pt_k (route) | machine_avail(mid) + pt_k (seq), "
            "T_ref = mean_k(pt_k); λ 由 infra 施加（本模块返回未乘 λ 的项）")


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def _sha16(path):
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()[:16]
    except OSError:
        return "?"


def wiring_line():
    """一行可留档的接线证据（含本文件 sha256 前 16 位 + infra 现读到的 λ）。"""
    import gp_module.reward_shaping as infra
    cfg = infra.shaping_config()
    return ("reward_shaping_s3[wiring] cid=s3 module=%s file_sha16=%s "
            "on=%s active=%s lam=%r infra_dispatch=%s"
            % (shape_s3.__module__, _sha16(os.path.abspath(__file__)),
               cfg.get("on"), cfg.get("active"), cfg.get("lam"),
               getattr(infra._DISPATCH.get(CID), "__module__", None)))


def install(verbose=True):
    """把 `_DISPATCH["s3"]` 指到本模块（幂等）；返回接线信息 dict。

    只在 T0 骨架缺"外部候选 registry"时才是必需的：`shape_term()` 调用时查表 ⇒ 运行期指派即生效。
    生产文件（`run_fjsp_final.py` / `reward_shaping.py`）本批一行不改。
    """
    import gp_module.reward_shaping as infra
    legal = tuple(getattr(infra, "LEGAL_CIDS", ()))
    if legal and CID not in legal:
        raise RuntimeError(f"install(): infra LEGAL_CIDS={legal} 不含 {CID!r}——接口对不上，拒绝接线")
    dispatch = getattr(infra, "_DISPATCH", None)
    if not isinstance(dispatch, dict):
        raise RuntimeError("install(): infra 没有 _DISPATCH 字典——接口对不上，拒绝接线")
    prev = dispatch.get(CID)
    replaced = prev is not None and getattr(prev, "__module__", None) != __name__
    dispatch[CID] = shape_s3
    info = {
        "cid": CID, "as": "gp_module.reward_shaping._DISPATCH[cid]",
        "prev": getattr(prev, "__qualname__", repr(prev)),
        "replaced_foreign": bool(replaced),
        "cfg": dict(infra.shaping_config()),
    }
    if verbose:
        print(wiring_line(), flush=True)
        if info["replaced_foreign"]:
            print("[reward_shaping_s3] 注意：原 _DISPATCH['s3'] 由 %r 提供，已被本模块覆盖"
                  % info["prev"], flush=True)
    return info


def uninstall():
    """还原到 infra 自带的 `shape_s3` 桩（仅用于测试/对照）。"""
    import gp_module.reward_shaping as infra
    prev = infra._DISPATCH.get(CID)
    infra._DISPATCH[CID] = infra.shape_s3
    return getattr(prev, "__qualname__", repr(prev))


def is_installed():
    """当前 `_DISPATCH["s3"]` 是否就是本模块的实现。"""
    import gp_module.reward_shaping as infra
    return infra._DISPATCH.get(CID) is shape_s3


def assert_installed():
    """启动器在开跑前调用：没接上就**抛**（防"以为开了 shaping、其实是基线"）。"""
    if not is_installed():
        import gp_module.reward_shaping as infra
        got = getattr(infra._DISPATCH.get(CID), "__module__", None)
        raise RuntimeError(f"assert_installed(): _DISPATCH['s3'] 指向 {got!r}，不是 "
                           f"{__name__}.shape_s3 ⇒ 本臂无效，拒绝开跑")
    return True
