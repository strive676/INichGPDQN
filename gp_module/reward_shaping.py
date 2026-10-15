# -*- coding: utf-8 -*-
"""Shared configuration and dispatch utilities for optional GP reward shaping."""
import math
import os


ENV_MAIN = "DSH_REWARD_SHAPING"
ENV_CID = "DSH_SHAPE_CID"
ENV_LAM = "DSH_SHAPE_LAM"
ENV_STAMP = "DSH_SHAPE_STAMP"
REWARD_SHAPING_ENV_KEYS = (ENV_MAIN, ENV_CID, ENV_LAM, ENV_STAMP)


_ON = ("1", "on", "true", "yes")
_OFF = ("", "0", "off", "none", "false", "no")

CANDIDATE_IDS = ("s1", "s2", "s3", "s4")
CID_PROBE = "probe"
LEGAL_CIDS = CANDIDATE_IDS + (CID_PROBE,)


SHAPE_SCHEMA = "p8-2.0"


REWARD_SHAPING_LOG = []


STAMP_MODES = ("first", "all", "off")
_STAMP_EMITTED = [0]

_OFF_CFG = {"on": False, "active": False, "cid": "", "lam": 0.0}



def check_reward_shaping_env_names() -> None:
    """`DSH_REWARD_SHAPING` 家族的**拼写守卫**：含 `SHAP` 的 `DSH_*` 只准是白名单里那三个。

    与 `run_fjsp_final.check_return_audit_env_names()` 逐字同构（同一纪律）：
    名字写错 ⇒ 直接报错，**不是**静默关掉 shaping。
    """
    _legal = {str(x).upper() for x in REWARD_SHAPING_ENV_KEYS}
    bad = []
    for k in os.environ:
        _u = str(k).upper()
        if not _u.startswith("DSH_"):
            continue
        if _u in _legal:
            continue
        if "SHAP" in _u:
            bad.append(str(k))
    if bad:
        raise ValueError(
            "环境变量 %s 看起来是 `DSH_REWARD_SHAPING` 家族的**拼写错误**"
            "（合法名字：%s）⇒ 直接报错，而不是静默关掉奖励成形（shaping）。"
            % (sorted(bad), list(REWARD_SHAPING_ENV_KEYS)))


def shaping_on() -> bool:
    """主开关是否开着（只读一个键；词表外取值 ⇒ 报错）。"""
    raw = os.environ.get(ENV_MAIN, "")
    if raw is None:
        raw = ""
    v = str(raw).strip().lower()
    if v in _ON:
        return True
    if v in _OFF:
        return False
    raise ValueError(
        "%s=%r 不是合法取值（开 = %s；关 = %s）⇒ 直接报错，不静默当成关。"
        "候选 id 请写进 `%s`，不要写成 `%s=s2` 这种简写。"
        % (ENV_MAIN, raw, "/".join(_ON), "/".join(x or "空" for x in _OFF),
           ENV_CID, ENV_MAIN))


def shaping_config() -> dict:
    """解析本局的 shaping 配置 ⇒ `{"on", "active", "cid", "lam"}`（**每调用一次读一次环境**）。

    · `on`     = 主开关开着（λ=0 也算 on ⇒ 会打戳行）；
    · `active` = `on` 且 `λ != 0` ⇒ **真的加奖励**（λ=0 是精确对照臂：不加、不调用 `shape_term`）；
    · 主开关关着 ⇒ 立刻返回常量 `_OFF_CFG`（**一个 env 读 + 一次成员测试**，其余什么都不做）。

    ⚠ 本函数**不做深解析**：`active` 时 cid/λ 的合法性在下面立刻校验（非法 ⇒ 报错）。
    """
    raw = os.environ.get(ENV_MAIN, "")
    if raw is None:
        raw = ""
    v = str(raw).strip().lower()
    if v not in _ON:
        if v in _OFF:
            return dict(_OFF_CFG)

        raise ValueError(
            "%s=%r 不是合法取值（开 = %s；关 = %s）⇒ 直接报错，不静默当成关。"
            % (ENV_MAIN, raw, "/".join(_ON), "/".join(x or "空" for x in _OFF)))
    cid = str(os.environ.get(ENV_CID, "") or "").strip().lower()
    if cid not in LEGAL_CIDS:
        raise ValueError(
            "%s 开着，但 %s=%r 不是合法候选 id（合法：%s）⇒ 直接报错，"
            "而不是静默跑一个'什么都不加'的假 shaping。"
            % (ENV_MAIN, ENV_CID, os.environ.get(ENV_CID, ""), list(LEGAL_CIDS)))
    raw_lam = os.environ.get(ENV_LAM, "")
    if raw_lam is None or str(raw_lam).strip() == "":
        lam = 1.0
    else:
        try:
            lam = float(str(raw_lam).strip())
        except (TypeError, ValueError):
            raise ValueError("%s=%r 不是浮点数 ⇒ 直接报错（λ 是预注册的扫档变量，不能猜）。"
                             % (ENV_LAM, raw_lam))
    if not math.isfinite(lam):
        raise ValueError("%s=%r 不是有限浮点数 ⇒ 直接报错。" % (ENV_LAM, raw_lam))
    shaping_stamp_mode()
    return {"on": True, "active": (lam != 0.0), "cid": cid, "lam": float(lam)}


def shaping_stamp_mode() -> str:
    """戳行发射策略：`first`（默认，每次运行只打**一局评估**）/ `all`（每局）/ `off`（不打）。"""
    raw = os.environ.get(ENV_STAMP, "")
    v = str(raw or "").strip().lower() or "first"
    if v not in STAMP_MODES:
        raise ValueError("%s=%r 不是合法取值（合法：%s）⇒ 直接报错，不静默按默认策略跑。"
                         % (ENV_STAMP, raw, list(STAMP_MODES)))
    return v


def shaping_stamp_should_emit() -> bool:
    """本行是否该打（并累加计数器）。调用方约定：只在**评估局**且开关开着时问一次。"""
    mode = shaping_stamp_mode()
    if mode == "off":
        return False
    _STAMP_EMITTED[0] += 1
    return (mode == "all") or (_STAMP_EMITTED[0] == 1)


def shaping_stamp_reset() -> None:
    """把戳行计数器清零（测试用；正常运行不需要）。"""
    _STAMP_EMITTED[0] = 0


def shaping_active() -> bool:
    """`shaping_config()["active"]` 的便捷读法（热路径用；关着 ⇒ 一次 env 读 + 早退）。"""
    return bool(shaping_config()["active"])



def _new_bucket() -> dict:
    return dict(n=0, nz=0, s=0.0, sabs=0.0, vmin=float("inf"), vmax=-float("inf"),
                n_route=0, n_seq=0, s_route=0.0, s_seq=0.0,
                n_base=0, s_base=0.0, s_shaped=0.0, dmax=0.0)


_STATS = {}


def shaping_stats_reset() -> None:
    """清空统计（每局开头调一次；只在开关开着时才会被调用）。"""
    _STATS.clear()


def _finalize(b) -> dict:
    """统计桶 → 可直接打印的快照（空桶的 ±inf / 均值分母 0 都归一成 0.0）。"""
    d = dict(b)
    if d["n"] <= 0:
        d["vmin"] = 0.0
        d["vmax"] = 0.0
    if d["vmin"] == float("inf"):
        d["vmin"] = 0.0
    if d["vmax"] == -float("inf"):
        d["vmax"] = 0.0
    d["base_mean"] = (d["s_base"] / d["n_base"]) if d["n_base"] else 0.0
    d["shaped_mean"] = (d["s_shaped"] / d["n_base"]) if d["n_base"] else 0.0
    return d


def shaping_stats_snapshot() -> dict:
    """统计快照 ⇒ `{"exec": {...}, "aux": {...}}`（`aux` = `_point_rewards` 口径）。

    `exec` = 执行奖励的注入点（`rollout` 主路径的 `r += λ·term`）；
    `aux`  = `_point_rewards` 口径（P8-1 记录器的反事实探针 / `DSH_STATE_ACTREW` 的状态通道 /
             `reward_gate` 闸门 —— 都不是执行奖励）⇒ `aux_nz>0` 就是"G_n 也吃了新 reward"的直接证据。

    ⚠ 两个键**恒在**（没被碰过的桶返回全 0 快照）：下游（戳行 / S1~S4 报告）读
    `snap["aux"]["nz"]` 不会因为"这一局没走该通道"而 KeyError —— 缺桶会被误读成"跑了没效果"。
    """
    return {where: _finalize(_STATS.get(where) or _new_bucket())
            for where in ("exec", "aux")}


def _bump(where, is_route, r_pre, d) -> None:
    b = _STATS.get(where)
    if b is None:
        b = _STATS[where] = _new_bucket()
    b["n"] += 1
    if d != 0.0:
        b["nz"] += 1
    b["s"] += d
    b["sabs"] += abs(d)
    if d < b["vmin"]:
        b["vmin"] = d
    if d > b["vmax"]:
        b["vmax"] = d
    if is_route:
        b["n_route"] += 1
        b["s_route"] += d
    else:
        b["n_seq"] += 1
        b["s_seq"] += d
    if r_pre is not None:
        b["n_base"] += 1
        b["s_base"] += float(r_pre)
        b["s_shaped"] += float(r_pre) + float(d)
        if abs(d) > b["dmax"]:
            b["dmax"] = abs(d)



def shape_term(env, rules, is_route, sel, feats, op, cid=None) -> float:
    """**P8-2 唯一接入点**：按 cid 分派到 `shape_s1..s4`（或自检项 `probe`）。

    签名与纪律见模块 docstring §二/§三。本函数只做 3 件事：
    ① 解析 cid（`cid=None` ⇒ 读环境）；② 校验 `sel/K`（越界 ⇒ 报错）；③ 分派 + 有限性校验。
    `K == 0`（无候选）⇒ 返回 `0.0`。
    """
    _cid = cid
    if _cid is None:
        _cid = shaping_config()["cid"]
    _cid = str(_cid or "").strip().lower()
    fn = _DISPATCH.get(_cid)
    if fn is None:
        raise ValueError("未知 shape cid=%r（合法：%s）⇒ 直接报错，不静默返回 0。"
                         % (_cid, list(LEGAL_CIDS)))
    if is_route:
        if op is None:
            raise ValueError("route 点必须给 `op`（env.route_op）—— 这是 shape_term 的签名契约。")
        k_n = len(getattr(op, "candidates", ()) or ())
    else:
        if feats is None:
            raise ValueError("sequence 点必须给 `feats`（候选特征列表）—— 签名契约。")
        k_n = len(feats)
    if k_n <= 0:
        return 0.0
    try:
        s = int(sel)
    except (TypeError, ValueError):
        raise ValueError("sel=%r 不是整数下标（route: op.candidates 下标；sequence: feats 下标）。"
                         % (sel,))
    if not (0 <= s < k_n):
        raise ValueError("sel=%d 越界（本点 K=%d；route: op.candidates 下标；sequence: feats 下标）。"
                         % (s, k_n))
    v = fn(env, rules, is_route, s, feats, op)
    try:
        v = float(v)
    except (TypeError, ValueError):
        raise ValueError("shape cid=%s 返回了非浮点 %r ⇒ 直接报错。" % (_cid, v))
    if not math.isfinite(v):
        raise ValueError("shape cid=%s 返回了 %r（非有限）⇒ 直接报错（nan/inf 会毁掉整条奖励链）。"
                         % (_cid, v))
    return v


def shaping_delta(cfg, env, rules, is_route, sel, feats, op, r_pre=None, where="exec") -> float:
    """`λ·shape_term(...)`，并把统计记进 `where` 桶（`exec` = 执行奖励注入点；`aux` = `_point_rewards`）。

    `cfg["active"]` 为假 ⇒ 返回 `0.0`（**不调用** `shape_term`：λ=0 是精确对照臂）。
    `r_pre` 给"base → shaped"的均值/`max|Δ|` 用（可省）。
    """
    if not cfg or not cfg.get("active"):
        return 0.0
    t = shape_term(env, rules, is_route, sel, feats, op, cid=cfg["cid"])
    d = float(cfg["lam"]) * float(t)
    _bump(where, bool(is_route), r_pre, d)
    return d



def _fmt(v) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return "%d" % v
    if isinstance(v, str):
        return v
    try:
        return "%.6g" % float(v)
    except (TypeError, ValueError):
        return str(v)


def shaping_stamp(scene_name, tag, cfg=None) -> str:
    """`reward_shaping[...]` 戳行：**人读句 + `sh_` 机器键块**（只在开关开着时打印）。

    自证要点（下游按机器键复算，不解析散文）：
      · cid/λ 与"每一个数都从 `shape_term` 的**当前状态**只读量来"的口径声明；
      · `sh_exec_*` = 执行奖励注入点的调用数/非零数/Σ/极值/base→shaped 均值/max|Δ|
        ⇒ "reward 真的变了、变了多少"直接可读；
      · `sh_aux_*` = `_point_rewards` 口径（P8-1 记录器内部）⇒ "G_n 也吃了新 reward"；
      · `sh_on=1` 但 `sh_lam=0` ⇒ **对照臂**（不加奖励，只有本行留痕）。
    """
    c = dict(cfg) if cfg else shaping_config()
    snap = shaping_stats_snapshot()
    e = snap.get("exec") or _finalize(_new_bucket())
    a = snap.get("aux") or _finalize(_new_bucket())
    kv = dict(
        sh_schema=SHAPE_SCHEMA,
        sh_cid=(c.get("cid") or "-"),
        sh_lam=float(c.get("lam", 0.0) or 0.0),
        sh_on=int(1 if c.get("on") else 0),
        sh_active=int(1 if c.get("active") else 0),
        sh_exec_n=int(e["n"]), sh_exec_nz=int(e["nz"]),
        sh_exec_n_rt=int(e["n_route"]), sh_exec_n_sq=int(e["n_seq"]),
        sh_exec_sum=float(e["s"]), sh_exec_abs=float(e["sabs"]),
        sh_exec_min=float(e["vmin"]), sh_exec_max=float(e["vmax"]),
        sh_exec_nbase=int(e["n_base"]), sh_exec_base_mean=float(e["base_mean"]),
        sh_exec_shaped_mean=float(e["shaped_mean"]), sh_exec_dmax=float(e["dmax"]),
        sh_aux_n=int(a["n"]), sh_aux_nz=int(a["nz"]),
        sh_aux_n_rt=int(a["n_route"]), sh_aux_n_sq=int(a["n_seq"]),
        sh_aux_sum=float(a["s"]), sh_aux_abs=float(a["sabs"]),
        sh_aux_min=float(a["vmin"]), sh_aux_max=float(a["vmax"]),
    )
    if c.get("on"):
        REWARD_SHAPING_LOG.append(dict(kv))
    _txt = (
        "%s|reward_shaping[%s]|**P8-2 奖励成形（reward shaping）**（只加即时奖励：`r += λ·shape_term`；"
        "不改动作空间/网络/判据/窗口 H）"
        "｜cid=%s｜λ=%s｜开关=%s（%s）"
        "｜**口径**：逐点、纯读、**零前滚**（只用当前决策点的 O(1) 只读量：队列/WIP、剩余松弛、"
        "机器可用时刻(CT)、候选耦合；**绝不** deepcopy 前滚、**绝不** env.reset、**绝不**读拖期）"
        "｜**P8-1 审计**：`DSH_RETURN_AUDIT` 的 `r`/`G_n` 走 `_point_rewards` ⇒ **已自动含本项**"
        "（同源，不另算一套）"
        "｜**执行奖励注入点**：调用=%d（route %d / sequence %d），非零=%d，Σ=%s，Σ|·|=%s，"
        "min=%s，max=%s"
        "｜**r 变化**（注入点逐点）：base 均值=%s → shaped 均值=%s，max|Δ|=%s（n=%d）"
        "｜**反事实口径**（`_point_rewards`：P8-1 记录器内部 / 状态通道 / reward_gate 闸门）："
        "调用=%d，非零=%d，Σ=%s ⇒ `sh_aux_nz>0` ⟺ 记录器的 G_n 也吃到了新 reward"
        % (scene_name, tag, _fmt(kv["sh_cid"]), _fmt(kv["sh_lam"]),
           "开" if kv["sh_on"] else "关",
           "真的加奖励" if kv["sh_active"] else "λ=0 ⇒ 精确对照臂（不加奖励）",
           kv["sh_exec_n"], kv["sh_exec_n_rt"], kv["sh_exec_n_sq"], kv["sh_exec_nz"],
           _fmt(kv["sh_exec_sum"]), _fmt(kv["sh_exec_abs"]), _fmt(kv["sh_exec_min"]),
           _fmt(kv["sh_exec_max"]), _fmt(kv["sh_exec_base_mean"]),
           _fmt(kv["sh_exec_shaped_mean"]), _fmt(kv["sh_exec_dmax"]), kv["sh_exec_nbase"],
           kv["sh_aux_n"], kv["sh_aux_nz"], _fmt(kv["sh_aux_sum"])))
    _mach = " ".join("%s=%s" % (k, _fmt(v)) for k, v in kv.items())
    return "%s｜机器键：%s" % (_txt, _mach)



_STUB_TAIL = (
    "\n\n【接入契约（P8-2）】签名保持 `(env, rules, is_route, sel, feats, op) -> float`，"
    "只改函数体。要求：**纯读、逐点、零前滚**（只用当前决策点的 O(1) 只读量；绝不 deepcopy 前滚、"
    "绝不 env.reset、绝不 env.step_*、绝不读拖期）；返回值语义 = **越大越好**（`r += λ·term`，λ>0）；"
    "有限浮点（nan/inf 会报错）；在 docstring 里写清：算的是哪个量、哪个方向更好、量级尺度、"
    "用到的 env 只读量、在 route/sequence 两侧是否都定义（不定义的一侧请 `return 0.0` 并说明）。"
    "可用的 O(1) 只读量见模块 docstring §四。改完请自跑："
    "`.venv\\Scripts\\python.exe tools\\_p8_2_smoke.py`（含默认关对拍）与 "
    "`.venv\\Scripts\\python.exe -m pytest tests\\test_reward_shaping.py -q`。")


def _not_implemented(cid, what):
    return NotImplementedError(
        "shape_%s（%s）**尚未实现**（P8-2 的接入桩，由候选实现员补）。"
        "本项在写成之前**故意抛错**而不返回 0 —— 静默的 0 会让一个没实现的臂看起来像"
        "'跑了但没效果'，把预注册分叉判成'全平'。%s" % (cid, what, _STUB_TAIL))


def shape_s1(env, rules, is_route, sel, feats, op) -> float:
    """**S1：WIP / 队列拥塞变化项（Δqlen）** —— 待实现（`cid=s1`）。

    预注册语义（P8-2_PLAN.md）：**WIP/队列拥塞变化** ⇒ 候选量 = 本决策点的队列拥塞
    （推荐读 `env.machines[m].queue` 的长度/剩余加工时间、`feats[k].wiq/niq`、
    `env.machine_avail(m, include_queue=True)`），把它写成**"越大越好"**的形式
    （拥塞越低越好 ⇒ 常见写法是取负：`-Σ_qlen` 或 `-机器可用时刻`）。
    ⚠ route 点 `feats=None`、sequence 点 `op=None`（一侧不定义就 `return 0.0` 并说明）。
    """
    raise _not_implemented("s1", "WIP/队列拥塞变化 Δqlen")


def shape_s2(env, rules, is_route, sel, feats, op) -> float:
    """**S2：剩余松弛总量变化项** —— 待实现（`cid=s2`）。

    预注册语义：**剩余松弛（slack）总量** ⇒ 候选量 = 本点各候选/在制工件的剩余松弛
    （推荐读 `feats[k].slack`、`op.remaining`、`Job.remaining_workload()`、`job.due_date - env.time`），
    写成"松弛越大越好"（如 `+min_k slack` / `+Σ slack`，**不要**引入拖期）。
    ⚠ 别用 `_la_window`/lookahead（零前滚）；不定义的一侧 `return 0.0` 并说明。
    """
    raise _not_implemented("s2", "剩余松弛总量变化")


def shape_s3(env, rules, is_route, sel, feats, op) -> float:
    """**S3：completion-time 项（CT / tri_shaping 同族）** —— 待实现（`cid=s3`）。

    预注册语义：**完工时刻（completion time）项** ⇒ 与 `run_fjsp_final.tri_shaping()` 的
    CT 支同源：`ct_k = env.machine_avail(machines[k]) + proc_time(k)`，
    项 = `-(ct_sel - min_k ct_k) / T_ref`（越大越好 = 越早完工；`T_ref` 用候选加工时间均值等
    O(1) 尺度，别用全局常数猜）。在 route 点 `op.candidates/op.procs` 直接可算；
    sequence 点用 `feats[k].pt` + `env.machine_avail(env.decision_machine_id)`。
    """
    raise _not_implemented("s3", "completion-time 项（tri_shaping 同族）")


def shape_s4(env, rules, is_route, sel, feats, op) -> float:
    """**S4：跨工件耦合项** —— 待实现（`cid=s4`）。

    预注册语义：**跨工件耦合** ⇒ 参考 `代码/gp_module/reward_coupling.py::coupling_reward(
    feats, sel, lam)`（P8-R3 的现成实现：选中的工序"挡了别人多少"的逐点量，越大越好）：
    本桩可直接转调它（保持 `feats`/`sel` 口径一致），sequence 点天然有定义；
    route 点若要用 `env.route_candidate_feats()`/`op` 上的耦合量，请写清定义与尺度。
    """
    raise _not_implemented("s4", "跨工件耦合项")


def shape_probe(env, rules, is_route, sel, feats, op) -> float:
    """**基础设施自检项（`cid=probe`；不是 P8-2 候选，不参与预注册分叉）**。

    `term = -10·(sel + 0.5) / K`（K = 本点候选数）⇒ 取值域 `[-10, -10/K]`，
    **只依赖 sel/K**（O(1)、确定性、无 env 读取）⇒ 允许在单测里写出**精确恒等式**：
    `_point_rewards(开) − _point_rewards(关) == λ·term_vec`（逐位）。

    用途：验证"开关骨架真的把奖励接上了、G_n 真的用新 reward"，以及默认关逐位不变的对拍。
    **禁止**把它当作候选去跑预注册分叉（它是自检工具，不是信息源）。
    """
    if is_route:
        k_n = len(getattr(op, "candidates", ()) or ())
    else:
        k_n = len(feats) if feats is not None else 0
    if k_n <= 0:
        return 0.0
    s = int(sel)
    if not (0 <= s < k_n):
        raise ValueError("probe: sel=%d 越界（K=%d）" % (s, k_n))
    return -10.0 * (float(s) + 0.5) / float(k_n)


_DISPATCH = {"s1": shape_s1, "s2": shape_s2, "s3": shape_s3, "s4": shape_s4,
             CID_PROBE: shape_probe}