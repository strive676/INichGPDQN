"""S4 reward shaping for coupling related scheduling decisions."""

import math

from gp_module.reward_coupling import coupling_reward


S4_CID = "s4"

S4_NAME = "跨工件耦合项"

S4_LAM_INNER = 0.0

S4_FORMULA = "term(c) = -Σ_{k≠c} [ max(0, p_c - s_k) - max(0, -s_k) ]"

S4_SIDE = "sequence"
S4_ROUTE_VALUE = 0.0

S4_SCHEMA = "p8-2.s4.0"


_ORIG = {}




def _finite(v, what):
    """转 float 并校验有限性；失败 ⇒ 报错（**不静默当 0**）。"""
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ValueError("S4: %s=%r 不是浮点数 ⇒ 直接报错，不静默当 0。" % (what, v))
    if not math.isfinite(x):
        raise ValueError("S4: %s=%r 非有限（nan/inf 会毁掉整条奖励链）⇒ 直接报错。" % (what, v))
    return x


def check_feats(feats):
    """校验 `feats` 口径（`pt`/`slack` 两个标量都在且有限）⇒ 返回候选数 `K`。

    缺键/类型错/非有限/空 ⇒ `ValueError`（本模块**没有**"猜一个 0 顶上"的路径）。
    """
    if feats is None:
        raise ValueError(
            "S4: sequence 点的 `feats` 为 None ⇒ 直接报错。"
            "（`shape_term` 的签名契约：sequence 点必须给候选特征列表。）")
    try:
        n = len(feats)
    except TypeError:
        raise ValueError("S4: `feats`=%r 没有长度 ⇒ 不是候选特征列表。" % (feats,))
    if n == 0:
        raise ValueError(
            "S4: `feats` 为空（K=0）⇒ 直接报错。"
            "（K==0 时骨架 `shape_term` 已先行 `return 0.0`，不该走到本函数。）")
    for k in range(n):
        f = feats[k]
        for attr in ("pt", "slack"):
            if not hasattr(f, attr):
                raise ValueError(
                    "S4: feats[%d]=%r 缺属性 `%s` ⇒ 直接报错（口径不符，不静默当 0）。"
                    % (k, type(f).__name__, attr))
            _finite(getattr(f, attr), "feats[%d].%s" % (k, attr))
    return n


def check_sel(sel, k_n):
    """校验 `sel` 是 `[0, k_n)` 的整数下标 ⇒ 返回 int（越界/非整数 ⇒ 报错）。"""
    try:
        s = int(sel)
    except (TypeError, ValueError):
        raise ValueError("S4: sel=%r 不是整数下标（sequence: feats 下标）⇒ 直接报错。" % (sel,))
    if not (0 <= s < int(k_n)):
        raise ValueError("S4: sel=%d 越界（本点 K=%d）⇒ 直接报错。" % (s, int(k_n)))
    return s




def coupling_term(feats, sel) -> float:
    """**S4 的项**：纯耦合支 `coupling_reward(feats, sel, lam=S4_LAM_INNER)`。

    先按本模块的严格口径校验 `feats/sel`，再**转调** `reward_coupling.coupling_reward`
    （不重写公式 ⇒ 与 P8-R3 的离线口径逐字同源）。返回有限 float，越大越好。
    """
    k_n = check_feats(feats)
    s = check_sel(sel, k_n)
    v = float(coupling_reward(feats, s, S4_LAM_INNER))
    if not math.isfinite(v):
        raise ValueError("S4: coupling_reward 返回了 %r（非有限）⇒ 直接报错。" % (v,))
    return v


def term_vector(feats):
    """本点**每个候选**的项 ⇒ `list[float]`（长度 = K）。测试/尺度标定用（只读）。"""
    k_n = check_feats(feats)
    return [coupling_term(feats, k) for k in range(k_n)]


def shape_s4(env, rules, is_route, sel, feats, op) -> float:
    """**骨架分派入口**（签名与 `reward_shaping.shape_sN` 逐字一致）。

    · `is_route=True` ⇒ `0.0`（该侧无此语义，见模块 docstring）；
    · `is_route=False` ⇒ `coupling_term(feats, sel)`。

    ⚠ 本函数**不读 `env` / `rules` / `op`**（纯读的最强形式：连 env 都不碰）⇒
    "零前滚"是结构性的，不靠调用者自觉。`rules` 允许为 None（本项不依赖规则）。
    """
    if is_route:
        return float(S4_ROUTE_VALUE)
    return coupling_term(feats, sel)




def install(force: bool = False) -> dict:
    """把 S4 装进骨架的 `_DISPATCH["s4"]`（幂等）⇒ 返回 `{"installed", "cid", "prev"}`。

    · 已装且 `force=False` ⇒ `installed=False`（重复调用安全）；
    · `force=True` ⇒ 无论如何重装（并覆盖"别人装的 s4"——本批 s4 归本模块）。
    """
    import gp_module.reward_shaping as RS
    prev = RS._DISPATCH.get(S4_CID)
    if (not force) and getattr(prev, "_p8_2_s4", False):
        return {"installed": False, "cid": S4_CID, "prev": getattr(prev, "__name__", None)}
    if S4_CID not in _ORIG:
        _ORIG[S4_CID] = prev
        _ORIG["shape_attr"] = getattr(RS, "shape_s4", None)
    shape_s4._p8_2_s4 = True
    RS.shape_s4 = shape_s4
    RS._DISPATCH[S4_CID] = shape_s4
    return {"installed": True, "cid": S4_CID, "prev": getattr(prev, "__name__", None)}


def uninstall() -> dict:
    """还原成安装前的那一个槽位（测试用；正常运行不需要）。"""
    import gp_module.reward_shaping as RS
    if S4_CID not in _ORIG:
        return {"uninstalled": False, "reason": "never installed"}
    RS._DISPATCH[S4_CID] = _ORIG[S4_CID]
    if _ORIG.get("shape_attr") is not None:
        RS.shape_s4 = _ORIG["shape_attr"]
    return {"uninstalled": True, "cid": S4_CID}


def is_installed() -> bool:
    """骨架当前分派的 `s4` 是不是本模块的实现。"""
    import gp_module.reward_shaping as RS
    return bool(getattr(RS._DISPATCH.get(S4_CID), "_p8_2_s4", False))


def meta() -> dict:
    """本模块的口径元信息（报告/戳行/复核用）。"""
    return {"schema": S4_SCHEMA, "cid": S4_CID, "name": S4_NAME,
            "side": S4_SIDE, "formula": S4_FORMULA,
            "lam_inner": S4_LAM_INNER, "route_value": S4_ROUTE_VALUE,
            "installed": is_installed()}


if __name__ == "__main__":
    import json
    print(json.dumps(meta(), ensure_ascii=False, indent=2))
