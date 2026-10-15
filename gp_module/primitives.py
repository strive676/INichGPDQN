"""Primitive-set construction and terminal definitions for GP dispatching rules."""
import operator
import os
import random
from functools import partial
from deap import gp


def _protected_div(a: float, b: float) -> float:
    return a / b if abs(b) > 1e-6 else 1.0


def _if_then_else(c: float, a: float, b: float) -> float:
    return a if c > 0 else b


FEATURE_NAMES = ["PT", "WKR", "CR", "SLACK", "OWT", "TIS", "NOR",
                 "WIQ", "NIQ", "MRT", "MWT", "MBT", "ERC"]
CTX_NAMES = ["T_PROG", "U_GLOBAL", "CONG", "N_ARRIVE"]


def ctx_enabled() -> bool:
    return os.environ.get("DSH_PSET_CTX", "0").strip() == "1"


def pset_ext() -> bool:
    """是否启用扩展原语/常数域。**默认启用**（经实验验证的唯一有效增益项）。

    实测（HH/LL，同种子同实例同预算）：常数域 ±1 → 库内最优 **1689 / 1020**；
    ±10+±100 → **1601 / 919**（−5.2% / −9.9%），并使"学出规则优于最优手工组合"成立
    （vs 手工最优 1669 / 970）。再加 ±1000 反而差 22%；再加 4 个全局上下文终端无益。
    复现旧结果：设 `DSH_PSET=0`（或 off / none）。
    """
    return os.environ.get("DSH_PSET", "ext").strip().lower() not in ("0", "off", "none")


def build_pset() -> gp.PrimitiveSetTyped:
    names = FEATURE_NAMES + (CTX_NAMES if ctx_enabled() else [])
    pset = gp.PrimitiveSetTyped("MAIN", [float] * len(names), float)
    rename = {f"ARG{i}": name for i, name in enumerate(names)}
    pset.renameArguments(**rename)
    pset.addPrimitive(operator.add, [float, float], float, name="add")
    pset.addPrimitive(operator.sub, [float, float], float, name="sub")
    pset.addPrimitive(operator.mul, [float, float], float, name="mul")
    pset.addPrimitive(_protected_div, [float, float], float, name="div")
    pset.addPrimitive(max, [float, float], float, name="max")
    pset.addPrimitive(min, [float, float], float, name="min")
    pset.addPrimitive(_if_then_else, [float, float, float], float, name="ifte")
    pset.addEphemeralConstant("c", partial(random.uniform, -1.0, 1.0), float)





    if pset_ext():
        pset.addPrimitive(operator.neg, [float], float, name="neg")
        pset.addPrimitive(abs, [float], float, name="abs")



        for _ub in os.environ.get("DSH_PSET_C", "10,100").split(","):
            _ub = _ub.strip()
            if _ub:
                pset.addEphemeralConstant(f"c{_ub}",
                                          partial(random.uniform, -float(_ub), float(_ub)), float)
    return pset
