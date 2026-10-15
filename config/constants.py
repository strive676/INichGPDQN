"""Shared constants and runtime settings for scheduling, energy, and state features."""
import os

P0 = 10.0
P_IDLE = 2.0
IDLE_FACTOR = 0.2


SETUP_TIME = 10.0
SETUP_POWER_FRAC = 0.2
P_PROCESS = P0
SPEEDS = (0.7, 1.0, 1.3)
SPEED_L = 0.7
SPEED_M = 1.0
SPEED_H = 1.3
SPEED_THRESHOLD = 2.5
REWARD_THRESHOLD = 1.0
SLOWDOWN_EXTENSION = 1.0 / SPEED_L - 1.0
V_MEDIUM = 1.0
POWER_EXPONENT = 3.0




REWARD_MODE = "slack"



STATE_VERSION = os.environ.get("DSH_STATE_VERSION", "v2")











ENERGY_MODEL = os.environ.get("DSH_ENERGY_MODEL", "legacy").strip().lower()   # legacy | sec
PS_FRAC = float(os.environ.get("DSH_PS_FRAC", "0.6"))
ETA_LO = float(os.environ.get("DSH_ETA_LO", "0.7"))
ETA_HI = float(os.environ.get("DSH_ETA_HI", "1.4"))


def set_state_version(v: str):
    """运行时切换状态版本（供同规模消融对照使用）。"""
    global STATE_VERSION
    assert v in ("v1", "v2"), v
    STATE_VERSION = v


def get_state_version() -> str:
    return STATE_VERSION


def set_power_exponent(k: float):
    """设置功率函数指数（用于敏感性扫描）。k=3 为 Wu & Che 模型。"""
    global POWER_EXPONENT
    POWER_EXPONENT = float(k)


def processing_power(v: float) -> float:
    """功率函数 P(v) = P0 · v^k（k=3，Wu & Che 2019）。加工能耗 E = P0·v^{k-1}·p_s。"""
    return P0 * (v ** POWER_EXPONENT)


def processing_power_factor(v: float) -> float:
    """加工功率相对该机器基准功率 P_m 的倍数（**环境与后处理共用同一实现**，单一公式）。

    legacy: v^k              —— 单调 ⇒ 能耗与加工时间完全反相关
    sec   : ps + (1−ps)·v^k  —— U 形 ⇒ 存在内点最优速度 v*（能耗最优不再是"最慢机"）
    """
    k = v ** POWER_EXPONENT
    return (PS_FRAC + (1.0 - PS_FRAC) * k) if ENERGY_MODEL == "sec" else k


def select_speed(slack: float, p: float) -> float:
    """速度规则：只降速不加速。slack > 2.5·p 时降速至 L（节能），否则中速 M。"""
    return SPEED_L if slack > SPEED_THRESHOLD * p else SPEED_M
