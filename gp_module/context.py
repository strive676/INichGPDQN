"""Context features and regime definitions for genetic programming evaluation."""
from typing import Iterable, List, Sequence, Tuple

import numpy as np


CTX_CELLS: List[Tuple[int, int]] = [(cc, u) for cc in (1, 2, 3) for u in (0, 1)]
N_CTX = len(CTX_CELLS)
_IDX = {b: i for i, b in enumerate(CTX_CELLS)}


def ctx_of(n_cands: int, min_slack: float, slack_q: float) -> Tuple[int, int]:
    """把一个排序决策点映射到情境格。"""
    return (min(int(n_cands), 3), 0 if float(min_slack) < float(slack_q) else 1)


def ctx_index(b: Tuple[int, int]) -> int:
    """情境格 → 下标（0..5），供向量化使用。"""
    return _IDX[b]


def slack_quantile(min_slacks: Sequence[float], q: float = 0.25) -> float:
    """min-slack 的分位数做阈值（默认下四分位，与原 `measure_ctx_gain` 口径一致）。

    用 `np.quantile`（线性插值）而**不是**手写的"取下取整下标"：原尺子用的就是 np.quantile，
    口径必须与历史结论（可达增益 −2.1%~−3.5%）可比，否则"同策略类上限"的阈值定义被悄悄换掉。
    """
    xs = np.asarray([float(x) for x in min_slacks], dtype=float)
    if xs.size == 0:
        return 0.0
    return float(np.quantile(xs, q))


def ctx_weights(cells: Iterable[Tuple[int, int]]) -> "list[float]":
    """各情境格的出现频率（用于组合适应度里按格加权）。"""
    w = [0.0] * N_CTX
    n = 0
    for b in cells:
        w[ctx_index(b)] += 1.0
        n += 1
    return [x / n for x in w] if n else w








KINDS = ("route", "seq")
N_CTX_K = N_CTX * len(KINDS)


def ctx_of_point(kind: str, n_cands: int, min_slack: float, slack_q: float):
    """带决策类型的完整情境键：(kind, (选择余地, 是否紧迫))。"""
    return (kind, ctx_of(n_cands, min_slack, slack_q))


def ctx_index_point(kind: str, b: Tuple[int, int]) -> int:
    """(kind, b) → 0..11 的下标。"""
    return KINDS.index(kind) * N_CTX + ctx_index(b)

