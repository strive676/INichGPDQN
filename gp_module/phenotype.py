"""Behavioral metrics and phenotype utilities for comparing dispatching rules."""
import numpy as np


def compute_normalized_pc(rule, sampled_points) -> np.ndarray:
    pc = []
    for candidates in sampled_points:
        priorities = np.asarray([rule(f) for f in candidates], dtype=float)
        n = len(candidates)
        if n <= 1:
            pc.append(0.0)
            continue




        order = np.lexsort((np.arange(n), -priorities))
        ranks = np.empty(n, dtype=float)
        ranks[order] = np.arange(1, n + 1)
        normalized = (n - ranks) / (n - 1)
        pc.extend(normalized.tolist())
    return np.array(pc, dtype=float)


def top_choice(rule, candidates) -> int:
    """规则在该决策点选中的候选下标（优先级最高者）。"""
    pr = [rule(f) for f in candidates]
    return int(np.argmax(pr))


def compute_reference_pc(rule, sampled_points, ref_rule) -> np.ndarray:
    """Xu(2026) 口径的 PC：**每个决策点一个值** = 被评估规则的首选候选
    在**参考规则**下的排名（1 = 参考规则也认为最优）。

    论文原文：
      "Each value within the vector signifies the rank by the reference rule assigned to
       the candidate machine/operation where the calculated rule designates the highest
       priority at a given decision point."
    参考规则 = 每代**最优个体**演化出的 sequencing / routing 规则（逐代更新）。

    与 `compute_normalized_pc` 的区别：
      · 后者记录"该规则自己给全部候选的归一化排名"（每点 n 个值，无外部锚定）；
      · 本函数记录"该规则的首选，在**当前最优规则**眼里的排名"（每点 1 个值，有锚定、逐代可比）。
    论文 Table I 的算例：参考排名 [1,3,2]，被评估规则排名 [3,1,2] ⇒ 其首选 O2 在参考下的排名为 3
    ⇒ PC 第一个值为 3 ✓ 与本实现一致。
    """
    pc = []
    for candidates in sampled_points:
        n = len(candidates)
        if n <= 1:
            pc.append(1.0)
            continue
        k = top_choice(rule, candidates)
        ref_pr = np.asarray([ref_rule(f) for f in candidates], dtype=float)

        order = np.lexsort((np.arange(n), -ref_pr))
        ranks = np.empty(n, dtype=float)
        ranks[order] = np.arange(1, n + 1)
        pc.append(float(ranks[k]))
    return np.array(pc, dtype=float)


def quality_gap_of(f: float, f_best: float, scale: float, eps: float = 1e-12) -> float:
    """相对质量差距 `(f − f_best)/|f_best|`（`GPParams.quality_gap` 的文档口径）。

    修复记录（GP-4）：原先分母写作 `max(|f_best|, 1e-12)`，当最优适应度**恰为 0**
    （拖期=0，在低负载/宽松交期场景可达）时其余个体的 gap 被放大到 ~5e13 ⇒ 全部超过
    quality_gap ⇒ 近优集塌缩到 min_k（实测只返回 2 条规则）。
    修法：**仅在 f_best≈0 时**改用群体尺度 `scale`（调用方传入，通常为适应度中位数）作分母；
    f_best≠0 时保持原公式逐字不变 ⇒ 不改变既有结果的语义。
    """
    denom = abs(f_best) if abs(f_best) > eps else max(abs(scale), eps)
    return (f - f_best) / denom


def behaviour_difference(rules, sampled_points) -> float:
    """ρ：两两规则在决策点上所选候选的 **Hamming 差异率**均值（Xu 式 (10)，4 条规则共 6 对）。

    论文："each scheduling heuristic … is evaluated on these 100 decisions, with the Hamming
    distance utilized to calculate the behavioral difference between each pair of sequencing
    rules (a total of 6 pairs for 4 sequencing rules)" ⇒ 本函数返回平均值（0~1）。
    """
    K = len(rules)
    if K < 2:
        return float("nan")
    diffs = []
    for i in range(K):
        for j in range(i + 1, K):
            d = sum(1 for c in sampled_points if top_choice(rules[i], c) != top_choice(rules[j], c))
            diffs.append(d / max(1, len(sampled_points)))
    return float(np.mean(diffs))
