"""Coupled reward terms for coordinating job-level scheduling decisions."""


def coupling_reward(feats, sel: int, lam: float = 1.0) -> float:
    """耦合+紧迫奖励。`feats` = 候选特征列表（`CandidateFeatures`），`sel` = 被选候选下标。

    λ=0 → 纯耦合项；λ=1 → 耦合 + 自身紧迫度。返回越大越好。
    """
    if feats is None or len(feats) == 0:
        return 0.0
    sel = int(sel)
    if sel < 0 or sel >= len(feats):
        return 0.0
    p_c = float(feats[sel].pt)
    tot = 0.0
    for k, f in enumerate(feats):
        if k == sel:
            continue
        s = float(f.slack)
        tot += max(0.0, p_c - s) - max(0.0, -s)
    return -tot - float(lam) * max(0.0, -float(feats[sel].slack))
