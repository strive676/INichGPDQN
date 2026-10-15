"""Sampling utilities for selecting representative evaluation points."""
import numpy as np


def stratified_sampling(points, n_target=40, n_urgent=15, n_load=15, n_energy=10,
                        slack_threshold=None, load_threshold=None, energy_threshold=None,
                        rng=None, routing_ratio=0.0):
    """分层采样决策点。

    routing_ratio > 0（修复 2）：先按 `kind` 分池，sequencing 与 routing 各取
    n_target·(1−ratio) 与 n_target·ratio 个，再在各自池内按紧迫度/负载/能耗分层，
    从而保证**两类决策点都被 PC 描述子覆盖**（此前 routing 点数为 0）。
    routing_ratio = 0（默认）：保持旧行为完全不变。
    """
    rng = rng or np.random.default_rng(0)
    kinds = {p.get("kind") for p in points}
    if routing_ratio > 0 and "route" in kinds:
        n_route = int(round(n_target * routing_ratio))
        n_seq = n_target - n_route
        out = []
        for kind, want in (("sequence", n_seq), ("route", n_route)):
            pool = [p for p in points if p.get("kind", "sequence") == kind]
            if not pool or want <= 0:
                continue
            share = want / float(n_target)
            out += _stratify(pool, want, max(1, int(round(n_urgent * share))),
                             max(1, int(round(n_load * share))),
                             max(1, int(round(n_energy * share))),
                             slack_threshold, load_threshold, energy_threshold, rng)
        return out[:n_target]
    return _stratify(points, n_target, n_urgent, n_load, n_energy,
                     slack_threshold, load_threshold, energy_threshold, rng)


def _stratify(points, n_target, n_urgent, n_load, n_energy,
              slack_threshold, load_threshold, energy_threshold, rng):
    slacks = np.array([p["slack_mean"] for p in points])
    pt_vars = np.array([p["pt_var"] for p in points])
    energies = np.array([p["energy_diff"] for p in points])
    slack_threshold = slack_threshold if slack_threshold is not None else np.quantile(slacks, 0.25)
    load_threshold = load_threshold if load_threshold is not None else np.quantile(pt_vars, 0.75)
    energy_threshold = energy_threshold if energy_threshold is not None else np.quantile(energies, 0.75)

    urgent = [i for i, s in enumerate(slacks) if s < slack_threshold]
    load = [i for i, v in enumerate(pt_vars) if v > load_threshold]
    energy = [i for i, e in enumerate(energies) if e > energy_threshold]
    taken = set()
    out = []
    for idxs, cap in [(urgent, n_urgent), (load, n_load), (energy, n_energy)]:
        picks = rng.choice(idxs, size=min(cap, len(idxs)), replace=False).tolist()
        for i in picks:
            if i not in taken:
                out.append(points[i])
                taken.add(i)
    remaining = [p for i, p in enumerate(points) if i not in taken]
    need = n_target - len(out)
    if need > 0 and remaining:
        fill = rng.choice(len(remaining), size=min(need, len(remaining)), replace=False)
        for i in fill:
            out.append(remaining[i])



    if len(points) <= n_target:
        return list(points)
    while len(out) < n_target:
        out.append(points[int(rng.integers(0, len(points)))])
    return out[:n_target]
