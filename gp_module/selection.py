"""Population selection and behavioral-diversity strategies for GP evolution."""
import os

import numpy as np
from scipy.spatial.distance import pdist, squareform

from gp_module.phenotype import quality_gap_of

EPS_FIT = 1e-12




FIX_NEAR_POOL = os.environ.get("DSH_FIX_NEAR_POOL", "0").strip().lower()





def greedy_maxmin_selection(individuals, K: int):
    """（旧机制，保留作消融对照）以最优个体为种子，其余按表型最大最小距离挑多样性。"""
    if K <= 0 or not individuals:
        return []
    n = len(individuals)
    pc_matrix = np.array([np.asarray(ind.pc, dtype=float) for ind in individuals])
    if n > 1 and pc_matrix.shape[1] > 0:
        dist = squareform(pdist(pc_matrix, metric="euclidean"))
    else:
        dist = np.zeros((n, n))
    best_idx = min(range(n), key=lambda i: individuals[i].fitness.values[0])
    elite_idxs = [best_idx]
    remaining = set(range(n)) - {best_idx}
    while len(elite_idxs) < K and remaining:
        max_min = -1.0
        chosen = None
        for i in remaining:
            d = min(dist[i][j] for j in elite_idxs)
            if d > max_min:
                max_min = d
                chosen = i
        elite_idxs.append(chosen)
        remaining.remove(chosen)
    return [individuals[i] for i in elite_idxs]


def _behaviour_clusters(individuals, delta: float):
    """按表型 PC 行为向量聚类，返回 [(簇代表索引, 簇内成员数)]。

    δ=0：行为**完全相同**才算同簇。PC 由整数秩归一化而来（k/(n−1)），
    相同行为产生完全相同的浮点向量，因此可用精确哈希在 O(n) 内完成聚类。
    δ>0：按欧氏距离 <= δ 归为同簇（贪心序贯聚类），退化为"纳什半径"式去重。
    每簇保留适应度最好的个体作为代表。
    """
    pc = [np.asarray(ind.pc, dtype=float) for ind in individuals]
    n = len(individuals)

    def better(i, j):
        return individuals[i].fitness.values[0] < individuals[j].fitness.values[0]

    if delta <= 0.0:
        cluster_of = {}
        rep_of = {}
        for i in range(n):
            key = tuple(pc[i].tolist())
            if key not in rep_of:
                rep_of[key] = i
                cluster_of[key] = 1
            else:
                cluster_of[key] += 1
                if better(i, rep_of[key]):
                    rep_of[key] = i
        return [(rep_of[k], cluster_of[k]) for k in rep_of]

    reps, counts = [], []
    for i in range(n):
        placed = False
        for r in range(len(reps)):
            d = float(np.linalg.norm(pc[i] - pc[reps[r]]))
            if d <= delta:
                counts[r] += 1
                if better(i, reps[r]):
                    reps[r] = i
                placed = True
                break
        if not placed:
            reps.append(i)
            counts.append(1)
    return list(zip(reps, counts))


def xu_clearing_selection(individuals, K_max: int, quality_gap: float = 0.5,
                          delta: float = 0.0, min_k: int = 1):
    """Xu clearing 精英选择（替代 greedy_maxmin）。

    (1) 按行为聚类（δ=0 ⇒ 行为不同即不同簇，见 _behaviour_clusters）；
    (2) 每个簇只保留适应度最好的代表（clearing：同行为者被"清除"出候选）；
    (3) 代表按适应度升序排列，依次入选，直到某代表的相对质量差距
        gap_i = (f_i − f_best) / max(|f_best|, 1e-12) > quality_gap 时提前停止；
    (4) 结果数量限制在 [min_k, K_max]。

    与旧机制的区别：旧机制只固定"最优的 1 条"、其余纯按多样性挑（不看适应度），
    因此库内容易出现"1 条好规则 + 9 条差规则"；本机制用质量差距作为硬约束，
    保证入选规则彼此质量均衡（差距 <= quality_gap），K 随质量分布自适应。
    """
    if not individuals or K_max <= 0:
        return []
    reps = _behaviour_clusters(individuals, delta)
    reps.sort(key=lambda t: individuals[t[0]].fitness.values[0])
    f_best = individuals[reps[0][0]].fitness.values[0]

    scale = float(np.median([abs(individuals[i].fitness.values[0]) for i, _c in reps]))
    chosen, gaps = [], []
    for idx, _cnt in reps:
        f = individuals[idx].fitness.values[0]
        gap = quality_gap_of(f, f_best, scale)
        if (len(chosen) >= min_k) and gap > quality_gap:
            break
        chosen.append(idx)
        gaps.append(gap)
        if len(chosen) >= K_max:
            break
    return [individuals[i] for i in chosen], np.asarray(gaps, dtype=float)



def adaptive_clearing(individuals, ladder=(0.0, 1.0, 2.0, 4.0), kappa: int = 1,
                      K: int = 4, gap_max: float = 0.20):
    """**A2：带质量预算的自适应 δ**（固定 δ 的"每代自选"版本）。

    动机（实测，见 `docs/多样性注入进化_方案与可行性评估.md`）：
      · δ=0（论文口径）只排除"行为完全相同"的个体，收敛后输出库 ρ 只剩 1.5%；
      · 固定 δ=1 能买到多样性，但**代价因场景而异**：HH 上不付代价（端到端 −129，3/3 赢），
        LL 上要付代价（端到端 +21）。固定值无法自动区分这两种情形。

    机制：在候选 δ 阶梯上**从小到大**试，取"能通过两道闸"的**最大** δ：
      ① **库容量闸**：幸存者数 ≥ K（否则输出库凑不满 K 条动作，太激进）；
      ② **质量预算闸**：库（幸存者按适应度前 K 条）第 K 条的相对质量差距
         gap_K = (f_K − f_1) / max(|f_1|, eps) ≤ gap_max。
      任一闸不过 ⇒ **立即停止爬升**并回退到上一个通过的 δ
        （δ 越大 ⇒ 幸存者越分散 ⇒ 库尾只会更差，故无需继续试）。
      若**连 δ=0 都不过闸**（库尾本来就超出预算）⇒ 回退 δ=0：本代"买不起"多样性。

    默认 `gap_max=0.20` 由实测标定（10 代、δ=1 的库尾 gap）：
      HH 16% / LH 5.4% / HL 42% / LL 25%
      ⇒ 0.20 会**接受 HH 与 LH、拒绝 HL 与 LL**，与"存在性曲线"给出的
        "HH 有货可买、LL 没货"一致。

    返回 `(delta, survivors, n_cleared, gap_K)`；`survivors` 已按适应度升序。
    """
    if not individuals:
        return 0.0, [], 0, float("nan")
    K = max(1, int(K))
    cand = sorted({float(x) for x in ladder})
    if 0.0 not in cand:
        cand = [0.0] + cand
    best = None
    for d in cand:
        surv, n_clr = clearing_survivors(individuals, delta=d, kappa=kappa)
        if len(surv) < K:
            break
        f = [float(i.fitness.values[0]) for i in surv[:K]]
        f1 = f[0]
        scale = float(np.median([abs(float(i.fitness.values[0])) for i in surv]))
        denom = abs(f1) if abs(f1) > EPS_FIT else max(abs(scale), EPS_FIT)
        gap = (f[-1] - f1) / denom
        if gap > gap_max:
            break
        best = (d, surv, n_clr, gap)
    if best is None:
        surv, n_clr = clearing_survivors(individuals, delta=0.0, kappa=kappa)
        return 0.0, surv, n_clr, float("nan")
    return best


def adaptive_output_delta(individuals, ladder=(0.0, 1.0, 2.0, 4.0), kappa: int = 1,
                          K: int = 4, gap_max: float = 0.20, points=None, rule_of=None):
    """**A2'：只在输出阶段选一档 δ，判据 = 「预算内库 ρ 最大」**。

    与 A2(v1) 的两处关键差别（都由实测决定，见
    `docs/多样性注入进化_方案与可行性评估.md` §2/A2）：
      ① **判据改为"预算内容量（库 ρ）最大"**，而不是"最大的 δ"。
         v1 用"最大 δ"在 LL 上会**买进更差的东西**：末代 δ=1 花 20.2% 的质量代价，
         库 ρ 反而从 1.7% 掉到 0.5%。改成"ρ 最大"后 LL 会自动拒绝所有 δ>0。
      ② **只在末代输出阶段生效**（调用方在进化循环结束后调用），进化过程仍是论文口径
         δ=0 ⇒ 不扰动轨迹、也不引入逐代抖动（v1 实测逐代所选 δ 在 0/1/2 间剧烈抖动）。

    选择规则：在 δ 阶梯上取 `argmax ρ(库)`，受两道约束
      · 幸存者数 ≥ K；
      · 库尾相对质量差距 gap_K ≤ gap_max。
    平局时取**较小**的 δ（质量代价更小）。所有档都超预算 ⇒ 回退 δ=0。

    `points` 必须是 ρ 口径的 100 个排序决策点，`rule_of(ind) -> rule` 把个体编译成规则。
    返回 `(delta, survivors, info)`；`info["menu"]` 给出每一档的实测菜单，便于事后审计。
    """
    from gp_module.phenotype import top_choice

    if not individuals:
        return 0.0, [], {"menu": []}
    K = max(1, int(K))
    cand = sorted({float(x) for x in ladder})
    if 0.0 not in cand:
        cand = [0.0] + cand
    rule_of = rule_of or (lambda x: x)
    best = None
    menu = []
    for d in cand:
        surv, n_clr = clearing_survivors(individuals, delta=d, kappa=kappa)
        if len(surv) < K:
            menu.append((d, len(surv), float("nan"), float("nan")))
            continue
        lib = surv[:K]
        f = [float(i.fitness.values[0]) for i in lib]
        scale = float(np.median([abs(float(i.fitness.values[0])) for i in surv]))
        denom = abs(f[0]) if abs(f[0]) > EPS_FIT else max(abs(scale), EPS_FIT)
        gap = (f[-1] - f[0]) / denom
        sigs = [[top_choice(rule_of(i), c) for c in points] for i in lib]
        ds = [sum(1 for x, y in zip(sigs[a], sigs[b]) if x != y) / max(1, len(points))
              for a in range(len(sigs)) for b in range(a + 1, len(sigs))]
        rho = float(np.mean(ds)) if ds else float("nan")
        menu.append((d, len(surv), gap, rho))
        if gap > gap_max:
            continue
        if best is None or rho > best[3] + 1e-12:
            best = (d, surv, n_clr, rho, gap)
    if best is None:
        surv, n_clr = clearing_survivors(individuals, delta=0.0, kappa=kappa)
        return 0.0, surv, dict(menu=menu, reason="全部超预算，回退 δ=0",
                               n_cleared=n_clr, gap=float("nan"), rho=float("nan"))
    return best[0], best[1], dict(menu=menu, n_cleared=best[2], gap=best[4], rho=best[3])


def greedy_complementary_subset(T: np.ndarray, K: int):
    """按**互补性**（而非"单条最优 + 行为距离"）贪心挑选 K 条规则。

    输入 T：(n_rules, n_seeds) 每条规则在每个**选择种子**上的拖期（越小越好；
    必须用与评估互斥的种子，否则就是胜者诅咒）。

    目标（子模式）：最大化"逐种子取集合内最优"带来的改善
        F(S) = − mean_seeds( min_{i∈S} T[i, seed] )
    贪心：先取均值最优的一条；之后每次加入使 F(S∪{r}) 最大的规则。
    这是 max-coverage 型子模函数的贪心，具 (1−1/e) 近似保证。

    与旧机制的关键差别：旧机制要求成员**各自都接近最优**（quality_gap）并按**行为距离**
    拉开；本机制只看"加入后集合的联合表现"，因此**允许成员各自略差、但彼此互补**——
    实测这正是把上行空间从 14.4% 提到 32.6% 的关键（见 NichGP修复逐项验证.md）。

    返回：被选中的行索引列表（按加入顺序）。
    """
    T = np.asarray(T, dtype=float)
    n = T.shape[0]
    if n == 0 or K <= 0:
        return []
    K = min(K, n)
    chosen = [int(np.argmin(T.mean(axis=1)))]
    cur = T[chosen[0]].copy()
    while len(chosen) < K:
        best_i, best_v = None, None
        for i in range(n):
            if i in chosen:
                continue
            v = float(np.minimum(cur, T[i]).mean())
            if best_v is None or v < best_v:
                best_v, best_i = v, i
        if best_i is None or best_v >= float(cur.mean()) - 1e-9:
            break
        chosen.append(best_i)
        cur = np.minimum(cur, T[best_i])
    return chosen


def greedy_state_regret_subset(regrets: np.ndarray, K: int, anchor: int = None):
    """在同一批决策点上贪心挑选互补规则。

    ``regrets`` 的形状为 (n_rules, n_decision_points)，每项是该规则在对应状态
    所选候选动作相对于 rollout 最优候选的非负遗憾。集合代价定义为
    ``mean_points(min_rule regret)``。默认先取平均遗憾最小的规则；提供 ``anchor`` 时
    则固定该首条规则（用于保留质量最佳个体），再逐次加入能最大幅度降低集合代价的规则。
    与 ``greedy_complementary_subset`` 不同，这里覆盖的是
    **状态/决策点**，而不是整局实例；因此只能作为离线状态级动作库筛查指标。

    含 NaN 的决策点会被剔除；若没有任何有限标签则返回空列表。遗憾必须非负，
    输入中的微小负浮点误差会截为 0。
    """
    R = np.asarray(regrets, dtype=float)
    if R.ndim != 2 or R.shape[0] == 0 or K <= 0:
        return []
    valid = np.all(np.isfinite(R), axis=0)
    R = np.maximum(R[:, valid], 0.0)
    if R.shape[1] == 0:
        return []
    K = min(int(K), R.shape[0])
    means = R.mean(axis=1)
    first = int(np.argmin(means)) if anchor is None else int(anchor)
    if first < 0 or first >= R.shape[0]:
        return []
    chosen = [first]
    cur = R[chosen[0]].copy()
    while len(chosen) < K:
        best_i, best_v = None, None
        for i in range(R.shape[0]):
            if i in chosen:
                continue
            value = float(np.minimum(cur, R[i]).mean())
            if best_v is None or value < best_v:
                best_i, best_v = i, value
        if best_i is None:
            break
        chosen.append(best_i)
        cur = np.minimum(cur, R[best_i])
    return chosen


def clearing_survivors(individuals, delta: float = 0.0, kappa: int = 1):
    """Xu(2026) Algorithm 1 line 6 的 **clearing**：每代执行，返回"幸存者"。

    语义（论文）：
      · niche 半径 δ、容量 κ = "the number of high-quality individuals in fitness to be
        retained in the niche"；
      · "Clearing(pop)" —— "penalizing individuals within a niche that show poor performance
        in fitness by assigning them an **infinite fitness value**"；
      · δ=0：距离 > 0 视为可接受（不同 niche），距离 = 0（行为完全相同）归入同一 niche。
    实现：按适应度升序遍历，与已有 niche 中心 PC 距离 ≤ δ 则归入该 niche，
    容量未满则保留、已满则"被清除"（从幸存者中剔除，等价于赋 ∞ 适应度）；否则自建新 niche。

    返回 (survivors, n_cleared)。survivors 即"未被清除"的个体，
    也就是该代可用于繁殖与最终输出的高**质量且多样**集合。
    """
    if not individuals:
        return [], 0
    order = sorted(range(len(individuals)), key=lambda i: individuals[i].fitness.values[0])
    pc = [np.asarray(individuals[i].pc, dtype=float) for i in order]
    centers, counts, keep = [], [], []
    for pos, i in enumerate(order):




        if delta <= 0.0:
            tgt = None
            for c in range(len(centers)):
                if np.array_equal(pc[pos], pc[centers[c]]):
                    tgt = c
                    break
        else:
            tgt, best_d = None, float("inf")
            for c in range(len(centers)):
                d = float(np.linalg.norm(pc[pos] - pc[centers[c]]))
                if d <= delta and d < best_d:
                    best_d, tgt = d, c
        if tgt is None:
            centers.append(pos)
            counts.append(1)
            keep.append(i)
        elif counts[tgt] < kappa:
            counts[tgt] += 1
            keep.append(i)
    keep.sort(key=lambda i: individuals[i].fitness.values[0])
    return [individuals[i] for i in keep], len(individuals) - len(keep)


def nichgp_selection(individuals, K: int, quality_gap: float = 0.5, delta: float = 0.0,

                     min_k: int = 2):
    """NichGP 式选择：**先在近优集合内做行为多样性挑选**。

    步骤：① 行为聚类（δ=0 精确哈希 / δ>0 欧氏距离），每簇留最优代表（clearing）；
          ② 只保留 gap=(f_i-f_best)/|f_best| <= quality_gap 的代表（近优集合）；
          ③ 近优集合内以最优为种子，贪心取 PC 欧氏距离最大最小的 K 条（行为互补）；
          ④ 数量限制 [min_k, K]。

    与 greedy_maxmin 的区别：后者无质量约束（会挑进很差的规则）；
    与 xu_clearing 的区别：后者按质量顺次取，不主动拉开行为距离。
    """
    if not individuals or K <= 0:
        return []
    reps = [individuals[i] for i, _ in _behaviour_clusters(individuals, delta)]
    reps.sort(key=lambda ind: ind.fitness.values[0])
    f_best = reps[0].fitness.values[0]
    scale = float(np.median([abs(ind.fitness.values[0]) for ind in reps]))
    near = []
    for ind in reps:
        gap = quality_gap_of(ind.fitness.values[0], f_best, scale)
        if gap <= quality_gap or len(near) < min_k:
            near.append(ind)
    if FIX_NEAR_POOL and FIX_NEAR_POOL != "0" and len(near) <= K:



        want = K + 1 if FIX_NEAR_POOL == "k1" else max(K + 1, 2 * K)
        near = reps[:min(len(reps), want)]
    if len(near) <= K:
        return near
    pc = [np.asarray(ind.pc, dtype=float) for ind in near]
    chosen, remaining = [0], list(range(1, len(near)))
    while len(chosen) < K and remaining:
        best_i, best_d = None, -1.0
        for i in remaining:
            d = min(float(np.linalg.norm(pc[i] - pc[j])) for j in chosen)
            if d > best_d:
                best_d, best_i = d, i
        chosen.append(best_i)
        remaining.remove(best_i)
    return [near[i] for i in chosen]


def diverse_topk(individuals, K: int, points, min_dist: float = 0.05, rule_of=None,
                 route_points=None, route_min_dist: float = 0.0, route_rule_of=None):
    """按适应度排序贪心挑 K 条，并强制**两两行为距离 ≥ min_dist**（修"动作集塌缩"）。

    动机（实测）：LL 场景下返回的 top-4 规则常出现 ρ=0 —— 4 条规则在 100 个排序决策上
    **行为完全一致**，等价于只有 1 个动作 ⇒ Stage-2 的 DRL 选择器没有动作可选。
    根因：论文 L606 的"top p by fitness"在种群收敛后取到的就是一批近乎相同的个体。

    做法：按适应度升序遍历；候选必须与**已选集合**中每一条的行为距离（这些决策点上
    "首选候选不同"的比例）≥ min_dist 才入选；遍历完仍不足 K 条时，按"与已选集合的最小距离
    最大"补齐，并记录 `n_relaxed`（不静默返回重复动作）。

    ⚠ `individuals` 是**个体对象**（DEAP 个体，可能是双树），行为距离要用**编译后的规则**算，
    故必须传 `rule_of(ind) -> rule`（不传则假定个体本身可调用）。返回 (picked, info)。

    ---- 2026-09-28 新增：路由维的行为距离约束（`route_points` / `route_min_dist`）----
    动机（实测，`tools/probe_selector.py`，全规模 HH×3 种子）：在**路由决策点**上，
    库里 K 个个体的**路由树全同率高达 97.7%**（排序点 71.7%）
    ⇒ 「一个动作 = 一棵双树」设计里**路由那一半的动作空间完全退化**、不携带任何可选择性。
    根因：本函数原先只用 `points`（来自 `xu_rho_points` 的 **100 个排序决策点**）约束行为距离，
    **完全不约束路由行为** ⇒ 输出动作集在路由维上可以任意重合。

    做法：额外传入 `route_points`（**路由决策点**，建议同样 100 个，与 ρ 对称）与
    `route_min_dist`；候选必须**同时**满足两段约束才入选：

        ├ 排序块：min_j Hamming(seq_i, seq_j) ≥ min_dist
        └ 路由块：min_j Hamming(rt_i,  rt_j)  ≥ route_min_dist

    ⚠ **关键实现点（不要改成把两块拼成一个向量再取总距离）**：若拼成 120/200 维的大向量、
      用总 Hamming 距离约束，则**路由维会被排序维稀释**——排序块占 100 维，只要排序距离
      达标，总距离就达标，路由维形同虚设，又退回"只约束排序"的老问题。
      故这里两段**分别判定**，任一不满足即不选。

    ⚠ **默认逐字不变**：`route_min_dist <= 0` 或 `route_points` 为空时，所有分支与加该参数前
      **完全一致**（不计算路由签名、不做路由判定、`info` 里两个路由字段为 NaN/0）。
    """
    from gp_module.phenotype import top_choice

    _use_rt = (route_min_dist > 0.0) and bool(route_points)

    if K <= 0 or not individuals or not points:
        return list(individuals or [])[:max(0, K)], dict(
            n_relaxed=0, min_pair_dist=float("nan"),
            route_min_pair_dist=float("nan"), route_n_relaxed=0,
            route_min_dist=float(route_min_dist) if _use_rt else 0.0)
    rule_of = rule_of or (lambda x: x)
    order = sorted(individuals,
                   key=lambda i: (i.fitness.values[0] if hasattr(i, "fitness") else 0.0))
    sig = {id(ind): [top_choice(rule_of(ind), c) for c in points] for ind in order}

    def ham(a, b) -> float:
        return sum(1 for x, y in zip(a, b) if x != y) / max(1, len(points))



    sig_rt = None
    if _use_rt:
        _rt_of = route_rule_of or rule_of
        sig_rt = {}
        for ind in order:
            r = _rt_of(ind)
            row = []
            for pt in route_points:
                row.append(0 if len(pt) <= 1
                           else int(np.argmax([r(f) for f in pt])))
            sig_rt[id(ind)] = row

    def ham_rt(a, b) -> float:
        return sum(1 for x, y in zip(a, b) if x != y) / max(1, len(route_points))

    picked, relaxed = [], 0
    for ind in order:
        if len(picked) >= K:
            break
        d = min((ham(sig[id(ind)], sig[id(p)]) for p in picked), default=1.0)
        if d < min_dist:
            continue
        if _use_rt and picked:
            dr = min((ham_rt(sig_rt[id(ind)], sig_rt[id(p)]) for p in picked), default=1.0)
            if dr < route_min_dist:
                continue
        picked.append(ind)
    while len(picked) < min(K, len(order)):
        rest = [i for i in order if i not in picked]
        if not rest:
            break
        if _use_rt:


            best = max(rest, key=lambda i: (
                min((ham(sig[id(i)], sig[id(p)]) for p in picked), default=1.0),
                min((ham_rt(sig_rt[id(i)], sig_rt[id(p)]) for p in picked), default=1.0)))
        else:
            best = max(rest, key=lambda i: min((ham(sig[id(i)], sig[id(p)]) for p in picked),
                                               default=1.0))
        picked.append(best)
        relaxed += 1
    mpd = min((ham(sig[id(a)], sig[id(b)]) for i, a in enumerate(picked)
               for b in picked[i + 1:]), default=float("nan"))
    mpd_rt = float("nan")
    if _use_rt:
        mpd_rt = min((ham_rt(sig_rt[id(a)], sig_rt[id(b)]) for i, a in enumerate(picked)
                      for b in picked[i + 1:]), default=float("nan"))
    return picked[:K], dict(n_relaxed=relaxed, min_pair_dist=float(mpd),
                            route_min_pair_dist=mpd_rt,
                            route_n_relaxed=relaxed if _use_rt else 0,
                            route_min_dist=float(route_min_dist) if _use_rt else 0.0,
                            n_route_points=len(route_points) if _use_rt else 0)
