"""Offline schedule post-processing for machine-speed optimization."""
from collections import defaultdict

from config import (IDLE_FACTOR, P0, POWER_EXPONENT, SPEED_L, SPEED_M,
                    processing_power_factor)


def build_graph(log, arrival):
    """log: [(jid, oi, machine, start, end, proc_time)] → (nodes, preds, succs, dur, proc, release)。

    dur = 机器占用时长（含维修）；proc = 纯加工时间（中速基准）。降速时只缩放 proc，
    维修等非加工部分保持不变。

    release = {(jid, 0): arrival[jid]}：工件首道工序的最早开工下界，**必需**。
    缺失它会让 FF 偏大、降速决策失去安全性（见模块 docstring 的修复记录）。
    """
    if arrival is None:
        raise ValueError("build_graph 必须提供 arrival（工件释放约束）；"
                         "缺省（无释放约束）的松弛图会高估自由时差、使降速破坏拖期不变性")
    nodes, dur, proc = [], {}, {}
    for rec in log:
        jid, oi, _m, s, e = rec[0], rec[1], rec[2], rec[3], rec[4]
        n = (jid, oi)
        nodes.append(n)
        dur[n] = float(e - s)
        proc[n] = float(rec[5]) if len(rec) > 5 else float(e - s)
    release = {(jid, 0): float(arrival[jid]) for jid, _ois in _by_job(log).items()
               if jid in arrival}
    preds = defaultdict(list)
    succs = defaultdict(list)
    by_job = defaultdict(list)
    for rec in log:
        by_job[rec[0]].append(rec[1])
    for jid, ois in by_job.items():
        ois_sorted = sorted(ois)
        for a, b in zip(ois_sorted, ois_sorted[1:]):
            preds[(jid, b)].append((jid, a))
            succs[(jid, a)].append((jid, b))
    by_machine = defaultdict(list)
    for rec in log:
        by_machine[rec[2]].append((rec[3], rec[0], rec[1]))
    for _m, lst in by_machine.items():
        lst.sort()
        for (_, j1, o1), (_, j2, o2) in zip(lst, lst[1:]):
            preds[(j2, o2)].append((j1, o1))
            succs[(j1, o1)].append((j2, o2))
    return nodes, preds, succs, dur, proc, release


def _by_job(log):
    by_job = defaultdict(list)
    for rec in log:
        by_job[rec[0]].append(rec[1])
    return by_job


def durations_at_speed(dur, proc, speeds):
    """按速度得到新的机器占用时长（只缩放加工部分）。"""
    return {n: (dur[n] - proc[n]) + proc[n] / speeds[n] for n in dur}


def topo_order(nodes, preds, succs):
    """拓扑排序。**包含环/不可行输入时立刻报错**（修复 D7）。

    原先实现遇到环只返回一个短列表（甚至空），`compute_times` 随后静默给出
    `ms=0.0` 的空结果，直到更下游才 KeyError ⇒ 未来某次改动产生不可行 log 时
    会先产出无意义的 makespan/能耗而非报错。真实 schedule_log 必为 DAG，故该断言不改变现有行为。
    """
    indeg = {n: len(preds[n]) for n in nodes}
    stack = [n for n in nodes if indeg[n] == 0]
    order = []
    while stack:
        n = stack.pop()
        order.append(n)
        for s in succs[n]:
            indeg[s] -= 1
            if indeg[s] == 0:
                stack.append(s)
    if len(order) != len(nodes):
        raise ValueError(
            f"析取图存在环或孤立前驱（拓扑序 {len(order)} != 节点数 {len(nodes)}）："
            f"请检查 schedule_log 的工件链/机器链一致性")
    return order


def compute_times(order, preds, succs, dur_map, makespan_bound=None, release=None):
    """前向 ES/EF + 后向 LS/LF。

    release = {(jid, oi): 最早开工下界}（工件首道工序 = 到达时刻）。
    ⚠ 不传 release 等价于"所有工件在 t=0 全部到达"，是松弛模型，FF 会偏大 ——
    生产路径必须传（`build_graph` 的返回值就是它）。
    """
    ES, EF = {}, {}
    if len(order) != len(dur_map):
        raise ValueError(f"拓扑序不完整：order={len(order)} vs 节点数={len(dur_map)}")
    for n in order:
        ES[n] = max([EF[p] for p in preds[n]] or [0.0])
        if release is not None:
            lo = release.get(n)
            if lo is not None and lo > ES[n]:
                ES[n] = float(lo)
        EF[n] = ES[n] + dur_map[n]
    makespan = max(EF.values()) if EF else 0.0
    LS, LF = {}, {}
    for n in reversed(order):
        succ_ls = [LS[s] for s in succs[n]]
        LF[n] = min(succ_ls) if succ_ls else makespan
        LS[n] = LF[n] - dur_map[n]
    return ES, EF, LS, LF, makespan


def free_float(nodes, succs, ES, EF, makespan):
    FF = {}
    for n in nodes:
        ss = succs[n]
        FF[n] = (min(ES[s] for s in ss) - EF[n]) if ss else (makespan - EF[n])
    return FF


def job_ops(log):
    by_job = defaultdict(list)
    for rec in log:
        by_job[rec[0]].append(rec[1])
    return by_job


def apply_slowdown(log, FF, proc, speed_l=SPEED_L):
    """返回 {node: speed}：非工件末道且 FF ≥ (1/v−1)·p 的工序降速到 speed_l。"""
    ext = 1.0 / speed_l - 1.0
    term = set()
    for jid, ois in job_ops(log).items():
        term.add((jid, max(ois)))
    speeds = {}
    slowed = []
    for rec in log:
        jid, oi = rec[0], rec[1]
        n = (jid, oi)
        p = proc[n]
        if n in term:
            speeds[n] = SPEED_M
        elif p > 0 and FF[n] >= ext * p:
            speeds[n] = speed_l
            slowed.append(n)
        else:
            speeds[n] = SPEED_M
    return speeds, slowed


def greedy_slowdown(log, nodes, preds, succs, dur, proc, order, due, speed_l=SPEED_L,
                    release=None):
    """贪心版（比自由时差更宽松，但仍保证拖期不变）。

    约束从"不推迟任何后继最早开始"放宽为"任何工件的完工时刻不超过
    max(基线完工时刻, 交期)"——对按时工件允许吃到交期，对已拖期工件不允许变差。
    因此拖期严格不变，但可降速的工序更多。

    ⚠ release 同样必需（见模块 docstring 的修复记录）：缺它则基线完工时刻与
    cap 都建立在松弛时间上，实际会推迟工件完工时刻。
    """
    term = {jid: (jid, max(ois)) for jid, ois in job_ops(log).items()}
    base_dur = durations_at_speed(dur, proc, {n: SPEED_M for n in nodes})
    _, base_EF, _, _, _ = compute_times(order, preds, succs, base_dur, release=release)
    base_C = {jid: base_EF[node] for jid, node in term.items()}
    cap = {jid: max(base_C[jid], due[jid]) + 1e-9 for jid in base_C}
    term_nodes = set(term.values())
    _, _, _LS, LF, _ = compute_times(order, preds, succs, base_dur, release=release)

    speeds = {n: SPEED_M for n in nodes}
    accepted = []
    cand = [n for n in nodes if n not in term_nodes]
    cand.sort(key=lambda n: -(LF[n] - base_EF[n]))
    for n in cand:
        speeds[n] = speed_l
        cur = durations_at_speed(dur, proc, speeds)
        _, EF2, _, _, _ = compute_times(order, preds, succs, cur, release=release)
        if all(EF2[node] <= cap[j] + 1e-6 for j, node in term.items()):
            accepted.append(n)
        else:
            speeds[n] = SPEED_M
    return speeds, accepted


def plan_energy(log, speeds, makespan, power, idle_factor=IDLE_FACTOR,
                setup_energy=0.0, k=POWER_EXPONENT):
    """**冻结计划**的四分量能耗（唯一实现，供所有后处理路径复用）。

    参数
      power: 逐机器基准功率序列 P_m（FJSP 传 `env.machine_power`；DJSS 全为 P0）。
             —— 修复记录：旧 `schedule_energy` 用恒定 `P_IDLE` 功率处理 FJSP 调度，
             导致 PEc 偏差 −29%~+35%（依调度而异）。
      setup_energy: 预备能耗 REc（本计划不变，由调用方给出）。
    口径（**整局计划口径**，需与调用方声明一致）：
      PEc = Σ_ops P_m·φ(v)·(p/v)，φ 由 `processing_power_factor` 提供（legacy: v^k；sec: ps+(1−ps)·v^k）
      busy_m = Σ_ops p/v ；bd_m = Σ_ops (占用时长 − p)（= 维修时长，降速不改）
      IEc = Σ_m P_m·IDLE·max(0, makespan − busy_m − bd_m)   ← 维修时间**不计入空闲**
      REp = Σ_m P_m·IDLE·bd_m
      total = PEc + IEc + REp + REc
    修复记录：旧实现只返回 PEc+IEc 且把维修时间算进空闲、并完全漏掉 REc
    ⇒ 总量不落在任何仿真口径上（与整局口径差 −11.2%、与到件口径差 +10.0%），
    而 REc 本身随路由策略变化（4.8%~9.0%），跨臂比较因此失真。
    """
    pec, busy, bd = 0.0, defaultdict(float), defaultdict(float)
    for rec in log:
        jid, oi, m = rec[0], rec[1], rec[2]
        p = float(rec[5]) if len(rec) > 5 else float(rec[4] - rec[3])
        occ = float(rec[4] - rec[3])
        v = speeds[(jid, oi)]

        pec += power[m] * processing_power_factor(v) * (p / v)
        busy[m] += p / v
        bd[m] += max(0.0, occ - p)
    M = len(power)
    iec = sum(power[m] * idle_factor * max(0.0, makespan - busy.get(m, 0.0) - bd.get(m, 0.0))
              for m in range(M))
    rep = sum(power[m] * idle_factor * bd.get(m, 0.0) for m in range(M))
    return dict(pec=pec, iec=iec, rep=rep, rec=float(setup_energy),
                total=pec + iec + rep + float(setup_energy))


def apply_slowdown_graded(log, FF, proc, v_opt=0.0, v_floor=0.0):
    """**连续**降速：每道工序按**自己的自由时差上限**降到最低可行速度。

        v_i = max( p_i/(p_i + FF_i),  v_opt )

    不变性（与二档版同一证明，只是把"用满/不用"改成"**恰好用满**"）：
        降速引入的延迟 δ_i = p_i/v_i − p_i ≤ FF_i  ⇒ 不推迟任何后继的最早开始
        ⇒ 各工序 ES/EF、工件完工时刻、makespan **严格不变** ⇒ 拖期严格不变。
    与二档版（`apply_slowdown`）的区别：二档版只处理 "FF ≥ 0.429·p" 的工序，
    其余**一点不降**；本版把"时差不够降满"的工序也**按可用的量降**，故节能严格更多。

    v_opt：能量最优速度下界（legacy 下 E∝v² 单调 ⇒ 取 0，越低越省；
           sec 下 E(v) 呈 U 形、最优 v*≈0.909 ⇒ 低于 v* 反而更耗能，故取 v*）。
    v_floor：可行速度下界（设备/工艺限制，默认 0 表示不设限）。
    工件末道工序**不降速**（其完工时刻 = 工件完工时刻，降它会改变拖期）。
    """
    ext_cap = 1.0 / max(1e-9, v_floor) if v_floor > 0 else float("inf")
    term = set()
    for jid, ois in job_ops(log).items():
        term.add((jid, max(ois)))
    speeds, slowed, delay = {}, [], 0.0
    for rec in log:
        jid, oi = rec[0], rec[1]
        n = (jid, oi)
        p = proc[n]
        if n in term or p <= 0:
            speeds[n] = SPEED_M
            continue
        ff = max(0.0, float(FF[n]))
        v_min = max(v_opt, p / (p + ff))
        if ext_cap is not float("inf"):
            v_min = max(v_min, v_floor)
        speeds[n] = v_min if v_min < SPEED_M - 1e-9 else SPEED_M
        if speeds[n] < SPEED_M - 1e-9:
            slowed.append(n)
            delay += p / speeds[n] - p
    return speeds, slowed, delay


def schedule_energy(log, speeds, makespan, n_machines, power=None, setup_energy=0.0):
    """按冻结计划重估能耗，返回四分量字典（见 `plan_energy`）。

    power=None 时退回 DJSS 恒定功率 `[P0]*M`（保持 DJSS 调用方语义）。
    """
    return plan_energy(log, speeds, makespan, power if power is not None else [P0] * n_machines,
                       setup_energy=setup_energy)
