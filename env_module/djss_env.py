"""Dynamic job-shop simulation environment with job and machine events."""
import heapq
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from config import (P0, P_PROCESS, IDLE_FACTOR, SETUP_TIME, SETUP_POWER_FRAC,
                    POWER_EXPONENT, SPEED_M, REWARD_MODE, select_speed,
                    processing_power_factor)
from config import constants as _constants
from config.scenarios import Scenario, Scale
from env_module.reward_calculator import RewardCalculator
from env_module.state_extractor import StateExtractor


@dataclass
class CandidateFeatures:
    pt: float = 0.0
    wkr: float = 0.0
    cr: float = 0.0
    slack: float = 0.0
    owt: float = 0.0
    tis: float = 0.0
    nor: float = 0.0
    wiq: float = 0.0
    niq: float = 0.0
    mrt: float = 0.0
    mwt: float = 0.0
    mbt: float = 0.0
    pec: float = 0.0

    t_prog: float = 0.0
    u_global: float = 0.0
    cong: float = 0.0
    n_arrive: float = 0.0

    def to_list(self, ext_ctx: bool = False) -> List[float]:
        base = [self.pt, self.wkr, self.cr, self.slack, self.owt, self.tis,
                self.nor, self.wiq, self.niq, self.mrt, self.mwt, self.mbt, self.pec]
        if ext_ctx:
            base += [self.t_prog, self.u_global, self.cong, self.n_arrive]
        return base


@dataclass
class Operation:
    machine: int
    proc_time: float
    index: int
    remaining: float = field(init=False)
    release_time: float = 0.0


    candidates: List[int] = field(default_factory=list)
    procs: Dict[int, float] = field(default_factory=dict)
    jid: int = -1
    fail_count: int = 0

    def __post_init__(self):
        self.remaining = self.proc_time


@dataclass
class Job:
    id: int
    arrival: float
    due_date: float
    ops: List[Operation]
    next_op_idx: int = 0
    completed: bool = False
    completion_time: float = 0.0
    _wl_cache: Optional[float] = None

    def remaining_workload(self) -> float:
        """剩余工作量（Σ op.remaining）。

        记忆化：只在"完工(下一工序前移)"或"故障(当前工序 remaining 减少)"时失效，
        其余时刻直接返回缓存。实测这一步省掉约 55 万次/局的重复求和（占单步耗时 ~35%）。
        语义与原实现完全一致（缓存失效点覆盖所有会改变 remaining 的事件）。
        """
        if self._wl_cache is None:
            self._wl_cache = sum(o.remaining for o in self.ops[self.next_op_idx:])
        return self._wl_cache

    def invalidate_workload(self):
        self._wl_cache = None

    def remaining_ops(self) -> int:
        return len(self.ops) - self.next_op_idx


class Machine:
    def __init__(self, mid: int, power: float = P0):
        self.id = mid
        self.power = power
        self.queue: List[Tuple[int, int]] = []   # (job_id, op_idx)
        self.busy = False
        self.current: Optional[Tuple[int, int]] = None
        self.current_start = 0.0
        self.busy_until = 0.0
        self.breakdown = False
        self.breakdown_until = 0.0
        self.busy_time = 0.0
        self.setup_count = 0
        self.on = False
        self.idle_since = 0.0
        self.breakdown_count = 0


        self.count_start: Optional[float] = None
        self.busy_snap = 0.0
        self.bd_snap = 0.0
        self.current_speed = 1.0
        self.breakdown_time = 0.0
        self.breakdown_start = 0.0
        self.op_start = 0.0


class DJSSEnv:
    def __init__(self, scenario: Scenario, scale: Scale, seed: Optional[int] = None,
                 reward_mode: str = None, emit_state: bool = True,
                 state_version: str = None, crn_faults: bool = False,
                 repair_lo: Optional[float] = None, repair_hi: Optional[float] = None):
        self.scenario = scenario
        self.scale = scale

        self._repair_lo, self._repair_hi = repair_lo, repair_hi



        self.crn_faults = crn_faults
        self._seed = seed
        self.state_extractor = StateExtractor(scale.n_machines,
                                              _constants.STATE_VERSION
                                              if state_version is None
                                              else state_version)
        self.reward_calculator = RewardCalculator()
        self.reward_mode = REWARD_MODE if reward_mode is None else reward_mode



        self.emit_state = emit_state
        self.rng = np.random.default_rng(seed)
        self.reset(seed)

    @property
    def state_dim(self) -> int:
        return self.state_extractor.state_dim


    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self._seed = seed

        if self.crn_faults:
            base = 0 if self._seed is None else int(self._seed)
            self.fault_rngs = [np.random.default_rng(base * 1000003 + mid * 7919 + 12345)
                               for mid in range(self.scale.n_machines)]
        else:
            self.fault_rngs = None
        self.time = 0.0
        self.jobs: List[Job] = []
        self.machines = [Machine(i) for i in range(self.scale.n_machines)]
        self.event_queue: List[Tuple[float, int, str, object]] = []
        self.completed_jobs = 0
        self.decision_machine_id = -1
        self.energy_processing = 0.0
        self.energy_repair = 0.0
        self.energy_setup = 0.0

        self.energy_processing_counted = 0.0
        self.energy_setup_counted = 0.0
        self.schedule_log = []
        self._tard_running = 0.0
        self.warmup_end_time = float("inf")
        self._generate_jobs()
        self._warmup()
        self.warmup_end_time = self.time


        self._energy_base = self.energy_breakdown()
        self._energy_setup_base = self.energy_setup
        return self.state()

    def _generate_jobs(self):
        mu_p = (self.scale.proc_lo + self.scale.proc_hi) / 2.0
        mu_o = (self.scale.min_ops + self.scale.max_ops) / 2.0
        lam = self.scenario.utilization * self.scale.n_machines / (mu_p * mu_o)
        t = 0.0
        for jid in range(self.scale.n_jobs):
            t += self.rng.exponential(1.0 / lam)
            n_ops = int(self.rng.integers(self.scale.min_ops, self.scale.max_ops + 1))
            ops = []
            for oi in range(n_ops):
                m = int(self.rng.integers(0, self.scale.n_machines))
                pt = float(self.rng.uniform(self.scale.proc_lo, self.scale.proc_hi))
                ops.append(Operation(m, pt, oi))
            total = sum(o.proc_time for o in ops)
            due = t + self.scenario.due_date_factor * total
            self.jobs.append(Job(jid, t, due, ops))
            heapq.heappush(self.event_queue, (t, 0, "arrival", jid))

    def _warmup(self):

        while not all(j.completed for j in self.jobs[:self.scale.n_warmup]):
            if not self._advance_to_next_decision():
                return
            candidates = self._get_candidates()
            if not candidates:
                continue
            jid, oi = min(candidates, key=lambda c: self.jobs[c[0]].arrival)
            self._schedule((jid, oi))

        self._advance_to_next_decision()


    def _get_candidates(self) -> List[Tuple[int, int]]:
        if self.decision_machine_id < 0:
            return []
        return list(self.machines[self.decision_machine_id].queue)

    def _candidate_features(self, jid: int, oi: int, machine_ctx=None) -> CandidateFeatures:
        job = self.jobs[jid]
        op = job.ops[oi]
        m = self.machines[op.machine]
        f = CandidateFeatures()
        f.pt = op.proc_time
        f.wkr = job.remaining_workload()
        f.cr = oi / len(job.ops)
        f.slack = job.due_date - self.time - job.remaining_workload()
        f.owt = self.time - op.release_time
        f.tis = self.time - job.arrival
        f.nor = job.remaining_ops()

        if machine_ctx is None:
            machine_ctx = self._machine_context(m)
        f.wiq, f.niq, f.mrt, f.mwt, f.mbt = machine_ctx






        f.pec = float(m.power) * float(op.proc_time)


        f.t_prog, f.u_global, f.cong, f.n_arrive = self._global_ctx()
        return f

    def _global_ctx(self):
        """4 个**全局/局势**特征（现有 13 个终端全是候选级，没有任何全局量）。

        动机（实测）：库内 oracle 明显优于最佳单条规则 ⇒ 决策需要**依赖情境**；
        而"看情况换规则"必须写成 `ifte(全局量 − c, 规则A, 规则B)` ⇒ 条件里必须能出现全局量。
        现有终端无法表达"早期 EDD / 后期 SPT""松时保交期 / 紧时控流"这类切换。
        """
        t = self.time
        M = max(1, self.scale.n_machines)
        remaining_total = sum(j.remaining_workload() for j in self.jobs if not j.completed)
        denom = t + remaining_total / M
        t_prog = (t / denom) if denom > 1e-9 else 0.0
        busy = sum(m.busy_time for m in self.machines)
        u_global = busy / (M * t) if t > 0 else 0.0
        n_wip = sum(1 for j in self.jobs if (not j.completed) and j.arrival <= t)
        n_arrive = sum(1 for j in self.jobs if t < j.arrival <= t + 50.0) / M
        return (t_prog, u_global, n_wip / M, n_arrive)

    def _machine_context(self, m: Machine):
        """决策点所在机器的 5 个机器级特征（对所有候选相同）。"""
        return (sum(self.jobs[j].ops[k].remaining for (j, k) in m.queue),
                len(m.queue),
                max(0.0, m.busy_until - self.time),
                (self.time - m.idle_since) if not m.busy else 0.0,
                m.busy_time)

    def _decision_features(self):
        """当前决策点全部候选的特征（机器级特征只算一次）。"""
        candidates = self._get_candidates()
        ctx = self._machine_context(self.machines[self.decision_machine_id])
        return candidates, [self._candidate_features(j, o, ctx) for (j, o) in candidates]

    def auto_speed(self, feats, sel: int) -> float:
        """速度规则：按选中工序的松弛与基准加工时间决定速度（固定阈值 2.5p）。"""
        return select_speed(feats[sel].slack, feats[sel].pt)

    def step(self, rule: Callable[[CandidateFeatures], float], speed: float = None):
        candidates, feats = self._decision_features()
        priorities = [rule(f) for f in feats]
        sel = int(np.argmax(priorities))
        selected = candidates[sel]


        if speed is None:
            speed = SPEED_M
        tard_before = self._tard_running
        slack_r = self.reward_calculator.slack_reward(feats, sel)

        self._schedule(selected, speed)
        has_next = self._advance_to_next_decision()
        done = (not has_next) or (self.completed_jobs >= self.scale.n_jobs)
        step_tard = self._tard_running - tard_before

        if self.reward_mode == "tardiness":

            reward = -(step_tard / max(1.0, float(self.scale.n_jobs)))
        else:
            reward = slack_r
        state = self.state() if self.emit_state else None
        return state, reward, done, {"step_tardiness": step_tard,
                                     "slack_reward": slack_r,
                                     "cum_tardiness": self._tard_running}

    def run_rule(self, rule: Callable[[CandidateFeatures], float]):
        self.reset()
        done = False
        while not done:
            _, _, done, _ = self.step(rule)
        tard = self.total_tardiness()
        energy = self._get_total_energy()
        return tard, energy

    def _schedule(self, selected: Tuple[int, int], speed: float = 1.0):
        jid, oi = selected
        op = self.jobs[jid].ops[oi]
        m = self.machines[op.machine]


        counted = self._is_counted(jid)
        if counted and m.count_start is None:
            m.count_start = self.time
            m.busy_snap = m.busy_time
            m.bd_snap = m.breakdown_time
        was_on = m.on
        self._machine_on(m)
        if counted and not was_on:
            self.energy_setup_counted += m.power * SETUP_POWER_FRAC * SETUP_TIME

        m.queue.remove((jid, oi))
        m.busy = True
        m.current = (jid, oi)
        m.current_start = self.time
        m.op_start = self.time
        m.current_speed = speed
        self._start_processing(op.machine, jid, oi, speed)

    def _machine_on(self, m):
        """机器由关到开：α+1，累加一次启动能耗。"""
        if not m.on:
            m.on = True
            m.setup_count += 1
            self.energy_setup += m.power * SETUP_POWER_FRAC * SETUP_TIME

    def _machine_off(self, m):
        """机器由开到关：α+1，累加一次关闭能耗。"""
        if m.on:
            m.on = False
            m.setup_count += 1
            self.energy_setup += m.power * SETUP_POWER_FRAC * SETUP_TIME

    def _start_processing(self, mid: int, jid: int, oi: int, speed: float = 1.0):
        m = self.machines[mid]
        op = self.jobs[jid].ops[oi]
        actual_time = op.proc_time / speed
        if actual_time != op.remaining:
            op.remaining = actual_time
            self.jobs[jid].invalidate_workload()

        fail_rate = self._fail_rate_used() if hasattr(self, "_fail_rate_used") \
            else self.scenario.failure_rate
        p_fail = 1.0 - math.exp(-fail_rate * actual_time)
        rng = self.fault_rngs[mid] if self.fault_rngs is not None else self.rng
        if rng.random() < p_fail:
            fail_at = float(rng.uniform(0, actual_time))
            heapq.heappush(self.event_queue, (self.time + fail_at, 0, "breakdown", mid))
        else:
            heapq.heappush(self.event_queue, (self.time + actual_time, 0, "completion", mid))


    def _advance_to_next_decision(self) -> bool:
        while True:

            for m in self.machines:
                if (not m.busy) and (not m.breakdown) and m.queue:
                    self.decision_machine_id = m.id
                    return True

            if not self.event_queue:
                self.decision_machine_id = -1
                return False
            t, _seq, etype, payload = heapq.heappop(self.event_queue)
            self._advance_time(t)
            self._handle_event(etype, payload)

    def _advance_time(self, t: float):
        self.time = t

    def _handle_event(self, etype: str, payload):
        if etype == "arrival":
            self._push_first_op(payload)
        elif etype == "completion":
            self._complete(payload)
        elif etype == "breakdown":
            self._break(payload)
        elif etype == "repair_done":
            self._repair(payload)

    def _push_first_op(self, jid: int):
        job = self.jobs[jid]
        if job.next_op_idx < len(job.ops):
            op = job.ops[job.next_op_idx]
            op.release_time = self.time
            self.machines[op.machine].queue.append((jid, job.next_op_idx))

    def _complete(self, mid: int):
        m = self.machines[mid]
        dt = self.time - m.current_start
        m.busy_time += dt
        e_proc = m.power * processing_power_factor(m.current_speed) * dt
        self.energy_processing += e_proc
        jid, oi = m.current
        counted = self._is_counted(jid)
        if counted:
            self.energy_processing_counted += e_proc
        job = self.jobs[jid]
        self.schedule_log.append((jid, oi, mid, m.op_start, self.time,
                                  self.jobs[jid].ops[oi].proc_time))
        m.busy = False
        m.current = None
        m.idle_since = self.time
        if not m.queue:
            self._machine_off(m)
            if counted:
                self.energy_setup_counted += m.power * SETUP_POWER_FRAC * SETUP_TIME
        job.next_op_idx += 1
        job.invalidate_workload()
        if job.next_op_idx < len(job.ops):
            nxt = job.ops[job.next_op_idx]
            nxt.release_time = self.time
            self.machines[nxt.machine].queue.append((jid, job.next_op_idx))
        else:
            job.completed = True
            job.completion_time = self.time
            self.completed_jobs += 1
            if jid >= self.scale.n_warmup and self.time > self.warmup_end_time:
                self._tard_running += max(0.0, self.time - job.due_date)

    def _break(self, mid: int):
        m = self.machines[mid]
        dt = self.time - m.current_start
        m.busy_time += dt
        e_proc = m.power * processing_power_factor(m.current_speed) * dt
        self.energy_processing += e_proc
        m.breakdown = True
        m.breakdown_count += 1
        m.breakdown_start = self.time
        jid, oi = m.current
        op = self.jobs[jid].ops[oi]
        op.fail_count += 1
        if self._is_counted(jid):
            self.energy_processing_counted += e_proc
        op.remaining -= dt
        self.jobs[jid].invalidate_workload()
        m.busy = False
        repair = float((self.fault_rngs[mid] if self.fault_rngs is not None
                        else self.rng).uniform(self.repair_lo, self.repair_hi))
        m.breakdown_until = self.time + repair
        heapq.heappush(self.event_queue, (m.breakdown_until, 0, "repair_done", mid))

    def _repair(self, mid: int):
        m = self.machines[mid]
        repair_dur = self.time - m.breakdown_start
        m.breakdown_time += repair_dur
        self.energy_repair += m.power * IDLE_FACTOR * repair_dur
        m.breakdown = False
        m.breakdown_until = 0.0
        jid, oi = m.current
        m.busy = True
        m.current_start = self.time
        op = self.jobs[jid].ops[oi]
        heapq.heappush(self.event_queue, (self.time + op.remaining, 0, "completion", mid))


    @property
    def repair_lo(self) -> float:
        return self.scale.repair_lo if self._repair_lo is None else self._repair_lo

    @property
    def repair_hi(self) -> float:
        return self.scale.repair_hi if self._repair_hi is None else self._repair_hi

    @property
    def mtbf(self) -> float:
        """平均故障间隔时间 = 1/λ（λ 为有效故障率）。"""
        rate = self._fail_rate_used() if hasattr(self, "_fail_rate_used") \
            else self.scenario.failure_rate
        return float("inf") if rate <= 0 else 1.0 / rate

    def state(self) -> np.ndarray:
        return self.state_extractor.extract(self)

    def energy_breakdown(self):
        """返回 (加工能耗 PEc, 空闲能耗 IEc, 维修能耗)。含预热期（全剧口径）。"""
        energy_idle = sum(m.power * IDLE_FACTOR * max(0.0, self.time - m.busy_time - m.breakdown_time)
                          for m in self.machines)
        return self.energy_processing, energy_idle, self.energy_repair

    def energy_full(self):
        """按文献口径返回 (PEc 加工, IEc 空闲, REc 预备, 总能耗)。

        总能耗 = PEc + IEc + REc。
        REc = Σ_m P_m·SETUP_POWER_FRAC·SETUP_TIME·α_m（α=启停次数），
        在每次"机器空→有活"时累加（详见 _schedule）。
        """
        pec, iec, rep = self.energy_breakdown()
        return pec, iec, self.energy_setup, pec + iec + rep + self.energy_setup

    def energy_breakdown_window(self):
        """返回**测量窗口**（预热结束之后）的 (加工, 空闲, 维修) 能耗。"""
        ep, ei, er = self.energy_breakdown()
        base = getattr(self, "_energy_base", None)
        if base is None:
            return ep, ei, er
        return ep - base[0], ei - base[1], er - base[2]

    def window_energy_full(self):
        """测量窗口的 (PEc, IEc, REc, 总)。注意 REc 为单调累加量，基线在 reset 时记录。"""
        pec, iec, rep = self.energy_breakdown_window()
        base_setup = getattr(self, "_energy_setup_base", 0.0)
        rec = self.energy_setup - base_setup
        return pec, iec, rec, pec + iec + rep + rec

    def window_energy(self) -> float:
        """测量窗口总能耗（不含预热期）。节能率应以本口径计算。"""
        return self.window_energy_full()[3]

    def window_energy_rate(self) -> float:
        """测量窗口内的平均功率 (kW)：稳态量，不含预热常量、不随时间单调增长。"""
        span = self.time - self.warmup_end_time
        if span <= 1e-9:
            return 0.0
        return self.window_energy() / span

    def total_energy(self) -> float:
        return sum(self.energy_breakdown()) + self.energy_setup

    def _get_total_energy(self) -> float:
        return self.total_energy()

    def total_tardiness(self) -> float:


        tard = 0.0
        for j in self.jobs:
            if j.id >= self.scale.n_warmup and j.completion_time > self.warmup_end_time:
                tard += max(0.0, j.completion_time - j.due_date)
        return tard



    def _is_counted(self, jid: int) -> bool:
        return self.jobs[jid].arrival >= self.warmup_end_time

    def tardiness_arrival(self) -> float:
        """到件口径拖期：只计"到达时刻 >= 预热结束时刻"的工件的拖期。"""
        return sum(max(0.0, j.completion_time - j.due_date) for j in self.jobs
                   if j.completion_time > 0.0 and j.arrival >= self.warmup_end_time)

    def arrival_energy_full(self):
        """到件口径五分量 (PEc, IEc, REp, REc, 总)。

        PEc / REc：按**触发该笔能耗的工序所属工件**精确归属（工件级，无近似）。
        IEc / REp：机器级量，按"该机器第一次加工计数工件"起的时段归属（起算点快照，
                  故预热期内加工的那部分不计入）。五分量之和 = 计数工件的总能耗。
        """
        iec = rep = 0.0
        for m in self.machines:
            if m.count_start is None:
                continue
            span = self.time - m.count_start
            busy = m.busy_time - m.busy_snap
            bd = m.breakdown_time - m.bd_snap
            rep += m.power * IDLE_FACTOR * bd
            iec += m.power * IDLE_FACTOR * max(0.0, span - busy - bd)
        pec = self.energy_processing_counted
        rec = self.energy_setup_counted
        return pec, iec, rep, rec, pec + iec + rep + rec

    def arrival_energy(self) -> float:
        return self.arrival_energy_full()[4]

    def n_arrivals_counted(self) -> int:
        return sum(1 for j in self.jobs if j.arrival >= self.warmup_end_time)

    def max_fails_per_op(self) -> int:
        """校验设计约束"每道工序最多遇到一次故障"（应恒为 ≤1）。"""
        return max((o.fail_count for j in self.jobs for o in j.ops), default=0)
