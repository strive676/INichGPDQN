"""Flexible job-shop environment with routing and sequencing decisions."""
from typing import Callable, List, Optional, Tuple

import numpy as np

from config import (P0, POWER_EXPONENT, SPEED_M, SETUP_TIME, SETUP_POWER_FRAC, P_PROCESS,
                    ENERGY_MODEL, ETA_LO, ETA_HI, processing_power_factor)
from config.scenarios import Scenario, Scale
from env_module.djss_env import DJSSEnv, Job, Machine, Operation, CandidateFeatures

ROUTE_FEAT_DIM = 16
MAX_CAND = 10
ARRIVE_WINDOW = 50.0


class FJSPEnv(DJSSEnv):
    """FJSP：每工序有 1~M 台候选机器。"""

    def __init__(self, scenario: Scenario, scale: Scale, seed: Optional[int] = None,
                 emit_state: bool = True, reward_mode: str = None,
                 state_version: str = None,
                 f_lo: float = 0.80, f_hi: float = 1.25,
                 p_lo: float = 5.0, p_hi: float = 25.0,
                 cand_lo: int = 1, cand_hi: Optional[int] = None,
                 fail_rate: Optional[float] = None, crn_faults: bool = False,
                 repair_lo: Optional[float] = None, repair_hi: Optional[float] = None,
                 alpha_lo: Optional[float] = None, alpha_hi: Optional[float] = None):
        self.f_lo, self.f_hi = f_lo, f_hi
        self.p_lo, self.p_hi = p_lo, p_hi
        self.cand_lo = int(cand_lo)
        self.cand_hi = int(cand_hi) if cand_hi is not None else scale.n_machines



        self.alpha_lo, self.alpha_hi = alpha_lo, alpha_hi

        self._fail_rate = fail_rate
        super().__init__(scenario, scale, seed=seed, emit_state=emit_state,
                         reward_mode=reward_mode, state_version=state_version,
                         crn_faults=crn_faults, repair_lo=repair_lo, repair_hi=repair_hi)


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
        self.machines: List[Machine] = []
        self.event_queue = []
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
        self.route_op: Optional[Operation] = None
        self.pending: Optional[str] = None


        self.route_norm = None
        self._generate_machines()
        self._generate_jobs_fjsp()
        self._warmup()
        self.warmup_end_time = self.time
        self._energy_base = self.energy_breakdown()
        self._energy_setup_base = self.energy_setup
        self._advance()
        return self.state()

    def _fail_rate_used(self):
        return self.scenario.failure_rate if self._fail_rate is None else self._fail_rate

    def _generate_machines(self):
        f = self.rng.uniform(self.f_lo, self.f_hi, self.scale.n_machines)
        self.machine_time_factor = f




        if ENERGY_MODEL == "sec":
            base = 0 if self._seed is None else int(self._seed)
            eta_rng = np.random.default_rng(base * 7717 + 4211)
            self.machine_efficiency = eta_rng.uniform(ETA_LO, ETA_HI, self.scale.n_machines)
        else:
            self.machine_efficiency = np.ones(self.scale.n_machines)
        self.machine_power = self.machine_efficiency * P0 / f ** 3
        self.machines = [Machine(m, float(self.machine_power[m]))
                         for m in range(self.scale.n_machines)]

    def _generate_jobs_fjsp(self):
        M = self.scale.n_machines
        mean_f = float(np.mean(self.machine_time_factor))
        mean_p = (self.p_lo + self.p_hi) / 2.0
        mean_ops = (self.scale.min_ops + self.scale.max_ops) / 2.0
        lam = self.scenario.utilization * M / (mean_p * mean_f * mean_ops)




        import os as _os
        _amode = _os.environ.get("DSH_ARRIVAL_MODE", "poisson").strip().lower()
        t = 0.0
        for jid in range(self.scale.n_jobs):
            if _amode != "all0":
                t += self.rng.exponential(1.0 / lam)
            n_ops = int(self.rng.integers(self.scale.min_ops, self.scale.max_ops + 1))
            ops = []
            for oi in range(n_ops):
                p_bar = float(self.rng.uniform(self.p_lo, self.p_hi))
                nc = int(self.rng.integers(self.cand_lo, self.cand_hi + 1))
                cands = sorted(int(x) for x in self.rng.choice(M, size=nc, replace=False))










                procs = {m: float(self.rng.uniform(self.p_lo, self.p_hi)) for m in cands}
                est = float(np.mean(list(procs.values())))

                ops.append(Operation(machine=-1, proc_time=est, index=oi,
                                     candidates=cands, procs=procs, jid=jid))
            total = sum(o.proc_time for o in ops)

            if self.alpha_lo is not None:
                alpha = float(self.rng.uniform(self.alpha_lo, self.alpha_hi))
            else:
                alpha = self.scenario.due_date_factor
            due = t + alpha * total
            self.jobs.append(Job(jid, t, due, ops))
            import heapq
            heapq.heappush(self.event_queue, (t, 0, "arrival", jid))


    def _schedule(self, selected, speed: float = SPEED_M):
        super()._schedule(selected, speed)
        jid, oi = selected
        op = self.jobs[jid].ops[oi]
        self.machines[op.machine].busy_until = self.time + op.proc_time / speed

    def _break(self, mid: int):
        super()._break(mid)
        m = self.machines[mid]
        jid, oi = m.current
        m.busy_until = m.breakdown_until + self.jobs[jid].ops[oi].remaining

    def _repair(self, mid: int):
        super()._repair(mid)
        m = self.machines[mid]
        jid, oi = m.current
        m.busy_until = self.time + self.jobs[jid].ops[oi].remaining

    def machine_avail(self, m: int, include_queue: bool = True) -> float:
        """机器能真正开工的时刻（可选是否计入队列积压）。

        修复记录：`_break` 会把 `busy` 置 False，旧实现只看 `busy_until`
        ⇒ **维修中的机器被报成"现在可用"**（实测把可用时刻低估最多 43.6 时间单位，
        1803 次路由决策里 947 次选中了当时正处于维修的候选机，其中 666 次另有非维修候选）。
        现在维修期由 `breakdown_until` 计入。论文协议 failure_rate=0 ⇒ 该修复对论文结果无影响；
        有故障场景（run_fjsp_final 的 failure_rate=0.05、能耗实验的 λ=0.005）结果会变。
        """
        mc = self.machines[m]
        base = self.time
        if mc.breakdown:
            base = max(base, mc.breakdown_until)
        elif mc.busy:
            base = max(base, mc.busy_until)
        if include_queue:
            base += sum(self.jobs[j].ops[o].proc_time for (j, o) in mc.queue)
        return base

    def machine_avail_mean(self, op) -> float:
        """当前工序各候选机器可用时刻的均值（Liu RA 奖励的基准项）。"""
        return float(np.mean([self.machine_avail(m) for m in op.candidates]))


    def _warmup(self):
        while not all(j.completed for j in self.jobs[:self.scale.n_warmup]):
            if not self._advance():
                return
            if self.pending == "route":
                self.step_route(self._ea_choice())
            elif self.pending == "sequence":
                self.step_sequence_index(0)          # FIFO
        self._advance()

    def _ea_choice(self):
        """EA = **Earliest Available**（Xu 2026 手工路由规则表第 3 条）：
        把工序分配给"可用时刻最早（最早空出来）"的机器，**不计入本工序的加工时间**。

        ⚠ 修正记录：此前实现为 `machine_avail(m) + op.procs[m]`，与 ECT 逐字相同
        ⇒ 旧的 "EA" 实际是 ECT（实测两者拖期/能耗完全一致：2400/49350），
        而 Xu 论文要求"排序智能体训练时路由固定为 EA"，故此处必须区分。
        `machine_avail(m)` 默认已含队列积压（= 真正能开工的时刻）。
        """
        op = self.route_op
        return int(np.argmin([self.machine_avail(m) for m in op.candidates]))


    def _handle_event(self, etype, payload):
        if etype == "arrival":
            jid = payload
            op = self.jobs[jid].ops[0]
            op.release_time = self.time
        elif etype == "completion":
            self._complete_fjsp(payload)
        elif etype == "breakdown":
            self._break(payload)
        elif etype == "repair_done":
            self._repair(payload)

    def _complete_fjsp(self, mid: int):
        from config import processing_power  # noqa: F401
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

        else:
            job.completed = True
            job.completion_time = self.time
            self.completed_jobs += 1
            if jid >= self.scale.n_warmup and self.time > self.warmup_end_time:
                self._tard_running += max(0.0, self.time - job.due_date)

    def _advance(self) -> bool:
        import heapq
        while True:

            op = self._find_unrouted()
            if op is not None:
                self.route_op = op
                self.pending = "route"



                self.decision_machine_id = -1
                return True

            for m in self.machines:
                if (not m.busy) and (not m.breakdown) and m.queue:
                    self.decision_machine_id = m.id
                    self.pending = "sequence"
                    return True

            if not self.event_queue:
                self.pending = None
                self.route_op = None
                return False
            t, _seq, etype, payload = heapq.heappop(self.event_queue)
            self.time = max(self.time, t)
            self._handle_event(etype, payload)

    def _find_unrouted(self) -> Optional[Operation]:
        for jid, j in enumerate(self.jobs):
            if j.completed:
                continue
            if j.next_op_idx >= len(j.ops):
                continue
            op = j.ops[j.next_op_idx]
            if op.machine >= 0:
                continue
            if op.index == 0:
                if j.arrival > self.time:
                    continue
                return op
            if op.release_time > 0 and op.release_time <= self.time and j.ops[op.index - 1].machine >= 0:
                return op
        return None


    def step_route(self, k: int):
        """路由决策：把当前工序排入第 k 个候选机器的队列。"""
        op = self.route_op
        m = int(op.candidates[k])
        mc = self.machines[m]
        op.machine = m
        op.proc_time = op.procs[m]
        op.remaining = op.procs[m]
        self.jobs[op.jid].invalidate_workload()
        mc.queue.append((op.jid, op.index))
        self.route_op = None
        self.pending = None
        self._advance()

    def step_sequence_index(self, i: int):
        """排序决策：从当前机器队列中取第 i 个工序开工。"""
        m = self.machines[self.decision_machine_id]
        jid, oi = m.queue[i]
        self._schedule((jid, oi), speed=SPEED_M)
        self.pending = None
        self._advance()

    def step_sequence(self, rule: Callable, speed: float = SPEED_M):
        """排序决策：用规则（候选特征 → 优先级）选工序。"""
        cands = self._get_candidates()
        ctx = self._machine_context(self.machines[self.decision_machine_id])
        feats = [self._candidate_features(j, o, ctx) for (j, o) in cands]
        k = int(np.argmax([rule(f) for f in feats]))
        jid, oi = cands[k]
        self._schedule((jid, oi), speed=speed)
        self.pending = None
        self._advance()


    def route_candidate_feats(self):
        """把**路由决策点**表达为"每台候选机器一个候选"的特征列表。

        用途（修复 2）：让 PC 行为描述子同时覆盖 routing 与 sequencing 决策点。
        对当前 route_op 的每台候选机器 m 构造与排序决策同构的 CandidateFeatures：
        该规则在这组候选上的排名 = 它在**路由维度上的行为倾向**
        （此前 PC 只含 sequencing 点，路由行为完全不可见）。

        机器级特征按各自机器算；工件级特征对该工序的所有候选相同。
        """
        op = self.route_op
        if op is None:
            return []
        job = self.jobs[op.jid]
        wkr = sum(job.ops[k].proc_time for k in range(op.index, len(job.ops)))
        out = []
        for m in op.candidates:
            mc = self.machines[m]
            f = CandidateFeatures()
            f.pt = float(op.procs[m])
            f.wkr = float(wkr)
            f.cr = op.index / max(1, len(job.ops))
            f.slack = job.due_date - self.time - wkr
            f.owt = self.time - op.release_time
            f.tis = self.time - job.arrival
            f.nor = job.remaining_ops()
            f.wiq = sum(self.jobs[j].ops[k].remaining for (j, k) in mc.queue)
            f.niq = len(mc.queue)
            f.mrt = max(0.0, self.machine_avail(m) - self.time)
            f.mwt = (self.time - mc.idle_since) if not mc.busy else 0.0
            f.mbt = mc.busy_time





            f.pec = float(mc.power) * float(op.procs[m])
            out.append(f)
        return out

    def route_state(self) -> Tuple[np.ndarray, np.ndarray]:
        op = self.route_op
        feats = np.zeros((MAX_CAND, ROUTE_FEAT_DIM), dtype=np.float32)
        mask = np.zeros(MAX_CAND, dtype=np.float32)
        if op is None:
            return feats, mask
        M = self.scale.n_machines
        n_wip = sum(1 for j in self.jobs if not j.completed and j.arrival <= self.time)
        busy_sum = sum(m.busy_time for m in self.machines)
        u_global = busy_sum / (M * self.time) if self.time > 0 else 0.0
        bf = sum(1 for m in self.machines if m.breakdown) / M
        job = self.jobs[op.jid]
        wkr = sum(job.ops[k].proc_time for k in range(op.index, len(job.ops)))
        slack = job.due_date - self.time - wkr
        p_base = float(np.mean(list(op.procs.values())))
        ttd = job.due_date - self.time
        n_arriving = sum(1 for j in self.jobs
                         if self.time < j.arrival <= self.time + ARRIVE_WINDOW) / M
        for k, m in enumerate(op.candidates[:MAX_CAND]):
            mc = self.machines[m]
            q_work = sum(self.jobs[j].ops[o].proc_time for (j, o) in mc.queue)
            feats[k, 0] = q_work                                       # WIQ
            feats[k, 1] = len(mc.queue)                                # NIQ
            feats[k, 2] = max(0.0, self.machine_avail(m) - self.time)  # MRT/AM
            feats[k, 3] = (self.time - mc.idle_since) if not mc.busy else 0.0   # MWT
            feats[k, 4] = mc.busy_time                                 # MBT
            feats[k, 5] = op.procs[m]                                  # PT_on_machine
            feats[k, 6] = mc.power                                     # POWER_m
            feats[k, 7] = mc.power * op.procs[m]                       # energy_on_machine
            feats[k, 8] = u_global
            feats[k, 9] = n_wip / M                                    # CONG
            feats[k, 10] = bf
            feats[k, 11] = p_base
            feats[k, 12] = wkr
            feats[k, 13] = slack
            feats[k, 14] = ttd
            feats[k, 15] = n_arriving
            mask[k] = 1.0







        if self.route_norm is not None:
            feats = self.route_norm.transform(feats)
        return feats, mask


    @property
    def state_dim(self) -> int:
        return self.state_extractor.state_dim

    def route_dim(self) -> int:
        return ROUTE_FEAT_DIM



class RouteNormalizer:
    """路由特征的**按维标准化**（只用合法候选行 `mask>0` 拟合）。

    动机（实测，审计发现）：RA 原先直接吃 `route_state()` 原始值，量纲跨 3~4 个数量级
    （MBT 126~914、ENERGY 35~381、UGLOBAL 0.65~0.75 近常数、BF≡0）
    ⇒ 路由 Q 网络复现了 `normalization.py` 文档里记载的"尺度失控"。
    SA 侧本就有 `Normalizer`，RA 侧完全没有。

    用法：由 Stage-2 的 RA 训练在开始前用若干条 rollout 的 route_state 拟合，
    然后挂到 `env.route_norm` ⇒ `route_state()` 自动输出标准化后的特征。
    `mean is None`（未拟合）时 `transform` 原样返回 ⇒ 行为与修复前相同。
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, feats_list, mask_list):
        rows = [np.asarray(f, dtype=np.float64)[np.asarray(m) > 0]
                for f, m in zip(feats_list, mask_list) if np.any(np.asarray(m) > 0)]
        if not rows:
            return self
        X = np.concatenate(rows, axis=0)
        self.mean = X.mean(axis=0)
        s = X.std(axis=0)
        s[s < 1e-9] = 1.0
        self.std = s
        return self

    def transform(self, feats):
        if self.mean is None:
            return feats
        return ((np.asarray(feats, dtype=np.float64) - self.mean) / self.std).astype(np.float32)


def make_route_policy(name: str):
    def fn(env: FJSPEnv) -> int:
        op = env.route_op
        if len(op.candidates) == 1:
            return 0
        rng = env.rng
        if name == "random":
            return int(rng.integers(0, len(op.candidates)))
        if name == "EA":
            return env._ea_choice()
        if name == "min-load":
            return int(np.argmin([sum(env.jobs[j].ops[o].proc_time
                                      for (j, o) in env.machines[m].queue)
                                  for m in op.candidates]))
        if name == "min-energy":
            return int(np.argmin([env.machines[m].power * op.procs[m] for m in op.candidates]))
        if name == "MET":
            return int(np.argmin([op.procs[m] for m in op.candidates]))
        if name == "ECT":
            best, best_v = 0, None
            for k, m in enumerate(op.candidates):
                mc = env.machines[m]
                q = sum(env.jobs[j].ops[o].proc_time for (j, o) in mc.queue)
                v = env.machine_avail(m) + op.procs[m]
                if best_v is None or v < best_v:
                    best_v, best = v, k
            return best
        raise ValueError(name)
    return fn
