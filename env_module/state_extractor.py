"""State feature construction for routing and sequencing decisions."""
import os

import numpy as np

STATE_DIM_V1 = 19
STATE_DIM_V2 = 22
STATE_DIM_V3 = 25

MRT_MODE = os.environ.get("DSH_MRT_MODE", "remaining").strip().lower()


def state_route_on() -> bool:
    """P1 改动 1 开关 `DSH_STATE_ROUTE`（**运行时读取**，默认 0 = 逐位等于改动前）。

    ⚠ 为什么用函数而不是模块常量：本仓库有「import 期常量 ⇒ 环境变量在 import 之后再设
      完全无效」的历史坑（见 `run_fjsp_final.k_out_now`）。运行时读取使测试/脚本可以在
      同一进程内切换，且 `run_repeat_matrix` 的子进程写法也照常生效。
    """
    return os.environ.get("DSH_STATE_ROUTE", "0").strip() == "1"


class StateExtractor:
    def __init__(self, n_machines: int, version: str = "v2"):
        self.n_machines = n_machines
        self.version = version
        self.state_dim = {"v1": STATE_DIM_V1, "v3": STATE_DIM_V3}.get(version, STATE_DIM_V2)

    def extract(self, env) -> np.ndarray:
        if self.version == "v1":
            return self._extract_v1(env)
        v = self._extract_v2(env)
        return np.concatenate([v, self._extra_v3(env)]) if self.version == "v3" else v

    def _extra_v3(self, env) -> np.ndarray:
        """v3 新增的 3 个特征 —— 都是 v2 **完全没有**反映的"局势/紧迫度"信息。

        动机：DQN 的瓶颈实测在**动作区分能力**（动作-奖励混叠、SNR 0.063）。v2 的 22 维里
        没有任何一维刻画"当前整体做得有多糟"或"候选之间是否真有差别"，策略因此难以判断
        该激进还是该保守。故补：
          1) 累计拖期（到件口径，按工件数×平均加工时间归一）—— 全局局势
          2) 候选之间 slack 的标准差 ——"是否存在明显更优的候选"（决定该不该信任选择）
          3) 决策机器队列中"已过期工件"的占比 —— 局部紧迫度（v2 只有队列长度/工作量）
        """
        scale = env.scale
        mu_p = (getattr(env, "p_lo", scale.proc_lo) + getattr(env, "p_hi", scale.proc_hi)) / 2.0
        cum_tard = env.tardiness_arrival() / max(1.0, scale.n_jobs * max(1e-9, mu_p))
        mid = getattr(env, "decision_machine_id", -1)
        cands = env._get_candidates() if mid is not None and mid >= 0 else []
        if cands:
            ctx = env._machine_context(env.machines[mid])
            fs = [env._candidate_features(j, o, ctx) for (j, o) in cands]
            sl = np.array([f.slack for f in fs], dtype=float)
            slack_std = float(sl.std())
            q_urgent = float(np.mean([env.jobs[j].due_date < env.time for (j, _o) in cands]))
        else:
            slack_std = 0.0
            q_urgent = 0.0
        return np.array([cum_tard, slack_std, q_urgent], dtype=float)


    def _extract_v1(self, env) -> np.ndarray:
        t = env.time
        active = [j for j in env.jobs if (not j.completed) and j.arrival <= t]
        if active:
            avg_pt = np.mean([j.ops[j.next_op_idx].proc_time for j in active
                              if j.next_op_idx < len(j.ops)])
            avg_wkr = np.mean([j.remaining_workload() for j in active])
            avg_cr = np.mean([j.next_op_idx / len(j.ops) for j in active])
            avg_slack = np.mean([j.due_date - t - j.remaining_workload() for j in active])
            avg_owt = np.mean([t - j.ops[j.next_op_idx].release_time for j in active
                               if j.next_op_idx < len(j.ops)])
            avg_tis = np.mean([t - j.arrival for j in active])
            avg_nor = np.mean([j.remaining_ops() for j in active])
        else:
            avg_pt = avg_wkr = avg_cr = avg_slack = avg_owt = avg_tis = avg_nor = 0.0
        ms = env.machines
        avg_wiq = np.mean([sum(env.jobs[j].ops[k].remaining for (j, k) in m.queue) for m in ms])
        avg_niq = np.mean([len(m.queue) for m in ms])
        avg_mrt = np.mean([max(0.0, m.busy_until - t) for m in ms])
        avg_mwt = np.mean([(t - m.idle_since) if not m.busy else 0.0 for m in ms])
        avg_mbt = np.mean([m.busy_time for m in ms])
        util = sum(m.busy_time for m in ms) / (self.n_machines * t) if t > 0 else 0.0
        avg_util = np.mean([m.busy_time / t if t > 0 else 0.0 for m in ms])
        loads = [max(0.0, m.busy_until - t) + sum(env.jobs[j].ops[k].remaining for (j, k) in m.queue)
                 for m in ms]
        load_balance = float(np.mean(loads))
        load_std = float(np.std(loads))
        congestion = sum(len(m.queue) for m in ms) / self.n_machines
        breakdown_factor = sum(1 for m in ms if m.breakdown) / self.n_machines
        energy = env.window_energy_rate()
        return np.array([avg_pt, avg_wkr, avg_cr, avg_slack, avg_owt, avg_tis, avg_nor,
                         avg_wiq, avg_niq, avg_mrt, avg_mwt, avg_mbt,
                         util, avg_util, load_balance, load_std, congestion,
                         breakdown_factor, energy], dtype=float)


    def _extract_v2(self, env) -> np.ndarray:
        t = env.time
        ms = env.machines
        M = self.n_machines

        busy_sum = sum(m.busy_time for m in ms)
        u_global = busy_sum / (M * t) if t > 0 else 0.0
        u_m = [(m.busy_time / t if t > 0 else 0.0) for m in ms]
        u_range = (max(u_m) - min(u_m)) if u_m else 0.0
        bf = sum(1 for m in ms if m.breakdown) / M
        n_wip = sum(1 for j in env.jobs if (not j.completed) and j.arrival <= t)
        cong = n_wip / M
        remaining_total = sum(j.remaining_workload() for j in env.jobs if not j.completed)
        denom = t + remaining_total / M
        t_progress = (t / denom) if denom > 1e-9 else 0.0

        mid = env.decision_machine_id





        if (state_route_on() and mid is not None and mid < 0
                and getattr(env, "pending", None) == "route"
                and getattr(env, "route_op", None) is not None):
            m4, cs = self._route_block(env, t)
            return np.array([u_global, u_range, bf, cong, t_progress] + m4 + cs, dtype=float)

        if mid is not None and mid >= 0:
            m = ms[mid]
            niq = float(len(m.queue))
            wiq = float(sum(env.jobs[j].ops[k].remaining for (j, k) in m.queue))





            if MRT_MODE == "idle":
                mrt = float(max(0.0, env.time - m.idle_since))
            elif m.busy and m.current is not None:
                cj, co = m.current
                mrt = float(env.jobs[cj].ops[co].remaining)
            else:
                mrt = 0.0
            mbt = float(m.busy_time)
        else:
            niq = wiq = mrt = mbt = 0.0

        cands = env._get_candidates() if (mid is not None and mid >= 0) else []
        if cands:
            ctx = env._machine_context(ms[mid])
            fs = [env._candidate_features(j, o, ctx) for (j, o) in cands]
            pt = np.array([f.pt for f in fs], dtype=float)
            wkr = np.array([f.wkr for f in fs], dtype=float)
            sl = np.array([f.slack for f in fs], dtype=float)
            ttd = np.array([env.jobs[j].due_date - t for (j, o) in cands], dtype=float)
            owt = np.array([f.owt for f in fs], dtype=float)
            cs = [pt.min(), pt.max(), pt.mean(),
                  wkr.min(), wkr.max(),
                  sl.min(), sl.max(), sl.mean(), float((sl < 0).sum()),
                  ttd.min(), ttd.mean(),
                  owt.min(), owt.mean()]
        else:
            cs = [0.0] * 13

        return np.array([u_global, u_range, bf, cong, t_progress,
                         niq, wiq, mrt, mbt] + cs, dtype=float)




    # -----+------------------------------------+-----------------------------------------------





    # -----+------------------------------------+-----------------------------------------------




    # -----+------------------------------------+-----------------------------------------------













    # -----+------------------------------------+-----------------------------------------------








    # ==========================================================================================
    def _route_block(self, env, t: float):
        """路由点的 4 个机器统计槽 + 13 个候选槽（字段全部来自 `route_candidate_feats()`）。

        **动机**：路由点约占全部决策点的一半，而改动前这 17 维全 0 ⇒ 策略在路由点上
        只能看到 5 个全局量 + 1 个标志位（实测该点 AUC≈0.5）。这里填入的正是
        "选哪台机器"所需的量：候选机器的加工时间/可用时刻/队列/能耗分布，
        以及当前工序自身的紧迫度（slack / TTD）。
        **取值口径**：`pt = op.procs[m]`（该工序在第 m 台的加工时间）、
        `mrt = max(0, machine_avail(m) − t)`（含队列积压的真正可开工时刻 − 现在）、
        `niq/wiq = 该机器队列的长度/剩余工作量`、`pec = power(m)·proc(m)`（真加工能耗）。
        **不新增任何仿真**：全部是 `FJSPEnv.route_candidate_feats()` 已有的字段。
        """
        feats = env.route_candidate_feats()
        if not feats:


            return [0.0] * 4, [0.0] * 13
        op = env.route_op
        job = env.jobs[op.jid]
        wkr = sum(job.ops[k].proc_time for k in range(op.index, len(job.ops)))
        pt = np.array([f.pt for f in feats], dtype=float)
        wiq = np.array([f.wiq for f in feats], dtype=float)
        niq = np.array([f.niq for f in feats], dtype=float)
        mrt = np.array([f.mrt for f in feats], dtype=float)
        mbt = np.array([f.mbt for f in feats], dtype=float)
        pec = np.array([f.pec for f in feats], dtype=float)

        m4 = [float(niq.mean()), float(wiq.mean()), float(mrt.mean()), float(mbt.mean())]

        cs = [float(pt.min()), float(pt.max()), float(pt.mean()), float(pt.max() - pt.min()),
              float(mrt.min()), float(mrt.max()),
              float(niq.max()), float(wiq.max()),
              float(pec.min()), float(pec.max()), float(pec.mean()),
              float(job.due_date - t - wkr),
              float(job.due_date - t)]
        return m4, cs
