"""Reward calculations for scheduling performance and energy objectives."""
from config import RewardParams


class RewardCalculator:
    def __init__(self, params: RewardParams = None):
        self.p = params or RewardParams()

    @staticmethod
    def _critical_factor(s: float, beta: float) -> float:
        return 1.0 - s / (abs(s) + beta)

    def slack_reward(self, feats, sel: int) -> float:
        """r_t = F_s·avg(p_others) − F̄·p_s（基准加工时间，与速度无关）。

        F_s = 1 − S_s/(|S_s|+β)；F̄ = Σ_{i≠s} F_i/(|Q|−1)；|Q|=1 时 r_t = 0。
        """
        n = len(feats)
        F_s = self._critical_factor(feats[sel].slack, self.p.beta)
        p_s = feats[sel].pt
        others = [feats[i] for i in range(n) if i != sel]
        if others:
            avg_p_others = sum(f.pt for f in others) / len(others)
            F_bar = sum(self._critical_factor(f.slack, self.p.beta) for f in others) / len(others)
        else:
            avg_p_others = 0.0
            F_bar = 0.0
        return F_s * avg_p_others - F_bar * p_s

    def energy_reward(self, feats, sel: int) -> float:
        """能耗附加项：与 `slack_reward` **结构完全同构**，只把基准加工时间 `pt` 换成加工能耗 `pec`。

            F_s = 1 − S_s/(|S_s|+β)                  （与 slack_reward 同一个临界因子、同一个 β）
            F̄  = Σ_{i≠s} F_i/(|Q|−1)                 （|Q|=1 时两项均为 0）
            energy_reward = F_s·avg(pec_others) − F̄·pec_s

        含义与松弛奖励完全平行：**既偏好紧迫的工序，又偏好加工能耗低的工序**。

        ⚠ 依赖 `feats[i].pec` 是**真加工能耗**（机器功率 × 加工时长）。
        它原先在排序侧被写成全局常数 `P_PROCESS·pt` ⇒ 本项会退化成 `P0 · slack_reward`
        （加进去等于只是把松弛奖励乘了个常数，不含任何新信息）。
        已随本轮改动修好（`djss_env._candidate_features`），并有回归测试钉住：
        `tests/test_dual_action.py::test_energy_reward_is_not_proportional_to_slack`。

        ⚠ 本方法**只加在 SA 的 sequence 点奖励上**；route 点的能耗走既有的
        `liu_reward(lambda_e=...)` / `energy_shaping` 通道（规格 §4）。基础奖励
        **一律是松弛/Liu 奖励**，能耗只作附加项 —— 这是用户 2026-09-27 明确要求的。
        """
        n = len(feats)
        F_s = self._critical_factor(feats[sel].slack, self.p.beta)
        e_s = feats[sel].pec
        others = [feats[i] for i in range(n) if i != sel]
        if others:
            avg_e_others = sum(f.pec for f in others) / len(others)
            F_bar = sum(self._critical_factor(f.slack, self.p.beta) for f in others) / len(others)
        else:
            avg_e_others = 0.0
            F_bar = 0.0
        return F_s * avg_e_others - F_bar * e_s
