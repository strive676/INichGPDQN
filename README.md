# INichGPDQN
Using INichGPDQN to solve the Dynamic Flexible Job Shop Scheduling Problem.
# NichGPDRL 动态柔性作业车间调度项目

本项目研究动态柔性作业车间中的规则生成与在线决策：遗传规划（GP）离线演化排序和机器选择规则，Double DQN 根据系统状态选择规则，环境模块模拟作业到达、机器故障、加工与能耗。

## 核心模块

- `gp_module/`：GP 个体、遗传操作、规则选择、决策点采样与奖励成形。
- `dqn_module/`：Double DQN、Q 网络、经验回放及动作优势组件。
- `env_module/`：动态作业车间环境、状态特征、奖励计算与排程后处理。
- `config/`：算法参数、奖励参数、常量与场景配置。
