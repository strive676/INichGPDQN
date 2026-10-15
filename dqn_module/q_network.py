"""Neural Q-network architecture and hidden-layer specification utilities."""
import torch.nn as nn


def parse_hidden_spec(spec):
    """把「逗号分隔的每层宽度」字符串解析成整数列表（供 `DSH_SA_NET` 使用）。

    规格示例：`"3"` ⇒ `[3]`（**1 层** 3 节点）；`"48,36,36,24,24,12"` ⇒ 6 层。
    返回 `None` 表示「未指定」（None / 空串 / 全空白）—— 调用方应走 `hidden_dim` 的
    **历史两层**路径，从而与改动前逐位一致。

    ⚠ 非法值**直接抛异常**，不静默回退：本仓库已多次因「开关写错但静默失效」而拿到
    与预期不符的结果（如 `DSH_DUALTREE` 写错名字、`DSH_K_OUT` 在 import 后被忽略）。
    """
    if spec is None:
        return None
    s = str(spec).strip()
    if not s:
        return None
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"隐藏层规格里有空项：{spec!r}")
        try:
            w = int(part)
        except ValueError:
            raise ValueError(f"隐藏层规格必须是逗号分隔的整数：{spec!r}（坏项 {part!r}）")
        if w <= 0:
            raise ValueError(f"隐藏层每层宽度必须为正整数：{spec!r}（坏项 {part!r}）")
        out.append(w)
    return out


class QNetwork(nn.Module):
    """Q 网络（全连接 + ReLU）。

    **默认路径与改动前逐位一致**：不传 `hidden_dims` 时 = 2 个隐藏层、宽度都是 `hidden_dim`。
    `hidden_dims`（新增）非空时按它构造 ⇒ 支持**任意层数与每层宽度**
    （论文 Table III 的 `48×36×36×24×24×12`、`16×16×16×8×8` 都能表达）。

    逐位一致的依据：两种路径下 `nn.Linear` 的**创建顺序完全相同**
    （旧实现 `fc1(state→h) → fc2(h→h) → fc3(h→action)`，新实现按同一顺序 append）
    ⇒ 同一 `torch.manual_seed` 下参数取值、`named_parameters()` 顺序、`state_dict()` 键顺序
    全部一致。实测证据见 `tools/verify_paper_params_defaults.py` 的 [B] 节。
    """

    def __init__(self, state_dim=22, action_dim=10, hidden_dim=128, hidden_dims=None):
        super().__init__()
        widths = [int(w) for w in hidden_dims] if hidden_dims else [int(hidden_dim), int(hidden_dim)]
        self.hidden_dims = list(widths)
        layers = []
        prev = int(state_dim)
        for w in widths:
            layers.append(nn.Linear(prev, w))
            layers.append(nn.ReLU())
            prev = w
        layers.append(nn.Linear(prev, int(action_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, state):
        return self.net(state)
