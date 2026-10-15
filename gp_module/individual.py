"""Representation and construction utilities for single- and dual-tree GP individuals."""
import copy
import operator
from deap import base, creator, gp, tools

from gp_module.primitives import build_pset


def _ensure_creator():
    try:
        creator.FitnessMin
    except AttributeError:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("RuleIndividual", gp.PrimitiveTree, fitness=creator.FitnessMin)
    try:
        creator.DualIndividual
    except AttributeError:



        creator.create("DualIndividual", list, fitness=creator.FitnessMin)


def _mate_dual(lim: int):
    """双树交叉：**两棵树各自**做单点交叉，并各自施加深度上限。

    DEAP 的 `gp.staticLimit` 装饰器只对"返回元组里的单个个体"取 `height`，而双树个体是 list
    （没有 height 属性）⇒ 不能直接用装饰器。这里把 staticLimit 的语义
    （**超限则用父代深拷贝替换返回值**）在两棵树上分别实现，保持与原实现同义。
    """
    def mate(ind1, ind2):
        for t in range(len(ind1)):
            keep1, keep2 = copy.deepcopy(ind1[t]), copy.deepcopy(ind2[t])
            c1, c2 = gp.cxOnePoint(ind1[t], ind2[t])
            if c1.height > lim:
                ind1[t] = keep1
            if c2.height > lim:
                ind2[t] = keep2
        return ind1, ind2
    return mate


def _mutate_dual(lim: int, expr, pset):
    """双树变异：两棵树各自做均匀变异，并各自施加深度上限（同上）。"""
    def mutate(ind):
        for t in range(len(ind)):
            keep = copy.deepcopy(ind[t])
            c, = gp.mutUniform(ind[t], expr=expr, pset=pset)
            if c.height > lim:
                ind[t] = keep
        return ind,
    return mutate


def make_individual_dual(toolbox_expr):
    """构造一个双树个体（排序树 + 路由树）。

    ⚠ DEAP 的 `gp.genHalfAndHalf` 返回的是**普通 list**，必须用 `creator.RuleIndividual`
    包装成 `gp.PrimitiveTree` 才有 `.height`/`__str__`，交叉与变异也都要求树类型。
    """
    return creator.DualIndividual([creator.RuleIndividual(toolbox_expr()),
                                   creator.RuleIndividual(toolbox_expr())])


def make_toolbox(gp_params=None, dual: bool = False):
    """构造 DEAP toolbox。传入 gp_params 时其深度参数**生效**（修复记录：

    旧实现把 min_=2/max_=6/max_value=8 写死，`GPParams.min_init_depth/
    max_init_depth/max_tree_depth` 被静默忽略 ⇒ 论文里宣称的树深上限 8 并未执行。
    不传参数时保持历史默认值（2/6/8），故旧调用方行为不变。

    `dual=True`（本轮新增，Xu 2026 口径）：个体 = [排序树, 路由树]，交叉/变异/深度限制
    **分别作用于两棵树**；`dual=False`（默认）保持原单树行为不变。
    """
    _ensure_creator()
    pset = build_pset()
    min_d = 2 if gp_params is None else gp_params.min_init_depth
    max_d = 6 if gp_params is None else gp_params.max_init_depth
    lim = 8 if gp_params is None else gp_params.max_tree_depth
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=min_d, max_=max_d)
    if dual:
        toolbox.register("individual", make_individual_dual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", _mate_dual(lim))
        toolbox.register("mutate", _mutate_dual(lim, toolbox.expr, pset))
        return toolbox, pset
    toolbox.register("individual", tools.initIterate, creator.RuleIndividual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=lim))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=lim))
    return toolbox, pset


def compile_rule(ind, pset):
    """把 DEAP 个体（**单树**，或双树个体的某一棵树）编译成 rule(CandidateFeatures)->float。

    输入维数必须与 `build_pset` 一致：DSH_PSET_CTX=1 时终端含 4 个全局上下文特征（17 维），
    否则 13 维。两处必须同时切换，否则参数个数不匹配（会直接报错，不会静默错配）。
    """
    from gp_module.primitives import ctx_enabled
    f = gp.compile(ind, pset)
    ext = ctx_enabled()
    return lambda feat: f(*feat.to_list(ext_ctx=ext))


def is_dual(ind) -> bool:
    """是否双树个体。

    ⚠ 坑：DEAP 的 `gp.PrimitiveTree` **本身就是 list 的子类**，所以不能用
    `isinstance(ind, list)` 判断双树——单树个体会被误判，`ind[0]` 取到的是**根节点**
    （一个 Primitive 对象），再拿去 `gp.compile` 就会生成
    `lambda PT,...: <deap.gp.Primitive object at 0x...>` 这种非法表达式（SyntaxError）。
    真实踩到过：单树对照臂因此直接报错。
    """
    return isinstance(ind, list) and not isinstance(ind, gp.PrimitiveTree)


def compile_seq_rule(ind, pset):
    """取个体的**排序规则**：双树取 ind[0]，单树即其本身。"""
    return compile_rule(ind[0] if is_dual(ind) else ind, pset)


def compile_route_rule(ind, pset):
    """取个体的**路由规则**：双树取 ind[1]；单树无路由树 ⇒ 返回 None（调用方决定回退策略）。"""
    if is_dual(ind):
        return compile_rule(ind[1], pset)
    return None


def tree_str(ind) -> str:
    """个体的基因型字符串（双树时为两棵树的表达式串拼接）。"""
    if is_dual(ind):
        return " | ".join(str(t) for t in ind)
    return str(ind)
