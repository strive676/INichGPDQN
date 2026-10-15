"""Crossover and mutation operators for genetic programming individuals."""
import random


def crossover(p1, p2, toolbox, rate=0.9):
    """交叉。返回的个体必须来自 `toolbox.mate` 的**返回值**。

    修复记录：`toolbox.mate`/`toolbox.mutate` 被 `gp.staticLimit` 装饰，而 DEAP 的
    staticLimit 是"在**返回值**里把超限子代替换成父代深拷贝"，并不是原地约束。
    旧代码丢弃返回值 ⇒ 树深限制**静默失效**（实测 12 代后高度达 25、51/60 超限 8），
    且 `GPParams.max_tree_depth` 从未生效。
    """
    c1, c2 = toolbox.clone(p1), toolbox.clone(p2)
    if random.random() < rate:
        c1, c2 = toolbox.mate(c1, c2)
        del c1.fitness.values
        del c2.fitness.values
    return c1, c2


def mutate(ind, toolbox, rate=0.1):
    """变异。同上，必须接收 `toolbox.mutate` 的返回值（超限时它返回变异前个体的深拷贝）。"""
    mut = toolbox.clone(ind)
    if random.random() < rate:
        mut, = toolbox.mutate(mut)
        del mut.fitness.values
    return mut
