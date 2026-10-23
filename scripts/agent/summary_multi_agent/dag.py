"""提供 ExecutionPlan 的拓扑分层，供编排器按依赖安全调度。"""

from __future__ import annotations

from .schemas import ExecutionPlan, SubTask


def topological_layers(plan: ExecutionPlan) -> tuple[tuple[SubTask, ...], ...]:
    """把计划转换为稳定排序的拓扑层，供同层无依赖任务并发执行。

    返回值按 task id 稳定排序；如果无法访问全部任务，说明计划有环并抛出
    ``ValueError``，编排器不会在不确定依赖下启动 Worker。
    """
    task_by_id = {task.task_id: task for task in plan.tasks}
    indegree = {task.task_id: len(task.dependencies) for task in plan.tasks}
    dependents: dict[str, list[str]] = {task.task_id: [] for task in plan.tasks}
    for task in plan.tasks:
        for dependency in task.dependencies:
            dependents[dependency].append(task.task_id)

    ready = sorted(task_id for task_id, degree in indegree.items() if degree == 0)
    layers: list[tuple[SubTask, ...]] = []
    visited = 0
    while ready:
        current_ids = tuple(ready)
        layers.append(tuple(task_by_id[task_id] for task_id in current_ids))
        visited += len(current_ids)
        next_ready: list[str] = []
        for task_id in current_ids:
            for dependent in sorted(dependents[task_id]):
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    next_ready.append(dependent)
        ready = sorted(next_ready)

    if visited != len(plan.tasks):
        raise ValueError("execution plan contains a cycle")
    return tuple(layers)
