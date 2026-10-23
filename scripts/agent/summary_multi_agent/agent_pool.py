"""管理按任务类型复用的 Worker 实例、租约和运行级并发上限。"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Callable

from .schemas import TaskType
from .workers.base import BaseWorker


WorkerFactory = Callable[[], BaseWorker]


@dataclass(frozen=True)
class AgentPoolStats:
    """描述 Worker 实例总量、租约量及按任务类型的空闲/占用分布。"""

    created_total: int
    leased_total: int
    idle_total: int
    created_by_type: dict[str, int]
    leased_by_type: dict[str, int]
    idle_by_type: dict[str, int]


class AgentPool:
    """按 TaskType 复用 Worker，并以租约保证并发和资源释放。

    ``acquire`` 会等待总实例额度或驱逐其他类型的空闲实例；调用方必须通过
    ``lease`` 或成对调用 ``acquire/release``，取消和异常也必须释放租约。
    """

    def __init__(
        self,
        factories: dict[TaskType, WorkerFactory],
        *,
        max_total: int = 4,
        max_idle_per_type: int = 2,
    ) -> None:
        """注册 Worker 工厂并设置总实例数和每类空闲缓存上限。"""
        if max_total < 1:
            raise ValueError("max_total must be at least 1")
        if max_idle_per_type < 0:
            raise ValueError("max_idle_per_type must be non-negative")
        self._factories = dict(factories)
        self._max_total = max_total
        self._max_idle_per_type = max_idle_per_type
        self._idle: dict[TaskType, list[BaseWorker]] = defaultdict(list)
        self._created_by_type: dict[TaskType, int] = defaultdict(int)
        self._leased: dict[int, tuple[TaskType, BaseWorker]] = {}
        self._condition = asyncio.Condition()

    @property
    def registered_task_types(self) -> tuple[TaskType, ...]:
        """返回可创建 Worker 的任务类型，按值稳定排序。"""
        return tuple(sorted(self._factories, key=lambda item: item.value))

    async def acquire(self, task_type: TaskType) -> BaseWorker:
        """获取指定类型 Worker；资源不足时异步等待而不阻塞事件循环。"""
        if task_type not in self._factories:
            raise KeyError(f"no worker factory registered for {task_type.value}")
        async with self._condition:
            while True:
                if self._idle[task_type]:
                    worker = self._idle[task_type].pop()
                    self._leased[id(worker)] = (task_type, worker)
                    return worker
                if sum(self._created_by_type.values()) < self._max_total:
                    worker = self._factories[task_type]()
                    if worker.task_type != task_type:
                        raise ValueError(
                            f"factory for {task_type.value} created {worker.task_type.value}"
                        )
                    self._created_by_type[task_type] += 1
                    self._leased[id(worker)] = (task_type, worker)
                    return worker
                if self._evict_idle_worker(exclude=task_type):
                    continue
                await self._condition.wait()

    def _evict_idle_worker(self, *, exclude: TaskType) -> bool:
        """驱逐一个非目标类型的空闲 Worker，为新租约腾出总额度。"""
        for idle_type in sorted(self._idle, key=lambda value: value.value):
            if idle_type == exclude or not self._idle[idle_type]:
                continue
            self._idle[idle_type].pop()
            self._created_by_type[idle_type] -= 1
            return True
        return False

    async def release(self, worker: BaseWorker) -> None:
        """归还 Worker 租约，或在空闲缓存已满时丢弃实例。"""
        async with self._condition:
            lease = self._leased.pop(id(worker), None)
            if lease is None:
                raise ValueError("worker is not leased from this pool")
            task_type, leased_worker = lease
            if leased_worker is not worker:
                raise ValueError("worker lease identity mismatch")
            if len(self._idle[task_type]) < self._max_idle_per_type:
                self._idle[task_type].append(worker)
            else:
                self._created_by_type[task_type] -= 1
            self._condition.notify_all()

    @asynccontextmanager
    async def lease(self, task_type: TaskType) -> AsyncIterator[BaseWorker]:
        """提供异常/取消安全的 Worker 异步上下文租约。"""
        worker = await self.acquire(task_type)
        try:
            yield worker
        finally:
            await self.release(worker)

    async def stats(self) -> AgentPoolStats:
        """在条件锁内快照当前实例和租约统计，不改变池状态。"""
        async with self._condition:
            leased_by_type: dict[TaskType, int] = defaultdict(int)
            for task_type, _worker in self._leased.values():
                leased_by_type[task_type] += 1
            created = {
                task_type.value: count for task_type, count in self._created_by_type.items()
            }
            leased = {task_type.value: count for task_type, count in leased_by_type.items()}
            idle = {
                task_type.value: len(workers)
                for task_type, workers in self._idle.items()
                if workers
            }
            return AgentPoolStats(
                created_total=sum(created.values()),
                leased_total=sum(leased.values()),
                idle_total=sum(idle.values()),
                created_by_type=created,
                leased_by_type=leased,
                idle_by_type=idle,
            )
