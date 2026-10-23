"""初始化、检查点保存和严格加载单次运行的 M4 Working Memory。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from .raw_material import build_coverage_ledger
from .schemas import RawMaterialBundle, SummaryRequest, SummaryWorkingState


def initialize_working_state(
    request: SummaryRequest,
    raw_material: RawMaterialBundle,
    *,
    current_map_required: bool = True,
) -> SummaryWorkingState:
    """创建一次运行的 M4 Shared Working Memory，并绑定不可变当前原文。"""
    if request.current_material != raw_material.raw_text:
        raise ValueError("SummaryRequest and RawMaterialBundle text must match exactly")
    coverage = build_coverage_ledger(
        raw_material,
        required_stages=(
            ("current_fact_map", "synthesis_merge")
            if current_map_required else ("synthesis_merge",)
        ),
    )
    return SummaryWorkingState(
        request=request,
        raw_material=raw_material,
        coverage_ledger=coverage,
    )


def save_working_state_checkpoint(
    state: SummaryWorkingState,
    output_dir: Path,
    *,
    filename: str = "working_state.json",
) -> Path:
    """以临时文件加原子替换保存 checkpoint，避免留下半写状态。

    写入的是完整 Pydantic 状态；目录会按需创建，文件 I/O 异常向调用方传播。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / filename
    temporary = output_dir / f".{filename}.tmp"
    payload = state.model_dump(mode="json")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    # 先写临时文件再 replace，保证 Inspector 不会读到半份 JSON。
    os.replace(temporary, destination)
    return destination


def load_working_state_checkpoint(path: Path) -> SummaryWorkingState:
    """加载并严格校验 checkpoint，尤其拒绝被改变的完整当前材料。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    state = SummaryWorkingState.model_validate(payload)
    if state.request.current_material != state.raw_material.raw_text:
        raise ValueError(
            "working-state checkpoint changed the complete current material"
        )
    return state
