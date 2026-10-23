"""计算候选文本变化和 Issue 签名，用于 Review 收敛与振荡保护。"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from difflib import SequenceMatcher

from ..schemas import ReviewIssue


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_review_text(value: str) -> str:
    """统一候选文本的 Unicode、大小写和空白，仅供收敛比较使用。"""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return _WHITESPACE_RE.sub(" ", normalized).strip()


def calculate_text_change_ratio(previous: str, current: str) -> float:
    """计算两版候选的规范化文本变化比例，结果限制在 ``0..1``。"""
    previous_normalized = normalize_review_text(previous)
    current_normalized = normalize_review_text(current)
    if previous_normalized == current_normalized:
        return 0.0
    if not previous_normalized or not current_normalized:
        return 1.0
    similarity = SequenceMatcher(
        None,
        previous_normalized,
        current_normalized,
        autojunk=True,
    ).ratio()
    return round(max(0.0, min(1.0, 1.0 - similarity)), 6)


def build_issue_signature(issue: ReviewIssue) -> str:
    """根据 Issue 类型、定位和 Evidence 构造稳定签名，用于检测重现。"""
    evidence_ids = sorted(set(issue.evidence_ids))
    related_task_ids = sorted(set(issue.related_task_ids))
    segment_ids = sorted(set(issue.segment_ids))
    target_entities = sorted(
        {
            normalized
            for value in issue.target_entities
            if (normalized := normalize_review_text(value))
        }
    )
    payload: dict[str, object] = {
        "issue_type": normalize_review_text(issue.issue_type),
        "section": normalize_review_text(issue.section),
        "evidence_ids": evidence_ids,
        "related_task_ids": related_task_ids,
        "segment_ids": (
            segment_ids if not evidence_ids and not related_task_ids else []
        ),
        "target_entities": target_entities,
    }
    if not evidence_ids and not related_task_ids and not segment_ids:
        payload["ungrounded_description"] = normalize_review_text(issue.description)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
