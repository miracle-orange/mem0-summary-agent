"""运行有界 Red-Blue-Judge Shadow Review，并防止候选和 Issue 振荡。"""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any

from ..schemas import (
    BlueReviewResult,
    JudgeDecision,
    JudgeDisposition,
    RedReviewReport,
    ReviewIssueReappearance,
    ReviewRoundRecord,
    ReviewRunResult,
    ReviewSeverity,
    ReviewTermination,
    ReviewVerificationResult,
    RunConfig,
    SummaryWorkingState,
)
from ..tracing import TraceRecorder
from .blue import BlueAgent
from .convergence import build_issue_signature, calculate_text_change_ratio
from .judge import JudgeAgent
from .red import RedAgent
from .verify import ReviewVerifier


class ReviewLoop:
    """运行有界 Red-Verify-Blue-Judge Shadow Review，并保护候选收敛。

    Review 产出的推荐 candidate 只供审计和人工检查；正式 Pipeline candidate 0
    不会被本循环自动替换。Issue 重现、候选 hash 重复、低改进和异常都会触发
    明确终止原因或 warning。
    """

    def __init__(
        self,
        *,
        red_agent: RedAgent,
        blue_agent: BlueAgent,
        judge_agent: JudgeAgent,
        verifier: ReviewVerifier,
        config: RunConfig,
        trace_recorder: TraceRecorder,
    ) -> None:
        """注入 Red/Blue/Judge、验证器、配置和共享 Trace。"""
        self._red = red_agent
        self._blue = blue_agent
        self._judge = judge_agent
        self._verifier = verifier
        self._config = config
        self._trace = trace_recorder
        self._red_reports: list[RedReviewReport] = []
        self._verification_results: list[ReviewVerificationResult] = []
        self._blue_results: list[BlueReviewResult] = []
        self._judge_decisions: list[JudgeDecision] = []

    @property
    def model_traces(self) -> dict[str, list[dict[str, Any]]]:
        """返回三类 Review Agent 的模型调用 trace 投影。"""
        return {
            "red": list(self._red.traces),
            "blue": list(self._blue.traces),
            "judge": list(self._judge.traces),
        }

    @property
    def red_reports(self) -> tuple[RedReviewReport, ...]:
        """返回已产生的 Red 报告。"""
        return tuple(self._red_reports)

    @property
    def verification_results(self) -> tuple[ReviewVerificationResult, ...]:
        """返回已执行的 Issue 验证结果。"""
        return tuple(self._verification_results)

    @property
    def blue_results(self) -> tuple[BlueReviewResult, ...]:
        """返回已产生的 Blue 候选修订结果。"""
        return tuple(self._blue_results)

    @property
    def judge_decisions(self) -> tuple[JudgeDecision, ...]:
        """返回各轮 Judge 决策。"""
        return tuple(self._judge_decisions)

    async def run(
        self,
        state: SummaryWorkingState,
        original_summary: str,
    ) -> ReviewRunResult:
        """按轮次并发验证、局部修订和独立评价，直至接受、收敛或达到上限。"""
        self._red_reports.clear()
        self._verification_results.clear()
        self._blue_results.clear()
        self._judge_decisions.clear()
        if not state.candidate_summaries:
            state.candidate_summaries.append(original_summary)
        if self._config.max_review_rounds == 0:
            return ReviewRunResult(
                shadow_mode=self._config.review_shadow_mode,
                termination=ReviewTermination.MAX_ROUNDS,
                warnings=("review enabled with max_review_rounds=0",),
            )

        current_candidate = original_summary
        current_index = 0
        recommended_index = 0
        best_score = -1.0
        previous_score = 0.0
        low_convergence_rounds = 0
        seen_hashes = {self._hash(current_candidate)}
        resolved_issue_history: dict[str, tuple[int, str]] = {}
        rounds: list[ReviewRoundRecord] = []
        warnings: list[str] = []
        termination = ReviewTermination.MAX_ROUNDS

        await self._trace.emit(
            "review.started",
            run_id=state.run_id,
            payload={
                "shadow_mode": self._config.review_shadow_mode,
                "max_rounds": self._config.max_review_rounds,
            },
        )
        try:
            for round_index in range(1, self._config.max_review_rounds + 1):
                await self._trace.emit(
                    "review.round_started",
                    run_id=state.run_id,
                    payload={
                        "round_index": round_index,
                        "candidate_index": current_index,
                    },
                )
                red_report = await self._red.review(
                    state,
                    current_candidate,
                    candidate_index=current_index,
                    round_index=round_index,
                )
                self._red_reports.append(red_report)
                warnings.extend(red_report.warnings)
                issue_signatures = {
                    issue.issue_id: build_issue_signature(issue)
                    for issue in red_report.issues
                }
                issue_reappearances = tuple(
                    ReviewIssueReappearance(
                        signature=signature,
                        previous_round_index=resolved_issue_history[signature][0],
                        previous_issue_id=resolved_issue_history[signature][1],
                        current_issue_id=issue.issue_id,
                    )
                    for issue in red_report.issues
                    if (signature := issue_signatures[issue.issue_id])
                    in resolved_issue_history
                )
                verification_results = tuple(
                    # Issue 验证彼此独立，可并发；所有结果汇总后才交给 Blue/Judge。
                    await asyncio.gather(
                        *(
                            self._verifier.verify(
                                state, issue, round_index=round_index
                            )
                            for issue in red_report.issues
                        )
                    )
                )
                self._verification_results.extend(verification_results)
                warnings.extend(
                    result.warning
                    for result in verification_results
                    if result.warning
                )
                actionable = [
                    issue
                    for issue in red_report.issues
                    if issue.severity
                    in {
                        ReviewSeverity.CRITICAL,
                        ReviewSeverity.HIGH,
                        ReviewSeverity.MEDIUM,
                    }
                ]

                blue_result = None
                proposed_candidate = current_candidate
                output_index = current_index
                if actionable:
                    blue_result = await self._blue.repair(
                        state,
                        current_candidate,
                        red_report,
                        verification_results,
                        source_candidate_index=current_index,
                        round_index=round_index,
                    )
                    self._blue_results.append(blue_result)
                    warnings.extend(blue_result.warnings)
                    proposed_candidate = blue_result.candidate_summary
                    if proposed_candidate != current_candidate:
                        state.candidate_summaries.append(proposed_candidate)
                        output_index = len(state.candidate_summaries) - 1
                text_change_ratio = calculate_text_change_ratio(
                    current_candidate,
                    proposed_candidate,
                )

                decision = await self._judge.judge(
                    state,
                    current_candidate,
                    proposed_candidate,
                    red_report,
                    blue_result,
                    verification_results,
                    candidate_index=output_index,
                    round_index=round_index,
                    previous_score=previous_score,
                )
                self._judge_decisions.append(decision)
                state.review_history.append(decision)
                warnings.extend(decision.warnings)
                round_record = ReviewRoundRecord(
                    round_index=round_index,
                    input_candidate_index=current_index,
                    output_candidate_index=output_index,
                    text_change_ratio=text_change_ratio,
                    issue_reappearances=issue_reappearances,
                    red_report=red_report,
                    verification_results=verification_results,
                    blue_result=blue_result,
                    judge_decision=decision,
                )
                rounds.append(round_record)

                if decision.score > best_score or (
                    decision.score == best_score
                    and decision.disposition == JudgeDisposition.ACCEPT
                ):
                    best_score = decision.score
                    recommended_index = output_index

                remaining_by_id = {
                    issue.issue_id: issue for issue in red_report.issues
                }
                deferred_serious = [
                    issue
                    for issue in red_report.deferred_issues
                    if issue.severity
                    in {
                        ReviewSeverity.CRITICAL,
                        ReviewSeverity.HIGH,
                        ReviewSeverity.MEDIUM,
                    }
                ]
                remaining_issue_ids = set(decision.remaining_issue_ids)
                if blue_result is not None:
                    remaining_issue_ids.update(blue_result.unresolved_issue_ids)
                remaining_serious = [
                    remaining_by_id[issue_id]
                    for issue_id in remaining_issue_ids
                    if issue_id in remaining_by_id
                    and remaining_by_id[issue_id].severity
                    in {
                        ReviewSeverity.CRITICAL,
                        ReviewSeverity.HIGH,
                        ReviewSeverity.MEDIUM,
                    }
                ]
                if blue_result is not None:
                    applied_issue_ids = {
                        repair.issue_id
                        for repair in blue_result.repairs
                        if repair.applied
                    }
                    for issue_id in applied_issue_ids - remaining_issue_ids:
                        signature = issue_signatures.get(issue_id)
                        if signature is not None:
                            resolved_issue_history.setdefault(
                                signature,
                                (round_index, issue_id),
                            )
                if issue_reappearances:
                    termination = ReviewTermination.OSCILLATION
                    warnings.append(
                        f"review stopped after {len(issue_reappearances)} resolved "
                        "issue signature(s) reappeared"
                    )
                    break
                if (
                    decision.disposition == JudgeDisposition.ACCEPT
                    and not remaining_serious
                    and deferred_serious
                ):
                    warnings.append(
                        f"round {round_index} Judge accepted selected issues, but "
                        f"{len(deferred_serious)} deferred serious issue(s) remain"
                    )
                if (
                    decision.disposition == JudgeDisposition.ACCEPT
                    and not remaining_serious
                    and not deferred_serious
                    and decision.score >= self._config.review_accept_score
                ):
                    recommended_index = output_index
                    termination = ReviewTermination.ACCEPTED
                    break
                if decision.disposition == JudgeDisposition.STOP_WITH_WARNINGS:
                    termination = ReviewTermination.STOP_WITH_WARNINGS
                    break
                if not actionable:
                    termination = ReviewTermination.NO_ACTIONABLE_ISSUES
                    break
                if proposed_candidate == current_candidate:
                    termination = ReviewTermination.CONVERGED
                    warnings.append("review stopped because Blue produced no traceable change")
                    break

                proposed_hash = self._hash(proposed_candidate)
                if proposed_hash in seen_hashes:
                    termination = ReviewTermination.OSCILLATION
                    warnings.append("review stopped after candidate oscillation was detected")
                    break
                seen_hashes.add(proposed_hash)

                if (
                    decision.improvement < self._config.review_min_improvement
                    and text_change_ratio
                    < self._config.review_text_change_convergence_threshold
                ):
                    low_convergence_rounds += 1
                else:
                    low_convergence_rounds = 0
                if low_convergence_rounds >= 2:
                    termination = ReviewTermination.CONVERGED
                    warnings.append(
                        "review stopped after two rounds below both score-improvement "
                        "and text-change convergence thresholds"
                    )
                    break

                previous_score = decision.score
                current_candidate = proposed_candidate
                current_index = output_index
                if round_index == self._config.max_review_rounds:
                    termination = ReviewTermination.MAX_ROUNDS
        except Exception as exc:
            termination = ReviewTermination.REVIEW_FAILED
            warnings.append(f"review failed: {type(exc).__name__}: {exc}")

        result = ReviewRunResult(
            shadow_mode=self._config.review_shadow_mode,
            termination=termination,
            original_candidate_index=0,
            recommended_candidate_index=recommended_index,
            rounds=tuple(rounds),
            warnings=tuple(dict.fromkeys(warnings)),
        )
        await self._trace.emit(
            "review.finished",
            run_id=state.run_id,
            payload={
                "termination": termination.value,
                "round_count": len(rounds),
                "recommended_candidate_index": recommended_index,
            },
        )
        return result

    @staticmethod
    def _hash(candidate: str) -> str:
        """计算候选正文 hash，用于检测候选振荡。"""
        return hashlib.sha256(candidate.encode("utf-8")).hexdigest()
