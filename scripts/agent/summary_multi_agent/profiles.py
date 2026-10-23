"""Defaults for the user-facing cloud workflow."""

CLOUD_FAST_DEFAULTS = {
    "planner_max_tokens": 4096,
    "planner_repair_max_tokens": 4096,
    "planner_repair_attempts": 1,
    "replanner_max_tokens": 4096,
    "replanner_repair_max_tokens": 4096,
    "replanner_repair_attempts": 1,
    "worker_action_max_tokens": 4096,
    "max_concurrent_model_calls": 4,
    "max_concurrent_workers": 4,
    "worker_timeout_seconds": 600,
    "global_timeout_seconds": 1200,
    "max_replan_rounds": 1,
    "schema_repair_attempts": 1,
    "summary_max_tokens": 4096,
    "rich_summary_max_tokens": 8192,
    "raw_segment_max_chars": 12000,
    "raw_segment_overlap_ratio": 0.01,
    "minutes_output_mode": "rich",
    "evaluation_mode": "lite",
}
