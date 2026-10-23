"""Rebuild baseline/new prompts offline with identical business inputs."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import sys
import types
from datetime import datetime
from pathlib import Path

from summary_multi_agent import synthesis
from summary_multi_agent.model_input_audit import _TokenCounter
from summary_multi_agent.schemas import SynthesisContext
from summary_multi_agent.tools.template import MEETING_SCENES, TemplateAdaptationInput, resolve_template


BASELINE_REF = "baseline/pre-prompt-improvements-20260930"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _baseline_module(ref: str, relative: str, name: str):
    source = subprocess.check_output(
        ["git", "show", f"{ref}:{relative}"], cwd=REPO_ROOT,
    ).decode("utf-8")
    module = types.ModuleType(name)
    module.__package__ = name.rsplit(".", 1)[0]
    sys.modules[name] = module
    exec(compile(source, f"{ref}:{relative}", "exec"), module.__dict__)
    return module


def compare_inputs(context: SynthesisContext, current: str, *, baseline_ref: str, scene: str, style: str):
    old_synthesis = _baseline_module(
        baseline_ref, "scripts/agent/summary_multi_agent/synthesis.py",
        "summary_multi_agent._prompt_audit_baseline_synthesis",
    )
    old_template = _baseline_module(
        baseline_ref, "scripts/agent/summary_multi_agent/tools/template.py",
        "summary_multi_agent.tools._prompt_audit_baseline_template",
    )
    old = asyncio.run(old_template.TemplateAdaptationTool().execute(
        old_template.TemplateAdaptationInput(
            summary_request=context.summary_request, requested_style=style,
            preferences=context.personalization_preferences,
        ),
        None,
    ))
    new = resolve_template(TemplateAdaptationInput(
        summary_request=context.summary_request, requested_style=style, requested_scene=scene,
        preferences=context.personalization_preferences,
    ))
    counter = _TokenCounter()
    snapshots = {}
    prompts = {}
    views = {}
    ids = {}
    for label, module, template in (
        ("baseline", old_synthesis, old), ("new", synthesis, new),
    ):
        selected = context.model_copy(update={"template": template.model_dump(mode="json")})
        prompt, report = module.build_projected_summary_synthesis_prompt(selected, current_material=current)
        view, ids[label] = module.build_synthesis_model_input(selected, current_material=current)
        parts = {key: module._compact_json(value) for key, value in view.items()}
        parts["current_material"] = current
        parts["instructions"] = prompt.split("\n\n<summary_task>", 1)[0]
        snapshots[label] = {
            "prompt_version": report["prompt_version"],
            "template_name": template.template_name,
            "total": counter.measure(prompt),
            "sections": {key: counter.measure(value) for key, value in parts.items()},
        }
        prompts[label] = prompt
        views[label] = view
    rich_context = context.model_copy(
        update={"template": new.model_dump(mode="json")}
    )
    rich_prompt, rich_report, rich_aliases, rich_sections = (
        synthesis.build_rich_summary_synthesis_prompt(
            rich_context,
            current_material=current,
        )
    )
    snapshots["rich"] = {
        "prompt_version": rich_report["prompt_version"],
        "template_name": new.template_name,
        "total": counter.measure(rich_prompt),
        "sections": {
            key: counter.measure(value) for key, value in rich_sections.items()
        },
        "history_alias_count": len(rich_aliases),
    }
    prompts["rich"] = rich_prompt
    business_keys = ("summary_task", "historical_context", "current_temporal_results")
    if any(views["baseline"][key] != views["new"][key] for key in business_keys):
        raise ValueError("business inputs changed between baseline and new projections")
    if ids["baseline"] != ids["new"]:
        raise ValueError("visible history IDs changed")
    if tuple(rich_report["history_record_ids"]) != tuple(ids["new"]):
        raise ValueError("rich prompt changed the visible history set")
    if rich_prompt.count(current) != 1:
        raise ValueError("rich prompt must contain current material exactly once")
    original_hash = hashlib.sha256(current.encode("utf-8")).hexdigest()
    result = {
        "baseline_ref": baseline_ref,
        "current_material_sha256": original_hash,
        "history_record_ids": list(ids["new"]),
        "business_inputs_equal": True,
        "model_calls": 0,
        "snapshots": snapshots,
        "delta": {
            key: snapshots["new"]["total"][key] - snapshots["baseline"]["total"][key]
            for key in ("chars", "utf8_bytes", "tokens")
        },
        "rich_delta_vs_prompt_improvements": {
            key: snapshots["rich"]["total"][key] - snapshots["new"]["total"][key]
            for key in ("chars", "utf8_bytes", "tokens")
        },
        "rich_delta_vs_baseline": {
            key: snapshots["rich"]["total"][key] - snapshots["baseline"]["total"][key]
            for key in ("chars", "utf8_bytes", "tokens")
        },
    }
    return result, prompts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path, help="Saved current_material.txt and synthesis_context.json")
    source.add_argument("--regression-fixture", action="store_true", help="Synthetic nonempty-history test fixture, not business-average compression")
    parser.add_argument("--baseline-ref", default=BASELINE_REF)
    parser.add_argument("--scene", choices=MEETING_SCENES, default="general")
    parser.add_argument("--style", choices=("structured", "natural"), default="structured")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.run_dir:
        context = SynthesisContext.model_validate_json(
            (args.run_dir / "synthesis_context.json").read_text("utf-8")
        )
        current = (args.run_dir / "current_material.txt").read_text("utf-8")
        source_label = str(args.run_dir.resolve())
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent / "tests"))
        from test_summary_multi_agent_model_input_projection import make_long_context
        context, current = make_long_context()
        source_label = "synthetic make_long_context regression fixture; not a production run"
    report, prompts = compare_inputs(
        context, current, baseline_ref=args.baseline_ref, scene=args.scene, style=args.style,
    )
    report["input_source"] = source_label
    output = args.output_dir or Path("outputs/minutes_prompt_input_audit") / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True, exist_ok=True)
    (output / "prompt_length_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    for label, prompt in prompts.items():
        (output / f"{label}_prompt.txt").write_text(prompt, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"OUTPUT_DIR: {output.resolve()}")


if __name__ == "__main__":
    main()
