"""Persist the public rich-minutes bundle without raw prompts or source documents."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from .models import MinutesReportBundle, MinutesReportManifest


PUBLIC_REPORT_FILES = (
    "summary_multi_agent.md",
    "minutes_document.json",
    "minutes_sources.json",
    "minutes_report.html",
    "minutes_report_manifest.json",
)


def _json_text(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def persist_report_bundle(
    output_dir: Path,
    bundle: MinutesReportBundle,
) -> MinutesReportManifest:
    """Write independent report derivatives without invalidating the summary."""

    output_dir.mkdir(parents=True, exist_ok=True)
    statuses = dict(bundle.manifest.artifact_status)
    warnings = list(bundle.manifest.warnings)

    def remove_stale(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    def write_derivative(name: str, content: str) -> None:
        try:
            (output_dir / name).write_text(content, encoding="utf-8")
            statuses[name] = "completed"
        except Exception as exc:  # A derivative must not invalidate formal Markdown.
            remove_stale(output_dir / name)
            statuses[name] = "failed"
            warnings.append(
                f"{name} write failed: {type(exc).__name__}: {exc}"
            )

    write_derivative(
        "minutes_document.json",
        _json_text(bundle.document.model_dump(mode="json")),
    )
    write_derivative(
        "minutes_sources.json",
        _json_text(bundle.sources.model_dump(mode="json")),
    )
    if bundle.html:
        write_derivative("minutes_report.html", bundle.html)
    else:
        remove_stale(output_dir / "minutes_report.html")
        statuses["minutes_report.html"] = "failed"

    statuses["minutes_report_manifest.json"] = "completed"
    statuses["minutes_report_bundle.zip"] = "completed"
    manifest = bundle.manifest.model_copy(
        update={
            "artifact_status": statuses,
            "warnings": tuple(dict.fromkeys(warnings)),
        }
    )
    manifest_path = output_dir / "minutes_report_manifest.json"
    try:
        manifest_path.write_text(
            _json_text(manifest.model_dump(mode="json")), encoding="utf-8"
        )
    except Exception as exc:
        remove_stale(manifest_path)
        remove_stale(output_dir / "minutes_report_bundle.zip")
        statuses["minutes_report_manifest.json"] = "failed"
        statuses["minutes_report_bundle.zip"] = "failed"
        warnings.append(
            "minutes_report_manifest.json write failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return manifest.model_copy(
            update={
                "artifact_status": statuses,
                "warnings": tuple(dict.fromkeys(warnings)),
            }
        )

    archive_path = output_dir / "minutes_report_bundle.zip"
    temporary_path = output_dir / "minutes_report_bundle.zip.tmp"
    try:
        with zipfile.ZipFile(
            temporary_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            for name in PUBLIC_REPORT_FILES:
                path = output_dir / name
                if path.is_file():
                    archive.write(path, arcname=name)
        temporary_path.replace(archive_path)
    except Exception as exc:
        remove_stale(temporary_path)
        remove_stale(archive_path)
        statuses["minutes_report_bundle.zip"] = "failed"
        warnings.append(
            f"minutes report ZIP failed: {type(exc).__name__}: {exc}"
        )
        manifest = manifest.model_copy(
            update={
                "artifact_status": statuses,
                "warnings": tuple(dict.fromkeys(warnings)),
            }
        )
        try:
            manifest_path.write_text(
                _json_text(manifest.model_dump(mode="json")), encoding="utf-8"
            )
        except Exception as manifest_exc:
            remove_stale(manifest_path)
            statuses["minutes_report_manifest.json"] = "failed"
            warnings.append(
                "minutes_report_manifest.json rewrite failed: "
                f"{type(manifest_exc).__name__}: {manifest_exc}"
            )
            manifest = manifest.model_copy(
                update={
                    "artifact_status": statuses,
                    "warnings": tuple(dict.fromkeys(warnings)),
                }
            )
    return manifest
