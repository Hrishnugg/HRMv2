#!/usr/bin/env python3
"""
Download and summarize JSON result artifacts from HRM-related Modal volumes.

This intentionally pulls JSON files even when they are large. It avoids binary
checkpoint/dataset artifacts and produces a normalized summary that can be used
as evidence for experiment writeups.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import modal
except Exception:  # pragma: no cover - CLI fallback handles missing SDK.
    modal = None


DEFAULT_VOLUMES = [
    "residual-tasklora-v2-vol",
    "multitask-astar-heuristic-tasklora-v1-vol",
    "transfer-astar-heuristic-avg-condlora-basis-v1-vol",
    "transfer-astar-heuristic-clean-parallel-v3-vol",
    "transfer-astar-heuristic-clean-parallel-v2-vol",
    "transfer-astar-heuristic-clean-parallel-v1-vol",
    "transfer-astar-heuristic-imitation-v2-vol-v2",
    "transfer-astar-heuristic-imitation-v2-vol",
    "transfer-astar-heuristic-rl-vol",
    "onlstm-hrm-comparison-presetm-v2-vol",
    "onlstm-hrm-comparison-presetm-vol",
    "lstm-hrm-comparison-vol",
    "hrm-astar-volume",
    "hrm-8gpu-v2-obs-vol",
    "hrm-8gpu-v2-vol",
    "hrm-8gpu-vol",
    "hrm-mid-vol",
    "hrm-robust-vol",
    "hrm-research-vol",
]

JSON_SUFFIXES = (".json",)
OPTIONAL_SMALL_SUFFIXES = (".csv", ".npz")


def run_modal(args: List[str], timeout: Optional[int] = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    cmd = [sys.executable, "-m", "modal", *args]
    return subprocess.run(
        cmd,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        env=env,
    )


def ensure_remote_path(path: str) -> str:
    return path if path.startswith("/") else "/" + path


def safe_local_path(out_dir: Path, volume: str, remote_path: str) -> Path:
    parts = [p for p in remote_path.replace("\\", "/").split("/") if p]
    return out_dir / volume / Path(*parts)


def ls_volume(volume: str, remote_path: str) -> List[Dict[str, Any]]:
    cp = run_modal(["volume", "ls", "--json", volume, remote_path], timeout=120)
    if cp.returncode != 0:
        raise RuntimeError(f"modal volume ls failed for {volume}:{remote_path}\n{cp.stderr or cp.stdout}")
    text = cp.stdout.strip()
    if not text:
        return []
    return json.loads(text)


def walk_volume_cli(volume: str, roots: Iterable[str]) -> List[Dict[str, Any]]:
    seen_dirs = set()
    files: List[Dict[str, Any]] = []
    stack = [ensure_remote_path(root) for root in roots]
    while stack:
        current = stack.pop()
        if current in seen_dirs:
            continue
        seen_dirs.add(current)
        try:
            items = ls_volume(volume, current)
        except Exception as exc:
            files.append({
                "volume": volume,
                "remote_path": current,
                "type": "ls_error",
                "error": str(exc),
            })
            continue
        for item in items:
            name = item.get("Filename") or item.get("filename") or ""
            item_type = (item.get("Type") or item.get("type") or "").lower()
            remote = ensure_remote_path(str(name))
            if item_type == "dir":
                stack.append(remote)
            else:
                files.append({
                    "volume": volume,
                    "remote_path": remote,
                    "type": item_type or "file",
                    "modified": item.get("Created/Modified"),
                    "size": item.get("Size"),
                })
    return files


def format_mtime(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        return dt.datetime.fromtimestamp(float(value), tz=dt.timezone.utc).isoformat()
    except Exception:
        return str(value)


def sdk_entry_type(entry: Any) -> str:
    raw = getattr(entry, "type", None)
    name = getattr(raw, "name", None)
    if name:
        lowered = name.lower()
    else:
        lowered = str(raw or "").split(".")[-1].lower()
    if lowered in {"directory", "dir"}:
        return "dir"
    if lowered in {"file", "regular"}:
        return "file"
    return lowered or "file"


def walk_volume_sdk(volume_obj: Any, volume: str, roots: Iterable[str]) -> List[Dict[str, Any]]:
    files: List[Dict[str, Any]] = []
    seen_paths = set()
    for root in roots:
        root_path = ensure_remote_path(root)
        for entry in volume_obj.listdir(root_path, recursive=True):
            remote = ensure_remote_path(str(getattr(entry, "path", "")))
            if not remote or remote in seen_paths:
                continue
            seen_paths.add(remote)
            item_type = sdk_entry_type(entry)
            if item_type == "dir":
                continue
            files.append({
                "volume": volume,
                "remote_path": remote,
                "type": item_type,
                "modified": format_mtime(getattr(entry, "mtime", None)),
                "size": getattr(entry, "size", None),
            })
    return files


def should_download(remote_path: str, include_small: bool) -> bool:
    lower = remote_path.lower()
    if lower.endswith(JSON_SUFFIXES):
        return True
    return include_small and lower.endswith(OPTIONAL_SMALL_SUFFIXES)


def download_file_cli(volume: str, remote_path: str, local_path: Path, force: bool) -> Dict[str, Any]:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists() and not force:
        return {"status": "exists", "local_path": str(local_path)}
    cp = run_modal(["volume", "get", "--force", volume, remote_path, str(local_path)], timeout=300)
    status = "ok" if local_path.exists() else "failed"
    return {
        "status": status,
        "returncode": cp.returncode,
        "local_path": str(local_path),
        "stdout": cp.stdout[-1000:],
        "stderr": cp.stderr[-1000:],
    }


def download_file_sdk(volume_obj: Any, remote_path: str, local_path: Path, force: bool) -> Dict[str, Any]:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists() and not force:
        return {"status": "exists", "local_path": str(local_path)}
    tmp_path = local_path.with_name(local_path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        with tmp_path.open("wb") as f:
            bytes_written = volume_obj.read_file_into_fileobj(remote_path.lstrip("/"), f)
        tmp_path.replace(local_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return {
        "status": "ok" if local_path.exists() else "failed",
        "local_path": str(local_path),
        "bytes_written": bytes_written,
    }


def as_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if out != out:
        return None
    return out


def normalize_rows(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if not isinstance(obj, dict):
        return []
    if isinstance(obj.get("rows"), list):
        return [x for x in obj["rows"] if isinstance(x, dict)]
    if isinstance(obj.get("row"), dict):
        return [obj["row"]]
    if isinstance(obj.get("suites"), dict):
        rows: List[Dict[str, Any]] = []
        for suite, suite_payload in obj["suites"].items():
            if not isinstance(suite_payload, dict):
                continue
            models = suite_payload.get("models", suite_payload)
            if not isinstance(models, dict):
                continue
            for model, metrics in models.items():
                if isinstance(metrics, dict):
                    row = dict(metrics)
                    row.setdefault("suite", suite)
                    row.setdefault("model", model)
                    rows.append(row)
        if rows:
            return rows
    if isinstance(obj.get("models"), dict):
        rows = []
        for model, payload in obj["models"].items():
            if not isinstance(payload, dict):
                continue
            metric_like = any(k in payload for k in ("success_rate", "avg_expansions", "successes"))
            if metric_like:
                row = dict(payload)
                row.setdefault("model", model)
                rows.append(row)
                continue
            for suite, metrics in payload.items():
                if isinstance(metrics, dict):
                    row = dict(metrics)
                    row.setdefault("model", model)
                    row.setdefault("suite", suite)
                    rows.append(row)
        if rows:
            return rows
    model_metric_rows = []
    for model, metrics in obj.items():
        if isinstance(metrics, dict) and any(k in metrics for k in ("success_rate", "successes", "avg_expansions")):
            row = dict(metrics)
            row.setdefault("model", model)
            model_metric_rows.append(row)
    return model_metric_rows


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[str(row.get("model") or row.get("display_name") or "?")].append(row)

    model_summaries = []
    for model, vals in sorted(by_model.items()):
        succ_values = [as_float(v.get("success_rate")) for v in vals]
        exp_values = [as_float(v.get("avg_expansions")) for v in vals]
        steps_values = [as_float(v.get("avg_steps")) for v in vals]
        succ_values = [v for v in succ_values if v is not None]
        exp_values = [v for v in exp_values if v is not None]
        steps_values = [v for v in steps_values if v is not None]
        model_summaries.append({
            "model": model,
            "rows": len(vals),
            "suites": sorted({str(v.get("suite")) for v in vals if v.get("suite") is not None}),
            "budgets": sorted({int(v.get("budget")) for v in vals if isinstance(v.get("budget"), int) or str(v.get("budget", "")).isdigit()}),
            "episodes": sorted({int(v.get("episodes")) for v in vals if isinstance(v.get("episodes"), int) or str(v.get("episodes", "")).isdigit()}),
            "mean_success_rate": round(sum(succ_values) / len(succ_values), 6) if succ_values else None,
            "mean_avg_expansions": round(sum(exp_values) / len(exp_values), 3) if exp_values else None,
            "mean_avg_steps": round(sum(steps_values) / len(steps_values), 3) if steps_values else None,
            "nonfinite_pred_count": sum(int(v.get("nonfinite_pred_count") or 0) for v in vals if str(v.get("nonfinite_pred_count") or "0").lstrip("-").isdigit()),
        })
    return {
        "row_count": len(rows),
        "model_count": len(model_summaries),
        "models": model_summaries,
    }


def summarize_json_file(path: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
    }
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception as exc:
        info["parse_error"] = str(exc)
        return info
    info["top_level_type"] = type(obj).__name__
    if isinstance(obj, dict):
        info["top_level_keys"] = list(obj.keys())[:30]
    rows = normalize_rows(obj)
    info["normalized"] = summarize_rows(rows)
    return info


def write_markdown(summary: Dict[str, Any], out_path: Path) -> None:
    lines = [
        "# Modal Experiment JSON Survey",
        "",
        f"Generated: {summary['generated_at']}",
        f"Downloaded files: {summary['downloaded_file_count']}",
        f"Parsed JSON files: {len(summary['json_summaries'])}",
        "",
        "## Volumes",
        "",
    ]
    for volume, count in sorted(summary["volume_file_counts"].items()):
        lines.append(f"- `{volume}`: {count} downloaded file(s)")
    lines.extend(["", "## JSON Summaries", ""])
    for item in summary["json_summaries"]:
        rel = item["path"]
        norm = item.get("normalized", {})
        lines.append(f"### `{rel}`")
        lines.append("")
        lines.append(f"- Size: {item.get('size_bytes')} bytes")
        if item.get("parse_error"):
            lines.append(f"- Parse error: `{item['parse_error']}`")
            lines.append("")
            continue
        lines.append(f"- Top-level type: `{item.get('top_level_type')}`")
        lines.append(f"- Normalized rows: {norm.get('row_count', 0)}")
        models = norm.get("models") or []
        if models:
            lines.append("")
            lines.append("| Model | Rows | Suites | Budgets | Episodes | Mean success | Mean avg expansions | Nonfinite preds |")
            lines.append("| --- | ---: | ---: | --- | --- | ---: | ---: | ---: |")
            for model in models[:30]:
                lines.append(
                    f"| `{model['model']}` | {model['rows']} | {len(model['suites'])} | "
                    f"{model['budgets']} | {model['episodes']} | {model['mean_success_rate']} | "
                    f"{model['mean_avg_expansions']} | {model['nonfinite_pred_count']} |"
                )
        lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="modal_downloads/full_survey")
    parser.add_argument("--volumes", default=",".join(DEFAULT_VOLUMES))
    parser.add_argument("--include-small", action="store_true", help="Also download .csv and .npz files.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--backend", choices=["auto", "sdk", "cli"], default="auto")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent downloads per volume.")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    volumes = [v.strip() for v in args.volumes.split(",") if v.strip()]

    manifest: List[Dict[str, Any]] = []
    for volume in volumes:
        print(f"[survey] walking {volume}", flush=True)
        sdk_volume = None
        use_sdk = args.backend in {"auto", "sdk"} and modal is not None
        if use_sdk:
            try:
                sdk_volume = modal.Volume.from_name(volume, create_if_missing=False)
                items = walk_volume_sdk(sdk_volume, volume, ["/"])
            except Exception as exc:
                if args.backend == "sdk":
                    items = [{
                        "volume": volume,
                        "remote_path": "/",
                        "type": "walk_error",
                        "error": str(exc),
                    }]
                else:
                    print(f"[survey] SDK walk failed for {volume}; falling back to CLI: {exc}", flush=True)
                    sdk_volume = None
                    items = walk_volume_cli(volume, ["/"])
        else:
            if args.backend == "sdk" and modal is None:
                items = [{
                    "volume": volume,
                    "remote_path": "/",
                    "type": "walk_error",
                    "error": "Modal SDK is not importable",
                }]
            else:
                items = walk_volume_cli(volume, ["/"])

        volume_items: List[Dict[str, Any]] = []
        jobs: List[Tuple[int, str, Path]] = []
        for item in items:
            item = dict(item)
            remote = item.get("remote_path", "")
            if item.get("type") in {"ls_error", "walk_error"}:
                volume_items.append(item)
                continue
            item["download"] = should_download(remote, args.include_small)
            if item["download"] and not args.manifest_only:
                local = safe_local_path(out_dir, volume, remote)
                item["local_path"] = str(local)
                jobs.append((len(volume_items), remote, local))
            volume_items.append(item)

        def download_job(job: Tuple[int, str, Path]) -> Tuple[int, Dict[str, Any]]:
            idx, remote_path, local_path = job
            try:
                if sdk_volume is not None:
                    result = download_file_sdk(sdk_volume, remote_path, local_path, args.force)
                else:
                    result = download_file_cli(volume, remote_path, local_path, args.force)
            except Exception as exc:
                result = {"status": "failed", "error": str(exc)}
            return idx, result

        if jobs:
            print(f"[survey] downloading {len(jobs)} file(s) from {volume} with {max(1, args.workers)} worker(s)", flush=True)
            completed = 0
            with cf.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
                futures = [pool.submit(download_job, job) for job in jobs]
                for future in cf.as_completed(futures):
                    idx, result = future.result()
                    volume_items[idx]["download_result"] = result
                    completed += 1
                    if completed == 1 or completed == len(jobs) or completed % max(1, args.progress_every) == 0:
                        print(
                            f"[survey] {volume}: {completed}/{len(jobs)} downloaded "
                            f"(latest {result.get('status')})",
                            flush=True,
                        )

        manifest.extend(volume_items)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    json_summaries = []
    for path in sorted(out_dir.glob("**/*.json")):
        if path.name in {"manifest.json", "summary.json"}:
            continue
        json_summaries.append(summarize_json_file(path))

    volume_file_counts: Dict[str, int] = defaultdict(int)
    for item in manifest:
        if item.get("download") and item.get("local_path"):
            volume_file_counts[item["volume"]] += 1

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "downloaded_file_count": sum(volume_file_counts.values()),
        "volume_file_counts": dict(volume_file_counts),
        "json_summaries": json_summaries,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, out_dir / "summary.md")
    print(f"[survey] wrote {manifest_path}")
    print(f"[survey] wrote {out_dir / 'summary.json'}")
    print(f"[survey] wrote {out_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
