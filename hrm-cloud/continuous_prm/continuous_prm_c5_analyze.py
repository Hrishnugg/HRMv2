#!/usr/bin/env python
"""Analyze C5 continuous PRM hard-map results.

The summary CSV is enough for success-rate deltas and a quick two-proportion
screen. When the raw evaluation CSV is also available, this script additionally
uses paired McNemar tests over matching ``suite/world_index/budget`` episodes.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


Z_95 = 1.959963984540054


def parse_float(value: Any, default: float = float("nan")) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if text == "":
        return default
    try:
        return float(text)
    except ValueError:
        return default


def parse_int(value: Any, default: int = 0) -> int:
    val = parse_float(value, float(default))
    if not math.isfinite(val):
        return default
    return int(round(val))


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def method_label(row: Dict[str, Any]) -> str:
    method = str(row.get("method", ""))
    backbone = str(row.get("backbone", ""))
    expert_task = str(row.get("expert_task", ""))
    alpha = parse_float(row.get("alpha"), 0.0)
    if method == "euclidean":
        return "euclidean"
    parts = [method]
    if backbone:
        parts.append(backbone)
    if expert_task:
        parts.append(expert_task)
    if alpha:
        parts.append(f"alpha={alpha:g}")
    return "/".join(parts)


def summary_key(row: Dict[str, Any]) -> Tuple[str, str, str, str, str, int]:
    return (
        str(row.get("suite", "")),
        str(row.get("method", "")),
        str(row.get("backbone", "")),
        str(row.get("expert_task", "")),
        f"{parse_float(row.get('alpha'), 0.0):.12g}",
        parse_int(row.get("budget")),
    )


def euclidean_key(suite: str, budget: int) -> Tuple[str, str, str, str, str, int]:
    return (suite, "euclidean", "", "", "0", budget)


def wilson_interval(successes: int, n: int, z: float = Z_95) -> Tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    phat = successes / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt((phat * (1.0 - phat) / n) + (z * z) / (4.0 * n * n))
    return max(0.0, center - margin), min(1.0, center + margin)


def two_prop_z_pvalue(x1: int, n1: int, x0: int, n0: int) -> float:
    if n1 <= 0 or n0 <= 0:
        return float("nan")
    p1 = x1 / n1
    p0 = x0 / n0
    pooled = (x1 + x0) / (n1 + n0)
    se = math.sqrt(max(0.0, pooled * (1.0 - pooled) * ((1.0 / n1) + (1.0 / n0))))
    if se == 0.0:
        return 1.0 if p1 == p0 else 0.0
    z = (p1 - p0) / se
    return math.erfc(abs(z) / math.sqrt(2.0))


def exact_mcnemar_pvalue(b: int, c: int) -> float:
    n = b + c
    if n <= 0:
        return 1.0
    tail = min(b, c)
    prob = sum(math.comb(n, i) for i in range(tail + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * prob)


def bh_qvalues(pvalues: List[float]) -> List[float]:
    indexed = [(i, p) for i, p in enumerate(pvalues) if math.isfinite(p)]
    out = [float("nan")] * len(pvalues)
    if not indexed:
        return out
    indexed.sort(key=lambda x: x[1])
    m = len(indexed)
    prev = 1.0
    for rank_from_end, (i, p) in enumerate(reversed(indexed), start=1):
        rank = m - rank_from_end + 1
        q = min(prev, p * m / rank)
        out[i] = q
        prev = q
    return out


def build_raw_lookup(raw_rows: Iterable[Dict[str, str]]) -> Dict[Tuple[str, str, str, str, str, int], Dict[int, int]]:
    lookup: Dict[Tuple[str, str, str, str, str, int], Dict[int, int]] = {}
    for row in raw_rows:
        key = summary_key(row)
        world_index = parse_int(row.get("world_index"), -1)
        if world_index < 0:
            continue
        lookup.setdefault(key, {})[world_index] = parse_int(row.get("found"), 0)
    return lookup


def compare_rows(
    summary_rows: List[Dict[str, str]],
    raw_lookup: Optional[Dict[Tuple[str, str, str, str, str, int], Dict[int, int]]],
    target_min: float,
    target_max: float,
    min_practical_delta: float,
) -> List[Dict[str, Any]]:
    by_key = {summary_key(row): row for row in summary_rows}
    out: List[Dict[str, Any]] = []

    for row in summary_rows:
        method = str(row.get("method", ""))
        if method == "euclidean":
            continue
        suite = str(row.get("suite", ""))
        budget = parse_int(row.get("budget"))
        base = by_key.get(euclidean_key(suite, budget))
        if not base:
            continue

        n = parse_int(row.get("episodes"))
        base_n = parse_int(base.get("episodes"))
        rate = parse_float(row.get("success_rate"))
        base_rate = parse_float(base.get("success_rate"))
        successes = parse_int(rate * n)
        base_successes = parse_int(base_rate * base_n)
        lo, hi = wilson_interval(successes, n)
        base_lo, base_hi = wilson_interval(base_successes, base_n)
        z_p = two_prop_z_pvalue(successes, n, base_successes, base_n)

        paired_n = 0
        b = 0
        c = 0
        paired_delta = float("nan")
        paired_p = float("nan")
        if raw_lookup is not None:
            learned_raw = raw_lookup.get(summary_key(row), {})
            base_raw = raw_lookup.get(euclidean_key(suite, budget), {})
            common_worlds = sorted(set(learned_raw).intersection(base_raw))
            paired_n = len(common_worlds)
            for world_index in common_worlds:
                learned_found = int(learned_raw[world_index])
                base_found = int(base_raw[world_index])
                if learned_found == 1 and base_found == 0:
                    b += 1
                elif learned_found == 0 and base_found == 1:
                    c += 1
            if paired_n:
                paired_delta = (b - c) / paired_n
                paired_p = exact_mcnemar_pvalue(b, c)

        p_for_q = paired_p if math.isfinite(paired_p) else z_p
        exp_mean = parse_float(row.get("expansions_mean"))
        base_exp_mean = parse_float(base.get("expansions_mean"))
        out.append({
            "suite": suite,
            "budget": budget,
            "baseline_success": base_rate,
            "method_success": rate,
            "delta": rate - base_rate,
            "baseline_episodes": base_n,
            "method_episodes": n,
            "method": method_label(row),
            "method_raw": method,
            "backbone": str(row.get("backbone", "")),
            "expert_task": str(row.get("expert_task", "")),
            "alpha": parse_float(row.get("alpha"), 0.0),
            "baseline_wilson_low": base_lo,
            "baseline_wilson_high": base_hi,
            "method_wilson_low": lo,
            "method_wilson_high": hi,
            "two_prop_p": z_p,
            "paired_worlds": paired_n,
            "paired_gain_only": b,
            "paired_loss_only": c,
            "paired_delta": paired_delta,
            "mcnemar_p": paired_p,
            "p_for_q": p_for_q,
            "baseline_expansions_mean": base_exp_mean,
            "method_expansions_mean": exp_mean,
            "expansions_delta": exp_mean - base_exp_mean if math.isfinite(exp_mean) and math.isfinite(base_exp_mean) else float("nan"),
            "baseline_in_target_band": target_min <= base_rate <= target_max,
            "practical_delta": (rate - base_rate) >= min_practical_delta,
            "deployable_method": method != "oracle_tasklora",
            "full_episode_comparison": bool(n == base_n and (raw_lookup is None or paired_n == base_n)),
            "nonfinite_heuristic_rows": parse_int(row.get("nonfinite_heuristic_rows")),
            "heuristic_max_mean": parse_float(row.get("heuristic_max_mean")),
            "delta_mean_mean": parse_float(row.get("delta_mean_mean")),
            "correction_abs_mean": parse_float(row.get("correction_abs_mean")),
        })

    qvals = bh_qvalues([parse_float(row["p_for_q"]) for row in out])
    for row, q in zip(out, qvals):
        row["bh_q"] = q
        row["significant_q05"] = bool(math.isfinite(q) and q <= 0.05)
        row["target_claim_candidate"] = bool(
            row["deployable_method"]
            and row["full_episode_comparison"]
            and row["baseline_in_target_band"]
            and row["practical_delta"]
            and row["significant_q05"]
        )

    out.sort(key=lambda r: (not r["target_claim_candidate"], not r["baseline_in_target_band"], r["suite"], r["budget"], -r["delta"], r["method"]))
    return out


def fmt_float(value: Any, digits: int = 3) -> str:
    val = parse_float(value)
    if not math.isfinite(val):
        return "nan"
    return f"{val:.{digits}f}"


def fmt_p(value: Any) -> str:
    val = parse_float(value)
    if not math.isfinite(val):
        return "nan"
    if val < 0.001:
        return f"{val:.2e}"
    return f"{val:.3f}"


def markdown_table(rows: List[Dict[str, Any]], limit: int) -> str:
    fields = [
        ("suite", "Suite"),
        ("budget", "Budget"),
        ("baseline_success", "Euclid"),
        ("method", "Method"),
        ("method_success", "Method"),
        ("delta", "Delta"),
        ("paired_gain_only", "Gain"),
        ("paired_loss_only", "Loss"),
        ("mcnemar_p", "McNemar p"),
        ("two_prop_p", "Z p"),
        ("bh_q", "BH q"),
        ("expansions_delta", "Exp Delta"),
    ]
    lines = ["|" + "|".join(label for _, label in fields) + "|", "|" + "|".join("---" for _ in fields) + "|"]
    for row in rows[:limit]:
        vals: List[str] = []
        for key, _ in fields:
            val = row.get(key, "")
            if key in {"baseline_success", "method_success", "delta", "expansions_delta"}:
                vals.append(fmt_float(val))
            elif key in {"mcnemar_p", "two_prop_p", "bh_q"}:
                vals.append(fmt_p(val))
            else:
                vals.append(str(val))
        lines.append("|" + "|".join(vals) + "|")
    return "\n".join(lines)


def write_markdown(path: Path, comparisons: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    target_rows = [r for r in comparisons if r["baseline_in_target_band"]]
    claim_rows = [r for r in comparisons if r["target_claim_candidate"]]
    sig_rows = [r for r in comparisons if r["significant_q05"] and r["delta"] > 0]

    lines = [
        "# C5 Hard-Map Significance Analysis",
        "",
        "## Inputs",
        "",
        f"- Summary CSV: `{Path(args.summary_csv).as_posix()}`",
        f"- Raw CSV: `{Path(args.raw_csv).as_posix() if args.raw_csv else '<not provided>'}`",
        f"- Target Euclidean band: {args.target_min:g}-{args.target_max:g}",
        f"- Minimum practical delta: {args.min_practical_delta:g}",
        "",
        "## Claim Candidates",
        "",
    ]
    if claim_rows:
        lines.append(markdown_table(claim_rows, args.markdown_limit))
    else:
        lines.append("No method met the target-band, practical-delta, and BH-q<=0.05 filters.")
    lines.extend([
        "",
        "## Target-Band Comparisons",
        "",
    ])
    if target_rows:
        lines.append(markdown_table(target_rows, args.markdown_limit))
    else:
        lines.append("No Euclidean baseline rows fell in the requested target band.")
    lines.extend([
        "",
        "## Positive Significant Comparisons",
        "",
    ])
    if sig_rows:
        lines.append(markdown_table(sig_rows, args.markdown_limit))
    else:
        lines.append("No positive method-vs-Euclidean comparisons were significant after BH correction.")
    lines.extend([
        "",
        "## Notes",
        "",
        "- Claim candidates exclude `oracle_tasklora`, which is a post-hoc diagnostic rather than a deployable heuristic.",
        "- Claim candidates require the method and Euclidean rows to cover the same episode count; with raw rows, they also require all worlds to be paired.",
        "- `McNemar p` is the preferred test when raw rows are available because each method is evaluated on the same worlds.",
        "- `Z p` is a summary-level two-proportion screen and is less appropriate than paired testing for these paired evaluations.",
        "- `BH q` applies Benjamini-Hochberg correction across all method-vs-Euclidean comparisons in this report.",
        "- `Exp Delta` is method mean expansions minus Euclidean mean expansions; negative values mean fewer expansions.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary-csv", type=Path, required=True)
    p.add_argument("--raw-csv", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--target-min", type=float, default=0.50)
    p.add_argument("--target-max", type=float, default=0.70)
    p.add_argument("--min-practical-delta", type=float, default=0.10)
    p.add_argument("--markdown-limit", type=int, default=40)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summary_rows = read_csv(args.summary_csv)
    raw_lookup = build_raw_lookup(read_csv(args.raw_csv)) if args.raw_csv else None
    comparisons = compare_rows(
        summary_rows,
        raw_lookup=raw_lookup,
        target_min=float(args.target_min),
        target_max=float(args.target_max),
        min_practical_delta=float(args.min_practical_delta),
    )

    out_dir = args.out_dir or args.summary_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "continuous_prm_c5_significance.csv"
    md_path = out_dir / "continuous_prm_c5_significance.md"
    write_csv(csv_path, comparisons)
    write_markdown(md_path, comparisons, args)

    target_rows = [r for r in comparisons if r["baseline_in_target_band"]]
    claim_rows = [r for r in comparisons if r["target_claim_candidate"]]
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    print(f"comparisons={len(comparisons)} target_band={len(target_rows)} claim_candidates={len(claim_rows)}")
    if target_rows:
        print()
        print(markdown_table(target_rows, min(12, args.markdown_limit)))


if __name__ == "__main__":
    main()
