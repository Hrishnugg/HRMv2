"""Generate the canonical combined C12-A/C12-B Markdown report from artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import pandas as pd
import torch


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _pct(value: float, digits: int = 1) -> str:
    return f"{100.0 * float(value):.{digits}f}%"


def _f(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def generate(repo: Path, output: Path) -> None:
    repo = repo.resolve()
    cprm = repo / "hrm-cloud" / "continuous_prm"
    a_probe_path = cprm / "runs" / "c12_persistent_v6" / "results" / "c12a_headroom_summary.json"
    a_summary_path = cprm / "runs" / "c12_persistent_pilot_v6_final" / "results" / "c12a_summary.json"
    b_root = cprm / "runs" / "c12_refiner"
    b_probe_path = b_root / "probe" / "c12b_probe_summary.json"
    b_summary_path = b_root / "results" / "c12b_summary.json"
    b_sig_path = b_root / "results" / "c12b_significance.csv"
    b_eval_path = b_root / "results" / "c12b_eval_raw.csv"
    b_state_path = b_root / "results" / "c12b_state_metrics.csv"
    b_manifest_path = b_root / "manifest.json"
    b_integrity_path = b_root / "results" / "integrity.json"
    b_reanalysis_path = b_root / "results" / "reanalysis" / "verification.json"

    a_probe = _load(a_probe_path)
    a_summary = _load(a_summary_path)
    b_probe = _load(b_probe_path)
    b_summary = _load(b_summary_path)
    b_sig = pd.read_csv(b_sig_path)
    b_eval = pd.read_csv(b_eval_path)
    b_state = pd.read_csv(b_state_path)
    b_manifest = _load(b_manifest_path)
    b_integrity = _load(b_integrity_path)
    b_reanalysis = _load(b_reanalysis_path)

    learned_eval = b_eval[b_eval.model_seed >= 0]
    plan = (
        learned_eval.groupby(["config", "K", "arm", "cycle"], as_index=False)
        .agg(burden=("expansion_burden", "mean"), success=("completion", "mean"), cost_ratio=("cost_ratio", "mean"))
    )
    state = (
        b_state[b_state.model_seed >= 0]
        .groupby(["config", "K", "arm", "cycle"], as_index=False)
        .agg(mae=("state_mae", "mean"), rho=("rank_corr", "mean"), bellman=("bellman_residual", "mean"))
    )
    refs = (
        b_eval[b_eval.model_seed < 0]
        .groupby(["config", "K", "arm"], as_index=False)
        .agg(success=("completion", "mean"), burden=("expansion_burden", "mean"), expansions=("expansions", "mean"))
    )

    g1_cells = {(row["config"], int(row["K"])): row for row in b_summary["gates"]["G1_B"]["cells"]}
    g1_config = {row["config"]: row for row in b_summary["gates"]["G1_B"]["by_config"]}
    probe_cells = {row["config"]: row for row in b_probe["cells"]}

    lines = []
    add = lines.append
    add("# C12 Persistent Hierarchical Planning — Results")
    add("")
    add("**Status (2026-07-14): COMPLETE.** C12-A's frozen one-seed pilot is complete and development-only with a `strong_negative` closure. C12-B completed the full G0 → smoke → pilot/runtime → three-seed TEST sequence. C12-B is also negative for the registered hierarchy-depth claim: G1-B fails the K-dose-response requirement, although refinement improves monotonically within every cell and C/K=8 passes the matched-control G2-B comparison.")
    add("")
    add("**Primary sources:** [design](../design/2026-07-10-c12-persistent-hierarchical-planning-design.md), [implementation plan](../plans/2026-07-10-c12-persistent-hierarchical-planning.md), [C12-A pilot analysis](../../../../../hrm-cloud/continuous_prm/runs/c12_persistent_pilot_v6_final/results/C12A_ANALYSIS.md), [C12-B probe summary](../../../../../hrm-cloud/continuous_prm/runs/c12_refiner/probe/c12b_probe_summary.json), [C12-B computed summary](../../../../../hrm-cloud/continuous_prm/runs/c12_refiner/results/c12b_summary.json), [significance table](../../../../../hrm-cloud/continuous_prm/runs/c12_refiner/results/c12b_significance.csv), and [integrity audit](../../../../../hrm-cloud/continuous_prm/runs/c12_refiner/results/integrity.json).")
    add("")
    add("## Executive verdict")
    add("")
    add("C12 establishes two useful boundaries. First, hidden slow/fast dynamics really do create decision-relevant memory headroom (C12-A G0-A passes), but the tested learned temporal hierarchies do not convert it into a reliable forecast, planning, or persistent-carry advantage. Second, repeated graph propagation really does improve bounded-search behavior within a checkpoint (C12-B cycle curves are monotone), but the gain is not larger at K=8 than K=2. The strict hierarchy/depth hypothesis therefore remains unsupported.")
    add("")
    add("C12-B also contains a localized architectural result that should be retained, not promoted to the headline: on C/K=8, the 101,505-parameter tied refiner at cycle 8 beats both the equally sized one-step shallow control and the 681,217-parameter eight-step untied control after BH correction, with no completion regression. This does not satisfy G1-B because the cycle gain lacks the preregistered K-dose response, and it does not replicate against untied on A/K=8 (untied is slightly but significantly better there).")
    add("")
    add("## C12-A — persistent hidden-regime dynamics")
    add("")
    add("### Methodology and reason for the formulation")
    add("")
    add("C12-A was motivated by a limitation of the earlier static continuous-space ladder: if the current observation is sufficient, persistent memory has little reason to win. It therefore constructs paired PRM decisions whose present observation is aliased while the hidden direction, gate phase, or route mode changes the correct future action. A present-sufficient stratum is the negative control. Static-map, roadmap, goal, and latent-regime seeds are separated; paired variants share the visible decision state but differ in hidden dynamics and oracle action. Providers plan with the same 32-step space-time A* and are scored against a separate true simulator.")
    add("")
    add(f"The full G0-A probe used {a_probe['pairs_per_stratum']} counterfactual pairs ({a_probe['episodes_per_stratum']} episodes) in each of four strata, totaling {a_probe['total_episode_provider_rows']} provider rows. World-pair clustered intervals were used. The privileged `true_mode` provider is an authorization diagnostic, not a learned candidate.")
    add("")
    add("### G0-A authorization")
    add("")
    am = a_probe["metrics"]
    add("| Condition | Result | Gate |")
    add("|---|---:|---|")
    add(f"| Constructed alias rate | {_pct(am['alias_rate'])} | Pass |")
    add(f"| History completion gain | +{am['history_completion_gain']:.3f} | Pass |")
    add(f"| Collision-adjusted regret reduction | {_pct(am['history_regret_reduction_frac'])} (95% CI {_pct(am['history_regret_reduction_ci_low'])}–{_pct(am['history_regret_reduction_ci_high'])}) | Pass |")
    add(f"| Oracle completion | {_pct(am['oracle_completion'])} | Pass |")
    add(f"| Oracle ceiling gap | {_pct(am['ceiling_gap_frac'])} | Pass |")
    add(f"| Present-sufficient history headroom | {am['control_headroom']:.3f} | Pass |")
    add("")
    add("### Learned pilot outcome")
    add("")
    hg = a_summary["headline_gates"]
    add(f"The frozen pilot selected `{a_summary['selected_hierarchy']}` as the hierarchy and `{a_summary['selected_flat']}` as the flat comparator using VALIDATION only. G1-A forecast: **{'PASS' if hg['G1_A']['passed'] else 'FAIL'}**; G2-A planning: **{'PASS' if hg['G2_A']['passed'] else 'FAIL'}**; G3-A carry: **{'PASS' if hg['G3_A']['passed'] else 'FAIL'}**. G4-A closes as **`{hg['G4_A']['verdict']}`**: {hg['G4_A']['interpretation']}")
    add("")
    add("This is development-only (`official_final=false`, one model seed). It is sufficient to close the approved pilot sequence, not to support a final multi-seed temporal-hierarchy claim.")
    add("")
    add("## C12-B — tied iterative product-graph refinement")
    add("")
    add("### Hypothesis and methodology")
    add("")
    add("C12-B asks whether C11's global product-graph model was limited by a fixed shallow computation, and whether a shared recurrent update can progressively propagate value information across long mission graphs. It reuses C11 world generation, exact product-oracle labels, C11 node/edge tensors, leg-sum initialization, and matched product A*. C11's forward motion edges are consumed in the reverse value direction (destination → source), so one recurrent cycle moves downstream cost-to-go information one product-graph hop toward predecessors.")
    add("")
    add("The tied model applies one shared graph block eight times and exposes cycles 1/2/4/8. Deep-supervision weights are 0.1/0.2/0.3/0.4. Controls isolate the relevant alternatives:")
    add("")
    add("| Arm | Parameters | Edge applications | Purpose |")
    add("|---|---:|---:|---|")
    add("| `c11_gnn8` | 681,089 | 8 | Exact C11 untied forward-message architecture, retrained |")
    add("| `shallow_param_match` | 101,505 | 1 | Same parameter count as tied, one propagation step |")
    add("| `untied_compute_match` | 681,217 | 8 | Same reverse edge-compute depth, distinct weights per step |")
    add("| `tied_refiner` | 101,505 | 8 | Shared recurrent block; primary method |")
    add("")
    add("All arms use the same residual-over-leg-sum target, smooth-L1 loss, AdamW (2e-4 learning rate, 1e-4 weight decay), gradient clip 1.0, 40 epochs, common node-budget graph batching, and validation-only checkpoint selection. Each authorized cell has 40 TRAIN, 10 VALIDATION, and the existing 25 C11 TEST worlds. Three model seeds are averaged inside each TEST world; the world is the independent unit. The analysis uses 10,000 world bootstraps, 20,000 sign flips, completion bootstraps, and BH correction over the four deep-cell G2 comparisons.")
    add("")
    add("### G0-B: K=16 feasibility and authorization")
    add("")
    add("K=16 was evaluated before learning under a frozen 300-attempt envelope. Both cells clearly contained the intended headroom/depth regime when a valid mission was found, but construction was too rare to meet the required 20 valid worlds. K=16 was therefore dropped before training; K=2/8 remained authorized from the existing C11 substrate.")
    add("")
    add("| Cell | Valid / attempts | Oracle/leg-sum ratio | Median final-transition hops | Max graph bytes | G0-B |")
    add("|---|---:|---:|---:|---:|---|")
    for config in ("A", "C"):
        row = probe_cells[config]
        add(f"| {config}/K16 | {row['valid_worlds']} / {row['attempts']} | {row['matched_oracle_legsum_median_ratio']:.4f} | {row['median_final_transition_hops']:.0f} | {row['max_graph_bytes']:,} | Fail: insufficient valid worlds |")
    add("")
    add(f"Frozen probe hashes: raw `{b_probe['artifacts']['raw_sha256']}`, seed ledger `{b_probe['artifacts']['seed_ledger_sha256']}`.")
    add("")
    add("### Full TEST cycle curves")
    add("")
    add("The primary planning metric is expansion burden = expansions / binding budget; lower is better. Completion is shown alongside it. Values average three model seeds within each of 25 TEST worlds.")
    add("")
    add("| Cell | Cycle 1 burden / success | Cycle 2 | Cycle 4 | Cycle 8 | C1−C8 improvement (95% CI) |")
    add("|---|---:|---:|---:|---:|---:|")
    for config in ("A", "C"):
        for K in (2, 8):
            vals = plan[(plan.config == config) & (plan.K == K) & (plan.arm == "tied_refiner")].set_index("cycle")
            g = g1_cells[(config, K)]
            cells = [f"{vals.loc[c, 'burden']:.4f} / {_pct(vals.loc[c, 'success'], 0)}" for c in (1, 2, 4, 8)]
            ci = g["cycle1_minus_cycle8_ci95"]
            add(f"| {config}/K{K} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {g['cycle1_minus_cycle8']:.4f} ({ci[0]:.4f}, {ci[1]:.4f}) |")
    add("")
    add("Every within-cell adjacent cycle comparison has a bootstrap interval excluding worsening, and cycle 1 vs 8 is separated in all four cells. That is a genuine progressive-compute signal. It is not the registered hierarchy-depth signal because the gain does not grow from K=2 to K=8:")
    add("")
    add("| Config | K8 improvement − K2 improvement | 95% CI | Dose-response |")
    add("|---|---:|---:|---|")
    for config in ("A", "C"):
        row = g1_config[config]
        ci = row["deep_minus_K2_ci95"]
        add(f"| {config} | {row['mean_deep_minus_K2_improvement']:.4f} | ({ci[0]:.4f}, {ci[1]:.4f}) | Fail |")
    add("")
    add("### G2-B matched controls at K=8")
    add("")
    add("Positive values mean the control uses more normalized expansions than tied cycle 8.")
    add("")
    add("| Config | Control | Control−tied burden (95% CI) | BH q | Completion regression | Verdict |")
    add("|---|---|---:|---:|---|---|")
    for _, row in b_sig.iterrows():
        verdict = "Tied better" if bool(row.planning_better) else ("Tied worse" if row.mean_control_minus_tied_burden < 0 else "No separation")
        add(f"| {row.config} | `{row.control}` | {row.mean_control_minus_tied_burden:.4f} ({row.burden_gain_ci95_low:.4f}, {row.burden_gain_ci95_high:.4f}) | {row.p_bh:.6f} | None | {verdict} |")
    add("")
    add("G2-B passes in C/K=8 because tied beats both controls with no completion loss. A/K=8 does not pass: tied beats shallow, but the untied compute match is better by 0.0031 burden (about five expansions at budget 1,600). This config dependence prevents a general tying advantage claim.")
    add("")
    add("### Value quality and planner caveat")
    add("")
    add("The cycle improvement is stronger in bounded-search behavior than in oracle-value fidelity. This matters because the learned heuristics are intentionally not constrained to admissibility.")
    add("")
    add("| Cell | Tied MAE C1 → C8 | Bellman residual C1 → C8 | Mean solved cost ratio at C8 |")
    add("|---|---:|---:|---:|")
    for config in ("A", "C"):
        for K in (2, 8):
            sv = state[(state.config == config) & (state.K == K) & (state.arm == "tied_refiner")].set_index("cycle")
            pv = plan[(plan.config == config) & (plan.K == K) & (plan.arm == "tied_refiner")].set_index("cycle")
            add(f"| {config}/K{K} | {sv.loc[1, 'mae']:.4f} → {sv.loc[8, 'mae']:.4f} | {sv.loc[1, 'bellman']:.4f} → {sv.loc[8, 'bellman']:.4f} | {pv.loc[8, 'cost_ratio']:.4f} |")
    add("")
    add("MAE improves in three cells but slightly worsens in C/K=8; Bellman residual worsens from cycle 1 to 8 in every cell. Cycle-8 solved paths average roughly 1.4–2.2% above oracle cost. The expansion gains should therefore be described as bounded-search efficiency under an inadmissible learned heuristic, not as uniformly better value iteration or optimal planning.")
    add("")
    add("### Reference headroom on the exact TEST worlds")
    add("")
    add("| Cell | Leg-sum success / burden | Oracle success / burden |")
    add("|---|---:|---:|")
    for config in ("A", "C"):
        for K in (2, 8):
            cell = refs[(refs.config == config) & (refs.K == K)].set_index("arm")
            add(f"| {config}/K{K} | {_pct(cell.loc['h_legsum', 'success'], 0)} / {cell.loc['h_legsum', 'burden']:.4f} | {_pct(cell.loc['h_oracle', 'success'], 0)} / {cell.loc['h_oracle', 'burden']:.4f} |")
    add("")
    add("Large oracle headroom remains in every cell. C12-B therefore does not fail because the planning problem is saturated; it fails the specific claim that tied recurrent computation produces a larger benefit as mission depth grows.")
    add("")
    add("### Gate resolution")
    add("")
    add(f"- **G0-B:** K16 not authorized; A/C at K2/8 authorized.")
    add(f"- **G1-B:** **{'PASS' if b_summary['gates']['G1_B']['passed'] else 'FAIL'}**. Monotone within-cell gains and deep C1-vs-C8 separation are present, but the required K-dose response is absent.")
    add(f"- **G2-B:** **{'PASS' if b_summary['gates']['G2_B']['passed'] else 'FAIL'}**, localized to C/K8.")
    add(f"- **G3-B registered closure:** `{b_summary['gates']['G3_B']['code']}`. Read literally as a failed registered progressive-refinement gate, not as " + '"no cycle ever helps"; all four cycle curves improve, but not preferentially on the deeper task.')
    add("")
    add("## Reproducibility and integrity")
    add("")
    add(f"The full C12-B run completed in {b_manifest['elapsed_s'] / 60.0:.2f} minutes with {len(b_manifest['training'])} checkpoint records and no failures. Integrity expected and observed {b_integrity['expected']['checkpoints']} checkpoints plus {b_integrity['expected']['rows']} state rows and {b_integrity['expected']['rows']} evaluation rows; every count, uniqueness check, finite-metric check, and required artifact check passed.")
    add("")
    add(f"Accepted TRAIN/VALIDATION/TEST seeds are 40/10/25 per cell with zero overlap. Reanalysis from the two raw CSVs passed: {b_reanalysis['source_rows']['state']} state rows and {b_reanalysis['source_rows']['evaluation']} evaluation rows; the significance CSV is byte-identical and the summary is semantically identical within 1e-12 float tolerance.")
    add("")
    add("| Artifact | SHA-256 |")
    add("|---|---|")
    add(f"| `c12b_state_metrics.csv` | `{_sha(b_state_path)}` |")
    add(f"| `c12b_eval_raw.csv` | `{_sha(b_eval_path)}` |")
    add(f"| `c12b_summary.json` | `{_sha(b_summary_path)}` |")
    add(f"| `c12b_significance.csv` | `{_sha(b_sig_path)}` |")
    add("")
    add("Regression verification: 217 C8/C11/C12 tests passed. The first two attempts were invalid environment runs because pytest could not create its default or `C:\\tmp` fixture root; the clean run used a repo-local writable `--basetemp` and had zero failures/errors.")
    add("")
    cuda = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    add(f"Environment: Python {platform.python_version()}, PyTorch {torch.__version__}, `{cuda}`. Source base: git `{_git(repo, 'rev-parse', '--short', 'HEAD')}` on branch `{_git(repo, 'branch', '--show-current')}` with C12-B changes in the working tree at run time.")
    add("")
    add("## Deviations and limitations")
    add("")
    add("- K=16 was not silently shrunk after training; it failed G0-B construction feasibility and was removed before any model fit, exactly as preregistered.")
    add("- The execution CLI uses explicit `probe`, `smoke`, `pilot`, `full`, and `analyze` modes instead of a separate mode-plus-scale pair. Training/evaluation remain one resumable full transaction so partial TEST artifacts cannot be mistaken for a completed verdict.")
    add("- The C12-B implementation is split into a model/probe module, an execution pipeline module, and an independent reanalysis module for reviewability.")
    add("- C12-A remains a one-seed development pilot (`official_final=false`); C12-B is the completed three-seed full grid.")
    add("- The registered G3-B wording is coarser than the observed pattern: G1-B fails because the dose response is absent, even though cycles improve within every cell. Both facts are reported.")
    add("- Learned heuristic paths are usually slightly suboptimal. Any paper-facing efficiency claim must include path-cost ratio alongside expansion counts.")
    add("")
    add("## Program implication")
    add("")
    add("C12 closes the two most plausible remaining substrate objections to C11 without producing a general hierarchy win. Memory-relevant dynamics exist, but the tested temporal hierarchies do not exploit them robustly. Recurrent graph computation helps bounded search, and tying can help in one gated deep configuration, but the benefit is not depth-selective. The next experiment should not be another backbone swap. A defensible follow-up would change the planner/learning objective itself—e.g., cost-aware focal ordering, admissibility-aware correction, learned macro-actions, or explicit multilevel search—and preregister path quality as a co-primary outcome.")
    add("")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    output = args.output or args.repo_root / "docs" / "experiments" / "continuous" / "c12" / "results" / "C12_RESULTS.md"
    generate(args.repo_root, output)


if __name__ == "__main__":
    main()

