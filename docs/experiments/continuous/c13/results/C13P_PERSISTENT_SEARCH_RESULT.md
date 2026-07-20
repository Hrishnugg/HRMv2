# C13-P result: persistent HRM search state

**Preregistration frozen:** 2026-07-19
**Completed:** 2026-07-20
**Design:** [2026-07-19-c13p-persistent-search-state.md](../design/2026-07-19-c13p-persistent-search-state.md)
**Implementation plan:** [2026-07-19-c13p-persistent-search-state.md](../plans/2026-07-19-c13p-persistent-search-state.md)
**Generated report:** [C13P_RESULT.md](../../../../../hrm-cloud/continuous_prm/runs/c13_persistent_search/results/C13P_RESULT.md)
**Verdict:** `c13p_no_persistent_ranking_signal`
**Self-bootstrap:** not authorized and not run
**Untouched confirmation:** not authorized and not run

## Bottom line

The persistent-carry arm does **not** satisfy the preregistered offline ranking
gate against the same selected checkpoint with its carry reset at every event.
The study is mechanically valid: G0-P passes, duplicate traces and deterministic
result projections match, the independent raw-artifact reanalysis reproduces
every gate primitive, and a separate `--verify-only` pass exits successfully.

The primary comparison is persistent minus reset world-macro MRR:

- point estimate: `-0.029419`;
- world-clustered 95% CI: `[-0.059848, -0.003029]`;
- world-macro top-1 delta: `-0.027962`;
- suites with positive MRR delta: `3/6`.

Because G1-P fails, the frozen final verdict is
`c13p_no_persistent_ranking_signal`. G2-P is reported descriptively but cannot
rescue or supersede the failed offline mechanism gate.

The claim-safe conclusion is:

> Under the frozen stationary path-frontier target, causal event representation,
> recurrent architecture, and training protocol, persistent HRM carry does not
> improve held-out frontier ranking over the identical checkpoint with per-event
> reset state. This is a negative result for this C13-P mechanism instantiation,
> not a general rejection of HRM, memory, or stateful search.

## Frozen intervention

C13-P changes the unit of recurrence from a static node example to an ordered
planning query. One HRM carry is initialized per world and updated after each
causally observed expansion event. The matched reset arm uses the same model,
checkpoint, event stream, candidate set, and scorer, but recreates zero carry at
every event while preserving the true event cadence. The third offline control
is the frozen C13-M base rank; the third online control is unchanged static
C13-M search.

The target is stationary: the positive frontier item is the first not-yet-closed
node on the frozen C13-M teacher's successful returned path. The flat C13-J node
encoder and C13-M local-Bellman rank remain frozen. No future search state,
teacher path, Dijkstra value, raster, or self-bootstrap label is a model input.

Before the final run, named mechanical defects were repaired without changing
the scientific protocol: fresh-history startup, exact float32 CE replay,
canonical raw-CSV float semantics, and restoration of the preregistered actual
world `side_len` scaling. The final run uses a fresh fingerprint and output
directory; no failed-attempt artifact is treated as evidence.

## Frozen identities and evidence counts

| Item | Value |
|---|---|
| Source commit | `cf36fd6` |
| Preregistration SHA-256 | `a3c14395a00c0ef61cdad641ddb98027a7eeb70fbcfdbee6b01d863205e479ba` |
| Implementation SHA-256 | `9e511c9732bf29bf2435a589a13662f6ffa4b2723f0576d13311b24aa82a4aa5` |
| Evaluation implementation SHA-256 | `e7537fb7e32b09659e059e8c8c12ef683ff579448d174b501248084e2d846390` |
| Frozen source checkpoint SHA-256 | `39d1e145bf5deb67e7d3281b784dc36810c06e5a8e6193a68590f012132c91c4` |
| Selected C13-P checkpoint SHA-256 | `938b78b337247870644c268373881d3dfebb2b220fcc9f2a6cddbde58f0848f0` |
| Selected model-state SHA-256 | `1928519b073bd52531185f507fcd9d51e223cf16462f9ab79aef43f0acc5e923` |
| Train / validation / development worlds | `96 / 24 / 24` |
| Train / validation / development events | `8,220 / 2,133 / 1,642` |
| Offline raw rows | `4,926` (`1,642 x 3` arms) |
| Online raw rows | `72` (`24 x 3` arms) |
| Integrity inventory | `44/44` artifacts |

Train and validation trace hashes are
`b073664c121ba80d67a63ec7971ada528dfc2b2e75015602f0dcef34c081c186`
and `237793439c42942da5e74732d2e3871b49fdbfaf1005dd061d7de40bf8cfdac6`.
The development trace hash is
`027422bd1a21d029ce8be71b024d9729a6675e5648859afc78d410c18e4523d9`.

## Training and checkpoint selection

Training used the frozen maximum of 20 epochs, patience four, TBPTT length 32,
AdamW configuration, and earliest-minimum validation-CE selection rule. It
stopped after epoch 5 because epochs 2-5 were four consecutive non-improvements
over epoch 1.

| Epoch | Train event-weighted CE | Validation CE | Validation world-macro MRR | Validation world-macro top-1 |
|---:|---:|---:|---:|---:|
| 1 | 2.511795 | **2.322235** | 0.615545 | 0.443546 |
| 2 | 2.308715 | 2.454265 | 0.620127 | 0.451295 |
| 3 | 2.321084 | 2.453143 | 0.632332 | 0.462624 |
| 4 | 2.880669 | 2.671721 | 0.636659 | 0.475640 |
| 5 | 2.321461 | 2.426591 | 0.649704 | 0.489480 |

Epoch 1 is the selected checkpoint. Later ranking metrics improve while CE does
not, but the selector was frozen before evaluation; choosing a later epoch
post hoc would be a different study.

## G0-P integrity gate

G0-P passes. The final verification establishes:

- exact replay of all frozen cohorts and all 144 feature-cache hashes;
- byte-identical duplicate train, validation, and development trace shards;
- byte-identical deterministic ranking and search projections;
- one selected checkpoint/model identity shared by persistent and reset arms;
- complete causal-field and forbidden-information checks over 1,642 development
  events;
- 72/72 valid returned paths and finite, nonnegative timing fields;
- independent reconstruction of G0-P, G1-P, G2-P, and the final verdict from
  promoted raw CSVs;
- exact 44-artifact integrity inventory and a successful fresh verify-only pass.

No tolerance was loosened to obtain this result. Canonical CSV writes preserve
binary64 values and every derived summary/gate is recomputed from the promoted
raw artifacts.

## G1-P offline ranking gate

| Suite | Persistent minus reset MRR |
|---|---:|
| hard bugtrap | +0.010746 |
| hard maze | +0.018219 |
| hard maze dense | -0.074296 |
| hard rooms | -0.058762 |
| hard rooms large | +0.016586 |
| hard spiral | -0.089006 |
| **Pooled** | **-0.029419** |

The pooled 95% CI excludes zero in the harmful direction. Only three suites
are positive, below the frozen robustness requirement. Persistent carry is not
merely unproven here; its held-out world-macro ranking is worse than reset under
the selected checkpoint and frozen evaluation.

## G2-P free-running search (descriptive after G1 failure)

| Arm | Mean expansions | Mean graph-optimal cost ratio | Maximum cost ratio |
|---|---:|---:|---:|
| persistent HRM | 67.083 | 1.089706 | 1.306528 |
| reset HRM | 60.458 | 1.100457 | 1.309618 |
| static C13-M | 69.417 | 1.013142 | 1.038347 |

Persistent minus reset expansions are `+6.625`, with world-clustered 95% CI
upper endpoint `+17.792`; only `2/6` suite deltas favor persistent. Persistent
minus C13-M expansions are `-2.333`, but the CI upper endpoint is `+10.833` and
only `3/6` suites favor persistent. The persistent arm also fails both frozen
path-quality margins. G2-P therefore fails independently of the G1-P stop.

## Post-hoc stability diagnosis (not a preregistered gate)

A read-only diagnostic of the final promoted raw logits shows the same
long-sequence instability that motivated the deeper audit:

- persistent maximum absolute logit: `1.57e20`;
- reset maximum absolute logit: `7.33`;
- persistent events with all candidate logits exactly equal: `348/1,642`;
- after event 64: `289/501` all-equal persistent events;
- maximum persistent event CE: `1.759e13`;
- maximum reset event CE: `4.015`.

This diagnostic does not alter the frozen verdict. It provides a concrete
mechanistic explanation for why deterministic reproduction is concerning:
the recurrence repeatedly reproduces unbounded common-mode state/logit growth,
after which float32 candidate distinctions collapse. Earlier checkpoint replay
also found finite parameters and optimizer moments, locating the failure in the
forward recurrent carry rather than an NaN/Inf optimizer crash.

Representation and integration remain plausible co-factors. The causal event
update omits newly relaxed neighbor/parent identities and open-set content
beyond scalar counts; TBPTT limits long-range credit; the pointwise learned
score fully replaces the safer C13-M queue priority in free-running search.
The valid G1 result nevertheless resolves the frozen study before those
post-hoc hypotheses: this particular persistent state does not help offline.

## Decision and claim boundary

C13-P is complete as a valid negative preregistered pilot. Do not retune the
selected checkpoint on the 24 development worlds, do not self-bootstrap, and
do not open confirmation. Preserve C13-M as the completed positive current-state
result and C13-P as a failed persistent-state mechanism instantiation.

A follow-up, if pursued, must be separately preregistered. The smallest clean
sequence is to add diagnostic state/logit bounds, test causal-state sufficiency
with newly relaxed/parent/frontier summaries, then change only one stability or
representation mechanism at a time. Require a stable offline persistent-over-
reset signal before free-running integration or on-policy training.

C13-P does not establish bounded A*, map-free navigation, a wall-clock speedup,
equivalence, or a general result about HRM or memory.
