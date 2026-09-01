"""Exact C11 mission-arm parameter counts from the live constructors.

v11-review item 11: the appendix architecture table gave 0.5-3.5M ranges for
the mission arms. Counts do not vary by config (the I/O contract is fixed in
C11MissionConfig), so each arm has one exact total. This script instantiates
every directly compared arm via the same build path the experiments used and
prints exact totals.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(
    HERE, "..", "..", "..", "hrm-cloud", "continuous_prm")))

import continuous_prm_c11_mission as M  # noqa: E402
try:  # registers the hrmv2_act provider builder if present
    import continuous_prm_c11_hrmv2_arm  # noqa: F401,E402
    HRMV2 = ["hrmv2_act"]
except Exception as e:  # pragma: no cover
    print("hrmv2_act unavailable:", e)
    HRMV2 = []

cfg = M.C11MissionConfig()
arms = list(M.ARM_NAMES) + HRMV2 + list(M.BIG_ARM_NAMES)
out = {}
for name in arms:
    model = M.build_arm(name, cfg)
    total = sum(p.numel() for p in model.parameters())
    out[name] = total
    print(f"{name:16s} {total:>12,}")

open(os.path.join(HERE, "c11_param_counts.json"), "w").write(
    json.dumps(out, indent=1))
