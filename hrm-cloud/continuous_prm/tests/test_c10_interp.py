import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
HERE = Path(__file__).resolve().parents[1]
import numpy as np
import continuous_prm_c10_interp as C10
import continuous_prm_common as C


def test_family_grid_specs_and_bracketing():
    specs = C10.c10_family_specs()
    for fam in C10.SOURCE_FAMILIES + C10.TARGET_FAMILIES:
        assert fam in specs
        s = specs[fam]
        assert s.mode == C10.HARD_MODE and s.name in ("C_hard_maze", "C_hard_rooms")
    src_cent = {f: C10.family_descriptor_centroid(specs[f], n=6, seed=1) for f in C10.SOURCE_FAMILIES}
    for tgt in C10.TARGET_FAMILIES:
        z_t = C10.family_descriptor_centroid(specs[tgt], n=6, seed=2)
        ok, viol = C10.bracketing_ok(z_t, list(src_cent.values()))
        assert isinstance(ok, bool) and isinstance(viol, list)
    z = C10.family_descriptor_centroid(specs["C10_maze_tgt"], n=8, seed=3)
    ok, viol = C10.bracketing_ok(z, [C10.family_descriptor_centroid(specs[f], n=8, seed=3) for f in C10.SOURCE_FAMILIES])
    assert isinstance(ok, bool)
