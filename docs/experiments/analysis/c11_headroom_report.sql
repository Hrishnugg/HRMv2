-- Reviewed C11 G0-H rows used by the Master Experiment Evidence Synthesis report.
-- Values come from C11_HEADROOM.md and were independently recomputed from
-- hrm-cloud/continuous_prm/runs/c11_probe/c11_probe_records.csv at the
-- documented binding budgets.

WITH c11_headroom(
    config,
    config_code,
    mission_length,
    K,
    expansion_ratio,
    oracle_cut,
    binding_budget,
    matched_n,
    worlds
) AS (
    VALUES
        ('A · maze waypoints',        'A', 'K=2', 2, 0.155, 0.845,  200,  6, 25),
        ('A · maze waypoints',        'A', 'K=4', 4, 0.121, 0.879,  400,  3, 25),
        ('A · maze waypoints',        'A', 'K=8', 8, 0.082, 0.918, 1600, 20, 25),
        ('B · rooms-large waypoints', 'B', 'K=2', 2, 0.225, 0.775,  100, 12, 25),
        ('B · rooms-large waypoints', 'B', 'K=4', 4, 0.208, 0.792,  200,  4, 25),
        ('B · rooms-large waypoints', 'B', 'K=8', 8, 0.103, 0.897,  800, 16, 25),
        ('C · maze keys/doors',       'C', 'K=2', 2, 0.144, 0.856,  200,  5, 25),
        ('C · maze keys/doors',       'C', 'K=4', 4, 0.128, 0.872,  400,  3, 25),
        ('C · maze keys/doors',       'C', 'K=8', 8, 0.084, 0.916, 1600, 23, 25)
)
SELECT
    config,
    config_code,
    mission_length,
    K,
    expansion_ratio,
    oracle_cut,
    binding_budget,
    matched_n,
    worlds
FROM c11_headroom
ORDER BY K, config_code;
