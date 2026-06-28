#!/usr/bin/env python3
"""Local CPU timing harness for residual_tasklora_v2 eval (no Modal, no GPU).

Usage:
  python hrm-cloud/bench_eval_episode.py --suite ID_A64_static --seeds 3 --budget 500
  python hrm-cloud/bench_eval_episode.py --suite OOD_A256_static --seeds 1 --budget 2000

Reports per-episode wall-clock. Pass --profile to see the cProfile breakdown,
where compute_true_cost_to_goal (the diagnostics-only DP) should dominate on
large maps — the Phase-1 hypothesis.
"""
import argparse, time, cProfile, pstats, io
import residual_tasklora_v2 as R


def _suite(suite_id):
    for s in R.build_eval_suites(include_stretch=True, eval_episodes=100):
        if s.suite_id == suite_id:
            return s
    raise SystemExit(f"unknown suite {suite_id}; pick from build_eval_suites()")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="ID_A64_static")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--budget", type=int, default=500)
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()
    suite = _suite(args.suite)

    # model=None isolates pure search + diagnostics DP (no NN); good enough for the
    # Phase-1 hypothesis (compute_true_cost_to_goal dominates large maps).
    def run():
        for i in range(args.seeds):
            t0 = time.time()
            res = R.run_policy_episode(suite, seed=i, model=None, alpha=1.0,
                                       max_expansions=args.budget, device="cpu")
            print(f"  seed={i} steps={res['steps']} exp={res['expansions']} "
                  f"wall={time.time()-t0:.2f}s")

    print(f"[bench] suite={suite.suite_id} n={suite.size} max_steps={suite.max_steps} "
          f"budget={args.budget} seeds={args.seeds}")
    if args.profile:
        pr = cProfile.Profile(); pr.enable(); run(); pr.disable()
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(15)
        print(s.getvalue())
    else:
        t0 = time.time(); run()
        print(f"[bench] total {time.time()-t0:.2f}s")


if __name__ == "__main__":
    main()
