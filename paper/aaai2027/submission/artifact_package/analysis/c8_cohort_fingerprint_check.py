import csv

BASE = r"C:\Users\hrish\Code Projects\HRMv2\hrm-cloud\continuous_prm\runs"

fresh = {}
with open(BASE + r"\c8r_fresh_eval\results\continuous_prm_c8_eval_raw.csv",
          newline="") as f:
    for r in csv.DictReader(f):
        if (r["provider"] == "euclid" and r["mode"] == "astar"
                and r["found"] in ("True", "1", "true")):
            k = (r["suite"], int(float(r["world_index"])))
            fresh[k] = int(float(r["optimal_arrival"]))

sipp = {}
with open(BASE + r"\c8r_sipp\confirmation_raw.csv", newline="") as f:
    for r in csv.DictReader(f):
        if r["found"] in ("True", "1", "true"):
            sipp[(r["suite"], int(r["world_index"]))] = int(r["optimal_arrival"])

common = sorted(set(fresh) & set(sipp))
match = sum(1 for k in common if fresh[k] == sipp[k])
print(f"common worlds: {len(common)}; matching optimal_arrival: {match}")
per = {}
for k in common:
    per.setdefault(k[0], [0, 0])[1] += 1
    if fresh[k] == sipp[k]:
        per[k[0]][0] += 1
print({s.replace("C_dyn_", ""): f"{a}/{b}" for s, (a, b) in per.items()})
for k in common[:6]:
    print(k, "fresh", fresh[k], "sipp", sipp[k])
