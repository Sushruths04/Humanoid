#!/usr/bin/env bash
# Aggregate results/*.json into the Markdown tables in vault/03_Results/Results_Live.md.
set -euo pipefail
cd "$(dirname "$0")/.."
python - <<'PY'
import json, glob, os
rows = []
for f in sorted(glob.glob("results/*.json")):
    try:
        d = json.load(open(f))
        rows.append((os.path.basename(f), d.get("v_pull_kmh"), d.get("success_rate"),
                     d.get("fall_rate"), d.get("mean_time_to_stand_s"),
                     d.get("board_angle_adherence")))
    except Exception as e:
        print("skip", f, e)
print("| file | v_pull | success | fall | t_stand | board_adh |")
print("|---|---|---|---|---|---|")
for r in rows:
    print("| " + " | ".join("" if x is None else str(x) for x in r) + " |")
print("\n>> paste the above into vault/03_Results/Results_Live.md (per DOC_PROTOCOL.md)")
PY
