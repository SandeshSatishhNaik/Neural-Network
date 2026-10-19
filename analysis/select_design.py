"""
select_design.py - step 4 of the workflow: for each geometry and target, keep the joint (Nu+Sh) model
or the per-target model, whichever has the lower VALIDATION RMSE. Test rows are never read.

Usage: python analysis/select_design.py results_pasr --out results_final/final_config.csv
"""
import argparse, json, os
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("run_dir"); ap.add_argument("--out", default="results_final/final_config.csv")
args = ap.parse_args()

rows = []
for g in ["cone", "plate", "wedge"]:
    ds = f"data_{g}"
    for t in ["Nu", "Sh"]:
        cand = []
        for source, folder in (("joint", ds), ("per-target", f"{ds}__{t}")):
            p = os.path.join(args.run_dir, folder)
            m = pd.read_csv(os.path.join(p, "metrics.csv"))
            v = m[(m.split == "val") & (m.target == t)].iloc[0]
            cfg = json.load(open(os.path.join(p, "scaler.json")))
            cand.append(dict(geometry=g, target=t, source=source, design=cfg["design"], anchor=cfg["anchor"],
                             dual_encoding=cfg["dual_encoding"], folder=folder, val_RMSE=v.RMSE, val_MaxRelErr_pct=v.MaxRelErr_pct))
        best = min(cand, key=lambda r: r["val_RMSE"])
        other = [c for c in cand if c is not best][0]
        rows.append(dict(**best, other_source=other["source"], other_design=other["design"], other_val_RMSE=other["val_RMSE"],
                         val_RMSE_ratio_other_over_best=other["val_RMSE"] / best["val_RMSE"]))

df = pd.DataFrame(rows)
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
df.to_csv(args.out, index=False)
print(df[["geometry", "target", "source", "design", "val_RMSE", "other_source", "other_design", "other_val_RMSE",
          "val_RMSE_ratio_other_over_best"]].to_string(index=False))
