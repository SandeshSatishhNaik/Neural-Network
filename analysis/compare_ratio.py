"""
compare_ratio.py - VALIDATION comparison of (a) the final design (per-target quad anchor) and
(b) the same + --ratio-features, over split seeds 0-4. Test rows are never read.

(a) validation predictions are rebuilt from saved weights (rebuild_predictions.py) into results_ratio/baseline_val/;
(b) runs write predictions_val.csv themselves into results_ratio/split<s>/<target>/data_<geometry>/.

Decision rule (fixed before any (b) result was seen): (b) wins for a target if its validation RMSE is lower
in at least 12 of the 15 (geometry x split seed) pairs (one-sided sign test p = 0.018).

Usage: python analysis/compare_ratio.py --seeds 0 1 2 3 4 --out results_ratio
"""
import argparse, os
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4]); ap.add_argument("--out", default="results_ratio")
ap.add_argument("--win-count", type=int, default=12)
args = ap.parse_args()


def stats(p, t):
    e, y = p[f"{t}_error"].values, p[f"{t}_true"].values
    rel = 100 * np.abs(e) / np.abs(y); hi = (p.Nt / p.Nb).values > 4
    return dict(RMSE=np.sqrt((e ** 2).mean()), RMSE_ratio_gt4=np.sqrt((e[hi] ** 2).mean()), RMSE_ratio_le4=np.sqrt((e[~hi] ** 2).mean()),
                MAPE_ratio_gt4=rel[hi].mean(), n_ratio_gt4=int(hi.sum()),
                MedRelErr_pct=np.median(rel), P95RelErr_pct=np.percentile(rel, 95), MaxRelErr_pct=rel.max())


rows = []
for s in args.seeds:
    for g in ["cone", "plate", "wedge"]:
        for t in ["Nu", "Sh"]:
            for design, d in (("a", f"{args.out}/baseline_val/split{s}/{t}/data_{g}"), ("b", f"{args.out}/split{s}/{t}/data_{g}")):
                rows.append(dict(split_seed=s, geometry=g, target=t, design=design, **stats(pd.read_csv(f"{d}/predictions_val.csv"), t)))
R = pd.DataFrame(rows); R.to_csv(f"{args.out}/val_comparison_all.csv", index=False)

W = R.pivot_table(index=["split_seed", "geometry", "target"], columns="design", values=["RMSE", "RMSE_ratio_gt4"]).reset_index()
W.columns = ["split_seed", "geometry", "target", "a_RMSE", "b_RMSE", "a_RMSE_gt4", "b_RMSE_gt4"]
W["b_better"] = W.b_RMSE < W.a_RMSE; W["a_over_b"] = W.a_RMSE / W.b_RMSE
W.to_csv(f"{args.out}/val_comparison_paired.csv", index=False)

COLS = ["RMSE", "RMSE_ratio_gt4", "RMSE_ratio_le4", "MAPE_ratio_gt4", "MedRelErr_pct", "P95RelErr_pct", "MaxRelErr_pct"]
S = R.groupby(["geometry", "target", "design"])[COLS].agg(["mean", "std"])
S.columns = [f"{c}_{k}" for c, k in S.columns]; S = S.reset_index(); S.to_csv(f"{args.out}/val_comparison_summary.csv", index=False)

f = lambda v, c: f"{v:.3f}" if "pct" in c or "MAPE" in c else f"{v:.2e}"
hdr = ["Geometry", "Target", "Design", "Val RMSE", "Val RMSE, Nt/Nb > 4", "Val RMSE, Nt/Nb ≤ 4", "Median rel. err. %", "95th pct rel. err. %", "Max rel. err. %"]
lines = ["| " + " | ".join(hdr) + " |", "| " + " | ".join(["---"] * len(hdr)) + " |"]
for r in S.itertuples():
    name = "(a) final design" if r.design == "a" else "(b) + log(Nt/Nb)"
    cells = [f"{f(getattr(r, c + '_mean'), c)} ± {f(getattr(r, c + '_std'), c)}" for c in ["RMSE", "RMSE_ratio_gt4", "RMSE_ratio_le4", "MedRelErr_pct", "P95RelErr_pct", "MaxRelErr_pct"]]
    lines.append("| " + " | ".join([r.geometry, r.target, name] + cells) + " |")
dec = []
for t in ["Nu", "Sh"]:
    w = W[W.target == t]; k = int(w.b_better.sum()); n = len(w)
    dec.append(dict(target=t, b_better_pairs=k, pairs=n, b_wins=k >= args.win_count, median_a_over_b=w.a_over_b.median(),
                    median_a_over_b_gt4=(w.a_RMSE_gt4 / w.b_RMSE_gt4).median()))
D = pd.DataFrame(dec); D.to_csv(f"{args.out}/val_decision.csv", index=False)
dl = ["| Target | Pairs where (b) has lower val RMSE | (b) wins | Median (a)/(b) val RMSE | Median (a)/(b), Nt/Nb > 4 |", "| --- | --- | --- | --- | --- |"]
dl += [f"| {r.target} | {r.b_better_pairs} / {r.pairs} | {'yes' if r.b_wins else 'no'} | {r.median_a_over_b:.3f} | {r.median_a_over_b_gt4:.3f} |" for r in D.itertuples()]
open(f"{args.out}/val_comparison.md", "w").write("\n".join(lines) + "\n\n" + "\n".join(dl) + "\n")
print("\n".join(lines)); print(); print(D.to_string(index=False))
