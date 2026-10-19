"""
final_report.py - step 7: paper table (split seed 0 and mean ± sd over split seeds) and the five checks
from CLAUDE.md, for the final configuration in final_config.csv.

Split seed 0 models are read from results_pasr/ (--seed0-dir), split seeds >= 1 from results_split<s>/ (see split_commands.py).

Usage: python analysis/final_report.py results_final/final_config.csv --seeds 0 1 2 3 4 --out results_final
"""
import argparse, os
import numpy as np, pandas as pd

# reference test RMSE from docs/RESULTS_AND_REVIEW.md: section 1 (plain MLP) and section 3 (PASR-Net ensemble)
REF_MLP = {("cone", "Nu"): 4.53e-4, ("cone", "Sh"): 6.47e-4, ("plate", "Nu"): 4.44e-4,
           ("plate", "Sh"): 6.02e-4, ("wedge", "Nu"): 4.52e-4, ("wedge", "Sh"): 6.04e-4}
REF_PASR = {("cone", "Nu"): 1.11e-4, ("cone", "Sh"): 4.25e-4, ("plate", "Nu"): 9.39e-5,
            ("plate", "Sh"): 3.39e-4, ("wedge", "Nu"): 9.38e-5, ("wedge", "Sh"): 3.06e-4}
COLS = ["R2", "RMSE", "MAE", "MAPE_pct", "MaxRelErr_pct", "Within_0p1pct", "Within_0p5pct"]

ap = argparse.ArgumentParser()
ap.add_argument("config"); ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
ap.add_argument("--out", default="results_final")
ap.add_argument("--seed0-dir", default="results_pasr")
args = ap.parse_args()
cfg = pd.read_csv(args.config); os.makedirs(args.out, exist_ok=True)


def folder(r, s):
    if s == 0: return os.path.join(args.seed0_dir, r.folder)
    return os.path.join(f"results_split{s}", "joint" if r.source == "joint" else r.target, f"data_{r.geometry}")


def loss_checks(h):
    """Checks 2 and 5 for one seed's loss history."""
    a, l = h[h.phase == "adam"], h[h.phase == "lbfgs"]
    fin = h[h.phase == "final"].iloc[-1]
    val_start, val_min = l.val_loss.iloc[0], min(l.val_loss.min(), fin.val_loss)
    return dict(val_falls_in_lbfgs=bool(fin.val_loss < val_start and fin.val_loss <= 1.5 * val_min),
                val_over_train=fin.val_loss / fin.train_loss,
                loss_drops_after_switch=bool(fin.train_loss < a.train_loss.iloc[-1]),
                adam_end_train=a.train_loss.iloc[-1], final_train=fin.train_loss, final_val=fin.val_loss)


table, checks, worst = [], [], []
for r in cfg.itertuples():
    ref = (REF_MLP if r.design == "MLP" else REF_PASR)[(r.geometry, r.target)]
    for s in args.seeds:
        d = folder(r, s)
        m = pd.read_csv(os.path.join(d, "metrics.csv"))
        t = m[(m.split == "test") & (m.target == r.target)].iloc[0]
        table.append(dict(geometry=r.geometry, target=r.target, design=f"{r.source}: {r.design}", split_seed=s,
                          **{c: t[c] for c in COLS}))
        p = pd.read_csv(os.path.join(d, "predictions_test.csv"))
        e = p[f"{r.target}_error"]
        h = pd.read_csv(os.path.join(d, "loss_history.csv"))
        lc = [loss_checks(h[h.seed == k]) for k in sorted(h.seed.unique())]
        checks.append(dict(geometry=r.geometry, target=r.target, split_seed=s, folder=d,
                           c1_R2_ge_0p9999=bool(t.R2 >= 0.9999), c1_RMSE_over_ref=t.RMSE / ref,
                           c1_pass=bool(t.R2 >= 0.9999 and t.RMSE <= 2 * ref),
                           c2_pass=all(x["val_falls_in_lbfgs"] for x in lc),
                           c2_val_over_train_max=max(x["val_over_train"] for x in lc),
                           c3_mean_over_sd=abs(e.mean()) / e.std(), c3_pass=bool(abs(e.mean()) < 0.1 * e.std()),
                           c4_MaxRelErr_pct=t.MaxRelErr_pct, c4_pass=bool(t.MaxRelErr_pct < 1.0),
                           c5_pass=all(x["loss_drops_after_switch"] for x in lc)))
        if t.MaxRelErr_pct >= 1.0 or s == 0:
            p["rel_err_pct"] = 100 * e.abs() / p[f"{r.target}_true"].abs()
            for _, w in p.nlargest(5, "rel_err_pct").iterrows():
                worst.append(dict(geometry=r.geometry, target=r.target, split_seed=s, row=int(w["row"]),
                                  true=w[f"{r.target}_true"], pred=w[f"{r.target}_pred"], rel_err_pct=w.rel_err_pct))

T = pd.DataFrame(table); C = pd.DataFrame(checks)
T.to_csv(os.path.join(args.out, "paper_table_all_splits.csv"), index=False)
C.to_csv(os.path.join(args.out, "checks.csv"), index=False)
pd.DataFrame(worst).to_csv(os.path.join(args.out, "worst_test_rows.csv"), index=False)

out = []
for (g, t), grp in T.groupby(["geometry", "target"], sort=False):
    s0 = grp[grp.split_seed == 0].iloc[0]
    row = dict(geometry=g, target=t, design=s0.design, n_split_seeds=len(grp))
    for c in COLS: row[f"{c}_seed0"] = s0[c]
    for c in COLS: row[f"{c}_mean"], row[f"{c}_sd"] = grp[c].mean(), grp[c].std(ddof=1)
    out.append(row)
P = pd.DataFrame(out); P.to_csv(os.path.join(args.out, "paper_table.csv"), index=False)


def fmt(c, v):
    return f"{v:.7f}" if c == "R2" else f"{v:.2e}" if c in ("RMSE", "MAE") else f"{v:.3f}" if c in ("MAPE_pct", "MaxRelErr_pct") else f"{v:.1f}"


hdr = ["Geometry", "Target", "Design", "Split", "R²", "RMSE", "MAE", "MAPE %", "Max rel. err. %", "Within 0.1 %", "Within 0.5 %"]
lines = ["| " + " | ".join(hdr) + " |", "| " + " | ".join(["---"] * len(hdr)) + " |"]
for r in out:
    lines.append("| " + " | ".join([r["geometry"], r["target"], r["design"], "seed 0"] + [fmt(c, r[f"{c}_seed0"]) for c in COLS]) + " |")
    lines.append("| " + " | ".join(["", "", "", f"mean ± sd ({r['n_split_seeds']} seeds)"] +
                                   [f"{fmt(c, r[f'{c}_mean'])} ± {fmt(c, r[f'{c}_sd'])}" for c in COLS]) + " |")
open(os.path.join(args.out, "paper_table.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print("\nChecks (False = fail):")
print(C.drop(columns="folder").to_string(index=False))
