"""
paper_figures.py - paper deliverables from the final design on split seeds 5-9 (results_paper/).

Reads the two final_report.py outputs (Nu: final design (a); Sh: (a) + log(Nt/Nb)) and writes
  paper_table.csv / paper_table.md   one table, both targets, mean ± sd over split seeds 5-9
  figures/<geometry>_<target>_{loss_curve,error_histogram,parity}.png   split seed 5 runs
  figures/summary_rmse.png            test RMSE mean ± sd per model across split seeds 5-9

Usage: python analysis/paper_figures.py
"""
import os, shutil
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results_paper"; FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)
SRC = {"Nu": "results_confirm/a/split{s}/Nu/data_{g}", "Sh": "results_confirm/b/split{s}/Sh/data_{g}"}
GEOS, TARGETS, SEEDS = ["cone", "plate", "wedge"], ["Nu", "Sh"], [5, 6, 7, 8, 9]
COLS = ["R2", "RMSE", "MAE", "MAPE_pct", "MedRelErr_pct", "P95RelErr_pct", "MaxRelErr_pct", "Within_0p1pct", "Within_0p5pct"]

# ---------------------------------------------------------------- table
T = pd.concat([pd.read_csv(os.path.join(OUT, f"report_{t}", "paper_table_all_splits.csv")) for t in TARGETS], ignore_index=True)
assert sorted(T.split_seed.unique()) == SEEDS
rows = []
for g in GEOS:
    for t in TARGETS:
        grp = T[(T.geometry == g) & (T.target == t)]
        row = dict(geometry=g, target=t, design=grp.design.iloc[0].replace("per-target: ", ""), n_split_seeds=len(grp))
        for c in COLS: row[f"{c}_mean"], row[f"{c}_sd"] = grp[c].mean(), grp[c].std(ddof=1)
        rows.append(row)
P = pd.DataFrame(rows); P.to_csv(os.path.join(OUT, "paper_table.csv"), index=False)


def fmt(c, v):
    return f"{v:.7f}" if c == "R2" else f"{v:.2e}" if c in ("RMSE", "MAE") else f"{v:.3f}" if c.endswith("RelErr_pct") or c == "MAPE_pct" else f"{v:.1f}"


hdr = ["Geometry", "Target", "Design", "R²", "RMSE", "MAE", "MAPE %", "Median rel. err. %", "95th pct rel. err. %",
       "Max rel. err. %", "Within 0.1 %", "Within 0.5 %"]
lines = ["| " + " | ".join(hdr) + " |", "| " + " | ".join(["---"] * len(hdr)) + " |"]
for r in rows:
    lines.append("| " + " | ".join([r["geometry"], r["target"], r["design"]] +
                                   [f"{fmt(c, r[c + '_mean'])} ± {fmt(c, r[c + '_sd'])}" for c in COLS]) + " |")
caption = ("Test-set accuracy of the final surrogates, mean ± sd over five random 70/10/20 splits (split seeds 5–9, "
           "1,100 test rows each), each model a 3-network ensemble. RMSE and MAE are absolute (dimensionless Nu, Sh). "
           "No design or setting was chosen using these test sets.")
open(os.path.join(OUT, "paper_table.md"), "w").write("\n".join(lines) + "\n\n" + caption + "\n")

# ---------------------------------------------------------------- split seed 5 figures
for g in GEOS:
    for t in TARGETS:
        d = SRC[t].format(s=5, g=g)
        for f in ["loss_curve", "error_histogram", "parity"]:
            shutil.copy(os.path.join(d, f"{f}.png"), os.path.join(FIG, f"{g}_{t}_{f}.png"))

# ---------------------------------------------------------------- summary figure
INK, MUTED, GRID = "#1f1f1e", "#5f5e5a", "#e6e5e0"
STYLE = {"Nu": dict(color="#2a78d6", marker="o", label="Nu (quad anchor)"),
         "Sh": dict(color="#eb6834", marker="s", label="Sh (quad anchor + log(Nt/Nb))")}
fig, ax = plt.subplots(figsize=(6.4, 3.6))
for i, t in enumerate(TARGETS):
    st = STYLE[t]
    for k, g in enumerate(GEOS):
        v = T[(T.geometry == g) & (T.target == t)].RMSE.values; x = k + (i - 0.5) * 0.28
        ax.scatter(np.full(len(v), x) + np.linspace(-0.04, 0.04, len(v)), v, s=14, color=st["color"], alpha=.35, lw=0, zorder=2)
        ax.errorbar(x, v.mean(), yerr=v.std(ddof=1), fmt=st["marker"], ms=7, color=st["color"], ecolor=st["color"],
                    elinewidth=2, capsize=0, mec="white", mew=1.5, zorder=3, label=st["label"] if k == 0 else None)
        ax.annotate(f"{v.mean():.2e}", (x, v.mean() + v.std(ddof=1)), xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7, color=MUTED)
ax.set_xticks(range(len(GEOS))); ax.set_xticklabels(GEOS, color=INK); ax.set_xlim(-0.6, len(GEOS) - 0.4)
ax.set_ylim(0, None); ax.set_ylabel("test RMSE (absolute)", color=INK)
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax.grid(axis="y", color=GRID, lw=.8); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color(MUTED)
ax.tick_params(colors=MUTED)
ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK)
ax.set_title("Test RMSE per model, mean ± sd over split seeds 5–9 (faint dots: individual splits)", loc="left", fontsize=9, color=INK)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "summary_rmse.png"), dpi=300); plt.close(fig)
print("\n".join(lines)); print("\nfigures:", sorted(os.listdir(FIG)))
