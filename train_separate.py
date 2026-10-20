"""
train_separate.py - one independent neural network per dataset (and optionally per target).

Each dataset file is trained in its own run: its own scaler, its own train/val/test split,
its own model, its own outputs folder. Nothing is shared between datasets.

Per model it writes:
  metrics.csv            R2, RMSE, MAE, MaxAE, MAPE, median / 95th-percentile / max relative error (%),
                         within 0.1% / 0.5%, for train / val / test, per target
  loss_curve.png         train + validation loss vs iteration (log scale), Adam -> L-BFGS boundary
  error_histogram.png    histogram of test errors (predicted - true) per target
  parity.png             predicted vs true on the test set, per target
  predictions_test.csv   inputs, true and predicted values for every test row
  predictions_val.csv    the same for every validation row
  loss_history.csv       iteration, phase, train loss, val loss
  model.pt + scaler.json trained weights and the exact scaling needed to reuse the model
And one summary_metrics.csv + summary_r2_rmse.png across all datasets.

Data formats accepted (auto-detected):
  A) .xlsx with a sheet named "Input" (inputs) and one sheet per target (e.g. "Nu", "Sh")
  B) .xlsx where the first sheet holds inputs and every other sheet holds one target column
  C) .csv / .xlsx single table: pass --targets Nu Sh (all other numeric columns are inputs)

Usage:
  python train_separate.py data_cone.xlsx data_plate.xlsx data_wedge.xlsx
  python train_separate.py data/*.xlsx --per-target            # one model per dataset AND per target
  python train_separate.py data_cone.xlsx --seeds 3             # 3-seed ensemble (better, 3x slower)
  python train_separate.py data_cone.xlsx --quick               # smoke test, ~20 s per network
  python train_separate.py data/*.xlsx --per-target --auto      # PASR-Net, design chosen per target on validation
  python train_separate.py data_cone.xlsx --anchor quad          # PASR-Net with quadratic anchor
  python train_separate.py data_cone.xlsx --anchor quad --ratio-features   # + standardised log(Nt/Nb) input
"""
import argparse, json, os, time, glob
import numpy as np, pandas as pd, torch, torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)


# ============================================================ data
def load_dataset(path, targets=None):
    """Return X (DataFrame of inputs), Y (DataFrame of targets)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
        if not targets: raise SystemExit(f"{path}: CSV needs --targets <col> [<col> ...]")
        return df.drop(columns=targets).select_dtypes("number"), df[targets]
    xl = pd.ExcelFile(path)
    sheets = xl.sheet_names
    if targets and len(sheets) == 1:                                   # format C (single sheet)
        df = pd.read_excel(xl, sheets[0])
        return df.drop(columns=targets).select_dtypes("number"), df[targets]
    in_sheet = "Input" if "Input" in sheets else sheets[0]             # format A or B
    X = pd.read_excel(xl, in_sheet).select_dtypes("number")
    Ys = {}
    for s in sheets:
        if s == in_sheet: continue
        d = pd.read_excel(xl, s).select_dtypes("number")
        for c in d.columns:
            name = c if d.shape[1] > 1 else (c if c not in Ys else s)
            Ys[name] = d[c].values
    Y = pd.DataFrame(Ys)
    if targets: Y = Y[targets]
    if len(X) != len(Y): raise SystemExit(f"{path}: inputs have {len(X)} rows, targets {len(Y)}")
    bad = X.isna().any(axis=1) | Y.isna().any(axis=1)
    if bad.any():
        print(f"  dropping {int(bad.sum())} rows with missing values")
        X, Y = X[~bad].reset_index(drop=True), Y[~bad].reset_index(drop=True)
    return X, Y


def split_indices(n, test_frac, val_frac, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_te, n_va = int(round(test_frac * n)), int(round(val_frac * n))
    return idx[n_te + n_va:], idx[n_te:n_te + n_va], idx[:n_te]     # train, val, test


class Scaler:
    """Inputs -> [-1, 1] (min-max on TRAIN rows). Targets -> optional log, then z-score on TRAIN rows."""
    def fit(self, X, Y, log_cols):
        self.x_cols, self.y_cols = list(X.columns), list(Y.columns)
        self.x_min, self.x_max = X.min().values, X.max().values
        self.log_cols = [c for c in log_cols if c in self.y_cols]
        T = self._fwd(Y.values)
        self.y_mu, self.y_sd = T.mean(0), T.std(0)
        return self
    def _fwd(self, Yv):
        T = np.array(Yv, dtype=float).copy()
        for c in self.log_cols: T[:, self.y_cols.index(c)] = np.log(T[:, self.y_cols.index(c)])
        return T
    def x(self, X):
        rng = np.where(self.x_max > self.x_min, self.x_max - self.x_min, 1.0)
        return torch.tensor(2 * (X[self.x_cols].values - self.x_min) / rng - 1)
    def y(self, Y): return torch.tensor((self._fwd(Y[self.y_cols].values) - self.y_mu) / self.y_sd)
    def inv(self, T):
        Y = np.asarray(T) * self.y_sd + self.y_mu
        for c in self.log_cols: Y[:, self.y_cols.index(c)] = np.exp(Y[:, self.y_cols.index(c)])
        return Y
    def to_json(self):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}


# ============================================================ model
class MLP(nn.Module):
    def __init__(self, n_in, n_out, width=128, depth=4, act="silu"):
        super().__init__()
        A = {"silu": nn.SiLU, "tanh": nn.Tanh, "gelu": nn.GELU, "relu": nn.ReLU}[act]
        layers, d = [], n_in
        for _ in range(depth): layers += [nn.Linear(d, width), A()]; d = width
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)


def train(model, xtr, ytr, xva, yva, adam_steps, lbfgs_iter, lr, log_every=10):
    """Full-batch Adam (cosine LR) then L-BFGS (strong Wolfe). Returns loss history rows."""
    mse = lambda a, b: ((a - b) ** 2).mean()
    hist = []
    def record(it, phase, ltr):
        with torch.no_grad(): lva = float(mse(model(xva), yva)) if len(xva) else float("nan")
        hist.append((it, phase, float(ltr), lva))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(adam_steps, 1))
    for i in range(adam_steps):
        opt.zero_grad(); l = mse(model(xtr), ytr); l.backward(); opt.step(); sch.step()
        if i % log_every == 0: record(i, "adam", l.item())
    if lbfgs_iter > 0:
        opt = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=lbfgs_iter, history_size=50,
                                tolerance_grad=1e-12, tolerance_change=1e-15, line_search_fn="strong_wolfe")
        k = [adam_steps]
        def closure():
            opt.zero_grad(); l = mse(model(xtr), ytr); l.backward()
            if k[0] % log_every == 0: record(k[0], "lbfgs", l.item())
            k[0] += 1; return l
        opt.step(closure)
    with torch.no_grad(): record(hist[-1][0] + 1 if hist else 0, "final", mse(model(xtr), ytr).item())
    return hist


# ============================================================ metrics + plots
def metrics(yt, yp):
    """All metrics in physical units. Relative errors use |true| (Nu and Sh are > 0 in these data)."""
    e = yp - yt
    rel = np.abs(e) / np.maximum(np.abs(yt), 1e-12)
    return dict(R2=float(1 - (e ** 2).sum() / ((yt - yt.mean()) ** 2).sum()),
                RMSE=float(np.sqrt((e ** 2).mean())), MAE=float(np.abs(e).mean()),
                MaxAE=float(np.abs(e).max()),
                MAPE_pct=float(100 * rel.mean()),
                MedRelErr_pct=float(100 * np.median(rel)),
                P95RelErr_pct=float(100 * np.percentile(rel, 95)),
                MaxRelErr_pct=float(100 * rel.max()),
                Within_0p1pct=float(100 * (rel <= 1e-3).mean()),
                Within_0p5pct=float(100 * (rel <= 5e-3).mean()))


def plot_loss(hists, path, title, adam_steps=None):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for s, h in enumerate(hists):
        h = np.array([[r[0], r[2], r[3]] for r in h], float)
        ls = ["-", "--", ":", "-."][s % 4]
        ax.semilogy(h[:, 0], h[:, 1], color="#1f4e8c", ls=ls, lw=1.4, label=f"train (seed {s})")
        if np.isfinite(h[:, 2]).any():
            ax.semilogy(h[:, 0], h[:, 2], color="#d0552a", ls=ls, lw=1.4, label=f"validation (seed {s})")
    adam_end = adam_steps if adam_steps else None
    if adam_end is not None:
        ax.axvline(adam_end, color="#777", ls="--", lw=0.8)
        ax.text(adam_end, ax.get_ylim()[1], " L-BFGS →", va="top", fontsize=8, color="#555")
        ax.text(adam_end, ax.get_ylim()[1], "← Adam ", va="top", ha="right", fontsize=8, color="#555")
    ax.set_xlabel("iteration (Adam steps, then L-BFGS function evaluations)")
    ax.set_ylabel("MSE on standardised targets")
    ax.set_title(title, loc="left", fontsize=10); ax.grid(alpha=.3); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def plot_hist(yt, yp, names, path, title):
    k = len(names); fig, axs = plt.subplots(1, k, figsize=(4.2 * k, 3.4), squeeze=False)
    for j, n in enumerate(names):
        e = yp[:, j] - yt[:, j]; ax = axs[0, j]
        ax.hist(e, bins=50, color="#1f4e8c", alpha=.85, edgecolor="white", lw=.4)
        ax.axvline(0, color="#d0552a", lw=1)
        ax.set_title(f"{n}   mean {e.mean():.2e}   sd {e.std():.2e}", fontsize=9, loc="left")
        ax.set_xlabel("error (predicted − true)"); ax.set_ylabel("count" if j == 0 else "")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0)); ax.grid(alpha=.3)
    fig.suptitle(title, x=.01, ha="left", fontsize=10); fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def plot_parity(yt, yp, names, path, title, m):
    k = len(names); fig, axs = plt.subplots(1, k, figsize=(4 * k, 3.9), squeeze=False)
    for j, n in enumerate(names):
        ax = axs[0, j]; lo, hi = min(yt[:, j].min(), yp[:, j].min()), max(yt[:, j].max(), yp[:, j].max())
        ax.plot([lo, hi], [lo, hi], "--", color="#777", lw=.8)
        ax.scatter(yt[:, j], yp[:, j], s=5, color="#1f4e8c", alpha=.6, lw=0)
        ax.set_title(n, loc="left", fontsize=10)
        ax.text(.04, .96, f"R² = {m[n]['R2']:.6f}\nRMSE = {m[n]['RMSE']:.3e}", transform=ax.transAxes, va="top", fontsize=8)
        ax.set_xlabel("true (solver)"); ax.set_ylabel("predicted"); ax.grid(alpha=.3)
    fig.suptitle(title, x=.01, ha="left", fontsize=10); fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


# ============================================================ PASR-Net pieces
def log_features(Xdf, Xref):
    """Standardised log of each input (offset of 1% of the range keeps inputs near 0 finite)."""
    lo, hi = Xref.min().values, Xref.max().values
    off = 0.01 * np.where(hi > lo, hi - lo, 1.0)
    L, Lr = np.log(np.abs(Xdf.values) + off), np.log(np.abs(Xref.values) + off)
    sd = Lr.std(0); sd[sd == 0] = 1.0
    return torch.tensor((L - Lr.mean(0)) / sd)


def ratio_feature(Xdf, tr):
    """log(Nt/Nb), standardised with training rows: one extra column for the network and the anchor."""
    if not {"Nt", "Nb"} <= set(Xdf.columns): raise SystemExit("--ratio-features needs input columns Nt and Nb")
    L = np.log(Xdf["Nt"].values / Xdf["Nb"].values)
    mu, sd = L[tr].mean(), L[tr].std()
    return torch.tensor((L - mu) / sd).unsqueeze(1), (float(mu), float(sd))


def quad_features(x):
    """[1, x_i, x_i*x_j (i<=j)] -> full quadratic response surface (1 + d + d(d+1)/2 terms)."""
    cols = [torch.ones(len(x), 1), x]
    for i in range(x.shape[1]):
        for j in range(i, x.shape[1]): cols.append((x[:, i] * x[:, j]).unsqueeze(1))
    return torch.cat(cols, 1)


def fit_model(X, Y, tr, va, args, anchor, dual, seed):
    """One PASR-Net (or plain MLP when anchor='none', dual=False). Returns predictions + artefacts.
       prediction(standardised) = p(x) + s * NN(features);  p = least-squares quadratic on train rows."""
    log_cols = [c for c in Y.columns if any(c.lower().startswith(q.lower()) for q in args.log_targets) and (Y[c] > 0).all()]
    sc = Scaler().fit(X.iloc[tr], Y.iloc[tr], log_cols)
    x, y = sc.x(X), sc.y(Y)
    z = torch.cat([x, log_features(X, X.iloc[tr])], 1) if dual else x
    rf, ratio_stats = ratio_feature(X, tr) if args.ratio_features else (None, None)
    if rf is not None: z = torch.cat([z, rf], 1)
    if anchor == "quad":
        Q = quad_features(x)
        if rf is not None: Q = torch.cat([Q, rf], 1)          # log(Nt/Nb) as an extra linear anchor term
        A = Q[tr]
        coef = torch.linalg.solve(A.T @ A + 1e-10 * torch.eye(A.shape[1]), A.T @ y[tr]); base = Q @ coef
    else:
        coef, base = None, torch.zeros_like(y)
    r = y - base; s = r[tr].std(0) if anchor != "none" else torch.ones(y.shape[1])
    torch.manual_seed(seed)
    m = MLP(z.shape[1], y.shape[1], args.width, args.depth, args.act)
    hist = train(m, z[tr], r[tr] / s, z[va], r[va] / s, args.adam_steps, args.lbfgs_iter, args.lr)
    with torch.no_grad(): t = base + s * m(z)
    return dict(sc=sc, pred=sc.inv(t.numpy()), hist=hist, net=m, coef=coef, scale=s, anchor=anchor, dual=dual,
                ratio_stats=ratio_stats)


CANDIDATES = [("none", False), ("quad", False), ("none", True), ("quad", True)]
CAND_NAME = {("none", False): "MLP", ("quad", False): "PASR quad anchor", ("none", True): "PASR dual encoding",
             ("quad", True): "PASR quad anchor + dual encoding"}


# ============================================================ one independent training run
def run_one(name, X, Y, args, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    tr, va, te = split_indices(len(X), args.test_frac, args.val_frac, args.split_seed)
    t0 = time.time(); base_seed = 1000 * args.split_seed
    if args.auto:       # choose the design by VALIDATION RMSE (physical units, mean over targets / target std)
        sel = []
        for a, d in CANDIDATES:
            f = fit_model(X, Y, tr, va, args, a, d, base_seed)
            v = float(np.mean([np.sqrt(((f["pred"][va, j] - Y.values[va, j]) ** 2).mean()) / Y.values[tr, j].std()
                               for j in range(Y.shape[1])]))
            sel.append(dict(design=CAND_NAME[(a, d)], anchor=a, dual=d, val_nrmse=v, fit=f))
            print(f"    {name}: candidate {CAND_NAME[(a, d)]:34s} validation NRMSE {v:.3e}")
        best = min(sel, key=lambda r: r["val_nrmse"]); anchor, dual = best["anchor"], best["dual"]
        pd.DataFrame([{k: v for k, v in r.items() if k != "fit"} for r in sel]).to_csv(os.path.join(out_dir, "design_selection.csv"), index=False)
        print(f"    {name}: selected '{best['design']}' (lowest validation error)")
        fits = [best["fit"]]
    else:
        anchor, dual, fits = args.anchor, args.dual_encoding, []
    while len(fits) < args.seeds:
        fits.append(fit_model(X, Y, tr, va, args, anchor, dual, base_seed + len(fits)))
    for s, f in enumerate(fits):
        torch.save({"net": f["net"].state_dict(), "anchor_coef": f["coef"], "scale": f["scale"]}, os.path.join(out_dir, f"model_seed{s}.pt"))
        print(f"    {name}: seed {s} done, final train loss {f['hist'][-1][2]:.2e}, val {f['hist'][-1][3]:.2e}")
    hists = [f["hist"] for f in fits]; sc = fits[0]["sc"]
    Pall = np.mean([f["pred"] for f in fits], 0); P = {"train": Pall[tr], "val": Pall[va], "test": Pall[te]}
    design = CAND_NAME[(anchor, dual)] + (" + log(Nt/Nb)" if args.ratio_features else "")
    rows = []
    for k, ix in (("train", tr), ("val", va), ("test", te)):
        for j, c in enumerate(Y.columns):
            rows.append(dict(model=name, design=design, target=c, split=k, n=len(ix), **metrics(Y.values[ix, j], P[k][:, j])))
    M = pd.DataFrame(rows); M.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    mt = {r["target"]: r for r in rows if r["split"] == "test"}
    yt = Y.values[te]
    lbl = "" if anchor == "none" else " (residual network, rescaled)"
    plot_loss(hists, os.path.join(out_dir, "loss_curve.png"), f"{name} [{design}]: training and validation loss{lbl}", args.adam_steps)
    plot_hist(yt, P["test"], list(Y.columns), os.path.join(out_dir, "error_histogram.png"), f"{name} [{design}]: test errors (n = {len(te)})")
    plot_parity(yt, P["test"], list(Y.columns), os.path.join(out_dir, "parity.png"), f"{name} [{design}]: predicted vs true, test set", mt)
    for k, ix in (("test", te), ("val", va)):
        pt = X.iloc[ix].copy()
        for j, c in enumerate(Y.columns): pt[f"{c}_true"] = Y.values[ix, j]; pt[f"{c}_pred"] = P[k][:, j]; pt[f"{c}_error"] = P[k][:, j] - Y.values[ix, j]
        pt.to_csv(os.path.join(out_dir, f"predictions_{k}.csv"), index_label="row")
    pd.DataFrame([dict(seed=s, iteration=r[0], phase=r[1], train_loss=r[2], val_loss=r[3]) for s, h in enumerate(hists) for r in h]
                 ).to_csv(os.path.join(out_dir, "loss_history.csv"), index=False)
    json.dump({"scaler": sc.to_json(), "design": design, "anchor": anchor, "dual_encoding": dual,
               "ratio_features": bool(args.ratio_features), "ratio_mean_sd": fits[0]["ratio_stats"],
               "residual_scale_per_seed": [f["scale"].tolist() for f in fits],
               "args": vars(args), "inputs": list(X.columns), "targets": list(Y.columns),
               "n_train": len(tr), "n_val": len(va), "n_test": len(te), "seconds": time.time() - t0},
              open(os.path.join(out_dir, "scaler.json"), "w"), indent=1)
    for c in Y.columns:
        r = mt[c]; print(f"    {name} | {c:>6s} | {design} | test R2 {r['R2']:.6f}  RMSE {r['RMSE']:.3e}  MaxRelErr {r['MaxRelErr_pct']:.2f}%")
    return M


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="dataset files (.xlsx or .csv); globs allowed")
    ap.add_argument("--targets", nargs="*", default=None, help="target columns (required for single-table files)")
    ap.add_argument("--per-target", action="store_true", help="also train a separate model for each target")
    ap.add_argument("--out", default="results", help="output folder")
    ap.add_argument("--width", type=int, default=128); ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--act", default="silu", choices=["silu", "tanh", "gelu", "relu"])
    ap.add_argument("--adam-steps", type=int, default=2500); ap.add_argument("--lbfgs-iter", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seeds", type=int, default=1, help="networks per model, averaged (3 = final results)")
    ap.add_argument("--log-targets", nargs="*", default=["Nu"], help="targets (by name prefix) trained in log space")
    ap.add_argument("--test-frac", type=float, default=0.2); ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    ap.add_argument("--quick", action="store_true", help="smoke test: 300 Adam steps, 100 L-BFGS iterations")
    ap.add_argument("--anchor", default="none", choices=["none", "quad"],
                    help="PASR-Net: 'quad' = least-squares quadratic anchor + network on the rescaled residual")
    ap.add_argument("--dual-encoding", action="store_true", help="PASR-Net: feed the network both x and log x")
    ap.add_argument("--ratio-features", action="store_true",
                    help="append standardised log(Nt/Nb) to the network inputs and the quadratic anchor's linear terms")
    ap.add_argument("--auto", action="store_true",
                    help="try MLP and the 3 PASR-Net variants, keep the one with the lowest VALIDATION error (4x slower)")
    args = ap.parse_args()
    if args.quick: args.adam_steps, args.lbfgs_iter = 300, 100
    if args.threads: torch.set_num_threads(args.threads)
    files = sorted({f for p in args.files for f in (glob.glob(p) or [p])})
    os.makedirs(args.out, exist_ok=True); allm = []
    for f in files:
        ds = os.path.splitext(os.path.basename(f))[0]
        X, Y = load_dataset(f, args.targets)
        print(f"\n=== {ds}: {len(X)} rows, inputs {list(X.columns)}, targets {list(Y.columns)}")
        allm.append(run_one(ds, X, Y, args, os.path.join(args.out, ds)))
        if args.per_target and Y.shape[1] > 1:
            for c in Y.columns:
                allm.append(run_one(f"{ds}__{c}", X, Y[[c]], args, os.path.join(args.out, f"{ds}__{c}")))
    S = pd.concat(allm, ignore_index=True); S.to_csv(os.path.join(args.out, "summary_metrics.csv"), index=False)
    T = S[S.split == "test"].reset_index(drop=True)
    if len(T):
        labels = [f"{a}\n{b}" for a, b in zip(T.model, T.target)]
        fig, axs = plt.subplots(1, 2, figsize=(max(8, 1.1 * len(T)), 4))
        axs[0].bar(range(len(T)), 1 - T.R2, color="#1f4e8c"); axs[0].set_yscale("log"); axs[0].set_ylabel("1 − R²  (lower is better)")
        axs[1].bar(range(len(T)), T.RMSE, color="#d0552a"); axs[1].set_yscale("log"); axs[1].set_ylabel("RMSE  (lower is better)")
        for ax in axs: ax.set_xticks(range(len(T))); ax.set_xticklabels(labels, fontsize=7); ax.grid(alpha=.3, axis="y")
        fig.suptitle("Test-set accuracy, every model", x=.01, ha="left"); fig.tight_layout()
        fig.savefig(os.path.join(args.out, "summary_r2_rmse.png"), dpi=300); plt.close(fig)
    print(f"\nAll results in {os.path.abspath(args.out)}")
    print(T[["model", "design", "target", "n", "R2", "RMSE", "MAE", "MAPE_pct", "MedRelErr_pct", "P95RelErr_pct", "MaxRelErr_pct", "Within_0p1pct", "Within_0p5pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
