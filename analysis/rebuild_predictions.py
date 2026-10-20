"""
rebuild_predictions.py - recompute a trained model's predictions on every split from its saved weights
(model_seed*.pt + scaler.json), for runs made before train_separate.py wrote predictions_val.csv.

The rebuilt test predictions are compared with the saved predictions_test.csv; the script stops if they differ.

Usage: python analysis/rebuild_predictions.py <model folder> <data file> <output folder>
       writes <output folder>/predictions_val.csv in the same format as train_separate.py
"""
import json, os, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_separate as T

folder, data, out = sys.argv[1:4]
cfg = json.load(open(os.path.join(folder, "scaler.json"))); a = cfg["args"]
X, Y = T.load_dataset(data); Y = Y[cfg["targets"]]
tr, va, te = T.split_indices(len(X), a["test_frac"], a["val_frac"], a["split_seed"])
log_cols = [c for c in Y.columns if any(c.lower().startswith(q.lower()) for q in a["log_targets"]) and (Y[c] > 0).all()]
sc = T.Scaler().fit(X.iloc[tr], Y.iloc[tr], log_cols)
x = sc.x(X)
z = torch.cat([x, T.log_features(X, X.iloc[tr])], 1) if cfg["dual_encoding"] else x
rf = T.ratio_feature(X, tr)[0] if cfg.get("ratio_features") else None
if rf is not None: z = torch.cat([z, rf], 1)
Q = T.quad_features(x)
if rf is not None: Q = torch.cat([Q, rf], 1)
preds = []
for s in range(a["seeds"]):
    ck = torch.load(os.path.join(folder, f"model_seed{s}.pt"), weights_only=False)
    m = T.MLP(z.shape[1], Y.shape[1], a["width"], a["depth"], a["act"]); m.load_state_dict(ck["net"])
    base = Q @ ck["anchor_coef"] if cfg["anchor"] == "quad" else torch.zeros(len(X), Y.shape[1])
    with torch.no_grad(): preds.append(sc.inv((base + ck["scale"] * m(z)).numpy()))
P = np.mean(preds, 0)

saved = pd.read_csv(os.path.join(folder, "predictions_test.csv"))
for j, c in enumerate(Y.columns):
    d = np.abs(saved[f"{c}_pred"].values - P[te, j]).max()
    if d > 1e-9 * np.abs(P[te, j]).max(): raise SystemExit(f"{folder}: rebuilt test predictions differ from saved ones (max {d:.2e})")
os.makedirs(out, exist_ok=True)
pv = X.iloc[va].copy()
for j, c in enumerate(Y.columns): pv[f"{c}_true"] = Y.values[va, j]; pv[f"{c}_pred"] = P[va, j]; pv[f"{c}_error"] = P[va, j] - Y.values[va, j]
pv.to_csv(os.path.join(out, "predictions_val.csv"), index_label="row")
print(f"{folder}: test predictions reproduced, wrote {out}/predictions_val.csv")
