# Project history and decisions

Chronological record of the work done in Claude (chat, 16–19 Oct 2026) and in Claude Code, with the reason for each decision. Numbers are test-set values unless stated.

## 1. First dataset (`Data.xlsx`, superseded)

- 4,800 rows, 10 inputs (P3, λ, δ₁, Hs, δ₂, Cr, Ea, B1, B2, n), outputs Cf, Nu, Sh.
- Found to be **one-factor-at-a-time**: 4,798 of 4,800 rows change exactly one parameter from a single baseline. No interaction information exists in such data.
- Consequence: random splits give R² ≈ 1 trivially (test points sit between training neighbours on smooth 1-D curves). A per-curve cubic spline beat every network on random splits.
- Holding out the top 20% of each sweep (extrapolation) exposed the real gap. Nu also had a cliff as Hs → 1 (Nu falls from −0.02 to −2.8).
- A structured additive network ("SweepNet") was built for that design; it was not clearly better than a plain MLP.
- **Decision:** generate space-filling data instead. The user supplied the cone/plate/wedge files.

## 2. Literature research (deep-research report)

Key conclusions used later:
- Smooth, noise-free simulation data favour smooth-activation MLPs over gradient-boosted trees (Grinsztajn et al., NeurIPS 2022 explains why trees win on irregular real tables; solver data are the opposite case).
- Adam followed by L-BFGS in float64 is the best-documented route to very low loss on full-batch problems (Rathore et al., ICML 2024).
- R² alone is inadequate for nonlinear models (Spiess & Neumeyer, BMC Pharmacology 2010); report RMSE, maximum error and held-out designs.
- Newer optimisers (Lion, Sophia, Muon, SOAP) have little evidence on small regression problems.

## 3. Cone / plate / wedge data, shared model (one network, 6 outputs)

- Data verified as an exact Latin hypercube; random split is fair.
- Round-1 ablation (single seed): SiLU beat tanh by ~30%; shared trunk + one head per geometry beat a flat 6-output layer; six separate networks were no better than one joint network at 6× the parameters; log Nu cut worst-case Nu relative error from ~5% to ~1%; Padé activation and residual blocks did not help; gradient boosting was 20–30× worse.
- Final shared model: trunk 11→128×4 SiLU, heads 128→64→2 per geometry, 76,230 parameters, 3-seed ensemble, 80/20 split.
  - Random split: mean R² 0.999998, RMSE 2.8e-4 – 4.7e-4.
  - Corners split (outer 15% of the input cube, never trained on): mean R² 0.999988, worst relative error 1.1%. Sh errors in the corners were shared by all seeds (systematic, not fixable by more seeds).
- Deliverables: results page, paper figures, a mathematical-foundations document (all equations and justifications).

## 4. Separate models per dataset (user's request)

- `train_separate.py` written: one independent model per dataset (optionally per target), 70/10/20 split, outputs R², RMSE, loss curve, error histogram, parity plot.
- Reference run (single seed, plain MLP 4×128): Nu RMSE 4.4–4.5e-4, Sh RMSE 6.0–6.5e-4, max relative error ≤ 0.85%.
- As expected from the ablation, separate models were about as accurate as the shared one, not more.

## 5. User's tuned run in Claude Code

The user tuned the pipeline in Claude Code and reported the table in `docs/RESULTS_AND_REVIEW.md` §2 (max relative errors 0.23–0.45%). What the tuning changed was not recorded. See the review points there.

## 6. Comparison with an RNN (Discover Nano 21:515, 2026, trapezoidal fin paper)

- That paper's RNN steps along the fin coordinate X; our outputs are scalars per parameter set, so there is no natural sequence.
- Tested on cone data, same split and training: RNN/GRU reading the parameters one at a time were 2.3–3.4× worse in RMSE and 3–10× slower; shuffling the parameter order changed the results (order is arbitrary); an RNN with all inputs at once is a one-layer tanh MLP and was worst.
- **Decision:** no recurrent models for these data. Mention the comparison in the paper.

## 7. New design search

Tested on cone data, judged on validation error (single seed), then confirmed with 3 seeds on all geometries:

| Idea | Outcome |
| --- | --- |
| Bigger MLP 6×256 | no better: capacity is not the limit |
| Power-law correlation (Nu ∝ ∏xᵢ^aᵢ) alone | 215% max error: Nu is not a power law here |
| Power-law backbone + neural residual | no real gain |
| Multi-stage residual network (Wang & Lai 2023) | mixed; a second stage on top of the best design added nothing (matches their note that the method slows beyond 2-D) |
| Cubic response-surface anchor | overfits |
| **Quadratic response-surface anchor + rescaled residual network** | **Nu error 2.5–3.2× lower, confirmed on all geometries** |
| **Dual encoding (x and log x as inputs)** | **Sh error 1.3–1.7× lower, confirmed** |
| log-transforming Sh | worse |

The combination was named PASR-Net (Polynomial-Anchored, Scaled-Residual network) in the chat. Its components are not new (response surface methodology, residual hybrids, multi-stage rescaling from Wang & Lai); the combination and its validation on this problem are the contribution. The name is close to an existing network ("PASSRnet", CVPR 2019), so the user may rename it.

## 8. Documentation produced in chat (not in this repo)

- Results page with parity, residual, loss, importance and partial-dependence plots.
- Mathematical foundations document (data analysis, transforms, network equations, optimisation, ensembling, metrics, worked example).
- Animated workflow page and four block diagrams of the final design.

## 9. Final configuration and fresh test sets (Claude Code, 19 Oct 2026)

- Data checked against CLAUDE.md: all facts hold (plate–wedge Sh correlation is 0.99972 rather than 0.9999).
- The user's tuned pipeline (§5) could not be recovered: no earlier version in git and the user no longer has the code. Its table is marked not reproducible; the planned "tuned settings + quad anchor" experiment was dropped.
- Main run `--per-target --auto --seeds 3` (split seed 0). Joint vs per-target chosen on validation RMSE (`analysis/select_design.py`): **per-target with the quadratic anchor for every geometry and target**. Margins are below run-to-run noise except cone Nu (joint model 1.9× worse, because its design was chosen for both targets and picked dual encoding).
- Two earlier beliefs not confirmed: Sh is not better in the joint model once the PASR designs are available, and dual encoding does not beat the quadratic anchor for Sh (it beat the plain MLP).
- Final design fixed and rerun on split seeds 1–4 (`analysis/split_commands.py`). Paper table with mean ± sd over seeds 0–4 in `results_final/` (`analysis/final_report.py`).
- Checks: three failures, all Sh on split 3 (RMSE > 2× reference on plate and wedge, max relative error 1.51% on cone), all caused by a handful of rows with Nb at its lower bound and Nt/Nb ≈ 5–7. New open item: Nt/Nb as an input feature (needs the user's agreement; to be judged on validation and confirmed on unused split seeds 5–9, because the problem was found on test rows).
- Process notes: the three geometries were trained as parallel single-thread processes (1.6× faster than one 4-thread process); `results_pasr/summary_metrics.csv` was rebuilt from the per-model files because each process overwrote it. While checking that the run had finished, the split-0 test table printed by `train_separate.py` was seen before the joint vs per-target choice; that choice is made by a script from validation rows only.

## 10. Nt/Nb as a physics-based feature (Claude Code, 20 Oct 2026)

Approved by the user after §9 showed Sh errors concentrated where Nt/Nb > 4.

- `train_separate.py`: new `--ratio-features` (log(Nt/Nb), standardised on training rows, appended to the network inputs and to the anchor's linear terms; default off, verified bit-identical to the earlier smoke test); median and 95th-percentile relative error added to `metrics.csv`; `predictions_val.csv` written for every model.
- `analysis/rebuild_predictions.py` rebuilds validation predictions of earlier runs from their weights (checked: all 30 reproduce their saved test predictions exactly), so the baseline did not need retraining.
- Protocol: (a) final design vs (b) + `--ratio-features`, both 3-seed, on VALIDATION over split seeds 0–4. Decision rule fixed before any (b) result: (b) wins for a target if its validation RMSE is lower in ≥ 12 of 15 (geometry × split) pairs. Only then confirm on unused split seeds 5–9 (test once), with (a) also run there for comparison.
- Check 1 replaced at the user's request by "RMSE within mean + 2 sd of split seeds 0–4 for that model". Note: on seeds 0–4 themselves this check cannot fail in practice (all 30 pass, including the two split-3 Sh runs that failed the old check), because the failing runs widen their own band. The R² ≥ 0.9999 floor is no longer part of check 1.
- **Result.** Validation (seeds 0–4): (b) better for Sh in 14/15 pairs (median 1.28×), for Nu in 8/15 (1.02×, no change). Test confirmation on unused seeds 5–9 (Sh only, (a) also run): (b) better in 14/15 pairs, median 1.12×; 95th-percentile relative error lower in 15/15, maximum relative error lower in only 6/15.
- **Interpretation.** The feature improves Sh everywhere by a similar fraction; it does not remove the small-Nb corner (still 6–7× harder; still ~6 of the 10 worst rows). Describe it in the paper as a general improvement, not as a fix for the corner.
- **Decision.** Final design: Nu per-target quad anchor (unchanged); Sh per-target quad anchor + `--ratio-features`. Paper numbers: Nu from split seeds 0–4, Sh from split seeds 5–9 (RESULTS_AND_REVIEW §8.5).
- **Claims from the chat sessions that did not hold:** "Sh does better in the joint model" and "dual encoding is best for Sh" (§9). Neither should appear in the paper.

- **Check 1 revised (user decision, 20 Oct 2026):** band from the reference model's split seeds 0–4, applied only to runs on new seeds, with R² ≥ 0.9999 restored (CLAUDE.md, `analysis/final_report.py`).
