# Prompt to paste into the Claude Code session

Copy everything inside the box below into the Claude Code session after the repository (with the three data files in `data/`) is open.

---

```
You are continuing my research project on neural-network surrogates for Nu and Sh (cone, plate, wedge).
Read CLAUDE.md, docs/PROJECT_HISTORY.md and docs/RESULTS_AND_REVIEW.md first. They contain the method,
every result so far, my earlier tuned results and the open review points. Follow the rules in CLAUDE.md,
especially: choose everything by validation error, report test error once at the end.

Do the following in order, and tell me after each step what you found before moving on.

1. Environment and data
   - Install requirements. Check that data/data_cone.xlsx, data/data_plate.xlsx, data/data_wedge.xlsx exist.
   - Print shapes, column names, min/max per column and missing values. Confirm they match the facts in
     CLAUDE.md. If anything differs, stop and tell me.

2. My earlier tuning
   - Check the git history and branches for an earlier tuned version of the pipeline. If you find one,
     diff it against train_separate.py and list exactly what the tuning changed.
   - If you cannot find it, ask me what I changed and whether I compared settings on validation or test error.

3. Smoke test
   python train_separate.py data/*.xlsx --quick --out results_quick
   Confirm all output files are written.

4. Main run (long: run in the background, report progress)
   python train_separate.py data/*.xlsx --per-target --auto --seeds 3 --out results_pasr
   This trains, for each geometry, one joint model (Nu+Sh, folder <dataset>) and one model per target
   (folders <dataset>__Nu, <dataset>__Sh). For each geometry and target, keep whichever has the lower
   VALIDATION error (split == val in metrics.csv). Sh has done better in the joint model before.

5. Combine with my tuning (only if step 2 found real changes)
   Apply my tuned settings via command-line flags (do not edit the method silently) together with the
   design chosen in step 4, and compare on validation error.

6. Fresh test set
   Rerun the final configuration with --split-seed 1 into a new folder, so the final numbers come from a
   test set that no decision has touched. Then repeat for split seeds 2, 3, 4 and report mean ± sd.

7. Final report
   Produce results_final/paper_table.csv and a markdown table with, per geometry and target:
   design, R² (decimal), RMSE, MAE, MAPE %, max relative error %, within 0.1 %, within 0.5 %,
   for split seed 0 and the mean ± sd over seeds 0–4. No "accuracy %" column.
   Run the five checks in CLAUDE.md and list any that fail.
   Point me to the loss curve, error histogram and parity plot for each final model.

8. Update docs/RESULTS_AND_REVIEW.md and docs/PROJECT_HISTORY.md with the new results and commit
   everything except the results folders' model weights.
```

---

## Notes for me (not for the prompt)

- The cloud session needs the data files inside the repository (`data/`), or uploaded at the start of the session.
- Step 4 takes roughly 1.5–3 hours on a small cloud machine (4 designs × 3 seeds × 2 targets × 3 geometries, plus the joint runs). If time is short, ask Claude Code to run cone first.
- If you want only the fixed final design without the search, replace `--auto` with `--anchor quad` for Nu and `--dual-encoding` for Sh, using `--targets Nu` / `--targets Sh` on separate runs.
