# Paper deliverables (final design, split seeds 5–9)

Regenerate:

```bash
python analysis/final_report.py results_paper/config_Nu.csv --seeds 5 6 7 8 9 \
  --template "results_confirm/a/split{s}/{target}/data_{geometry}" --out results_paper/report_Nu
python analysis/final_report.py results_paper/config_Sh.csv --seeds 5 6 7 8 9 \
  --template "results_confirm/b/split{s}/{target}/data_{geometry}" --design-suffix " + log(Nt/Nb)" \
  --band-template "results_ratio/split{s}/{target}/data_{geometry}" --out results_paper/report_Sh
python analysis/paper_figures.py
```

- `paper_table.md` / `paper_table.csv`: mean ± sd over split seeds 5–9, both targets.
- `report_<target>/`: per-split table, checks, worst test rows.
- `figures/`: split seed 5 loss curve, error histogram and parity plot per model; `summary_rmse.png`.
