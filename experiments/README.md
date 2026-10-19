# Archived experiments (chat session, 18–19 Oct 2026)

These scripts produced the comparisons in `docs/RESULTS_AND_REVIEW.md`. They are kept for reproducibility; the main pipeline is `train_separate.py`.
They import `train_separate.py` and expect the data files in the working directory, so run them from a folder that contains both, e.g. `cp data/*.xlsx . && PYTHONPATH=. python experiments/confirm.py data_cone.xlsx`.

| Script | What it tested | Results |
| --- | --- | --- |
| `rnn_test.py` | RNN / GRU (as in the Discover Nano 2026 fin paper) vs MLP, cone | `rnn_results.json` |
| `new_arch.py` | bigger MLP, power-law backbone, quadratic anchor, multi-stage residual, log inputs, cone | `new_arch_results.json` |
| `pasr.py` | anchor degree, dual encoding, two stages, cone | `pasr_data_cone.json` |
| `confirm.py` | final 3-seed confirmation, all geometries, one model per target | `confirm.json` |
