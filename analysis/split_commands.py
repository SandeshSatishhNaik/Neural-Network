"""
split_commands.py - step 6: print the training commands that rerun the final configuration
(design fixed by the split-seed-0 validation choice, no --auto) on fresh split seeds.

Output folders: results_split<s>/joint/data_<g>   (joint Nu+Sh model, design chosen for the joint model)
                results_split<s>/<target>/data_<g> (one model for that target only)

Usage: python analysis/split_commands.py results_final/final_config.csv 1 2 3 4 | xargs -P 3 -I{} sh -c "{}"
"""
import sys
import pandas as pd

cfg = pd.read_csv(sys.argv[1]); seeds = [int(s) for s in sys.argv[2:]]
for s in seeds:
    done = set()
    for r in cfg.itertuples():
        flags = (" --anchor quad" if r.anchor == "quad" else "") + (" --dual-encoding" if r.dual_encoding else "")
        if r.source == "joint":
            key, out, tgt = ("joint", r.geometry), f"results_split{s}/joint", ""
        else:
            key, out, tgt = (r.target, r.geometry), f"results_split{s}/{r.target}", f" --targets {r.target}"
        if key in done: continue
        done.add(key)
        print(f".venv/bin/python -u train_separate.py data/data_{r.geometry}.xlsx{tgt}{flags} --seeds 3 --split-seed {s} "
              f"--threads 1 --out {out} > logs/split{s}_{key[0]}_{r.geometry}.log 2>&1")
