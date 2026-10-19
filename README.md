# Nu–Sh neural surrogate

Neural-network surrogates for the Nusselt (Nu) and Sherwood (Sh) numbers of a boundary-layer heat and mass transfer problem on a cone, a plate and a wedge.

- `CLAUDE.md`: method, commands, checks and rules (read automatically by Claude Code).
- `PROMPT_FOR_CLAUDE_CODE.md`: the prompt to start a Claude Code session.
- `docs/PROJECT_HISTORY.md`: what was tried and why.
- `docs/RESULTS_AND_REVIEW.md`: all results so far and open review points.
- `train_separate.py`: the training pipeline. `python train_separate.py data/*.xlsx --per-target --auto --seeds 3`
- `experiments/`: archived scripts behind the design decisions.
