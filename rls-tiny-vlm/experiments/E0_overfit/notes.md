# E0 — Overfit one batch

**Question.** Can the pipeline learn at all? (bug check, not a result)
**Prediction.** Dropout off, 32 fixed samples, lr 1e-3: loss → ~0 in a few hundred steps and 100 % exact match
on that batch. If not, suspect: loss masking, target shift (position V-1 predicts letter 1), attention mask, optimizer.
**Command.** `python -m experiments.E0_overfit.overfit --device cpu` → `overfit_loss.png`, `results.json`.
**Result / log.** (paste the printed numbers and note any bug found along the way)
