# Superseded runs, kept for transparency

These runs finished correctly, but a later decision earlier in the greedy search changed the configuration they
build on. They are kept here, outside the experiment table, so the history stays visible.

## `a4_logmel_path/`: A4-R2-04 … A4-R2-11 and A4-R3-01 (seeds 42, 13, 7)

A4's R2 search is greedy: each step changes one factor of the current best configuration. These runs were
trained while A4-R2-03 (MFCC-40 + Δ + ΔΔ input) had collapsed to chance because of the augmentation fill bug
(see `results/failed_runs/`), so every later step built on **log-mel** inputs.

After the fix, A4-R2-03 reached log loss 0.133 (from 0.140) and was kept. Every later step therefore changes
an **MFCC-40** configuration, and these log-mel runs no longer answer the questions logged under their ids.
`ExperimentRunner` refuses to reuse a logged run whose parameters differ from the ones requested, so the runs
were moved here and retrained on the corrected path.

Best result on the superseded log-mel path, for reference (validation, seed 42):

| Run | Change | Log loss | Accuracy |
|---|---|---|---|
| A4-R2-05 | 2.0 s window | 0.129 | 96.5% |
| A4-R3-01 | same configuration, seeds 42 / 13 / 7 | 0.138 ± 0.009 | 96.8% ± 0.2 |

These runs are not used anywhere in the report's results. Their checkpoints were deleted.
