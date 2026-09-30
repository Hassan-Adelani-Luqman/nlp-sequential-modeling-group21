# Failed runs, kept for transparency

The plan logs every run, including failures. These runs failed because of a bug, so they were rerun after the fix
and are kept here, outside the experiment table, rather than deleted.

| Run | What happened | Fix |
|---|---|---|
| `A4-R2-03_globalfill` | A4 with MFCC-40 + Δ + ΔΔ inputs collapsed to chance: 8.3% accuracy, log loss 2.485 = ln 12. The model predicted uniform probabilities. | The "full" augmentation padded shifted and stretched clips with the clip's *global* minimum, and filled masks with its *global* mean. That is correct for log-mel (one dB scale), but not for MFCCs: c0 is often near −600, and writing it into the 119 delta channels created huge outliers. `LogMelAugment(fill="channel")` now uses each channel's own minimum and mean for MFCC-type inputs. The A4 notebook reran A4-R2-03 with this fix. |

The global fill stays the default, so every logged log-mel run (and A3's runs) reproduces exactly.
