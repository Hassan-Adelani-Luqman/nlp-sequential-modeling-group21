# AI-use log

The brief requires us to disclose significant AI use. Add a row whenever an AI tool
contributes to the work, and say how the output was checked. The report's
AI-disclosure section is written from this log.

| Date | Member | Tool | What it was used for | How we verified / what we changed |
|---|---|---|---|---|
| 2026-09-26 | M1 | Claude Code | Drafting the phased project plan (Plan.md) from the brief and rubric | Reviewed against the brief and rubric; dataset choice and model list discussed and agreed by the group |
| 2026-09-26 | M1 | Claude Code | Scaffolding Phase 1 code: `src/paths.py`, `src/data.py`, `src/utils.py`, `src/evaluate.py`, smoke tests, READMEs | Read and understood every function; smoke tests pass (`pytest -q`); split checked on the real data |
| 2026-09-26 | M1 | Claude Code | Rewriting the plan and Phase 1 code for the audio challenge (news data was inaccessible): audio loader, `src/features.py` (energy-window crop, log-mel/MFCC, caching), audio smoke tests; profiling the recordings | Checked file-format and duration stats on the real data; tests pass on synthetic and real data; split hash b74d294f… recorded |
| 2026-09-29 | M1 | Claude Code | Checking energy-window crops on real clips (energy captured, visual spectrogram check), fixing off-centre crops, adding shared feature presets, versioned cache keys and the feature-cache build | Inspected crop plots for all 12 words and the worst cases; measured centring over all 4,200 clips (90% within 0.2 s); tests pass; cache shapes checked against the manifest |
| 2026-09-30 | M1 | Claude Code | Phase 4 baselines: A1/A2 model code, temperature scaling, word-pair metric, shared plot style, the 02_baselines notebook and its observation notes; diagnosing the SVM-probability and HMM-NaN issues | Compared SVM probability methods on val before choosing; reproduced the HMM failure and verified the MAP-prior fix (0/24 degenerate); smoke-ran the notebook; checked every figure and every number quoted in the notes against the logged runs; corrected one unsupported claim (3-state HMMs being order-free) |
