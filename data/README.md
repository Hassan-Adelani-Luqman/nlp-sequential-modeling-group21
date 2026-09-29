# Data

Group copy on Kaggle (private, members only): <https://www.kaggle.com/datasets/luqmanhassanadelani/group21-swahili-audio>

We use the **Swahili Audio Classification** data from Zindi:
<https://zindi.world/competitions/swahili-audio-classification/data>

> **Competition rules:** only the competition data may be used, and it must **not be
> shared** with anyone who isn't a participant. So:
> - every group member **joins the Zindi competition**;
> - the Kaggle copy stays **private** and is shared only with the group;
> - no audio, features or model weights are ever committed to GitHub.
>
> Only the split files in `data/splits/` (clip ids + labels) are committed.

## What's in it

| File | Content |
|---|---|
| `Train.csv` | 4,200 rows: `Word_id` (`.wav` filename), `Swahili_word` (label), `English_translation` |
| `Test.csv` | 1,800 rows: `Word_id` (unlabelled) |
| `SampleSubmission.csv` | `Word_id` + 12 probability columns (evaluated with log loss) |
| `Swahili_words.zip` | 6,000 `.wav` files (506 MB zipped, ~900 MB unzipped) |
| `Swahili_Audio_StarterNotebook.ipynb` | Zindi's starter notebook (optional, not used) |

Measured properties:
- **Classes:** 12 words, 350 clips each (perfectly balanced): *moja … kumi* (1–10), *ndio* (yes), *hapana* (no).
- **Audio format:** all files are 16 kHz, mono, 16-bit PCM, with none unreadable.
- **Duration:** 2.2–62 s, median 4.3 s. The spoken word itself is well under a second, so most of each clip is silence or background noise.
- **Duplicates:** none byte-identical.

## 1. Download (once, M1)
1. Log in to Zindi, join the competition, and download `Train.csv`, `Test.csv`,
   `SampleSubmission.csv` and `Swahili_words.zip`.
2. Unzip so that you get `data/raw/Swahili_words/*.wav`. The loader also accepts a nested
   `Swahili_words/Swahili_words/` folder.

## 2. Share with the group on Kaggle (once, M1)
1. Kaggle → **Datasets → New Dataset**:
   - Upload the three CSVs and `Swahili_words.zip`. Kaggle extracts the zip.
   - Name it `group21-swahili-audio` and keep it **Private**.
2. Dataset page → **Settings → Sharing**: add the other three members.
3. In any notebook: **Add Input → Datasets → Your Datasets / Shared with you →
   group21-swahili-audio**. `src/paths.py` finds it automatically under `/kaggle/input/`.

## 3. Colab
The data is found automatically in either of these places:
- `/content/drive/MyDrive/group21-swahili-audio/`: upload the CSVs and the unzipped `Swahili_words/` to your Drive and mount it.
- `/content/data/raw/`, by downloading from Kaggle with your own API token:
  ```python
  # default is luqmanhassanadelani/group21-swahili-audio; override with SWN_KAGGLE_DATASET if needed
  # put kaggle.json in ~/.kaggle/ first (Kaggle → Settings → Create New Token)
  from src.paths import download_from_kaggle
  download_from_kaggle()
  ```

Anywhere else, set `SWN_DATA_DIR=/path/to/folder/with/Train.csv`.

## 4. The frozen split (`data/splits/`)
M1 created it once and it is committed. **Do not regenerate it.**

| File | Content |
|---|---|
| `train_ids.csv`, `val_ids.csv`, `test_ids.csv` | `id,label`: 2,940 / 630 / 630 clips, stratified 70/15/15, seed 42 |
| `label_map.json` | word → integer id (alphabetical, fixed) |
| `manifest.json` | seed, sizes, class counts per split, glosses, duplicate stats, **split hash** |

Split hash: **`b74d294fa65d9ae708be8f50b598e6eb`**. Check that your copy matches:
```bash
python -m src.data          # must print "OK - matches manifest"
```

If the split ever has to change, agree it with the whole group and re-run every experiment.
