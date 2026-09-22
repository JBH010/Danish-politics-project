# DK ↔ EP Title Embeddings — Design

**Date:** 2026-09-21  
**Status:** Approved for implementation after user review of this spec  
**Deliverable:** `notebooks/Embeddings_DK_EP.ipynb`

## Goal

Embed Danish Folketinget roll-call titles (already translated to English) and European Parliament vote titles in a shared English embedding space, save the vectors, and produce a first visual/statistical look at whether the two corpora sit close or far apart.

This is separate from the existing ANEY candidate-test ↔ DK vote embedding work (`Embeddings_ANEY_v*`).

## Decisions (locked)

| Choice | Decision |
|--------|----------|
| Scope | Embeddings + first-look analysis (centroid distance, similarity distributions, sample nearest neighbors, 2D plot) |
| Text | Titles only: DK `sag_titel_en` vs EP `display_title` |
| Model | `sentence-transformers/all-mpnet-base-v2` |
| EP filter | Prefer main votes: if `is_main` exists, keep `True` only; older periods without the column keep all rows |
| Structure | Single notebook (may be long) |

## Data inputs

### Danish Parliament

- Path: `data/temp_data/parliament/roll_calls_translated.csv`
- Embed column: `sag_titel_en`
- Identity: `afstemningid`
- Keep for meta: `sag_titel` (Danish original), `sag_titel_en`, `dato` if present
- Drop rows with missing/empty `sag_titel_en`

### European Parliament

- Paths: `EP/EP-data/{2009-2014,2014-2019,2019-2024,2024-2029}/ep_votes.csv`
- Embed column: `display_title`
- Identity: `id` + `period` (period folder name)
- Filter: `is_main == True` when column exists (2019+ currently; all rows there are already main). Periods without `is_main` (2009–2014, 2014–2019) include all rows.
- Keep for meta: `display_title`, `timestamp`/`dato` if present, `period`, `procedure_reference` if present
- Drop rows with missing/empty `display_title`

**Schema note:** Older EP CSVs have fewer columns (`id`, `timestamp`, `display_title`, …). Newer ones add `is_main`, `procedure_title`, etc. Loading must tolerate both schemas.

## Model & encoding

- Model: `sentence-transformers/all-mpnet-base-v2`
- Encode with `normalize_embeddings=True` so cosine similarity is a dot product
- No E5-style `"query: "` / `"passage: "` prefixes (mpnet does not use them)
- Batch size: start at 32–64; adjust if memory-constrained
- Device: auto (CUDA/MPS/CPU)

## Outputs

Directory: `data/temp_data/embeddings/` (create if missing)

| File | Contents |
|------|----------|
| `dk_title_embeddings.npy` | `(n_dk, d)` float32, L2-normalized |
| `dk_title_meta.csv` | Row-aligned metadata (ids, titles, date) |
| `ep_title_embeddings.npy` | `(n_ep, d)` float32, L2-normalized |
| `ep_title_meta.csv` | Row-aligned metadata (ids, titles, period, timestamp) |

Row `i` in each `.npy` must match row `i` in the corresponding meta CSV.

## First-look analysis (same notebook)

1. **Corpus sizes** — print `n_dk`, `n_ep`, embedding dim, model name  
2. **Centroid cosine** — mean DK vector vs mean EP vector (normalized means or cosine of means)  
3. **Similarity distributions** — histogram of cosines for:
   - random sample of DK–DK pairs
   - random sample of EP–EP pairs
   - random sample of DK–EP pairs  
   Use a fixed `random_state` and a manageable sample size (e.g. 5k–20k pairs each) so it runs quickly.  
4. **Nearest neighbors sanity check** — for a few random DK titles, show top-5 EP by cosine; for a few random EP titles, show top-5 DK  
5. **2D projection** — PCA (default; UMAP optional if available) on a balanced sample (e.g. all DK + random subsample of EP to similar count), scatter colored by parliament (`DK` vs `EP`)

Interpretation guidance in markdown: high within-corpus and lower cross-corpus similarity suggests separated agendas; overlapping clouds / high DK–EP cosines suggest topical overlap (not identity of specific votes).

## Out of scope

- Embedding amendment subjects or longer EP procedure text
- Full all-pairs retrieval tables for every vote
- Linking embeddings to individual MEP/MP vote choices
- Reusing or comparing against ANEY multilingual-e5 embeddings
- Automatic topic labels / clustering beyond the first-look plot

## Success criteria

- Notebook runs end-to-end from project `notebooks/` working directory
- Artifacts written under `data/temp_data/embeddings/` and reloadable
- At least one plot and the three similarity histograms are produced
- Sample nearest-neighbor tables are readable enough to spot obvious topical matches or nonsense

## Future extensions (not in this notebook unless trivial)

- This was done Expand EP text to `display_title` + `procedure_title`
- Include amendments (`is_main=False`)
- Restrict to overlapping calendar years with DK
- Switch or A/B models
