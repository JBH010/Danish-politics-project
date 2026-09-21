# DK ↔ EP Title Embeddings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `notebooks/Embeddings_DK_EP.ipynb` that embeds DK translated titles and EP main titles with `all-mpnet-base-v2`, saves artifacts, and produces a first-look similarity/PCA analysis.

**Architecture:** One exploratory notebook. Load and clean both corpora, encode with SentenceTransformer, persist `.npy` + aligned meta CSVs under `data/temp_data/embeddings/`, then compute centroid cosine, sampled pairwise similarity histograms, sample nearest neighbors, and a PCA scatter plot.

**Tech Stack:** Python, pandas, numpy, sentence-transformers, scikit-learn (PCA), matplotlib

## Global Constraints

- Spec: `docs/superpowers/specs/2026-09-21-dk-ep-title-embeddings-design.md`
- Model: `sentence-transformers/all-mpnet-base-v2` with `normalize_embeddings=True`
- DK text: `sag_titel_en` from `data/temp_data/parliament/roll_calls_translated.csv`
- EP text: `display_title` from all four `EP/EP-data/*/ep_votes.csv`; filter `is_main==True` only when column exists
- No E5 query/passage prefixes
- Notebook working directory assumed: `notebooks/` (paths use `Path("..")`)
- Do not modify ANEY embedding notebooks

---

### Task 1: Scaffold notebook — paths, loads, EP concat + filter

**Files:**
- Create: `notebooks/Embeddings_DK_EP.ipynb`

**Interfaces:**
- Produces: `df_dk` (columns at least `afstemningid`, `sag_titel`, `sag_titel_en`, `dato`), `df_ep` (columns at least `id`, `display_title`, `period`, `timestamp` if available), constants `PROJECT_DIR`, `DK_PATH`, `EP_PERIODS`, `OUT_DIR`

- [ ] **Step 1: Create the notebook with markdown goal + setup cell**

Create `notebooks/Embeddings_DK_EP.ipynb` with:

Markdown cell:
```markdown
# DK ↔ EP Title Embeddings

Embed translated Danish Folketinget roll-call titles and European Parliament vote titles with `all-mpnet-base-v2`, save vectors, and inspect whether the two corpora sit close or far in embedding space.

Spec: `docs/superpowers/specs/2026-09-21-dk-ep-title-embeddings-design.md`
```

Code cell:
```python
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_DIR = Path("..")
DK_PATH = PROJECT_DIR / "data" / "temp_data" / "parliament" / "roll_calls_translated.csv"
EP_DATA_DIR = PROJECT_DIR / "EP" / "EP-data"
EP_PERIODS = ["2009-2014", "2014-2019", "2019-2024", "2024-2029"]
OUT_DIR = PROJECT_DIR / "data" / "temp_data" / "embeddings"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
RANDOM_STATE = 42
```

- [ ] **Step 2: Load DK titles**

```python
df_dk = pd.read_csv(DK_PATH)
df_dk = df_dk.dropna(subset=["sag_titel_en"]).copy()
df_dk["sag_titel_en"] = df_dk["sag_titel_en"].astype(str).str.strip()
df_dk = df_dk[df_dk["sag_titel_en"].str.len() > 0].reset_index(drop=True)

print(f"DK titles: {len(df_dk):,}")
print(df_dk[["afstemningid", "sag_titel", "sag_titel_en"]].head(3))
```

- [ ] **Step 3: Load and concat EP votes with optional `is_main` filter**

```python
ep_frames = []
for period in EP_PERIODS:
    path = EP_DATA_DIR / period / "ep_votes.csv"
    if not path.exists():
        print(f"Missing: {path}")
        continue
    df = pd.read_csv(path)
    df["period"] = period
    if "is_main" in df.columns:
        before = len(df)
        df = df[df["is_main"] == True].copy()
        print(f"{period}: {before:,} → {len(df):,} (is_main=True)")
    else:
        print(f"{period}: {len(df):,} (no is_main column; keeping all)")
    ep_frames.append(df)

df_ep = pd.concat(ep_frames, ignore_index=True)
df_ep = df_ep.dropna(subset=["display_title"]).copy()
df_ep["display_title"] = df_ep["display_title"].astype(str).str.strip()
df_ep = df_ep[df_ep["display_title"].str.len() > 0].reset_index(drop=True)

print(f"EP titles: {len(df_ep):,}")
print(df_ep[["id", "period", "display_title"]].head(3))
```

- [ ] **Step 4: Smoke-check loads (run notebook cells)**

Expected: DK ≈ 2,128 rows; EP ≈ 19k (all older periods + main newer rows). No missing title columns. `OUT_DIR` exists.

- [ ] **Step 5: Commit**

```bash
git add notebooks/Embeddings_DK_EP.ipynb
git commit -m "Add DK-EP embeddings notebook scaffold and data loads"
```

---

### Task 2: Encode titles and save embeddings + meta

**Files:**
- Modify: `notebooks/Embeddings_DK_EP.ipynb`
- Create (at runtime): `data/temp_data/embeddings/dk_title_embeddings.npy`, `dk_title_meta.csv`, `ep_title_embeddings.npy`, `ep_title_meta.csv`

**Interfaces:**
- Consumes: `df_dk`, `df_ep`, `MODEL_NAME`, `OUT_DIR`
- Produces: `dk_emb` `(n_dk, 768)`, `ep_emb` `(n_ep, 768)`, both L2-normalized float32; meta CSVs row-aligned with embeddings

- [ ] **Step 1: Load model and encode both corpora**

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(MODEL_NAME)

dk_texts = df_dk["sag_titel_en"].tolist()
ep_texts = df_ep["display_title"].tolist()

dk_emb = model.encode(
    dk_texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True,
)
ep_emb = model.encode(
    ep_texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True,
)

dk_emb = np.asarray(dk_emb, dtype=np.float32)
ep_emb = np.asarray(ep_emb, dtype=np.float32)

print(MODEL_NAME)
print(f"DK embeddings: {dk_emb.shape}")
print(f"EP embeddings: {ep_emb.shape}")
assert dk_emb.shape[0] == len(df_dk)
assert ep_emb.shape[0] == len(df_ep)
assert np.allclose(np.linalg.norm(dk_emb, axis=1), 1.0, atol=1e-3)
assert np.allclose(np.linalg.norm(ep_emb, axis=1), 1.0, atol=1e-3)
```

- [ ] **Step 2: Save `.npy` and aligned meta CSVs**

```python
dk_meta_cols = [c for c in ["afstemningid", "sag_titel", "sag_titel_en", "dato"] if c in df_dk.columns]
ep_meta_cols = [c for c in ["id", "period", "display_title", "timestamp", "procedure_reference"] if c in df_ep.columns]

df_dk[dk_meta_cols].to_csv(OUT_DIR / "dk_title_meta.csv", index=False)
df_ep[ep_meta_cols].to_csv(OUT_DIR / "ep_title_meta.csv", index=False)
np.save(OUT_DIR / "dk_title_embeddings.npy", dk_emb)
np.save(OUT_DIR / "ep_title_embeddings.npy", ep_emb)

print(f"Wrote artifacts to {OUT_DIR.resolve()}")
for name in [
    "dk_title_embeddings.npy",
    "dk_title_meta.csv",
    "ep_title_embeddings.npy",
    "ep_title_meta.csv",
]:
    print(f"  {name}: {(OUT_DIR / name).stat().st_size:,} bytes")
```

- [ ] **Step 3: Reload check**

```python
dk_emb_reload = np.load(OUT_DIR / "dk_title_embeddings.npy")
ep_emb_reload = np.load(OUT_DIR / "ep_title_embeddings.npy")
dk_meta_reload = pd.read_csv(OUT_DIR / "dk_title_meta.csv")
ep_meta_reload = pd.read_csv(OUT_DIR / "ep_title_meta.csv")

assert dk_emb_reload.shape == dk_emb.shape
assert ep_emb_reload.shape == ep_emb.shape
assert len(dk_meta_reload) == dk_emb.shape[0]
assert len(ep_meta_reload) == ep_emb.shape[0]
print("Reload OK")
```

- [ ] **Step 4: Commit notebook (do not commit large `.npy` unless the project already tracks them)**

```bash
git add notebooks/Embeddings_DK_EP.ipynb
git commit -m "Encode DK and EP titles with mpnet and save embedding artifacts"
```

If `.npy` files are large and not wanted in git, leave them untracked (preferred). Meta CSVs are optional to commit.

---

### Task 3: First-look analysis — centroid, histograms, neighbors, PCA

**Files:**
- Modify: `notebooks/Embeddings_DK_EP.ipynb`

**Interfaces:**
- Consumes: `dk_emb`, `ep_emb`, `df_dk`, `df_ep`, `RANDOM_STATE`
- Produces: printed centroid cosine; three histogram panels; sample top-5 neighbor tables; PCA scatter figure

- [ ] **Step 1: Centroid cosine**

```python
def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)

dk_centroid = l2_normalize(dk_emb.mean(axis=0, keepdims=True))[0]
ep_centroid = l2_normalize(ep_emb.mean(axis=0, keepdims=True))[0]
centroid_cosine = float(dk_centroid @ ep_centroid)
print(f"Cosine between DK and EP centroids: {centroid_cosine:.4f}")
```

- [ ] **Step 2: Sampled pairwise similarity distributions**

```python
import matplotlib.pyplot as plt

rng = np.random.default_rng(RANDOM_STATE)
N_PAIRS = 10_000

def sample_pair_cosines(a: np.ndarray, b: np.ndarray, n: int, rng) -> np.ndarray:
    i = rng.integers(0, len(a), size=n)
    j = rng.integers(0, len(b), size=n)
    # avoid identical indices for within-corpus samples when same matrix
    if a is b:
        same = i == j
        while same.any():
            j[same] = rng.integers(0, len(b), size=same.sum())
            same = i == j
    return (a[i] * b[j]).sum(axis=1)

cos_dk_dk = sample_pair_cosines(dk_emb, dk_emb, N_PAIRS, rng)
cos_ep_ep = sample_pair_cosines(ep_emb, ep_emb, N_PAIRS, rng)
cos_dk_ep = sample_pair_cosines(dk_emb, ep_emb, N_PAIRS, rng)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
for ax, vals, title in zip(
    axes,
    [cos_dk_dk, cos_ep_ep, cos_dk_ep],
    ["DK–DK", "EP–EP", "DK–EP"],
):
    ax.hist(vals, bins=40, density=True, alpha=0.85)
    ax.set_title(f"{title}\nmean={vals.mean():.3f}")
    ax.set_xlabel("cosine")
axes[0].set_ylabel("density")
fig.suptitle("Sampled pairwise title similarities", y=1.05)
plt.tight_layout()
plt.show()
```

- [ ] **Step 3: Nearest-neighbor sanity check**

```python
def top_k_from_other(query_embs, other_embs, other_titles, query_idx, k=5):
    sims = other_embs @ query_embs[query_idx]
    top = np.argpartition(-sims, k)[:k]
    top = top[np.argsort(-sims[top])]
    return list(zip(sims[top], other_titles.iloc[top].tolist(), top.tolist()))

dk_sample_idx = rng.choice(len(df_dk), size=3, replace=False)
ep_sample_idx = rng.choice(len(df_ep), size=3, replace=False)

print("=== Random DK → top-5 EP ===")
for qi in dk_sample_idx:
    print(f"\nDK: {df_dk.loc[qi, 'sag_titel_en'][:160]}")
    for score, title, _ in top_k_from_other(dk_emb, ep_emb, df_ep["display_title"], qi, k=5):
        print(f"  {score:.3f}  {title[:140]}")

print("\n=== Random EP → top-5 DK ===")
for qi in ep_sample_idx:
    print(f"\nEP: {df_ep.loc[qi, 'display_title'][:160]}")
    for score, title, _ in top_k_from_other(ep_emb, dk_emb, df_dk["sag_titel_en"], qi, k=5):
        print(f"  {score:.3f}  {title[:140]}")
```

- [ ] **Step 4: Balanced PCA scatter**

```python
from sklearn.decomposition import PCA

n_dk = len(dk_emb)
ep_sample = rng.choice(len(ep_emb), size=min(n_dk, len(ep_emb)), replace=False)

X = np.vstack([dk_emb, ep_emb[ep_sample]])
labels = np.array(["DK"] * n_dk + ["EP"] * len(ep_sample))

xy = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X)

fig, ax = plt.subplots(figsize=(7, 6))
for lab, color in [("DK", "C0"), ("EP", "C1")]:
    mask = labels == lab
    ax.scatter(xy[mask, 0], xy[mask, 1], s=8, alpha=0.35, label=lab, c=color)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("PCA of title embeddings (all DK + EP subsample)")
ax.legend()
plt.tight_layout()
plt.show()
```

Add a short markdown cell after the plots:

```markdown
**How to read this:** If DK–DK and EP–EP similarities are clearly higher than DK–EP, the agendas occupy somewhat different regions. Overlapping PCA clouds and high DK–EP cosines suggest topical overlap in titles (not that specific votes are the same).
```

- [ ] **Step 5: Run analysis cells end-to-end**

Expected: centroid cosine printed; three histograms; neighbor printouts look topically plausible or clearly wrong (either is informative); PCA figure renders.

- [ ] **Step 6: Commit**

```bash
git add notebooks/Embeddings_DK_EP.ipynb
git commit -m "Add DK-EP embedding first-look similarity and PCA analysis"
```

---

## Spec coverage checklist

| Spec requirement | Task |
|------------------|------|
| Load DK `sag_titel_en` | Task 1 |
| Load all EP periods; `is_main` when present | Task 1 |
| `all-mpnet-base-v2`, normalized encode | Task 2 |
| Save npy + meta under `data/temp_data/embeddings/` | Task 2 |
| Centroid cosine | Task 3 |
| Pairwise similarity histograms | Task 3 |
| Sample nearest neighbors | Task 3 |
| PCA 2D plot | Task 3 |
| Out of scope (amendments, full retrieval, ANEY reuse) | Not implemented |
