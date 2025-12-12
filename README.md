# Benchmarking Multi-Omics Representation Learning on TCGA-BRCA

This repository benchmarks **multi-omics representation learning and classification** on the **TCGA BRCA**
cohor. We compare classification performance and biological interpretability of different integration strategies of baseline logistic regression (LR), PCA LR, **MOFA**-based LR, and **IntegrAO**. 

---

## Repository tree

```text
repo_root/
├─ README.md
├─ requirements.txt
│
├─ code/
│  ├─ notebooks/
│  │  ├─ 00_Convert2Pickle.ipynb
│  │  ├─ 10_biological_info.ipynb
│  │  ├─ 20_log_reg_baseline.ipynb
│  │  ├─ 30_MOFA_training_models.ipynb
│  │  ├─ 40_mofa_pca_comparison.ipynb
│  │  ├─ 50_integrao.ipynb
│  │  └─ utils/
│  │     └─ omic_helpers.py
│  │
│  └─ scripts/
│     └─ (optional helper scripts)
│
├─ data/
│  └─ (unzipped dataset goes here)
│
├─ models/
│  ├─ mofa/
│  │  └─ exports_500/
│  │     └─ (unzipped MOFA .hdf5 exports go here)
│  └─ integrao/
│     └─ (saved IntegrAO models go here)
│
├─ results/
│  └─ figures/
│     └─ (all figures used in the report)
│
├─ papers/
│  ├─ IntegrAO.pdf
│  ├─ MOFA.pdf
│  └─ ReviewPaperMultiOmics.pdf
│
└─ 

# Benchmarking Multi-Omics Representation Learning on TCGA-BRCA
Multi-omics analysis of the **TCGA BRCA** cohort (mRNA / DNA-methylation / RPPA), focusing on
**representation learning (PCA, MOFA, IntegrAO)** and **classification baselines** (logistic regression).
Subtypes are evaluated using **PAM50** labels (including a combined `PAM50_any` label across views).

All reusable utilities (preprocessing, feature selection, embeddings, plotting, and baselines) are
centralized in `code/notebooks/utils/` (details below; module documentation will be expanded once the
final `utils/` contents are fixed).

---

## Notebooks (run in this order)

All main notebooks are located in `code/notebooks/`.  
The recommended execution order is:

0. Download and unzip data and trained models as described:
  ### 1) Download data
  - Download: **data.zip** from Google Drive: https://drive.google.com/drive/folders/1o3YSKcZxi5Ie9GXMNwEPYmvqQg9rG1zJ?usp=sharing
  - Unzip into the repo so you get:
    - `data/ ...`

  ### 2) Download pretrained MOFA models 
  - Download: **models_mofa.zip** from Google Drive: https://drive.google.com/drive/folders/1TQRrggB6-CX7fGwo1YSQsur5J8-WR3wn?usp=sharing
  - Unzip into:
  - `models/mofa/exports_500/ ...`

1. **`10_Convert2Pickle.ipynb`**  
   Loads raw TCGA BRCA multi-omics data and converts it into a convenient Python format (e.g. a pickle
   containing a `data` dict) placed under `data/`.  
   **This preprocessing step is required by all downstream notebooks.**

2. **`20_biological_info.ipynb`**  
   Provides biological context about the dataset and labels to support interpretation of downstream
   results.

3. **`30_log_reg_baseline.ipynb`**  
   Explores:
   - patient and class distribution across omics views
   - basic feature “informativeness” (variance / ANOVA-style statistics, etc.)  
   Implements baseline classifiers, including:
   - early integration (concatenated raw features) for logistic regression
   - late fusion (per-view logistic regression) baseline

4. **`40_MOFA_training_models.ipynb`**  
   Trains MOFA models on selected views/subsets and saves fitted models to disk under
   `models/mofa/exports_500/`.  
   Also:
   - runs logistic regression on MOFA factors
   - inspects which features contribute to MOFA factors (via weights/loadings)

5. **`50_mofa_pca_comparison.ipynb`**  
   Compares PCA and MOFA embeddings:
   - variance explained per view and per factor
   - factor–class correlations
   - 2D/3D embeddings colored by PAM50
   - logistic regression on PCA components vs MOFA factors

6. **`60_integrao.ipynb`**  
   Trains an **IntegrAO** model and evaluates it for classification.  
   Also includes model analysis, including feature exploration / inspection of IntegrAO-learned feature importance 

---

## Shared helper utilities (`utils/`)

All notebooks import shared functionality from `code/notebooks/utils/`.

The unified helper module(s) provide:

- **Basic utilities**
  - z-scoring of DataFrames
  - concatenation of omic views with optional block scaling

- **PCA / embedding helpers**
  - per-view PCA and PCA on top-K most variable features
  - variance explained per view
  - pairwise silhouette scores and correlation matrices between embeddings
  - per-factor \(R^2\) across views

- **MOFA-related helpers**
  - 2D / 3D plots of MOFA factors
  - factor–class correlation matrices
  - feature-importance rankings (global and per-modality) based on MOFA weights

- **Labels**
  - PAM50 extraction from per-view metadata
  - combined `PAM50_any` labels across views

- **Per-feature statistics**
  - ANOVA per feature (F, p, η²)
  - variance per feature

- **Feature selection**
  - variance- and ANOVA-based `SelectKBest` score functions
  - small “view blocks” for sklearn Pipelines (ANOVA / most-variable)
  - helpers to select top features per view (same k or per-view k)

- **Plotting helpers**
  - 2D / 3D scatter plots for PCA and MOFA embeddings

- **Classification baselines**
  - logistic regression on precomputed factor embeddings (e.g. MOFA / PCA)
  - leakage-safe PCA + logistic regression on concatenated raw features
  - generic `GridSearchCV` wrapper with reporting
  - late-fusion (per-view) logistic regression baseline

> Note: Once the final contents of `utils/` are fixed, this section should be updated to list the
> exact helper files and their responsibilities.

---

## Setup

### Requirements
Install dependencies from **`requirements.txt`** (located at the repository root):

```bash
pip install -r requirements.txt
