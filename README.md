# Benchmarking multi-omics 

This repository contains code and notebooks for multi-omics analysis of the TCGA BRCA dataset.

All generic utilities have been consolidated into a single helper module:

- **`omics_helpers.py`** – unified helper module for the TCGA BRCA multi-omics experiments.

  It provides:

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
    - PAM50 extraction from per-view meta
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
    - PCA + logistic regression on concatenated raw features (no leakage)  
    - generic `GridSearchCV` wrapper with reporting  
    - late-fusion (per-view) logistic regression baseline


## Repository structure

- **`omics_helpers.py`**  
  All reusable code for preprocessing, feature selection, embeddings (PCA/MOFA), plotting and classification baselines.

- **`Convert2Pickle.ipynb`**  
  Loads the raw TCGA BRCA multi-omics data and converts it into a convenient Python format (e.g. pickle with a `data` dict used by the other notebooks).

- **`MOFA_training_models.ipynb`**  
  Trains MOFA models on the selected views / subsets and saves the fitted models and factor/loadings matrices to disk.
  Does logreg on MOFA factors.
  Looks into which features contribute to MOFA factors.

- **`mofa_pca_comaprison.ipynb`**  
  Compares PCA and MOFA embeddings:
  - variance explained per view and per factor  
  - factor–class correlations  
  - 2D/3D embeddings coloured by PAM50  
  - runs logreg on PCA components and MOFA factors

- **`log_reg_baseline.ipynb`**
  Looks into class and patient distrubtion among omics
  Looks at the 'informativeness' of features
  Classification baselines:
  - early and late integration on raw features for logreg

- **`biological_info.ipynb`**  
  Gives some biological info about datasets

In order to load mofa models with the code provided, place the unzipped models in the folder code/exports500/. Then there is no need to train mofa models and one can skip that cells when runnig MOFA_training_models.ipynb.

## Requirements

The code relies on the following Python packages:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `scanpy`
- `mofax`
- `muon`
- `matplotlib-venn`  (imported as `matplotlib_venn`)
- `palettable`       (for Wes Anderson colour palettes)
- `jupyter` / `notebook` (to run the `.ipynb` files)
  



