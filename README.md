
# Feature relevance XAI in NIDS

This repository contains the code for the paper "Evaluating feature relevance XAI in network intrusion detection" published as part of The first World Conference on eXplainable Artificial Intelligence (xAI 2023).

## Data

This repository uses the established CIDDS-001 dataset:

[Ring, Markus, et al. "Technical Report CIDDS-001 data set." J. Inf. Warfare 13 (2017).]

Fully preprocessed data can be downloaded [here](https://drive.google.com/drive/folders/1yMaciiXPSbqnJrJHRrLeoiwMIiCzRF_I?usp=sharing) and needs to be placed into `data_prep/onehot_quantized`. The original dataset can be downloaded from https://www.hs-coburg.de/cidds.

Ground truth explanations are found under `data/cidds/data_raw`.

## Training detectors
 
Parameter studies can be conducted using `ae_param_search.py`, `if_param_search.py`, `oc_param_search.py`
for Autoencoder, Isolation Forest, and One-Class SVM respectively.

The best performing models and hyperparameters used in the paper are available in `outputs/models/cidds`.

## Generating SHAP Explanations

Trained models can be explained using `run_cidds_xai.py`.

**Note**: Additional setup is required for running SHAP with *optimized* replacement data.
To integrate the optimization procedure directly within kernel-SHAP,
this implementation requires to manually override the `shap/explainer/_kernel.py` script within the SHAP package.
For this, either override the contents of `shap/explainer/_kernel.py` entirely
with the backup file provided in `xai/backups/shap_kernel_backup.py`
or add the small segments marked with `# NEWCODE` within `xai/backups/shap_kernel_backup.py` in the
original library file of `shap/explainer/_kernel.py`.
