# EchoFine

Official research code for:
**Robust Fine-Grained Echocardiographic View Classification with Supervised Contrastive Learning**  
(submitted to *Medical Image Analysis*).

---

## Overview

EchoFine is a supervised contrastive learning framework for **fine-grained transthoracic echocardiography (TTE) view classification**.

This repository contains code for:
- Contrastive pretraining and downstream fine-tuning.
- Robustness experiments under simulated annotation variability.
- Multi-expert agreement analysis.
- Statistical significance testing and reporting plots.

The work introduces and evaluates:
- **TTE47**: a 47-class fine-grained echo benchmark.
- Contrastive objectives including **SupCon**, **DropCon**, and **LogSum**.
- Representation-space robustness metrics (**DR**, **LRP**).

---

## Data and Models

- **TTE47 dataset (public release page)**:  
  https://www.thrive-centre.com/datasets/TTE47
- **Released pretrained/fine-tuned models (EchoForge)**:  
  https://github.com/thrive-centre/EchoForge/blob/main/echoforge/classification/models/EchoView47/README.md

> Note: The full training set used in the paper is restricted by data governance.  
> The public release focuses on the test benchmark and model artifacts.

---

## Repository Structure

- `view_classification_with_noise.py`  
  Main training/evaluation pipeline (pretraining, corrections, downstream training, inference).
- `noise_run_script.py`  
  Experiment launcher for running noise-level sweeps.
- `experiment_runner.py`  
  Shared command builder/execution utilities.
- `experiment_discovery.py`, `experiment_reporting.py`, `experiment_plots.py`  
  Experiment discovery, aggregation, and plotting/report generation.
- `expert_agreement.py`, `experts_analysis.py`  
  Multi-expert agreement utilities and expert-label handling.
- `confusion_analysis.py`, `mismatch_analysis.py`, `evaluation_analysis.py`  
  Confusion matrices, mismatch analysis, and evaluation wrappers.
- `statistical_significance_test.py`  
  Selection-aware permutation / McNemar / bootstrap significance analysis.
- `utils.py`  
  JSON/file utility helpers.

---

## Environment

Recommended:
- Python 3.10+
- TensorFlow 2.16.x
- CUDA-enabled GPU for training

Core Python dependencies:
- `tensorflow`, `tensorflow-addons`, `tensorflow-probability`
- `keras-cv`, `keras-cv-attention-models`
- `numpy`, `pandas`, `scikit-learn`
- `matplotlib`, `seaborn`
- `opencv-python`, `hdbscan`, `timm`

Install example:

```bash
pip install tensorflow tensorflow-addons tensorflow-probability keras-cv keras-cv-attention-models numpy pandas scikit-learn matplotlib seaborn opencv-python hdbscan timm
```

---

## Configuration Before Running

`view_classification_with_noise.py` contains project-specific paths and experiment settings in `main()`.  
Update these for your environment before running:

- `data_folder`
- `save_folder`
- `info_folder`
- `files_for_relabelling`
- `dual_files_csv`

The script also expects split/class metadata JSON files (`class_lookup.json`, `data_split.json`) in the configured `info_folder` unless `split_sets=True`.

---

## Usage

### 1) Run the default experiment schedule

```bash
python noise_run_script.py
```

### 2) Run individual stages directly

Pretraining only (example: Xception + LogSum, 0% noise):

```bash
python view_classification_with_noise.py --pretraining --noise_percentage 0 --temperature 0.3 --backbone 0 --loss 7
```

Downstream fine-tuning with pretrained weights:

```bash
python view_classification_with_noise.py --downstream_training --downstream_with_pretrainedweights --use_epoch_weights 0 --noise_percentage 0 --temperature 0.3 --backbone 0 --loss 7
```

ImageNet baseline:

```bash
python view_classification_with_noise.py --downstream_training --downstream_with_imagenet --noise_percentage 0 --backbone 0 --loss 7
```

Random-init baseline:

```bash
python view_classification_with_noise.py --downstream_training --downstream_with_rand_init --noise_percentage 0 --backbone 0 --loss 7
```

### Backbone / loss indices

Backbones (`--backbone`):
`0:xception, 1:resnet50, 2:resnet101, 3:densenet121, 4:convnextbase, 5:efficientnetv2s, 6:vit_base, 7:swintransformerv2base, 8:convnexttiny, 9:vit_small, 10:swintransformerv2tiny`

Losses (`--loss`):
`0:supcon, 1:dropcon, 2:compcon, 3:compcon_hybrid, 4:supcon_softscale, 5:supcon_sqmax, 6:supcon_weighted, 7:logsum, 8:logsum_weighted, 9:logsum_expavgsum, 10:logsum_neg_emph, 11:hybrid, 12:pairwise, 13:dcl_softmax, 14:sup_barlow, 15:supcon_adaptive`

---

## Outputs

Typical artifacts include:
- Model checkpoints (`ssl/`, downstream folders).
- Prediction JSON files (`all_predictions_*.json`).
- Metric reports (`results_*.json`, classification reports).
- Figures/CSVs in `results_plots/` (temperature sweeps, grouped best accuracy).

---

## Citation

If you use this repository, please cite the associated paper:

**Preshen Naidoo et al.**  
*Robust Fine-Grained Echocardiographic View Classification with Supervised Contrastive Learning.*  
Submitted to *Medical Image Analysis*.

