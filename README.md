# SL-SSNS

Official implementation of **A distribution-aware semi-supervised pipeline for cost-effective neuron segmentation** (*iScience*, 2026).

[![Paper](https://img.shields.io/badge/iScience-Paper-blue)](https://doi.org/10.1016/j.isci.2025.114507)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vPYYeaycpdQjDiu_TQD4LqQbjezf40yc?usp=sharing)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Overview](#overview) · [Demo](#interactive-demo) · [Installation](#installation) · [Data and models](#data-and-models) · [Usage](#usage) · [Citation](#citation)

## Overview

SL-SSNS addresses the distribution mismatch between sparsely labeled and unlabeled volume electron microscopy (EM) data through two complementary components:

- **Selective labeling:** coverage-based greedy selection (CGS) identifies representative subvolumes for annotation using embeddings from a self-supervised patch encoder.
- **Semi-supervised segmentation:** IIC-Net combines intra- and inter-dataset mixing with mixed-view consistency regularization.

The workflow is **raw EM volume → representative subvolume selection → annotation → semi-supervised training → neuron segmentation**. See the [paper](https://doi.org/10.1016/j.isci.2025.114507) for the method and experimental evaluation.

## Interactive demo

The [Colab demo](https://colab.research.google.com/drive/1vPYYeaycpdQjDiu_TQD4LqQbjezf40yc?usp=sharing) illustrates subvolume selection in the spatial and embedding domains. It provides a starting point for exploring selective labeling; the local workflow is documented below.

**Spatial domain — selected regions in an EM volume**

![Spatial view of subvolume selection](https://github.com/user-attachments/assets/30a81673-03c6-4fd5-af86-c4d0ab60c2e5)

**Embedding domain — coverage of the patch representations**

![Embedding view of subvolume selection](https://github.com/user-attachments/assets/a9028640-b4e4-48cc-9bb9-cf3cbebbe9f0)

## Repository structure

| Component | Contents | Guide |
| --- | --- | --- |
| `CGS/` | Selection script, encoder checkpoint, configuration and visualization utilities | [Subvolume selection](CGS/README.md) |
| `Pretrain/` | Self-supervised encoder pretraining script, learner and data provider | [Encoder pretraining](Pretrain/README.md) |
| `IIC-Net/` | Segmentation model, data providers, losses and postprocessing utilities | [Training release status](IIC-Net/README.md) |

> [!IMPORTANT]
> The current checkout includes the CGS entry point and pretraining source. The `warmup.py` and `semi_tuning.py` entry points are **not included**, so end-to-end IIC-Net training cannot be launched from this checkout. Pretraining also needs the configuration and compatibility checks described in its guide.

## Installation

Create an isolated Python environment for local CGS selection:

```bash
git clone https://github.com/yanchaoz/SL-SSNS.git
cd SL-SSNS
python -m venv .venv
```

Activate it with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in Windows PowerShell. Install a PyTorch build appropriate for your Python version and hardware using the [official installation selector](https://pytorch.org/get-started/locally/), then install the selection dependencies:

```bash
python -m pip install -r requirements-cgs.txt
```

This dependency list is derived from the CGS imports; it is **not a version-locked reproduction environment**. Dependency installation and full selection have not been validated across platforms. Pretraining has additional dependencies and compatibility requirements; see its [environment notes](Pretrain/README.md#environment-and-compatibility).

## Data and models

| Resource | Location | Notes |
| --- | --- | --- |
| Selection encoder | [`CGS/selection_model.ckpt`](CGS/selection_model.ckpt) | Included in the repository; used by `MODEL.model_pth` in the selection configuration |
| Local selection input | `CGS/data/ac3.tif` by default | Supply your own raw volume; EM datasets are not bundled |
| AC3/AC4 | [Original data package](https://lichtman.rc.fas.harvard.edu/vast/AC3AC4Package.zip) | Prepare inputs and splits for the relevant data provider |
| CREMI | [Dataset website](https://cremi.org/) | Original dataset and benchmark information |

For the complete dataset list and experimental splits, consult the paper's [key resources and methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC12805337/). Cite the original dataset publications when using their data. The selection checkpoint is an **embedding encoder**, not an IIC-Net neuron-segmentation checkpoint.

## Usage

### 1. Select subvolumes for annotation

Prepare a single grayscale volume in `(z, y, x)` order and edit [`CGS/config/SL.yaml`](CGS/config/SL.yaml) to set the input path and selection budget. The loader divides intensities by 255, so prepare data on the expected 8-bit intensity scale.

Before running, read the [CGS guide](CGS/README.md): the current script selects `cuda:3` whenever CUDA is available, reads only the **first 100 z-slices**, and writes to a fixed `record/` directory. Adjust the device in `CGS.py` for your machine and preserve earlier outputs before another run.

From the repository root:

```bash
cd CGS
python CGS.py --cfg SL
```

`--cfg SL` selects `CGS/config/SL.yaml`; it is a configuration basename, not a file path. The output contains selected TIFF subvolumes, patch indices, coordinates and UMAP plots. These TIFFs are raw images to annotate, not predicted neuron labels.

### 2. Pretrain an encoder (optional)

The included selection checkpoint lets you start with CGS without retraining. To investigate encoder pretraining, follow the [pretraining guide](Pretrain/README.md), which describes the actual entry point, expected AC3/AC4 layout and outstanding compatibility issues.

### 3. Train IIC-Net

The released model, losses and data providers are in [`IIC-Net/`](IIC-Net/). See the [IIC-Net guide](IIC-Net/README.md) for the missing entry points and configuration mismatches that currently prevent a complete training run.

## Citation

If you use this work, please cite the [published article](https://doi.org/10.1016/j.isci.2025.114507):

```bibtex
@article{zhang2026slssns,
  title   = {A distribution-aware semi-supervised pipeline for cost-effective neuron segmentation},
  author  = {Zhang, Yanchao and Zhai, Hao and Guo, Jinyue and Liu, Jing and Xie, Qiwei and Han, Hua},
  journal = {iScience},
  volume  = {29},
  number  = {1},
  pages   = {114507},
  year    = {2026},
  doi     = {10.1016/j.isci.2025.114507}
}
```

The citation uses the 2026 journal issue year; the article was first published online in December 2025.

## Acknowledgments

This code builds on [SSNS-Net](https://github.com/weih527/SSNS-Net) (IEEE TMI, 2022) by Wei Huang et al. The postprocessing utilities build on [ELF](https://github.com/constantinpape/elf) and [waterz](https://github.com/funkey/waterz).

## License and support

The code is distributed under the [MIT License](LICENSE). For questions or reproducibility issues, please [open an issue](https://github.com/yanchaoz/SL-SSNS/issues) with your command, configuration, environment and error message.
