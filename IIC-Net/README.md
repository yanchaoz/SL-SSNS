# IIC-Net: semi-supervised neuron segmentation

[Back to the project overview](../README.md)

IIC-Net is the segmentation component of SL-SSNS. This directory contains the model, loss functions, data providers, augmentation routines and postprocessing helpers associated with the research workflow.

## Release status

The supervised warm-up entry point `warmup.py` and the semi-supervised training entry point `semi_tuning.py` are **not present in this checkout**. There is also no standalone inference command in this directory. The included components therefore do not constitute an executable end-to-end training or inference pipeline.

The following source is available for inspection and development:

| Path | Purpose |
| --- | --- |
| [`model/model_superhuman.py`](model/model_superhuman.py) | Segmentation network |
| [`dataloader/`](dataloader/) | Labeled, unlabeled, mixed-data and validation providers |
| [`loss/`](loss/) | Supervised and unlabeled loss functions |
| [`utils/`](utils/) | Augmentation, affinity processing and segmentation helpers |
| [`config/semi_snemi3d.yaml`](config/semi_snemi3d.yaml) | Research configuration |

## Configuration and data alignment

The supplied configuration and the data-provider branches are not aligned:

| Configuration value | Released provider |
| --- | --- |
| `DATA.dataset_name: 'ac4-all'` | [`data_provider_labeled.py`](dataloader/data_provider_labeled.py) accepts CREMI ROI/random-subset names |
| `DATA.unlabel_dataset: 'ac3_ac4'` | [`data_provider_unlabel.py`](dataloader/data_provider_unlabel.py) accepts CREMI-B/C subset names |
| `DATA.valid_dataset: 'snemi3d-ac3'` | [`provider_valid.py`](dataloader/provider_valid.py) accepts `cremi-A`, `cremi-B` or `cremi-C` |
| `MODEL.continue_train: True` with an empty `continue_path` | No usable continuation checkpoint is specified |

For example, the labeled provider's `cremi-roi-bc-160` branch reads `roi_0.tif` through `roi_3.tif` and matching `gt_0.tif` through `gt_3.tif` beneath `<DATA.data_folder>/cremi-r/roi-bc-160/`. Labels must be neuron-instance annotations aligned with the corresponding raw volumes. Choosing a different YAML name alone does not adapt these providers to a new dataset.

Completing a reproducible training release requires the missing entry points, aligned configurations and data layouts, a compatible environment, and validation with the intended data splits. The selection encoder in `CGS/` is not a trained IIC-Net segmentation model.

## Postprocessing helpers

The utilities include waterz and ELF-based postprocessing code. These require their own optional dependencies, including compiled segmentation libraries; they are not installed by [`requirements-cgs.txt`](../requirements-cgs.txt). Consult the [ELF](https://github.com/constantinpape/elf) and [waterz](https://github.com/funkey/waterz) projects for installation details and check compatibility with the helper APIs before use.

For a missing script or reproducibility question, [open an issue](https://github.com/yanchaoz/SL-SSNS/issues) identifying the repository revision and the stage you are trying to run.

## Maintained loss utilities

The weighted MSE now follows the input device and retains its original normalization by batch size times spatial voxel count (excluding channels). Unlabeled MSE/BCE and CBST BCE return zero loss and zero prediction gradient for an empty confidence mask. Nonempty-mask formulas are unchanged. These focused fixes have CPU regression coverage; they do not supply the missing training entry points.
