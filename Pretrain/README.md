# Self-supervised encoder pretraining

[Back to the project overview](../README.md)

This directory contains the patch encoder, the self-supervised learner and the pretraining loop. Pretraining is optional for the [CGS workflow](../CGS/README.md), which includes a selection checkpoint.

## Entry point and configuration

The source entry point is [`pretrain.py`](pretrain.py), and its default configuration is [`config/pretraining_snemi3d.yaml`](config/pretraining_snemi3d.yaml). After resolving the setup issues below, the invocation from the repository root is:

```bash
cd Pretrain
python pretrain.py --cfg pretraining_snemi3d
```

`--cfg` accepts a basename under `Pretrain/config/`, without `.yaml`. The directory is `Pretrain/`, not `Pretraining/`, and the script is `pretrain.py`, not `pretraining.py`.

## Prepare the data and configuration

[`pretrain_provider.py`](pretrain_provider.py) expects this layout beneath `DATA.data_folder`:

```text
<data_folder>/
└── snemi3d/
    ├── AC3_inputs.h5
    └── AC4_inputs.h5
```

Each HDF5 file must contain a raw 3D volume under `main`, in `(z, y, x)` order. The provider samples `18 × 160 × 160` patches and divides intensities by 255. Prepare the intended training split in advance: the provider loads the full stored arrays.

Before launching:

1. Replace the placeholder `DATA.data_folder` with your data root.
2. Add `DATA.unlabel_dataset: 'ac3_ac4'`. The provider reads this key, while the supplied YAML currently contains only `DATA.dataset_name`.
3. Keep `MODEL.pre_train: False` when starting from scratch. If enabling it, supply a compatible encoder checkpoint through `MODEL.pretrain_path`.
4. Set `TRAIN.batch_size`, `TRAIN.num_workers`, `TRAIN.cache_path` and `TRAIN.save_path` for your machine. The learner requires a batch size greater than one; multi-GPU batches must also divide evenly across the visible GPUs.

## Environment and compatibility

The pretraining source imports PyTorch, NumPy, PyYAML, `attrdict`, `tensorboardX`, HDF5/image I/O packages, Pillow, SciPy, scikit-image, OpenCV and Matplotlib. The [CGS requirements](../requirements-cgs.txt) alone do not cover this stage. No validated pretraining environment or dependency lock is supplied.

The following issues must be resolved in a training environment before treating the command above as runnable:

| Source detail | Consequence |
| --- | --- |
| `pretrain.py` sets `CUDA_VISIBLE_DEVICES='0,1'` at import time | It overrides a device selection made in the shell; edit this assignment for your machine |
| Model construction uses `cuda:0` unconditionally | Setting `TRAIN.if_cuda: False` does not create a CPU training workflow |
| The provider calls `self.data_iter.next()` | Check compatibility with your PyTorch version; current iterators may require `next(self.data_iter)` |
| Pretraining imports `attrdict`; CGS uses `addict.Dict` | Installing `addict` does not satisfy the pretraining import, and `attrdict` compatibility must be checked with your Python version |
| Augmented crops receive random offsets after the initial fixed-size crop | Returned spatial shapes can vary; review batching before a multi-sample training run |

This guide documents the released research code and its setup gaps. It does not establish successful retraining or benchmark reproduction.

## Outputs

The loop writes logs and TensorBoard records under the configured model directory and saves `model-NNNNNN.ckpt` files at `TRAIN.save_freq`. Saved files contain encoder `model_weights` and an iteration counter. Although the code creates a validation log and the YAML contains validation settings, the current loop does not implement held-out validation. It also does not save optimizer state or the complete self-supervised learner state, so checkpoint loading should not be interpreted as an exact training resume.
