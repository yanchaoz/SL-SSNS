# Self-supervised encoder pretraining

[Back to the project overview](../README.md)

This directory contains the patch encoder, self-supervised learner, data provider and training loop. The [CGS workflow](../CGS/README.md) includes a selection checkpoint, so pretraining is optional.

## Environment and compatibility

Use Python 3.10+ with an appropriate PyTorch 2.6+ build, then install from the repository root:

```bash
python -m pip install -r requirements-pretrain.txt
```

The dependency list is not a benchmark-reproduction lockfile. Synthetic CPU tests cover the learner, all seven augmentation branches, fixed-size batching and checkpoint round trips. Full research-data training and multi-GPU reproduction have not been validated.

## Prepare the data and configuration

Edit [`config/pretraining_snemi3d.yaml`](config/pretraining_snemi3d.yaml). `DATA.unlabel_dataset: 'ac3_ac4'` selects this layout beneath `DATA.data_folder`:

```text
<data_folder>/
└── snemi3d/
    ├── AC3_inputs.h5
    └── AC4_inputs.h5
```

Each HDF5 file must contain a raw 3D volume under `main`, in `(z, y, x)` order. The provider samples `18 × 160 × 160` patches and divides intensities by 255. Prepare the intended training split in advance: the full stored arrays are loaded into memory. Use the expected 8-bit intensity scale.

Replace the placeholder `DATA.data_folder`. Set `TRAIN.batch_size` (at least two), `TRAIN.num_workers` and `TRAIN.save_path` for your machine. Keep `MODEL.pre_train: False` when starting from scratch; enable it with `MODEL.pretrain_path` to initialize an encoder from compatible weights. Relative paths in the YAML are based on `Pretrain/`.

## Run pretraining

From the repository root:

```bash
python Pretrain/pretrain.py --cfg pretraining_snemi3d --device cuda:0
```

`--cfg` accepts a basename under `Pretrain/config/` or an explicit YAML path. Use `--device cpu` for a small CPU experiment; omitting it follows `TRAIN.if_cuda`. The script respects `CUDA_VISIBLE_DEVICES` and no longer overrides GPU visibility. With multiple visible GPUs and `cuda:0`, the encoder uses DataParallel and the batch size must divide evenly across GPUs. Restrict visibility before launch to select a single GPU.

Both augmented views retain a fixed patch shape. The original post-augmentation random offsets could truncate already cropped patches and have been removed. Augmentation types, learner loss, encoder architecture and optimizer defaults are retained.

## Outputs and resuming

Each invocation creates a new timestamped directory beneath `TRAIN.save_path`, containing training logs, TensorBoard records and `model-NNNNNN.ckpt` files. The loop stops at exactly `TRAIN.total_iters` and saves the final iteration even when it is not a multiple of `TRAIN.save_freq`.

Checkpoints contain CGS-compatible encoder `model_weights`, the complete `learner_weights`, optimizer state and iteration. Resume into a new run directory with:

```bash
python Pretrain/pretrain.py --cfg pretraining_snemi3d --device cuda:0 --resume /path/to/model-002000.ckpt
```

Use the same learner configuration and GPU topology. Older encoder-only checkpoints can initialize `MODEL.pre_train` but cannot restore a complete training state; `--resume` reports this explicitly. Random-number and data-worker states are not saved, so resumed runs are not guaranteed to be bitwise identical to uninterrupted training.

The current loop reports training loss only. Historical validation/cache fields remain in the YAML for reference; they do not implement held-out validation or cache generation. Paper-level evaluation still requires the intended datasets, splits and validation procedure.
