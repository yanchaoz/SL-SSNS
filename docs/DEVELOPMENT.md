# Development and validation

[Project overview](../README.md)

## Code organization

| File | Responsibility |
| --- | --- |
| `CGS/CGS.py` | CLI arguments and stage orchestration |
| `CGS/selection.py` | NumPy distance, coverage and greedy selection |
| `CGS/CGS_tools.py` | Volume loading, encoder inference, output and optional plots |
| `Pretrain/pretrain.py` | Training orchestration and iteration loop |
| `Pretrain/training_utils.py` | Configuration, learning-rate schedule and checkpoint I/O |
| `Pretrain/pretrain_provider.py` | Fixed-shape augmented pairs and batch iteration |
| `IIC-Net/loss/` | Original loss interfaces with focused numerical/device fixes |

The model architectures and state-dictionary layer names are retained. Historical stage-local `model` and `utils` modules remain separate; run each entry point in its own process. No missing IIC-Net training script has been reconstructed.

## Run the CPU regression suite

Install an appropriate PyTorch build, then:

```bash
python -m pip install -r requirements-pretrain.txt pytest
python -m pytest tests -q
python CGS/CGS.py --help
python Pretrain/pretrain.py --help
```

If an unrelated application has installed automatic pytest plugins in a shared environment, run in a clean environment or set `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`. The repository tests require no external plugins.

The suite covers frozen selection/coverage cases from commit `caa0ae6157be79fa1e39b047c10500c3c6f62cc0`, ties, rectangular-grid coordinates, invalid budgets, input normalization, slice limits, batch coordinates, output protection, augmentation shapes, learner backward/optimizer steps, checkpoint export/resume and empty loss masks. Synthetic volumes and checkpoint payloads stay in memory; temporary files contain only lightweight metadata/logs.

The GitHub Actions workflow runs the same CPU suite. Passing tests establish these contracts, not dataset accuracy, segmentation quality, GPU throughput or reproduction of the paper's benchmarks.

## Intentional behavior changes

- Correct rectangular-grid patch indexing and coverage colors in CGS plots.
- Reject multiple CGS inputs, invalid shapes/dtypes, impossible budgets and existing output directories instead of silently processing ambiguous inputs or overwriting results.
- Make GPU selection, slice limits and plotting explicit CLI options. Preserve the 100-slice default, intensity scaling, distance calculation, neighborhood ties and greedy traversal order.
- Keep pretraining pairs at a fixed size, use the supported iterator protocol, honor GPU visibility and stop exactly at the iteration budget.
- Save complete learner/optimizer checkpoints alongside encoder weights, and refuse encoder-only files for full resume.
- Define zero-mask losses and remove the weighted MSE's default-CUDA allocation.

Full training-state resume assumes the same model and GPU topology. RNG and data-worker states are not restored.
