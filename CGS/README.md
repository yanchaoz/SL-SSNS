# Coverage-based greedy selection (CGS)

[Back to the project overview](../README.md)

CGS extracts patch embeddings from a raw EM volume and selects groups of spatially adjacent patches for annotation. The included `selection_model.ckpt` provides the encoder weights.

## Prepare the input

Install the [CGS dependencies](../README.md#installation) and edit [`config/SL.yaml`](config/SL.yaml). Paths in this configuration are resolved from the working directory; run the script from `CGS/`.

| Setting | Default | Meaning |
| --- | --- | --- |
| `MODEL.folder_name` | `./data` | Input directory |
| `MODEL.data_list` | `['ac3.tif']` | Use one volume per run |
| `MODEL.model_pth` | `selection_model.ckpt` | Encoder checkpoint |
| `MODEL.crop_size` | `[18, 160, 160]` | Patch size in `(z, y, x)` voxels |
| `MODEL.stride` | `[8, 40, 40]` | Patch stride in `(z, y, x)` voxels |
| `CGS.subvolume_num` | `4` | Number of subvolumes to select |
| `CGS.n_neighbors_list` | `[30]` | Neighbor counts used for coverage scoring |
| `CGS.window_size` | `[1, 9, 9]` | Selected window size in patch-grid units |

The loader first attempts HDF5 input with dataset key `main`, then falls back to `imageio.volread` (for example, a multipage TIFF). Supply a grayscale 3D volume in `(z, y, x)` order without a channel or time axis. Intensities are converted to `float32` and divided by 255; arbitrary 16-bit values are not rescaled automatically.

The current loader keeps only `[:100]` along z. Prepare the intended selection region before running, especially when separating training and test data. Each input dimension must be at least the patch size, and the resulting patch grid must fit the requested selection window. The neighbor count must not exceed the number of patch embeddings. Ensure that the grid contains enough disjoint patch windows for the requested budget.

With the supplied configuration, the nominal output subvolume size is **18 × 480 × 480 voxels**, calculated per axis as `stride × (window_size − 1) + crop_size`. This describes the checked-in configuration, not every experiment in the paper. `CGS.point_cloud_size` in the YAML is not used by the entry point; grid dimensions are calculated from the input.

## Choose a device and run

[`CGS.py`](CGS.py) currently sets:

```python
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
```

`cuda:3` means the fourth GPU visible to PyTorch. On a single-GPU machine, change it to `cuda:0`; to use the CPU even when CUDA is available, set `torch.device('cpu')`. There is currently no command-line device option. Restricting `CUDA_VISIBLE_DEVICES` alone does not fix the hard-coded index because visible GPUs are renumbered.

From the repository root, after preparing the input and device:

```bash
cd CGS
python CGS.py --cfg SL
```

To use another configuration, place it in `CGS/config/` and pass its basename without `.yaml`.

## Understand the outputs

Results are written under `CGS/record/`. In the filenames below, `K` is the neighbor count and `i` is the zero-based selection iteration.

| File | Contents |
| --- | --- |
| `list_K_i.npy` | Patch indices for the subvolume selected at iteration `i` |
| `position_K_i.npy` | Corresponding patch-start coordinates in `(z, y, x)` order |
| `subvolume_K_i.tif` | Raw image crop for annotation |
| `umap_K_i.png` | UMAP view of selected, covered and remaining patch embeddings, with Matplotlib's default PNG format |

Inspect every crop before annotation. Patch indices are checked for overlap, but overlapping receptive fields can still produce voxel overlap between selected crops. Reflection padding may appear near the input boundaries. Coordinates refer to the loaded, truncated volume with padding appended at its high ends; offsets from an earlier manual crop are not recorded.

## Current implementation limits

- **One volume per run:** although the loader accepts a list, selection uses the first volume's grid and crop extraction reads `datasets[0]`. A multi-volume run is not a supported selection workflow.
- **Memory:** pairwise distances form an `N × N` matrix for `N` patches, with additional arrays for neighbor coverage. Start with a manageable region; memory grows quadratically with patch count.
- **Plots:** `plt.show(block=True)` pauses after each selection. Close the plot window to continue. For a non-interactive run, select a non-interactive Matplotlib backend, such as `Agg`, before launching Python.
- **Repeated runs:** output filenames are reused. Move or copy `record/` to a new location before rerunning if you need to retain previous selections.
- **Reproducibility:** UMAP sets `random_state=40`, but the repository has no locked selection environment. Record package versions, input region, configuration and checkpoint for each experiment.

These notes describe [`CGS_tools.py`](CGS_tools.py) as released. A successful import or help command does not verify selection quality or reproduce the paper's results.
