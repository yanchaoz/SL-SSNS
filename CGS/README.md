# Coverage-based greedy selection (CGS)

[Back to the project overview](../README.md)

CGS extracts patch embeddings from a raw EM volume and selects groups of spatially adjacent patches for annotation. The included `selection_model.ckpt` provides the encoder weights.

## Prepare the input

Install the [CGS dependencies](../README.md#installation) and edit [`config/SL.yaml`](config/SL.yaml). Relative input/checkpoint paths in this configuration are resolved from `CGS/`. The CLI can be launched from the repository root or from `CGS/`.

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

The loader detects HDF5 input and reads dataset key `main`; other inputs are read with `imageio.volread` (for example, a multipage TIFF). Missing HDF5 keys are reported directly. Supply a grayscale 3D `uint8` volume in `(z, y, x)` order without a channel or time axis. Intensities are converted to `float32` and divided by 255; arbitrary 16-bit values are not rescaled automatically.

By default the loader keeps only `[:100]` along z. Set `--max-slices N` to change this limit or `--max-slices 0` to read the full volume. Prepare the intended selection region before running, especially when separating training and test data. Each input dimension must be at least the patch size, and the resulting patch grid must fit the requested selection window. The neighbor count must not exceed the number of patch embeddings. Ensure that the grid contains enough disjoint patch windows for the requested budget.

With the supplied configuration, the nominal output subvolume size is **18 × 480 × 480 voxels**, calculated per axis as `stride × (window_size − 1) + crop_size`. This describes the checked-in configuration, not every experiment in the paper. `CGS.point_cloud_size` in the YAML is not used by the entry point; grid dimensions are calculated from the input.

## Choose a device and run

From the repository root:

```bash
python CGS/CGS.py --cfg SL --device cpu --output-dir runs/selection --no-plots
```

Use `--device cuda:0` for a CUDA GPU. The default `auto` uses the first visible CUDA device if available and otherwise CPU; it respects `CUDA_VISIBLE_DEVICES`. Run `python CGS/CGS.py --help` to see all options without loading the model or data.

`--cfg` accepts a basename in `CGS/config/` or an explicit YAML path. `--output-dir` is relative to the working directory (default: `CGS/record/`). Existing output paths are refused to prevent overwriting another run. Selection remains single-volume; multiple inputs are rejected explicitly.

Omit `--no-plots` to save UMAP plots. Plotting is non-interactive by default; add `--show-plots` to display the figures.

## Understand the outputs

Results are written to `--output-dir`. In the filenames below, `K` is the neighbor count and `i` is the zero-based selection iteration.

| File | Contents |
| --- | --- |
| `list_K_i.npy` | Patch indices for the subvolume selected at iteration `i` |
| `position_K_i.npy` | Corresponding patch-start coordinates in `(z, y, x)` order |
| `subvolume_K_i.tif` | Raw image crop for annotation |
| `umap_K_i.png` | Optional UMAP view of selected, covered and remaining embeddings |
| `selection.json` | Neighbor count, iteration and cumulative coverage score |

Inspect every crop before annotation. Patch indices are checked for overlap, but overlapping receptive fields can still produce voxel overlap between selected crops. Reflection padding may appear near the input boundaries. Coordinates refer to the loaded, truncated volume with padding appended at its high ends; offsets from an earlier manual crop are not recorded.

## Current implementation limits

- **One volume per run:** multiple inputs are rejected. Run selection separately for each volume.
- **Memory:** pairwise distances form an `N × N` matrix for `N` patches, with additional arrays for neighbor coverage. Start with a manageable region; memory grows quadratically with patch count.
- **Plots:** use `--no-plots` to avoid the UMAP step. `--show-plots` pauses for interactive viewing.
- **Repeated runs:** choose a new output directory each time. Impossible selection budgets raise an error before selection files are written.
- **Reproducibility:** UMAP sets `random_state=40`, but the repository has no locked selection environment. Record package versions, input region, configuration and checkpoint for each experiment.

These notes describe [`CGS_tools.py`](CGS_tools.py) as released. A successful import or help command does not verify selection quality or reproduce the paper's results.

## Numerical behavior

[`selection.py`](selection.py) contains the distance, coverage and greedy-selection functions. K-nearest coverage includes the patch itself and all ties at the Kth distance. Equal candidate scores keep the first candidate in z/y/x traversal order. Frozen reference cases verify the original behavior on square x/y grids.

Rectangular grids now use the x-grid length for the y-row stride. The original helper incorrectly used the y-grid length, which could select the wrong patch IDs. UMAP coverage colors now follow the same boolean coverage matrix used for selection. These are correctness fixes; they do not change the encoder architecture.
