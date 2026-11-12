"""Model, volume I/O and visualization helpers for CGS.

Numerical selection lives in selection.py. Paths in YAML retain the historical
CGS-directory base, so the CLI also works when launched from the repository root.
"""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import h5py
import imageio.v2 as imageio
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset
from tqdm import tqdm

from model.encoder import UNet_PNI_encoder
from selection import (
    euclidean_distances,
    make_hard,
    positive_triplet,
    select_subvolumes,
)


def compute_padding_and_num(size, patch_size, stride):
    if patch_size <= 0 or stride <= 0 or size < patch_size:
        raise ValueError("Each input dimension must fit a positive patch and stride")
    num = math.ceil((size - patch_size) / stride) + 1
    return stride * (num - 1) + patch_size - size, num


class ProviderValid(Dataset):
    """Single-volume, z/y/x patch provider with the original high-end padding."""

    def __init__(self, cfg, max_slices=100):
        self.crop_size = positive_triplet(cfg.MODEL.crop_size, "crop_size")
        self.stride = positive_triplet(cfg.MODEL.stride, "stride")
        if len(cfg.MODEL.data_list) != 1:
            raise ValueError("CGS supports exactly one volume per run")
        if max_slices < 0:
            raise ValueError("max_slices must be nonnegative")
        path = Path(cfg.MODEL.folder_name) / cfg.MODEL.data_list[0]
        if not path.is_file():
            raise FileNotFoundError(path)
        if h5py.is_hdf5(path):
            with h5py.File(path, "r") as handle:
                data = handle["main"][: max_slices or None]
        else:
            data = imageio.volread(path)[: max_slices or None]
        if data.ndim != 3 or data.dtype != np.uint8:
            raise ValueError("Input must be a 3D uint8 volume in (z, y, x) order")
        self.original_shape = data.shape
        geometry = [
            compute_padding_and_num(s, p, step)
            for s, p, step in zip(data.shape, self.crop_size, self.stride)
        ]
        self.valid_padding = [v for pair in geometry for v in pair]
        (
            self.padding_z,
            self.num_z,
            self.padding_y,
            self.num_y,
            self.padding_x,
            self.num_x,
        ) = self.valid_padding
        self.datasets = [
            np.pad(data, [(0, 2 * pad) for pad, _ in geometry], mode="reflect")
        ]
        self.padded_shape = self.datasets[0].shape
        self.total_patches = self.patches_per_volume = (
            self.num_z * self.num_y * self.num_x
        )

    def __len__(self):
        return self.total_patches

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        z, remainder = divmod(index, self.num_y * self.num_x)
        y, x = divmod(remainder, self.num_x)
        position = np.array(
            [v * s for v, s in zip((z, y, x), self.stride)], dtype=np.int32
        )
        slices = tuple(
            slice(start, start + size) for start, size in zip(position, self.crop_size)
        )
        patch = self.datasets[0][slices].astype(np.float32) / 255.0
        return np.ascontiguousarray(patch[None]), position


def load_config(cfg_name):
    base = Path(__file__).resolve().parent
    path = Path(cfg_name)
    if path.suffix not in (".yaml", ".yml"):
        path = base / "config" / f"{cfg_name}.yaml"
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    cfg = SimpleNamespace(
        **{
            key: SimpleNamespace(**value) if isinstance(value, dict) else value
            for key, value in raw.items()
        }
    )
    for field in ("folder_name", "model_pth"):
        value = Path(getattr(cfg.MODEL, field))
        setattr(cfg.MODEL, field, str(value if value.is_absolute() else base / value))
    return cfg


def load_model(cfg, checkpoint_path, device):
    model = UNet_PNI_encoder(
        in_planes=cfg.MODEL.input_nc,
        filters=cfg.MODEL.filters,
        pad_mode=cfg.MODEL.pad_mode,
        bn_mode=cfg.MODEL.bn_mode,
        relu_mode=cfg.MODEL.relu_mode,
        init_mode=cfg.MODEL.init_mode,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state = checkpoint.get("model_weights", checkpoint)
    model.load_state_dict(
        {key.removeprefix("module."): value for key, value in state.items()}
    )
    return model.eval()


def run_inference(model, val_loader, provider, device):
    features, positions = [], []
    with (
        torch.inference_mode(),
        tqdm(total=len(provider), desc="Extracting features") as progress,
    ):
        for data, pos in val_loader:
            features.append(model(data.to(device)).cpu().numpy())
            positions.append(pos.numpy())
            progress.update(len(data))
    # Keep historical (N, 1, 3) coordinates independently of loader batch size.
    return np.concatenate(features), np.concatenate(positions)[:, None, :]


def save_results(
    n_neighbors_list,
    subvolume_num,
    features,
    positions,
    window_size,
    point_cloud_size,
    output_dir="record",
    make_plots=True,
    show_plots=False,
):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    if not n_neighbors_list or len(set(n_neighbors_list)) != len(n_neighbors_list):
        raise ValueError("Provide distinct neighbor counts")
    distances = euclidean_distances(features, features)
    selections = {}
    for neighbors in n_neighbors_list:
        coverage = make_hard(distances, neighbors)
        selections[neighbors] = select_subvolumes(
            coverage, positions, point_cloud_size, window_size, subvolume_num
        )
    embedding = None
    if make_plots:
        import umap

        if len(features) < 4:
            raise ValueError(
                "UMAP requires at least four patches; use --no-plots for tiny inputs"
            )
        embedding = umap.UMAP(random_state=40).fit_transform(features)
    output_dir.mkdir(parents=True, exist_ok=False)
    summary = []
    for neighbors, results in selections.items():
        coverage = make_hard(distances, neighbors) if make_plots else None
        selected = []
        for iteration, result in enumerate(results):
            selected.extend(result.patch_ids)
            np.save(output_dir / f"list_{neighbors}_{iteration}.npy", result.patch_ids)
            np.save(
                output_dir / f"position_{neighbors}_{iteration}.npy", result.positions
            )
            if make_plots:
                plot_selection(
                    embedding,
                    selected,
                    coverage,
                    output_dir / f"umap_{neighbors}_{iteration}.png",
                    show=show_plots,
                )
            summary.append(
                dict(
                    neighbors=int(neighbors), iteration=iteration, covered=result.score
                )
            )
    (output_dir / "selection.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return selections


def extract_subvolume(
    cfg, window_size, stride, patch_size, volume, output_dir="record"
):
    output_dir = Path(output_dir)
    size = [s * (w - 1) + p for s, w, p in zip(stride, window_size, patch_size)]
    for neighbors in cfg.CGS.n_neighbors_list:
        for iteration in range(cfg.CGS.subvolume_num):
            positions = np.load(
                output_dir / f"position_{neighbors}_{iteration}.npy", allow_pickle=False
            )
            start = positions.reshape(-1, 3)[0]
            crop = volume[tuple(slice(int(p), int(p + n)) for p, n in zip(start, size))]
            if crop.shape != tuple(size):
                raise ValueError("Selected subvolume extends beyond the padded input")
            imageio.volwrite(
                output_dir / f"subvolume_{neighbors}_{iteration}.tif", crop
            )


def plot_selection(embedding, selected, coverage, path, show=False):
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected_mask = np.zeros(len(embedding), dtype=bool)
    selected_mask[selected] = True
    covered = np.any(coverage[selected], axis=0)
    fig, ax = plt.subplots()
    for mask, color, marker, label in (
        (~covered & ~selected_mask, "deepskyblue", ".", "Others"),
        (covered & ~selected_mask, "orange", "x", "Covered"),
        (selected_mask, "red", "*", "Selected"),
    ):
        ax.scatter(*embedding[mask].T, color=color, marker=marker, s=6, label=label)
    ax.set(xlabel="UMAP 1", ylabel="UMAP 2")
    ax.legend()
    fig.savefig(path)
    if show:
        plt.show()
    plt.close(fig)
