import pandas as pd
import yaml
import torch
from tqdm import tqdm
from addict import Dict as AttrDict
from collections import OrderedDict
from model.encoder import UNet_PNI_encoder
import os
import math
import h5py
import imageio
import numpy as np
from torch.utils.data import Dataset
import umap
import matplotlib.pyplot as plt


def compute_padding_and_num(size: int, patch_size: int, stride: int):
    """
    Compute the required padding and number of patches for a given dimension.

    Args:
        size (int): Original size of the dimension.
        patch_size (int): Size of the patch.
        stride (int): Stride length.

    Returns:
        padding (int): Padding required to fit patches evenly.
        num (int): Number of patches along this dimension.
    """
    num = math.ceil((size - patch_size) / stride) + 1
    total = stride * (num - 1) + patch_size
    padding = total - size
    return padding, num


class ProviderValid(Dataset):
    def __init__(self, cfg):
        """
        Dataset class for validation.

        Args:
            cfg (object): Configuration object containing dataset and model info.
        """
        self.cfg = cfg
        self.model_type = cfg.MODEL.model_type
        self.crop_size = cfg.MODEL.crop_size
        self.out_size = cfg.MODEL.out_size
        self.stride = cfg.MODEL.stride
        self.dataset_names = cfg.MODEL.data_list
        self.data_folder = cfg.MODEL.folder_name

        print(f"Validation dataset:", self.data_folder, self.dataset_names)

        # Load all volumes
        self.datasets = []
        for name in self.dataset_names:
            path = os.path.join(self.data_folder, name)
            print(f"Loading {name} ...")
            try:
                with h5py.File(path, 'r') as f:
                    data = f['main'][:][:100]
            except Exception:
                data = imageio.volread(path)[:100]
            print(f"Loaded shape: {data.shape}")
            self.datasets.append(data)

        # Original shape of the first volume
        self.original_shape = list(self.datasets[0].shape)

        # Compute padding and patch numbers for z, y, x
        self.valid_padding = [
            *compute_padding_and_num(self.original_shape[0], self.crop_size[0], self.stride[0]),  # z
            *compute_padding_and_num(self.original_shape[1], self.crop_size[1], self.stride[1]),  # y
            *compute_padding_and_num(self.original_shape[2], self.crop_size[2], self.stride[2]),  # x
        ]

        # Unpack padding and counts
        self.padding_z, self.num_z = self.valid_padding[0], self.valid_padding[1]
        self.padding_y, self.num_y = self.valid_padding[2], self.valid_padding[3]
        self.padding_x, self.num_x = self.valid_padding[4], self.valid_padding[5]

        # Apply symmetric reflection padding
        pad_width = (
            (0, 2 * self.padding_z),
            (0, 2 * self.padding_y),
            (0, 2 * self.padding_x),
        )
        self.datasets = [np.pad(d, pad_width, mode='reflect') for d in self.datasets]

        self.padded_shape = list(self.datasets[0].shape)
        self.patches_per_volume = self.num_z * self.num_y * self.num_x
        print(self.num_z, self.num_y, self.num_x)
        self.total_patches = self.patches_per_volume * len(self.datasets)

    def __len__(self):
        """
        Returns total number of patch samples.
        """
        return self.total_patches

    def __getitem__(self, index):
        """
        Retrieve a single patch based on index.

        Args:
            index (int): Linear index for patch retrieval.

        Returns:
            patch (np.ndarray): A single normalized patch of shape (1, D, H, W).
        """
        # Determine volume and local patch index
        volume_idx = index // self.patches_per_volume
        local_idx = index % self.patches_per_volume

        # Decode 3D position from index
        z_idx = local_idx // (self.num_y * self.num_x)
        rem = local_idx % (self.num_y * self.num_x)
        y_idx = rem // self.num_x
        x_idx = rem % self.num_x

        # Compute crop start and end positions
        start_z = z_idx * self.stride[0]
        start_y = y_idx * self.stride[1]
        start_x = x_idx * self.stride[2]

        end_z = start_z + self.crop_size[0]
        end_y = start_y + self.crop_size[1]
        end_x = start_x + self.crop_size[2]

        # Ensure boundaries don't exceed volume size
        if end_z > self.padded_shape[0]:
            start_z = self.padded_shape[0] - self.crop_size[0]
            end_z = self.padded_shape[0]
        if end_y > self.padded_shape[1]:
            start_y = self.padded_shape[1] - self.crop_size[1]
            end_y = self.padded_shape[1]
        if end_x > self.padded_shape[2]:
            start_x = self.padded_shape[2] - self.crop_size[2]
            end_x = self.padded_shape[2]

        pos = [start_z, start_y, start_x]
        # Extract and normalize the patch
        volume = self.datasets[volume_idx]
        patch = volume[start_z:end_z, start_y:end_y, start_x:end_x].copy()
        patch = patch.astype(np.float32) / 255.0
        patch = patch[np.newaxis, ...]  # Add channel dimension
        return np.ascontiguousarray(patch, dtype=np.float32), np.array(pos, dtype=np.int32)


def load_config(cfg_name):
    """Load configuration file as an AttrDict object."""
    config_path = os.path.join('./config', cfg_name + '.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    print(f'Loading config from: {config_path}')
    with open(config_path, 'r') as f:
        return AttrDict(yaml.safe_load(f))


def load_model(cfg, checkpoint_path, device):
    """Initialize the model and load pre-trained weights."""
    print('Initializing Superhuman model...')
    model = UNet_PNI_encoder(
        in_planes=cfg.MODEL.input_nc,
        filters=cfg.MODEL.filters,
        pad_mode=cfg.MODEL.pad_mode,
        bn_mode=cfg.MODEL.bn_mode,
        relu_mode=cfg.MODEL.relu_mode,
        init_mode=cfg.MODEL.init_mode
    ).to(device)

    print(f'Loading checkpoint: {checkpoint_path}')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_weights', checkpoint)

    # Remove 'module.' prefix if trained with DataParallel
    new_state_dict = OrderedDict((k.replace('module.', ''), v) for k, v in state_dict.items())
    model.load_state_dict(new_state_dict)
    model.eval()
    return model


def run_inference(model, val_loader, provider, device):
    """Run inference on validation data and extract embeddings and positions."""
    features, positions = [], []
    print(f'Running inference on {len(provider)} sub-volumes...')
    pbar = tqdm(total=len(provider), desc='Extracting Features')

    for data, pos in val_loader:
        inputs = data.to(device, non_blocking=True)
        with torch.no_grad():
            outputs = model(inputs)
            features.append(outputs.cpu().numpy())
            positions.append(pos.numpy())
        pbar.update(1)
    pbar.close()
    return np.concatenate(features, axis=0), np.array(positions)


def save_results(n_neighbors_list, subvolume_num, features, positions, window_size, point_cloud_size):
    """Perform CGS patch selection and save results."""
    print('Computing pairwise distances...')
    dist_matrix_soft = euclidean_distances(features, features)
    os.makedirs('./record', exist_ok=True)
    print('UMAP Projection...')
    reducer = umap.UMAP(random_state=40)
    x_dr = reducer.fit_transform(features)
    umap1, umap2 = x_dr[:][:, 0], x_dr[:][:, 1]
    for n_neighbors in n_neighbors_list:
        print(f'\nSelecting subvolumes with {n_neighbors} neighbors...')
        dist_matrix = make_hard(dist_matrix_soft, n_neighbors)
        already_selected = []

        for iteration in range(subvolume_num):
            best_score = -1
            best_patch = []
            best_positions = []

            for z in range(point_cloud_size[0] - window_size[0] + 1):
                for y in range(point_cloud_size[1] - window_size[1] + 1):
                    for x in range(point_cloud_size[2] - window_size[2] + 1):
                        start_idx = z * point_cloud_size[1] * point_cloud_size[2] + y * point_cloud_size[2] + x
                        candidate_ids, candidate_pos = generate_candidate_points(
                            start_idx, window_size, point_cloud_size, positions
                        )
                        score = get_score(candidate_ids + already_selected, dist_matrix)

                        if score > best_score and notoverlap(candidate_ids, already_selected):
                            best_score = score
                            best_patch = candidate_ids
                            best_positions = candidate_pos

            already_selected += best_patch
            umap_subv(umap1, umap2, features, already_selected, dist_matrix, n_neighbors, iteration)
            np.save(f'./record/list_{n_neighbors}_{iteration}.npy', best_patch)
            np.save(f'./record/position_{n_neighbors}_{iteration}.npy', best_positions)
            print(f'Saved subvolume {iteration + 1}/{subvolume_num} | Score: {best_score:.4f}')


def Get_List_Max_Index(list_, n):
    N_large = pd.DataFrame({'score': list_}).sort_values(by='score', ascending=[False])
    return list(N_large.index)[:n], sum(list(N_large.score)[:n])


def euclidean_distances(x, y, squared=False):
    assert isinstance(x, np.ndarray) and x.ndim == 2
    assert isinstance(y, np.ndarray) and y.ndim == 2
    assert x.shape[1] == y.shape[1]

    x_square = np.sum(x * x, axis=1, keepdims=True)
    if x is y:
        y_square = x_square.T
    else:
        y_square = np.sum(y * y, axis=1, keepdims=True).T
    distances = np.dot(x, y.T)
    distances *= -2
    distances += x_square
    distances += y_square
    np.maximum(distances, 0, distances)
    if x is y:
        distances.flat[::distances.shape[0] + 1] = 0.0
    if not squared:
        np.sqrt(distances, distances)
    return distances
import numpy as np

def negative_cosine_distances(x, y):
    assert isinstance(x, np.ndarray) and x.ndim == 2
    assert isinstance(y, np.ndarray) and y.ndim == 2
    assert x.shape[1] == y.shape[1]
    x_norm = x / np.linalg.norm(x, axis=1, keepdims=True)
    y_norm = y / np.linalg.norm(y, axis=1, keepdims=True)
    sim_matrix = np.dot(x_norm, y_norm.T)
    distances = -sim_matrix
    return distances


def get_score(label_list, dist_matrix):
    covered_list = []
    for sample in label_list:
        sub_coverlist = list(np.where(dist_matrix[sample] != 0)[0])
        covered_list += sub_coverlist
    covered_set = list(set(covered_list))
    return len(covered_set)


def get_cover_list(label_list, n_neighbors, dist_matrix):
    covered_list = []
    for sample in label_list:
        a, b = Get_List_Max_Index(-dist_matrix[sample], n_neighbors)
        sub_coverlist = a
        covered_list += sub_coverlist
    covered_set = list(set(covered_list))
    return covered_set


def make_hard(dist_matrix, n_neighbors):
    hard_list = []
    for index in range(dist_matrix.shape[0]):
        th = np.partition(-dist_matrix[index], -n_neighbors)[-n_neighbors]
        hard_sub_list = np.uint(-dist_matrix[index] >= th)
        hard_list.append(hard_sub_list)
    return np.array(hard_list)


def generate_candidate_points(start_point, window_size, point_cloud_size, position):
    window_list = []
    position_list = []
    for index_z in range(window_size[0]):
        for index_y in range(window_size[1]):
            for index_x in range(window_size[2]):
                window_list.append(
                    start_point + index_z * (point_cloud_size[1] * point_cloud_size[2]) + index_y * point_cloud_size[
                        1] + index_x)
                position_list.append(
                    position[start_point + index_z * (point_cloud_size[1] * point_cloud_size[2]) + index_y *
                             point_cloud_size[1] + index_x])
    return window_list, position_list


def notoverlap(start_list, already_list):
    for i in start_list:
        if i in already_list:
            return False
    return True


def extract_subvolume(cfg, window_size, stride, patch_size, volume):
    n_neighbors_list = cfg.CGS.n_neighbors_list
    # print(n_neighbors_list)
    for n_neighbors in n_neighbors_list:
        for iteration in range(cfg.CGS.subvolume_num):
            print('saving..')
            position_list = np.load(f'./record/position_{n_neighbors}_{iteration}.npy')
            subvol_size = [stride[i] * (window_size[i] - 1) + patch_size[i] for i in range(3)]
            z, y, x = position_list[0][0]
            subvolume = volume[z:z + subvol_size[0], y:y + subvol_size[1], x:x + subvol_size[2]]
            imageio.volwrite(f'./record/subvolume_{n_neighbors}_{iteration}.tif', subvolume)


def umap_subv(x1, x2, features, select_list, dist_matrix, n_neighbors, iteration):
    cover_list = get_cover_list(select_list, n_neighbors, dist_matrix)
    x1_e = []
    x2_e = []
    plt.figure()
    for k in range(features.shape[0]):
        if (k not in select_list) and (k not in cover_list):
            x1_e.append(x1[k])
            x2_e.append(x2[k])

    plt.scatter(x1_e, x2_e, color='deepskyblue', s=3, label='Others')
    x1_c = []
    x2_c = []
    for j in range(features.shape[0]):
        if (j not in select_list) and (j in cover_list):
            x1_c.append(x1[j])
            x2_c.append(x2[j])
    plt.scatter(x1_c, x2_c, color='orange', marker='x', s=6, label='Covered vector')
    x1_s = []
    x2_s = []
    for i in range(features.shape[0]):
        if (i in select_list):
            x1_s.append(x1[i])
            x2_s.append(x2[i])
    plt.scatter(x1_s, x2_s, color='red', marker='*', s=6, label='Selected vector')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.savefig(f'./record/umap_{n_neighbors}_{iteration}')
    plt.show(block=True)
