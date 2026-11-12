"""Coverage-based greedy selection, independent of model and file I/O.

The neighborhood includes the patch itself and all ties at the Kth distance.
Candidates are visited in z/y/x order; equal scores keep the first candidate.
Non-overlap is defined on patch IDs, as in the original implementation.
"""

from dataclasses import dataclass
from itertools import product

import numpy as np


def positive_triplet(values, name):
    values = tuple(values)
    if len(values) != 3 or any(
        isinstance(v, (bool, np.bool_))
        or not isinstance(v, (int, np.integer))
        or v <= 0
        for v in values
    ):
        raise ValueError(f"{name} must contain three positive integers (z, y, x)")
    return values


def euclidean_distances(x, y, squared=False):
    """Use the original dot-product distance calculation and clipping order."""
    x, y = np.asarray(x), np.asarray(y)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError(
            "Features must be two matrices with matching feature dimensions"
        )
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Features must be finite")
    if not np.issubdtype(x.dtype, np.floating) or not np.issubdtype(
        y.dtype, np.floating
    ):
        raise ValueError("Features must be floating point")
    x_square = np.sum(x * x, axis=1, keepdims=True)
    y_square = x_square.T if x is y else np.sum(y * y, axis=1, keepdims=True).T
    distances = np.dot(x, y.T)
    distances *= -2
    distances += x_square
    distances += y_square
    np.maximum(distances, 0, out=distances)
    if x is y:
        np.fill_diagonal(distances, 0)
    if not squared:
        np.sqrt(distances, out=distances)
    return distances


def make_hard(dist_matrix, n_neighbors):
    """Convert distances to a boolean coverage matrix, retaining Kth-place ties."""
    distances = np.asarray(dist_matrix)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("Distances must be a square matrix")
    if not np.isfinite(distances).all() or (distances < 0).any():
        raise ValueError("Distances must be finite and nonnegative")
    if isinstance(n_neighbors, (bool, np.bool_)) or not isinstance(
        n_neighbors, (int, np.integer)
    ):
        raise ValueError("Neighbor count must be an integer")
    if not 1 <= n_neighbors <= len(distances):
        raise ValueError("Neighbor count must be between 1 and the number of patches")
    # Process one row at a time to avoid a second dense N x N float allocation.
    coverage = np.empty(distances.shape, dtype=bool)
    for index, row in enumerate(distances):
        threshold = np.partition(row, n_neighbors - 1)[n_neighbors - 1]
        coverage[index] = row <= threshold
    return coverage


def get_score(label_list, coverage):
    """Number of embeddings covered by at least one selected patch."""
    return int(np.any(coverage[label_list], axis=0).sum()) if len(label_list) else 0


def generate_candidate_points(start_point, window_size, point_cloud_size, positions):
    ids = [
        start_point
        + z * point_cloud_size[1] * point_cloud_size[2]
        + y * point_cloud_size[2]
        + x
        for z, y, x in product(*(range(v) for v in window_size))
    ]
    return ids, [positions[i] for i in ids]


def notoverlap(candidate, selected):
    return set(candidate).isdisjoint(selected)


@dataclass
class Selection:
    patch_ids: list
    positions: np.ndarray
    score: int


def select_subvolumes(coverage, positions, grid_shape, window_size, count):
    """Select disjoint patch windows by cumulative coverage.

    Raise if the greedy sequence cannot satisfy the requested budget; never
    return empty selections that would later fail during crop extraction.
    """
    grid = positive_triplet(grid_shape, "grid_shape")
    window = positive_triplet(window_size, "window_size")
    if any(w > g for w, g in zip(window, grid)):
        raise ValueError("Selection window does not fit the patch grid")
    if (
        isinstance(count, (bool, np.bool_))
        or not isinstance(count, (int, np.integer))
        or count <= 0
    ):
        raise ValueError("Subvolume count must be a positive integer")
    n = int(np.prod(grid))
    positions = np.asarray(positions)
    if coverage.shape != (n, n) or positions.shape not in ((n, 3), (n, 1, 3)):
        raise ValueError("Coverage and positions must match the patch grid")
    if count * int(np.prod(window)) > n:
        raise ValueError("Requested budget exceeds the number of available patches")
    selected, results = [], []
    for _ in range(count):
        best = None
        for z, y, x in product(*(range(g - w + 1) for g, w in zip(grid, window))):
            start = z * grid[1] * grid[2] + y * grid[2] + x
            ids, coords = generate_candidate_points(start, window, grid, positions)
            if not notoverlap(ids, selected):
                continue
            score = get_score(ids + selected, coverage)
            if best is None or score > best.score:
                best = Selection(ids, np.asarray(coords), score)
        if best is None:
            raise ValueError(
                f"No disjoint candidate remains after {len(results)} selections; reduce the budget"
            )
        selected.extend(best.patch_ids)
        results.append(best)
    return results
