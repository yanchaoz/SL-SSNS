import json
from pathlib import Path

import numpy as np
import pytest


def test_original_selection_and_coverage(stage_import):
    selection = stage_import("CGS", "selection")
    fixtures = json.loads(
        (Path(__file__).parent / "fixtures/selection.json").read_text()
    )
    for case in fixtures["cases"]:
        features = (
            np.random.RandomState(case["seed"])
            .normal(size=case["feature_shape"])
            .astype(np.float32)
        )
        if case["all_tied"]:
            features[:] = 1
        distances = selection.euclidean_distances(features, features)
        coverage = selection.make_hard(distances, case["neighbors"])
        np.testing.assert_array_equal(
            coverage, [[int(v) for v in row] for row in case["coverage"]]
        )
        positions = np.array(list(np.ndindex(*case["grid"])))[:, None, :]
        result = selection.select_subvolumes(
            coverage, positions, case["grid"], case["window"], case["count"]
        )
        assert [{"patch_ids": r.patch_ids, "score": r.score} for r in result] == case[
            "selected"
        ]


@pytest.mark.parametrize("neighbors", [0, -1, 5, 1.5, True])
def test_invalid_neighbors(stage_import, neighbors):
    selection = stage_import("CGS", "selection")
    with pytest.raises(ValueError):
        selection.make_hard(np.zeros((4, 4)), neighbors)


def test_rectangular_grid_candidates_follow_xyz_coordinates(stage_import):
    selection = stage_import("CGS", "selection")
    positions = np.array(list(np.ndindex(2, 3, 4)))
    ids, coords = selection.generate_candidate_points(
        13, (1, 2, 2), (2, 3, 4), positions
    )
    assert ids == [13, 14, 17, 18]
    np.testing.assert_array_equal(coords, [[1, 0, 1], [1, 0, 2], [1, 1, 1], [1, 1, 2]])


def test_exhausted_greedy_budget_is_explicit(stage_import):
    selection = stage_import("CGS", "selection")
    # A middle window wins first, leaving no disjoint window of size two.
    coverage = np.array(
        [[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=bool
    )
    with pytest.raises(ValueError, match="No disjoint candidate"):
        selection.select_subvolumes(coverage, np.zeros((4, 3)), (1, 1, 4), (1, 1, 2), 2)


def test_provider_normalization_padding_and_slice_limit(stage_import, monkeypatch):
    tools = stage_import("CGS", "CGS_tools")
    cfg = tools.load_config("SL")
    cfg.MODEL.crop_size, cfg.MODEL.stride = [2, 4, 4], [2, 2, 2]
    raw = np.arange(7 * 5 * 5, dtype=np.uint8).reshape(7, 5, 5)
    monkeypatch.setattr(tools.Path, "is_file", lambda _: True)
    monkeypatch.setattr(tools.h5py, "is_hdf5", lambda _: False)
    monkeypatch.setattr(tools.imageio, "volread", lambda _: raw)
    provider = tools.ProviderValid(cfg, max_slices=3)
    assert provider.original_shape == (3, 5, 5)
    assert len(provider) == 8
    expected = np.pad(raw[:3], ((0, 2), (0, 2), (0, 2)), mode="reflect")
    patch, position = provider[7]
    np.testing.assert_array_equal(position, [2, 2, 2])
    np.testing.assert_allclose(patch[0], expected[2:4, 2:6, 2:6] / 255, rtol=1e-7)
    assert tools.ProviderValid(cfg, max_slices=0).original_shape == raw.shape
    cfg.MODEL.data_list *= 2
    with pytest.raises(ValueError, match="exactly one"):
        tools.ProviderValid(cfg)


def test_save_and_extract_without_overwriting(stage_import, monkeypatch, tmp_path):
    tools = stage_import("CGS", "CGS_tools")
    cfg = tools.load_config("SL")
    cfg.CGS.n_neighbors_list, cfg.CGS.subvolume_num = [1], 1
    stored, crops = {}, {}
    monkeypatch.setattr(
        tools.np,
        "save",
        lambda path, value: stored.update({str(path): np.array(value)}),
    )
    monkeypatch.setattr(tools.np, "load", lambda path, **_: stored[str(path)])
    monkeypatch.setattr(
        tools.imageio, "volwrite", lambda path, value: crops.update({str(path): value})
    )
    features = np.array([[0.0], [1.0], [3.0], [7.0]], dtype=np.float32)
    positions = np.array([[[0, 0, 0]], [[0, 0, 2]], [[0, 0, 4]], [[0, 0, 6]]])
    output = tmp_path / "selection"
    tools.save_results(
        [1], 1, features, positions, [1, 1, 2], [1, 1, 4], output, make_plots=False
    )
    volume = np.arange(2 * 4 * 10, dtype=np.uint8).reshape(2, 4, 10)
    tools.extract_subvolume(cfg, [1, 1, 2], [1, 1, 2], [2, 4, 4], volume, output)
    np.testing.assert_array_equal(next(iter(crops.values())), volume[:, :, :6])
    assert next(iter(crops.values())).shape == (2, 4, 6)
    with pytest.raises(FileExistsError):
        tools.save_results(
            [1], 1, features, positions, [1, 1, 2], [1, 1, 4], output, make_plots=False
        )


def test_inference_batch_coordinates(stage_import):
    import torch

    tools = stage_import("CGS", "CGS_tools")
    dataset = torch.utils.data.TensorDataset(
        torch.arange(12, dtype=torch.float32).reshape(4, 1, 3),
        torch.arange(12).reshape(4, 3),
    )
    model = torch.nn.Flatten(1)
    features, positions = tools.run_inference(
        model, torch.utils.data.DataLoader(dataset, batch_size=2), dataset, "cpu"
    )
    assert features.shape == (4, 3)
    assert positions.shape == (4, 1, 3)
    np.testing.assert_array_equal(positions[:, 0], np.arange(12).reshape(4, 3))


def test_plot_colors_follow_coverage(stage_import, monkeypatch, tmp_path):
    tools = stage_import("CGS", "CGS_tools")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    original_subplots = plt.subplots
    figures = []

    def capture():
        fig, ax = original_subplots()
        figures.append((fig, ax))
        return fig, ax

    monkeypatch.setattr(plt, "subplots", capture)
    embedding = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    coverage = np.eye(4, dtype=bool)
    coverage[0, 1] = True
    output = tmp_path / "coverage.png"
    tools.plot_selection(embedding, [0], coverage, output)
    groups = {
        collection.get_label(): collection.get_offsets()
        for collection in figures[0][1].collections
    }
    np.testing.assert_array_equal(groups["Covered"], [[1, 1]])
    np.testing.assert_array_equal(groups["Selected"], [[0, 0]])
    np.testing.assert_array_equal(groups["Others"], [[2, 2], [3, 3]])
    assert output.is_file()
