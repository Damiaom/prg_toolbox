"""Tests for prg_toolbox.pipelines: run_PRG and the directory/parallel drivers."""

import dataclasses
import hashlib
import json
import os
import pickle

import matplotlib

matplotlib.use("Agg")  # headless backend, must be set before importing pyplot

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from prg_toolbox import (  # noqa: E402
    MSVariables,
    mean_variance,
    momentum_space_distribution,
    pipelines,
)
from prg_toolbox.config import AnalysisParams  # noqa: E402
from prg_toolbox.plotting import plot_momentum_space_distribution  # noqa: E402


@pytest.fixture
def small_binary_dir(tmp_path, rng):
    """A directory of 3 small synthetic .npy binary recordings."""
    data_dir = tmp_path / "recordings"
    data_dir.mkdir()
    for i in range(3):
        binary = (rng.random((16, 300)) < 0.3).astype(int)
        np.save(data_dir / f"rec{i}.npy", binary)
    return data_dir


@pytest.fixture
def params():
    p = AnalysisParams()
    p.rg_steps = 3
    p.observables = [mean_variance]
    p.verbose = "silent"
    return p


class TestRunPRG:
    def test_returns_populated_observable(self, rng, params):
        binary = (rng.random((16, 300)) < 0.3).astype(int)
        result = pipelines.run_PRG(binary, user_params=params)
        assert set(result.keys()) == {"mean_variance"}
        assert np.isfinite(result["mean_variance"].exponent)

    def test_rejects_wrong_params_type(self, rng):
        binary = (rng.random((16, 300)) < 0.3).astype(int)
        with pytest.raises(TypeError, match="AnalysisParams"):
            pipelines.run_PRG(binary, user_params={"not": "a dataclass"})

    def test_nsamples_greater_than_one_does_not_crash(self, rng, params):
        # average_observable_sample_values used to pass an
        # invalid kwarg to get_scaling_exponent for eigenvalue observables
        # whenever nsamples > 1.
        binary = (rng.random((16, 300)) < 0.3).astype(int)
        params.subsampling.nsamples = 2
        result = pipelines.run_PRG(binary, user_params=params)
        assert np.isfinite(result["mean_variance"].exponent)


class TestRunPRGInDirectory:
    def test_accepts_directory_path(self, small_binary_dir, params):
        pipelines.run_PRG_in_directory(
            str(small_binary_dir), user_params=params, save_results=False
        )

    def test_accepts_explicit_file_list(self, small_binary_dir, params):
        files = [str(p) for p in sorted(small_binary_dir.iterdir())]
        pipelines.run_PRG_in_directory(files, user_params=params, save_results=False)

    def test_save_results_writes_one_pkl_per_file(
        self, tmp_path, monkeypatch, small_binary_dir, params
    ):
        monkeypatch.chdir(tmp_path)
        pipelines.run_PRG_in_directory(
            str(small_binary_dir), user_params=params, save_results=True
        )

        pkl_files = [
            f
            for _, _, files in os.walk(tmp_path / "results")
            for f in files
            if f.endswith(".pkl")
        ]
        assert sorted(pkl_files) == ["rec0.pkl", "rec1.pkl", "rec2.pkl"]

    def test_skips_files_by_index_and_name(self, small_binary_dir, params, capsys):
        params.verbose = "warnings"  # skip messages are suppressed under "silent"
        pipelines.run_PRG_in_directory(
            str(small_binary_dir),
            skipped_files_list=[1, "rec2.npy"],
            user_params=params,
            save_results=False,
        )
        out = capsys.readouterr().out
        assert "rec0.npy skipped" in out
        assert "rec2.npy skipped" in out
        assert "rec1.npy skipped" not in out


class TestRunPRGInDirectoryParallel:
    def test_accepts_directory_path_not_just_a_list(self, small_binary_dir, params):
        # Regression test: this used to only accept a pre-filtered list of
        # .gdf files (`file_directory.endswith('.gdf')`), silently dropping
        # every other format and rejecting a plain directory path.
        pipelines.run_PRG_in_directory_parallel(
            str(small_binary_dir),
            user_params=params,
            save_results=False,
            num_cores_to_use=2,
        )

    def test_processes_non_gdf_files(
        self, small_binary_dir, params, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        pipelines.run_PRG_in_directory_parallel(
            str(small_binary_dir),
            user_params=params,
            save_results=True,
            num_cores_to_use=2,
        )
        pkl_files = [
            f
            for _, _, files in os.walk(tmp_path / "results")
            for f in files
            if f.endswith(".pkl")
        ]
        assert sorted(pkl_files) == ["rec0.pkl", "rec1.pkl", "rec2.pkl"]

    def test_load_data_call_does_not_crash(self, small_binary_dir, params):
        # Regression test: process_single_file used to call
        # load_data(path, load_params=prg_params), but load_data's real
        # kwarg is named user_params -- this raised a TypeError inside every
        # worker process.
        files = [str(p) for p in sorted(small_binary_dir.iterdir())]
        pipelines.run_PRG_in_directory_parallel(
            files, user_params=params, save_results=False, num_cores_to_use=2
        )

    def test_matches_sequential_results(
        self, small_binary_dir, params, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        seq_dir = tmp_path / "seq"
        par_dir = tmp_path / "par"
        seq_dir.mkdir()
        par_dir.mkdir()

        monkeypatch.chdir(seq_dir)
        pipelines.run_PRG_in_directory(
            str(small_binary_dir), user_params=params, save_results=True
        )
        monkeypatch.chdir(par_dir)
        pipelines.run_PRG_in_directory_parallel(
            str(small_binary_dir),
            user_params=params,
            save_results=True,
            num_cores_to_use=2,
        )

        seq_pkls = sorted(
            f
            for _, _, files in os.walk(seq_dir / "results")
            for f in files
            if f.endswith(".pkl")
        )
        par_pkls = sorted(
            f
            for _, _, files in os.walk(par_dir / "results")
            for f in files
            if f.endswith(".pkl")
        )
        assert seq_pkls == par_pkls


class TestRunPRGMomentumSpace:
    @pytest.fixture
    def ms_params(self, params):
        params.observables = [momentum_space_distribution]
        return params

    def test_momentum_space_observable_alone(self, rng, ms_params):
        binary = (rng.random((16, 300)) < 0.3).astype(int)
        result = pipelines.run_PRG(binary, user_params=ms_params)
        distribution = result["momentum_space_distribution"]
        assert len(distribution.avg_across_windows) == ms_params.rg_steps + 1
        assert distribution.cutoffs.tolist() == [16, 8, 4, 2]

    def test_matches_direct_call(self, rng, ms_params):
        binary = (rng.random((16, 300)) < 0.3).astype(int)
        result = pipelines.run_PRG(binary, user_params=ms_params)
        direct = momentum_space_distribution(MSVariables(binary, rg_steps=3))
        for from_pipeline, expected in zip(
            result["momentum_space_distribution"].avg_across_windows,
            direct.values,
            strict=True,
        ):
            np.testing.assert_allclose(from_pipeline, expected)

    def test_alongside_real_space_observables(self, rng, ms_params):
        ms_params.observables = [mean_variance, momentum_space_distribution]
        binary = (rng.random((16, 300)) < 0.3).astype(int)
        result = pipelines.run_PRG(binary, user_params=ms_params)
        assert set(result) == {"mean_variance", "momentum_space_distribution"}
        assert np.isfinite(result["mean_variance"].exponent)

    def test_averages_over_time_windows(self, rng, ms_params):
        ms_params.time_slicing.window_duration_ms = 100
        binary = (rng.random((16, 300)) < 0.3).astype(int)
        result = pipelines.run_PRG(binary, user_params=ms_params)
        distribution = result["momentum_space_distribution"]
        assert distribution.values[1].shape == (3, 200)  # windows x bins
        np.testing.assert_allclose(
            distribution.avg_across_windows[1], distribution.values[1].mean(axis=0)
        )
        assert np.any(distribution.std_across_windows[1] > 0)

    def test_uses_bins_from_params(self, rng, ms_params):
        ms_params.momentum_space_bins = np.linspace(-6, 6, 25)
        binary = (rng.random((16, 300)) < 0.3).astype(int)
        result = pipelines.run_PRG(binary, user_params=ms_params)
        np.testing.assert_allclose(
            result["momentum_space_distribution"].bin_edges, np.linspace(-6, 6, 25)
        )

    def test_saved_results_can_be_plotted(
        self, tmp_path, monkeypatch, small_binary_dir, ms_params
    ):
        monkeypatch.chdir(tmp_path)
        ms_params.momentum_space_bins = np.linspace(-6, 6, 25)
        pipelines.run_PRG_in_directory(
            str(small_binary_dir), user_params=ms_params, save_results=True
        )
        pkl_path = next(
            os.path.join(root, f)
            for root, _, files in os.walk(tmp_path / "results")
            for f in files
            if f == "rec0.pkl"
        )
        with open(pkl_path, "rb") as f:
            saved = pickle.load(f)["results"]["momentum_space_distribution"]
        assert saved["bin_centers"].shape == (24,)
        fig, ax = plt.subplots()
        plot_momentum_space_distribution(saved, ax=ax)
        plt.close(fig)

    def test_unset_bins_do_not_change_the_results_folder(
        self, tmp_path, monkeypatch, small_binary_dir, params
    ):
        # The folder name is a hash of the parameters; adding the (unset)
        # momentum_space_bins option must not rename existing analyses.
        monkeypatch.chdir(tmp_path)
        pipelines.run_PRG_in_directory(
            str(small_binary_dir), user_params=params, save_results=True
        )
        clean = dataclasses.asdict(params)
        clean["observables"] = [o.__name__ for o in params.observables]
        del clean["momentum_space_bins"]
        hash_params = {
            k: v for k, v in clean.items() if k not in ("verbose", "save_path")
        }
        expected = hashlib.sha256(
            json.dumps(hash_params, sort_keys=True).encode("utf-8")
        ).hexdigest()[:40]
        folders = [
            d
            for _, dirs, _ in os.walk(tmp_path / "results")
            for d in dirs
            if d.startswith("prg_analysis_")
        ]
        assert folders == [f"prg_analysis_{expected}"]
