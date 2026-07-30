"""Tests for prg_toolbox.pipelines: run_PRG and the directory/parallel drivers."""

import os

import numpy as np
import pytest

from prg_toolbox import pipelines
from prg_toolbox import mean_variance
from prg_toolbox.config import AnalysisParams


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
        pipelines.run_PRG_in_directory(str(small_binary_dir), user_params=params, save_results=False)

    def test_accepts_explicit_file_list(self, small_binary_dir, params):
        files = [str(p) for p in sorted(small_binary_dir.iterdir())]
        pipelines.run_PRG_in_directory(files, user_params=params, save_results=False)

    def test_save_results_writes_one_pkl_per_file(self, tmp_path, monkeypatch, small_binary_dir, params):
        monkeypatch.chdir(tmp_path)
        pipelines.run_PRG_in_directory(str(small_binary_dir), user_params=params, save_results=True)

        pkl_files = [f for _, _, files in os.walk(tmp_path / "results") for f in files if f.endswith(".pkl")]
        assert sorted(pkl_files) == ["rec0.pkl", "rec1.pkl", "rec2.pkl"]

    def test_skips_files_by_index_and_name(self, small_binary_dir, params, capsys):
        params.verbose = "warnings"  # skip messages are suppressed under "silent"
        pipelines.run_PRG_in_directory(
            str(small_binary_dir), skipped_files_list=[1, "rec2.npy"], user_params=params, save_results=False
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
            str(small_binary_dir), user_params=params, save_results=False, num_cores_to_use=2
        )

    def test_processes_non_gdf_files(self, small_binary_dir, params, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        pipelines.run_PRG_in_directory_parallel(
            str(small_binary_dir), user_params=params, save_results=True, num_cores_to_use=2
        )
        pkl_files = [f for _, _, files in os.walk(tmp_path / "results") for f in files if f.endswith(".pkl")]
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

    def test_matches_sequential_results(self, small_binary_dir, params, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        seq_dir = tmp_path / "seq"
        par_dir = tmp_path / "par"
        seq_dir.mkdir()
        par_dir.mkdir()

        monkeypatch.chdir(seq_dir)
        pipelines.run_PRG_in_directory(str(small_binary_dir), user_params=params, save_results=True)
        monkeypatch.chdir(par_dir)
        pipelines.run_PRG_in_directory_parallel(
            str(small_binary_dir), user_params=params, save_results=True, num_cores_to_use=2
        )

        seq_pkls = sorted(f for _, _, files in os.walk(seq_dir / "results") for f in files if f.endswith(".pkl"))
        par_pkls = sorted(f for _, _, files in os.walk(par_dir / "results") for f in files if f.endswith(".pkl"))
        assert seq_pkls == par_pkls
