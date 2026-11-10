"""Tests for prg_toolbox.plotting.plot_imports: shared styling helpers."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from prg_toolbox import MSVariables, momentum_space_distribution  # noqa: E402
from prg_toolbox.plotting import plot_momentum_space_distribution  # noqa: E402
from prg_toolbox.plotting.plot_imports import (  # noqa: E402
    set_colors_from_palette,
    set_default_kwargs,
)


class TestSetDefaultKwargsColors:
    def test_none_uses_all_defaults(self):
        *_, colors = set_default_kwargs(None)
        assert colors == {
            "data": "tab:purple",
            "surrogate": "tab:blue",
            "reference": "0.7",
        }

    def test_string_shorthand_overrides_only_data(self):
        *_, colors = set_default_kwargs("green")
        assert colors["data"] == "green"
        assert colors["surrogate"] == "tab:blue"
        assert colors["reference"] == "0.7"

    def test_partial_dict_overrides_only_given_keys(self):
        *_, colors = set_default_kwargs({"surrogate": "black"})
        assert colors["data"] == "tab:purple"
        assert colors["surrogate"] == "black"
        assert colors["reference"] == "0.7"

    def test_full_dict_overrides_everything(self):
        custom = {"data": "red", "surrogate": "blue", "reference": "green"}
        *_, colors = set_default_kwargs(custom)
        assert colors == custom

    def test_invalid_type_raises_typeerror(self):
        with pytest.raises(TypeError, match="colors must be"):
            set_default_kwargs(["red", "blue", "green"])


class TestSetColorsFromPalette:
    def test_none_uses_default_data_colormap(self):
        colors = set_colors_from_palette(3, None, data_or_surrogate="data")
        expected = set_colors_from_palette(
            3, {"data": "magma"}, data_or_surrogate="data"
        )
        assert colors == expected

    def test_string_shorthand_overrides_only_data_colormap(self):
        colors_str = set_colors_from_palette(3, "plasma", data_or_surrogate="data")
        colors_dict = set_colors_from_palette(
            3, {"data": "plasma"}, data_or_surrogate="data"
        )
        assert colors_str == colors_dict

    def test_string_shorthand_does_not_affect_surrogate_colormap(self):
        # A 'data'-only string override should leave 'surrogate' at its default.
        colors_default_surrogate = set_colors_from_palette(
            3, None, data_or_surrogate="surrogate"
        )
        colors_after_string = set_colors_from_palette(
            3, "plasma", data_or_surrogate="surrogate"
        )
        assert colors_default_surrogate == colors_after_string

    def test_partial_dict_overrides_only_given_key(self):
        colors_default_surrogate = set_colors_from_palette(
            3, None, data_or_surrogate="surrogate"
        )
        colors_after_partial = set_colors_from_palette(
            3, {"data": "plasma"}, data_or_surrogate="surrogate"
        )
        assert colors_default_surrogate == colors_after_partial

    def test_invalid_type_raises_typeerror(self):
        with pytest.raises(TypeError, match="palette must be"):
            set_colors_from_palette(3, ("magma", "viridis"))


class TestPlotMomentumSpaceDistribution:
    @pytest.fixture
    def result(self, random_binary_array):
        msvars = MSVariables(random_binary_array, rg_steps=4)
        return momentum_space_distribution(msvars)

    @pytest.fixture
    def ax(self):
        fig, ax = plt.subplots()
        yield ax
        plt.close(fig)

    def test_one_curve_per_step_plus_gaussian(self, result, ax):
        plot_momentum_space_distribution(result, ax=ax)
        assert len(ax.get_lines()) == result.rg_steps + 2
        assert ax.get_yscale() == "log"
        assert [t.get_text() for t in ax.get_legend().get_texts()] == [
            r"$(K = N/16)$",
            "Gaussian",
        ]

    def test_gaussian_reference_is_standard_normal(self, result, ax):
        plot_momentum_space_distribution(result, ax=ax)
        gaussian = ax.get_lines()[-1]
        x, y = gaussian.get_xdata(), gaussian.get_ydata()
        np.testing.assert_allclose(y, np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi))

    def test_empty_bins_are_not_drawn(self, result, ax):
        plot_momentum_space_distribution(result, ax=ax)
        for line in ax.get_lines()[:-1]:
            assert np.all(line.get_ydata() > 0)

    def test_accepts_dictionary_input(self, result, ax):
        data = {
            "bin_centers": result.bin_centers,
            "avg_across_windows": result.avg_across_windows,
            "std_across_windows": result.std_across_windows,
        }
        plot_momentum_space_distribution(data, ax=ax)
        assert len(ax.get_lines()) == result.rg_steps + 2

    def test_surrogate_adds_curves_and_legend_entry(self, result, ax):
        plot_momentum_space_distribution(result, surrogate_data=result, ax=ax)
        assert len(ax.get_lines()) == 2 * (result.rg_steps + 1) + 1
        assert "Surrogate" in [t.get_text() for t in ax.get_legend().get_texts()]

    def test_rejects_mismatched_surrogate_type(self, result, ax):
        with pytest.raises(ValueError, match="same type"):
            plot_momentum_space_distribution(result, surrogate_data={}, ax=ax)

    def test_rejects_unknown_data_type(self, ax):
        with pytest.raises(ValueError, match="momentum_space_distribution"):
            plot_momentum_space_distribution([1, 2, 3], ax=ax)

    def test_upper_limit_ignores_transparent_first_curve(self, result, ax):
        plot_momentum_space_distribution(result, ax=ax)
        visible_max = max(np.max(y) for y in result.values[1:])
        assert ax.get_ylim()[1] == pytest.approx(2 * visible_max)
