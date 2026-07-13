"""Tests for prg_toolbox.plotting.plot_imports: shared styling helpers."""

import pytest

from prg_toolbox.plotting.plot_imports import set_default_kwargs, set_colors_from_palette


class TestSetDefaultKwargsColors:
    def test_none_uses_all_defaults(self):
        *_, colors = set_default_kwargs(None)
        assert colors == {"data": "tab:purple", "surrogate": "tab:blue", "reference": "0.7"}

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
        import matplotlib.pyplot as plt
        colors = set_colors_from_palette(3, None, data_or_surrogate="data")
        expected = set_colors_from_palette(3, {"data": "magma"}, data_or_surrogate="data")
        assert colors == expected

    def test_string_shorthand_overrides_only_data_colormap(self):
        colors_str = set_colors_from_palette(3, "plasma", data_or_surrogate="data")
        colors_dict = set_colors_from_palette(3, {"data": "plasma"}, data_or_surrogate="data")
        assert colors_str == colors_dict

    def test_string_shorthand_does_not_affect_surrogate_colormap(self):
        # A 'data'-only string override should leave 'surrogate' at its default.
        colors_default_surrogate = set_colors_from_palette(3, None, data_or_surrogate="surrogate")
        colors_after_string = set_colors_from_palette(3, "plasma", data_or_surrogate="surrogate")
        assert colors_default_surrogate == colors_after_string

    def test_partial_dict_overrides_only_given_key(self):
        colors_default_surrogate = set_colors_from_palette(3, None, data_or_surrogate="surrogate")
        colors_after_partial = set_colors_from_palette(3, {"data": "plasma"}, data_or_surrogate="surrogate")
        assert colors_default_surrogate == colors_after_partial

    def test_invalid_type_raises_typeerror(self):
        with pytest.raises(TypeError, match="palette must be"):
            set_colors_from_palette(3, ("magma", "viridis"))
