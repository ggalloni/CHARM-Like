from typing import Callable, Dict, List

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib import gridspec
from matplotlib.axes import Axes

from charm_like.compute_statistics import plot_statistics
from charm_like.settings_class import Settings
from overall_plots import (
    get_settings_dict,
    plot_mismatch,
    plot_wrongfid,
    produce_row_plot,
)
from utils.common_functions import configure_plt, join, parse_args


def main_mosaic(
    settings: Dict[str, Settings],
    function: Callable,
    want_statistics=True,
    is_gaussian=False,
):
    configure_plt()

    if want_statistics:
        fig = plt.figure(figsize=(15, 12))
        gs_main = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.25)
    else:
        fig = plt.figure(figsize=(15, 7.5))
        gs_main = gridspec.GridSpec(1, 1)

    gs_grid = gridspec.GridSpecFromSubplotSpec(
        3,
        3,
        subplot_spec=gs_main[0],
        hspace=0.0,
        wspace=0.0,
    )

    grid_axes: List[Axes] = np.empty((3, 3), dtype=object)
    for i in range(3):
        for j in range(3):
            grid_axes[i, j] = fig.add_subplot(gs_grid[i, j])

    kwargs = {
        "settings": settings,
        "want_legend": False if "mismatch" in function.__name__ else True,
        "want_offset": settings["EE_correct"].want_offset,
        "fig": fig,
        "axess": grid_axes,
        "is_gaussian": is_gaussian,
    }
    function(**kwargs)

    for col in range(3):
        ref_ax = grid_axes[2, col]
        for row in range(2):
            grid_axes[row, col].get_shared_x_axes().joined(ref_ax, grid_axes[row, col])
            grid_axes[row, col].set_xticklabels([])

    for col in range(3):
        for row in range(3):
            if row == 1 and col == 0:
                grid_axes[row, col].set_ylabel("Relative Probability")
            else:
                grid_axes[row, col].set_ylabel("")

    for row in range(3):
        ref_ax = grid_axes[row, 0]
        for col in range(1, 3):
            grid_axes[row, col].get_shared_y_axes().joined(ref_ax, grid_axes[row, col])
            grid_axes[row, col].set_yticklabels([])

    params = settings["EE_correct"]
    if want_statistics:
        gs_grid = gridspec.GridSpecFromSubplotSpec(
            1,
            3,
            subplot_spec=gs_main[1],
            hspace=0.3,
            wspace=0.33,
        )

        grid_axes = np.empty((1, 3), dtype=object)
        for i in range(1):
            for j in range(3):
                grid_axes[i, j] = fig.add_subplot(gs_grid[i, j])
        correct_keys = [key for key in settings.keys() if "correct" in key]
        for i in range(3):
            params = settings[correct_keys[i]]

            extra = f"{params.offset_type}_offset_" if params.want_offset else ""

            cross_like = (
                "gaussian_cross"
                if is_gaussian
                else f"{params.offset_type}_offset_cross"
            )
            full_like = "full_gaussian" if is_gaussian else f"{extra}HL"
            marg_like = "gaussian_without_auto" if is_gaussian else f"{extra}mHL"

            cHL_chi2s = np.load(
                params.chi2_folder
                + f"{cross_like}_chi2_"
                + f"{params.extra_chi2s}{params.name}.npy"
            )
            HL_chi2 = np.load(
                params.chi2_folder
                + f"{full_like}_chi2_{params.extra_chi2s}{params.name}.npy"
            )
            mHL_chi2 = np.load(
                params.chi2_folder
                + f"{marg_like}_chi2_{params.extra_chi2s}{params.name}.npy"
            )

            hybrid_HL_chi2 = None
            if params.custom_idxs is not None:
                hybrid_HL_chi2 = np.load(
                    params.chi2_folder
                    + f"{extra}mHL_hybrid_chi2_{params.extra_chi2s}{params.name}.npy"
                )

            plot_statistics(
                params,
                HL_chi2,
                mHL_chi2,
                cHL_chi2s,
                hybrid_HL_chi2,
                want_offset=params.want_offset,
                in_ax=grid_axes[0, i],
            )
            leg = grid_axes[0, i].get_legend()
            handles = leg.legend_handles
            if i == 1:
                grid_axes[0, i].legend(
                    handles=handles,
                    loc="upper center",
                    ncol=len(handles),
                    frameon=True,
                    bbox_to_anchor=(0.5, 1.25),
                )
            else:
                grid_axes[0, i].legend(handles=[])
            grid_axes[0, i].set_xticklabels([])
            grid_axes[0, i].set_xticks([])

    func_name = function.__name__.replace("plot_", "")
    extra = f"{params.offset_type}_offset_" if params.want_offset else ""
    if is_gaussian:
        extra = "gaussian_"
    if params.save_plots:
        plt.savefig(
            params.plots_folder
            + f"../{extra}{params.extra_plots}"
            + f"{func_name}_mosaic_{params.N_chs}ch.png"
        )

    if params.show_plots:
        plt.show()


def collapsed_mosaic(
    settings: Dict[str, Settings], want_statistics=True, is_gaussian=False
):
    configure_plt()

    if want_statistics:
        fig = plt.figure(figsize=(18, 7))
        gs_main = gridspec.GridSpec(2, 1, height_ratios=[1, 1.1], hspace=0.5)
    else:
        fig = plt.figure(figsize=(18, 7.5))
        gs_main = gridspec.GridSpec(1, 1)

    gs_grid = gridspec.GridSpecFromSubplotSpec(
        1,
        3,
        subplot_spec=gs_main[0],
        hspace=0.0,
        wspace=0.0,
    )

    grid_axes: List[Axes] = np.empty((1, 3), dtype=object)
    for i in range(1):
        for j in range(3):
            grid_axes[i, j] = fig.add_subplot(gs_grid[i, j])

    produce_row_plot(
        settings["EE_correct"],
        settings["BB_correct"],
        settings["BB_notens_correct"],
        grid_axes[0],
        ["EE", "BB", "BB_notens"],
        want_legend=False,
        want_offset=settings["EE_correct"].want_offset,
        lss=["-", "-", "-"],
        is_gaussian=is_gaussian,
    )

    for col in range(3):
        for row in range(1):
            if row == 1 and col == 0:
                grid_axes[row, col].set_ylabel("Relative Probability")
            else:
                grid_axes[row, col].set_ylabel("")

    for row in range(1):
        ref_ax = grid_axes[row, 0]
        for col in range(1, 3):
            grid_axes[row, col].get_shared_y_axes().joined(ref_ax, grid_axes[row, col])
            grid_axes[row, col].set_yticklabels([])

    handles = [
        mpatches.Patch(color="red", label="Gauss." if is_gaussian else "HL"),
        mpatches.Patch(
            color="dodgerblue", label="Marg. Gauss." if is_gaussian else "mHL"
        ),
        mpatches.Patch(
            color="forestgreen", label="Cross Gauss." if is_gaussian else "cHL"
        ),
    ]
    if settings["EE_correct"].want_pixel_based:
        handles.insert(0, mpatches.Patch(color="goldenrod", label="Pixel-based"))

    if settings["EE_correct"].want_single_field_HL:
        handles.insert(1, mpatches.Patch(color="maroon", label="Single-field"))

    grid_axes[0, 0].set_ylabel(r"Relative Probability")

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=len(handles),
        frameon=False,
    )

    off = -0.02 if settings["EE_correct"].want_pixel_based else 0.08
    if is_gaussian:
        off = off - 0.08
    if settings["EE_correct"].want_single_field_HL:
        off = off - 0.03
    fig.patches.append(
        mpatches.Rectangle(
            (0.3 + off, 0.89),  # bottom-left corner (x0, y0)
            (0.5 - (0.3 + off)) * 2,  # width
            0.045,  # height
            linewidth=1,
            edgecolor="black",
            facecolor="none",
            transform=fig.transFigure,
            zorder=5,
            alpha=0.2,
        )
    )

    params = settings["EE_correct"]
    if want_statistics:
        gs_grid = gridspec.GridSpecFromSubplotSpec(
            1,
            3,
            subplot_spec=gs_main[1],
            hspace=0.3,
            wspace=0.4,
        )

        grid_axes = np.empty((1, 3), dtype=object)
        for i in range(1):
            for j in range(3):
                grid_axes[i, j] = fig.add_subplot(gs_grid[i, j])
        correct_keys = [key for key in settings.keys() if "correct" in key]
        for i in range(3):
            params = settings[correct_keys[i]]

            extra = f"{params.offset_type}_offset_" if params.want_offset else ""

            cross_like = (
                "gaussian_cross"
                if is_gaussian
                else f"{params.offset_type}_offset_cross"
            )
            full_like = "full_gaussian" if is_gaussian else f"{extra}HL"
            marg_like = "gaussian_without_auto" if is_gaussian else f"{extra}mHL"

            cHL_chi2s = np.load(
                params.chi2_folder
                + f"{cross_like}_chi2_"
                + f"{params.extra_chi2s}{params.name}.npy"
            )
            HL_chi2 = np.load(
                params.chi2_folder
                + f"{full_like}_chi2_{params.extra_chi2s}{params.name}.npy"
            )
            mHL_chi2 = np.load(
                params.chi2_folder
                + f"{marg_like}_chi2_{params.extra_chi2s}{params.name}.npy"
            )

            hybrid_HL_chi2 = None
            if params.custom_idxs is not None:
                hybrid_HL_chi2 = np.load(
                    params.chi2_folder
                    + f"{extra}mHL_hybrid_chi2_{params.extra_chi2s}{params.name}.npy"
                )

            plot_statistics(
                params,
                HL_chi2,
                mHL_chi2,
                cHL_chi2s,
                hybrid_HL_chi2,
                want_offset=params.want_offset,
                in_ax=grid_axes[0, i],
            )
            leg = grid_axes[0, i].get_legend()
            handles = leg.legend_handles
            if i == 1:
                grid_axes[0, i].legend(
                    handles=handles,
                    loc="upper center",
                    ncol=len(handles),
                    frameon=True,
                    bbox_to_anchor=(0.5, 1.25),
                )
            else:
                grid_axes[0, i].legend(handles=[])
            grid_axes[0, i].set_xticklabels([])
            grid_axes[0, i].set_xticks([])

    extra = f"{params.offset_type}_offset_" if params.want_offset else ""
    if is_gaussian:
        extra = "gaussian_"
    if params.save_plots:
        plt.savefig(
            params.plots_folder
            + f"../{extra}{params.extra_plots}"
            + f"collapsed_mosaic_{params.N_chs}ch.png"
        )

    if params.show_plots:
        plt.show()


if __name__ == "__main__":
    yaml.add_constructor("!join", join)

    parsed_args = parse_args()
    config_path = parsed_args.config_path
    field = parsed_args.field
    num_chs = parsed_args.num_chs

    is_gaussian = parsed_args.want_gaussian

    settings = get_settings_dict(config_path, num_chs)

    configure_plt()

    main_mosaic(settings, plot_mismatch, want_statistics=True, is_gaussian=is_gaussian)
    main_mosaic(settings, plot_wrongfid, want_statistics=False, is_gaussian=is_gaussian)

    collapsed_mosaic(settings, want_statistics=True, is_gaussian=is_gaussian)
