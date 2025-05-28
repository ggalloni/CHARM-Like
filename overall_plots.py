import os
import sys
from typing import Dict

import yaml

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import matplotlib.patches as mpatches
import numpy as np
from matplotlib import pyplot as plt

from charm_like.settings_class import Settings
from utils.common_functions import configure_plt, join, parse_args


def combined_HL_posterior_plot(
    params: Settings,
    default_chi2s,
    nodef1_chi2s,
    nodef2_chi2s,
    label,
    colors,
    ax=None,
    lss=["-", "--", ":"],
):
    if params.want_pixel_based:
        pixel_chi2s = params.get_pixel_based_chi2()
        grid = params.get_pixel_based_grid()
        ax.plot(
            grid,
            np.exp(
                -0.5
                * (np.mean(pixel_chi2s, axis=0) - np.min(np.mean(pixel_chi2s, axis=0)))
            ),
            color="goldenrod",
            ls="-",
            alpha=1,
            label="Pixel-based",
            zorder=-10,
        )

    # if params.want_single_field_HL:
    #     harm_like = (
    #         f"{params.offset_type}_offset_SFHL" if want_offset else "SFHL"
    #     )
    #     SFHL_chi2 = np.load(
    #         params.chi2_folder
    #         + f"{harm_like}_chi2_{params.extra_chi2s}{params.name}.npy"
    #     )
    #     ax.plot(
    #         params.grid,
    #         np.exp(
    #             -0.5
    #             * (
    #                 np.mean(SFHL_chi2, axis=0)
    #                 - np.min(np.mean(SFHL_chi2, axis=0))
    #             )
    #         ),
    #         color="maroon",
    #         ls="-",
    #         # lw=1,
    #         alpha=1,
    #         label="Pixel-based",
    #         zorder=-20,
    #     )

    if isinstance(colors, str):
        colors = [colors] * 3

    ax.plot(
        params.grid,
        np.exp(
            -0.5
            * (np.mean(default_chi2s, axis=0) - np.min(np.mean(default_chi2s, axis=0)))
        ),
        label=label,
        color=colors[0],
        ls=lss[0],
    )

    ax.plot(
        params.grid,
        np.exp(
            -0.5
            * (np.mean(nodef1_chi2s, axis=0) - np.min(np.mean(nodef1_chi2s, axis=0)))
        ),
        color=colors[1],
        ls=lss[1],
    )

    ax.plot(
        params.grid,
        np.exp(
            -0.5
            * (np.mean(nodef2_chi2s, axis=0) - np.min(np.mean(nodef2_chi2s, axis=0)))
        ),
        color=colors[2],
        ls=lss[2],
    )


def produce_row_plot(
    EE_settings: Settings,
    BB_settings: Settings,
    BB_notens_settings: Settings,
    axess: plt.axes,
    fields: list,
    want_legend=False,
    want_offset=False,
    lss=["-", "--", ":"],
    is_gaussian=False,
):
    extra = f"{EE_settings.offset_type}_offset_" if want_offset else ""

    colors = ["red", "dodgerblue", "forestgreen"]

    # HL chi2s
    full_like = "full_gaussian" if is_gaussian else f"{extra}HL"
    marg_like = "gaussian_without_auto" if is_gaussian else f"{extra}mHL"
    cross_like = (
        "gaussian_cross" if is_gaussian else f"{EE_settings.offset_type}_offset_cross"
    )

    file = (
        EE_settings.chi2_folder
        + f"{full_like}_chi2_{EE_settings.extra_chi2s}{EE_settings.name}.npy"
    )
    EE_HL_chi2 = np.load(file)

    file = (
        EE_settings.chi2_folder
        + f"{marg_like}_chi2_"
        + f"{EE_settings.extra_chi2s}{EE_settings.name}.npy"
    )
    EE_mHL_chi2 = np.load(file)

    file = (
        EE_settings.chi2_folder
        + f"{cross_like}_chi2_"
        + f"{EE_settings.extra_chi2s}{EE_settings.name}.npy"
    )
    EE_cHL_chi2 = np.load(file)

    combined_HL_posterior_plot(
        EE_settings,
        EE_HL_chi2,
        EE_mHL_chi2,
        EE_cHL_chi2,
        "",
        colors,
        ax=axess[0],
        lss=lss,
    )

    file = (
        BB_settings.chi2_folder
        + f"{full_like}_chi2_{BB_settings.extra_chi2s}{BB_settings.name}.npy"
    )
    BB_HL_chi2 = np.load(file)

    file = (
        BB_settings.chi2_folder
        + f"{marg_like}_chi2_"
        + f"{BB_settings.extra_chi2s}{BB_settings.name}.npy"
    )
    BB_mHL_chi2 = np.load(file)

    file = (
        BB_settings.chi2_folder
        + f"{cross_like}_chi2_"
        + f"{BB_settings.extra_chi2s}{BB_settings.name}.npy"
    )
    BB_cHL_chi2 = np.load(file)

    combined_HL_posterior_plot(
        BB_settings,
        BB_HL_chi2,
        BB_mHL_chi2,
        BB_cHL_chi2,
        "mHL",
        colors,
        ax=axess[1],
        lss=lss,
    )

    file = (
        BB_notens_settings.chi2_folder
        + f"{full_like}_chi2_{BB_notens_settings.extra_chi2s}"
        + f"{BB_notens_settings.name}.npy"
    )
    BB_notens_HL_chi2 = np.load(file)

    file = (
        BB_notens_settings.chi2_folder
        + f"{marg_like}_chi2_"
        + f"{BB_notens_settings.extra_chi2s}{BB_notens_settings.name}.npy"
    )
    BB_notens_mHL_chi2 = np.load(file)

    file = (
        BB_notens_settings.chi2_folder
        + f"{cross_like}_chi2_"
        + f"{BB_notens_settings.extra_chi2s}{BB_notens_settings.name}.npy"
    )
    BB_notens_cHL_chi2 = np.load(file)

    combined_HL_posterior_plot(
        BB_notens_settings,
        BB_notens_HL_chi2,
        BB_notens_mHL_chi2,
        BB_notens_cHL_chi2,
        "cHL",
        colors,
        ax=axess[2],
        lss=lss,
    )

    for i, ax in enumerate(axess):
        if fields[i] == "EE":
            N = (EE_settings.grid_steps - 1) // 160 + 1
            ticks = np.array([round(EE_settings.grid[i * 160], 4) for i in range(N)])
            ticks = ticks[:-2]
            ax.set_xticks(
                ticks,
                labels=[ticks[i] for i in range(len(ticks))],
            )
            ax.vlines(EE_settings.fiducial_value, 0, 1.1, color="black", ls="--")
            ax.set_xlim(0.04, 0.08)
            ax.set_xlabel(EE_settings.param_latex)
        elif fields[i] == "BB":
            N = (BB_settings.grid_steps - 1) // 200 + 1
            ticks = np.array([round(BB_settings.grid[i * 200], 4) for i in range(N)])
            ticks = ticks[:-1]
            ax.vlines(BB_settings.fiducial_value, 0, 1.1, color="black", ls="--")
            ax.set_xticks(
                ticks,
                labels=[ticks[i] for i in range(len(ticks))],
            )
            ax.set_xlim(0.0005 - 0.0005, 0.0195 + 0.0005)
            ax.set_xlabel(BB_settings.param_latex)
        elif fields[i] == "BB_notens":
            N = (BB_notens_settings.grid_steps - 1) // 200 + 1
            ticks = np.array(
                [round(BB_notens_settings.grid[i * 200], 4) for i in range(N)]
            )
            ticks = ticks[:-1]
            ax.vlines(BB_notens_settings.fiducial_value, 0, 1.1, color="black", ls="--")
            ax.set_xticks(
                ticks,
                labels=[ticks[i] for i in range(len(ticks))],
            )
            ax.set_xlim(0.0, 0.002 + 0.00005)
            ax.set_xticks(
                ticks / 10,
                labels=[ticks[i] / 10 for i in range(len(ticks))],
            )
            ax.set_xlabel(BB_notens_settings.param_latex)
        ax.set_ylim(0, 1.1)


def produce_column_plot(
    correct: Settings,
    incorrect_1: Settings,
    incorrect_2: Settings,
    axess: plt.axes,
    field: str,
    want_legend=False,
    want_offset=False,
    is_gaussian=False,
):
    extra = f"{correct.offset_type}_offset_" if want_offset else ""

    # HL chi2s
    like = "full_gaussian" if is_gaussian else f"{extra}HL"
    file = correct.chi2_folder + f"{like}_chi2_{correct.extra_chi2s}{correct.name}.npy"
    correct_HL_chi2 = np.load(file)

    file = (
        incorrect_1.chi2_folder
        + f"{like}_chi2_{incorrect_1.extra_chi2s}{incorrect_1.name}.npy"
    )
    incorrect_1_HL_chi2 = np.load(file)

    file = (
        incorrect_2.chi2_folder
        + f"{like}_chi2_{incorrect_2.extra_chi2s}{incorrect_2.name}.npy"
    )
    incorrect_2_HL_chi2 = np.load(file)

    combined_HL_posterior_plot(
        correct,
        correct_HL_chi2,
        incorrect_1_HL_chi2,
        incorrect_2_HL_chi2,
        "Gaussian" if is_gaussian else "HL",
        "red",
        ax=axess[0],
    )

    # mHL chi2s
    like = "gaussian_without_auto" if is_gaussian else f"{extra}mHL"
    file = (
        correct.chi2_folder
        + f"{like}_chi2_"
        + f"{correct.extra_chi2s}{correct.name}.npy"
    )
    correct_mHL_chi2 = np.load(file)

    file = (
        incorrect_1.chi2_folder
        + f"{like}_chi2_"
        + f"{incorrect_1.extra_chi2s}{incorrect_1.name}.npy"
    )
    incorrect_1_mHL_chi2 = np.load(file)

    file = (
        incorrect_2.chi2_folder
        + f"{like}_chi2_"
        + f"{incorrect_2.extra_chi2s}{incorrect_2.name}.npy"
    )
    incorrect_2_mHL_chi2 = np.load(file)

    combined_HL_posterior_plot(
        correct,
        correct_mHL_chi2,
        incorrect_1_mHL_chi2,
        incorrect_2_mHL_chi2,
        "Marg. Gaussian" if is_gaussian else "mHL",
        "dodgerblue",
        ax=axess[1],
    )

    # cHL chi2s
    like = "gaussian_cross" if is_gaussian else f"{correct.offset_type}_offset_cross"
    file = (
        correct.chi2_folder
        + f"{like}_chi2_"
        + f"{correct.extra_chi2s}{correct.name}.npy"
    )
    correct_cHL_chi2 = np.load(file)

    file = (
        incorrect_1.chi2_folder
        + f"{like}_chi2_"
        + f"{incorrect_1.extra_chi2s}{incorrect_1.name}.npy"
    )
    incorrect_1_cHL_chi2 = np.load(file)

    file = (
        incorrect_2.chi2_folder
        + f"{like}_chi2_"
        + f"{incorrect_2.extra_chi2s}{incorrect_2.name}.npy"
    )
    incorrect_2_cHL_chi2 = np.load(file)

    combined_HL_posterior_plot(
        correct,
        correct_cHL_chi2,
        incorrect_1_cHL_chi2,
        incorrect_2_cHL_chi2,
        "Cross Gaussian" if is_gaussian else "cHL",
        "forestgreen",
        ax=axess[2],
    )

    grid = correct.grid
    N = (correct.grid_steps - 1) // 200 + 1
    ticks = np.array([round(grid[i * 200], 4) for i in range(N)])
    ticks = ticks[:-1]
    for ax in axess:
        ax.vlines(grid[correct.fid_idx], 0, 1.1, color="black", ls="--")
        ax.set_xticks(
            ticks,
            labels=[ticks[i] for i in range(len(ticks))],
        )
        if field == "EE":
            N = (correct.grid_steps - 1) // 160 + 1
            ticks = np.array([round(grid[i * 160], 4) for i in range(N)])
            ticks = ticks[:-2]
            ax.set_xticks(
                ticks,
                labels=[ticks[i] for i in range(len(ticks))],
            )
            ax.set_xlim(0.04, 0.08)
        elif field == "BB":
            ax.set_xlim(0.0005 - 0.0005, 0.0195 + 0.0005)
        elif field == "BB_notens":
            ax.set_xlim(0.0, 0.002 + 0.00005)
            ax.set_xticks(
                ticks / 10,
                labels=[ticks[i] / 10 for i in range(len(ticks))],
            )
        ax.set_ylim(0, 1.1)

    axess[-1].set_xlabel(correct.param_latex)
    # if field == "BB_notens":
    #     axess[-1].set_xlabel(r"$10 \times$" + correct.param_latex)

    if want_legend:
        handles = [
            plt.Line2D([0], [0], color="black", ls="-", label=f"True {correct.param}"),
            plt.Line2D(
                [0],
                [0],
                color="black",
                ls="--",
                label=f"{correct.param} = {incorrect_1.wrong_fiducial_value}",
            ),
            plt.Line2D(
                [0],
                [0],
                color="black",
                ls=":",
                label=f"{correct.param} = {incorrect_2.wrong_fiducial_value}",
            ),
        ]
        axess[0].legend(handles=handles, loc="upper right")


def plot_mismatch(
    settings: Dict[str, Settings],
    want_offset=False,
    fig=None,
    axess=None,
    want_legend=False,
    is_mosaic=False,
    is_gaussian=False,
):
    configure_plt()

    if axess is None:
        fig, axes = plt.subplots(
            3,
            3,
            figsize=(18, 12),
            sharex="col",
            sharey="row",
            gridspec_kw={"hspace": 0.0, "wspace": 0.03},
        )
    else:
        axes = axess
    fig.set_tight_layout(False)
    fig.subplots_adjust(
        hspace=0.0, wspace=0.0, top=0.90, bottom=0.055, left=0.04, right=0.98
    )

    produce_column_plot(
        settings["EE_correct"],
        settings["EE_mismatch"],
        settings["EE_mismatch_2"],
        axess=axes[:, 0],
        field="EE",
        want_offset=want_offset,
        is_gaussian=is_gaussian,
    )

    produce_column_plot(
        settings["BB_correct"],
        settings["BB_mismatch"],
        settings["BB_mismatch_2"],
        axess=axes[:, 1],
        field="BB",
        want_offset=want_offset,
        is_gaussian=is_gaussian,
    )

    produce_column_plot(
        settings["BB_notens_correct"],
        settings["BB_notens_mismatch"],
        settings["BB_notens_mismatch_2"],
        axess=axes[:, 2],
        field="BB_notens",
        want_offset=want_offset,
        is_gaussian=is_gaussian,
    )

    patch_handles = [
        mpatches.Patch(color="red", label="Gauss." if is_gaussian else "HL"),
        mpatches.Patch(
            color="dodgerblue", label="Marg. Gauss." if is_gaussian else "mHL"
        ),
        mpatches.Patch(
            color="forestgreen", label="Cross Gauss." if is_gaussian else "cHL"
        ),
    ]
    if settings["EE_correct"].want_pixel_based:
        patch_handles.insert(0, mpatches.Patch(color="goldenrod", label="Pixel-based"))

    if settings["EE_correct"].want_single_field_HL:
        patch_handles.insert(1, mpatches.Patch(color="maroon", label="Single-field"))

    line_handles = [
        plt.Line2D(
            [0],
            [0],
            color="black",
            ls="-",
            label=r"$\widehat{N}_\ell = {N}_\ell^{\rm True}$",
        ),
        plt.Line2D(
            [0],
            [0],
            color="black",
            ls="--",
            label=r"$\widehat{N}_\ell = 0.8^2\ {N}_\ell^{\rm True}$",
        ),
        plt.Line2D(
            [0],
            [0],
            color="black",
            ls=":",
            label=r"$\widehat{N}_\ell = 1.2^2\ {N}_\ell^{\rm True}$",
        ),
    ]
    axes[0, 0].set_ylabel(r"Relative Probability")
    axes[1, 0].set_ylabel(r"Relative Probability")
    axes[2, 0].set_ylabel(r"Relative Probability")

    _ = fig.legend(
        handles=patch_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(patch_handles),
        frameon=False,
        handletextpad=0.5,
        columnspacing=1.0,
    )

    _ = fig.legend(
        handles=line_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=len(line_handles),
        frameon=False,
        handletextpad=0.5,
        columnspacing=1.0,
    )

    off = -0.01 if settings["EE_correct"].want_pixel_based else 0.05
    if is_gaussian:
        off = off - 0.07
    if settings["EE_correct"].want_single_field_HL:
        off = off - 0.07
    fig.patches.append(
        mpatches.Rectangle(
            (0.3 + off, 0.91),  # bottom-left corner (x0, y0)
            (0.5 - (0.3 + off)) * 2,  # width
            0.08,  # height
            linewidth=1,
            edgecolor="black",
            facecolor="none",
            transform=fig.transFigure,
            zorder=5,
            alpha=0.2,
        )
    )

    fsky = int(settings["BB_correct"].fsky)
    extra = (
        f"{settings['BB_correct'].offset_type}_offset_fsky{fsky}_"
        if want_offset
        else f"fsky{fsky}_"
    )
    if is_gaussian:
        extra = f"gaussian_fsky{fsky}_"
    if axess is None:
        plt.savefig(
            settings["EE_correct"].plots_folder
            + f"../{extra}QML_complete_joined_mismatch_"
            + f"{settings['EE_correct'].N_chs}ch.png"
        )

        plt.show()


def plot_wrongfid(
    settings: Dict[str, Settings],
    want_legend=False,
    want_offset=False,
    fig=None,
    axess=None,
    is_mosaic=False,
    is_gaussian=False,
):
    configure_plt()

    if axess is None:
        fig, axes = plt.subplots(
            3,
            3,
            figsize=(15, 7.5),
            sharex="col",
            sharey="row",
            gridspec_kw={"hspace": 0.0, "wspace": 0.03},
        )
    else:
        axes = axess
    fig.set_tight_layout(False)
    fig.subplots_adjust(
        hspace=0.0, wspace=0.0, top=0.95, bottom=0.055, left=0.04, right=0.98
    )

    produce_column_plot(
        settings["EE_correct"],
        settings["EE_wrongfid"],
        settings["EE_wrongfid_2"],
        axess=axes[:, 0],
        field="EE",
        want_legend=want_legend,
        want_offset=want_offset,
        is_gaussian=is_gaussian,
    )

    produce_column_plot(
        settings["BB_correct"],
        settings["BB_wrongfid"],
        settings["BB_wrongfid_2"],
        axess=axes[:, 1],
        field="BB",
        want_legend=want_legend,
        want_offset=want_offset,
        is_gaussian=is_gaussian,
    )

    produce_column_plot(
        settings["BB_notens_correct"],
        settings["BB_notens_wrongfid"],
        settings["BB_notens_wrongfid_2"],
        axess=axes[:, 2],
        field="BB_notens",
        want_legend=want_legend,
        want_offset=want_offset,
        is_gaussian=is_gaussian,
    )

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
    axes[0, 0].set_ylabel(r"Relative Probability")
    axes[1, 0].set_ylabel(r"Relative Probability")
    axes[2, 0].set_ylabel(r"Relative Probability")

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.012),
        ncol=len(handles),
        frameon=False,
    )

    off = -0.02 if settings["EE_correct"].want_pixel_based else 0.08
    if is_gaussian:
        off = off - 0.08
    fig.patches.append(
        mpatches.Rectangle(
            (0.3 + off, 0.957),  # bottom-left corner (x0, y0)
            (0.5 - (0.3 + off)) * 2,  # width
            0.04,  # height
            linewidth=1,
            edgecolor="black",
            facecolor="none",
            transform=fig.transFigure,
            zorder=5,
            alpha=0.2,
        )
    )

    if axess is None:
        fsky = int(settings["BB_correct"].fsky)
        extra = (
            f"{settings['EE_correct'].offset_type}_offset_fsky{fsky}_"
            if want_offset
            else f"fsky{fsky}_"
        )
        if is_gaussian:
            extra = f"gaussian_fsky{fsky}_"
        plt.savefig(
            settings["EE_correct"].plots_folder
            + f"../{extra}QML_complete_joined_wrongfid_"
            + f"{settings['EE_correct'].N_chs}ch.png"
        )

        plt.show()


def get_settings_dict(config_path: str, num_chs: int) -> Dict[str, Settings]:
    yaml.add_constructor("!join", join)
    config_dir = os.path.dirname(os.path.abspath(config_path)) + "/configs/"

    name = f"{num_chs}ch/QML_EE_{num_chs}ch"
    names = [f"{name}_config.yaml"]

    name = f"{num_chs}ch/QML_BB_{num_chs}ch"
    names += [f"{name}_config.yaml"]

    name = f"{num_chs}ch/QML_BB_{num_chs}ch_notens"
    names += [f"{name}_config.yaml"]

    types = [
        "",
        "_mismatch",
        "_mismatch_2",
        "_wrongfid",
        "_wrongfid_2",
    ]

    configs_collection = [
        config_dir + f"{n.replace('_config.yaml', '')}{t}_config.yaml"
        for n in names
        for t in types
    ]

    settings: Dict[str, Settings] = {}
    for i, config in enumerate(configs_collection):
        n = (
            config.split("/")[-1]
            .replace("QML_", "")
            .replace("_config.yaml", "")
            .replace(f"{num_chs}ch_", "")
            .replace(f"_{num_chs}ch", "")
        )
        if i == 0:
            n = "EE_correct"
        elif i == 5:
            n = "BB_correct"
        elif i == 10:
            n = "BB_notens_correct"
        settings[n] = Settings(config, read_theory_spectra=False)

    return settings


def main():
    yaml.add_constructor("!join", join)

    parsed_args = parse_args()
    config_path = parsed_args.config_path
    field = parsed_args.field
    num_chs = parsed_args.num_chs

    is_gaussian = parsed_args.want_gaussian

    settings = get_settings_dict(config_path, num_chs)

    # ========== Launching scripts ==========
    msg = f"PLOTTING STUFF for {field} with {num_chs} CHANNELS!".center(40)
    msg = f"°`°º¤ø,,ø¤°º¤ø,,ø¤º°`° {msg} °`°º¤ø,,ø¤°º¤ø,,ø¤º°`°".center(100)
    print(f"\n{msg}\n")

    plot_mismatch(
        settings,
        settings["EE_correct"].want_offset,
        is_gaussian=is_gaussian,
    )

    plot_wrongfid(
        settings,
        want_legend=True,
        want_offset=settings["EE_correct"].want_offset,
        is_gaussian=is_gaussian,
    )


if __name__ == "__main__":
    main()
