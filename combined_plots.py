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
    color,
    ax=None,
):
    ax.plot(
        params.grid,
        np.exp(
            -0.5
            * (np.mean(default_chi2s, axis=0) - np.min(np.mean(default_chi2s, axis=0)))
        ),
        label=label,
        color=color,
        ls="-",
    )

    ax.plot(
        params.grid,
        np.exp(
            -0.5
            * (np.mean(nodef1_chi2s, axis=0) - np.min(np.mean(nodef1_chi2s, axis=0)))
        ),
        color=color,
        ls="--",
    )

    ax.plot(
        params.grid,
        np.exp(
            -0.5
            * (np.mean(nodef2_chi2s, axis=0) - np.min(np.mean(nodef2_chi2s, axis=0)))
        ),
        color=color,
        ls=":",
    )


def produce_mismatch_plot(
    correct: Settings,
    mismatch: Settings,
    mismatch_2: Settings,
    loc="upper left",
    set_xlim=False,
):
    configure_plt()

    fig, axes = plt.subplots(3, 1, figsize=(7, 12), sharex=True)
    fig.subplots_adjust(hspace=0)

    extra_handles = [
        plt.Line2D(
            [0],
            [0],
            color="black",
            ls="-",
            label=r"$\widehat{N}_\ell = \widebar{N}_\ell$",
        ),
        plt.Line2D(
            [0],
            [0],
            color="black",
            ls="--",
            label=r"$\widehat{N}_\ell = 0.8^2\ \widebar{N}_\ell$",
        ),
        plt.Line2D(
            [0],
            [0],
            color="black",
            ls=":",
            label=r"$\widehat{N}_\ell = 1.2^2\ \widebar{N}_\ell$",
        ),
    ]

    # HL chi2s
    file = correct.chi2_folder + f"HL_chi2_{correct.name}.npy"
    correct_HL_chi2 = np.load(file)

    file = mismatch.chi2_folder + f"HL_chi2_{mismatch.name}.npy"
    mismatch_HL_chi2 = np.load(file)

    file = mismatch_2.chi2_folder + f"HL_chi2_{mismatch_2.name}.npy"
    mismatch_2_HL_chi2 = np.load(file)

    combined_HL_posterior_plot(
        correct,
        correct_HL_chi2,
        mismatch_HL_chi2,
        mismatch_2_HL_chi2,
        "HL",
        "red",
        ax=axes[0],
    )

    # mHL chi2s
    file = correct.chi2_folder + f"mHL_chi2_{correct.name}.npy"
    correct_mHL_chi2 = np.load(file)

    file = mismatch.chi2_folder + f"mHL_chi2_{mismatch.name}.npy"
    mismatch_mHL_chi2 = np.load(file)

    file = mismatch_2.chi2_folder + f"mHL_chi2_{mismatch_2.name}.npy"
    mismatch_2_mHL_chi2 = np.load(file)

    combined_HL_posterior_plot(
        correct,
        correct_mHL_chi2,
        mismatch_mHL_chi2,
        mismatch_2_mHL_chi2,
        "mHL",
        "dodgerblue",
        ax=axes[1],
    )

    # cHL chi2s
    file = correct.chi2_folder + f"cross_chi2_{correct.name}.npy"
    correct_cHL_chi2 = np.load(file)

    file = mismatch.chi2_folder + f"cross_chi2_{mismatch.name}.npy"
    mismatch_cHL_chi2 = np.load(file)

    file = mismatch_2.chi2_folder + f"cross_chi2_{mismatch_2.name}.npy"
    mismatch_2_cHL_chi2 = np.load(file)

    combined_HL_posterior_plot(
        correct,
        correct_cHL_chi2,
        mismatch_cHL_chi2,
        mismatch_2_cHL_chi2,
        "cHL",
        "forestgreen",
        ax=axes[2],
    )

    grid = correct.grid
    N = (correct.grid_steps - 1) // 100 + 1
    ticks = np.array([round(grid[i * 100], 4) for i in range(N)])
    for ax in axes:
        ax.vlines(grid[correct.fid_idx], 0, 1.1, color="black", ls="--")
        ax.set_ylabel(r"Relative Probability")
        handles, labels = ax.get_legend_handles_labels()
        handles = [mpatches.Patch(color=handles[0].get_color(), label=labels[0])]
        handles += extra_handles
        ax.legend(handles=handles, loc=loc)
        ax.set_ylim(0, 1.1)
        ax.set_xticks(
            ticks,
            labels=[ticks[i] for i in range(len(ticks))],
        )
        if "EE" in correct.name:
            N = (correct.grid_steps - 1) // 160 + 1
            ticks = np.array([round(grid[i * 160], 4) for i in range(N)])
            ax.set_xticks(
                ticks,
                labels=[ticks[i] for i in range(len(ticks))],
            )
            ax.set_xlim(0.04 - 0.0005, 0.08 + 0.0005)
        if set_xlim:
            ax.set_xlim(-0.00005, 0.002 + 0.00005)
            ax.set_xticks(
                ticks / 10,
                labels=[ticks[i] for i in range(len(ticks))],
            )

    axes[-1].set_xlabel(correct.param_latex)
    if set_xlim:
        axes[-1].set_xlabel(r"$10 \times$" + correct.param_latex)

    extra = (
        f"{correct.offset_type}_offset_fsky{correct.fsky}"
        if correct.want_offset
        else f"fsky{correct.fsky}_"
    )
    if correct.save_plots:
        plt.savefig(
            correct.plots_folder
            + f"{extra}QML_complete_{correct.extra}mismatch_"
            + f"comparison_{correct.name}.png"
        )

    if correct.show_plots:
        plt.show()


def produce_wrongfid_plot(
    correct: Settings,
    wrongfid: Settings,
    wrongfid_2: Settings,
    loc="upper left",
    set_xlim=False,
):
    configure_plt()

    _, axes = plt.subplots(
        3, 1, figsize=(7, 12), sharex=True, gridspec_kw={"hspace": 0.0}
    )

    extra_handles = [
        plt.Line2D([0], [0], color="black", ls="-", label=f"True {correct.param}"),
        plt.Line2D(
            [0],
            [0],
            color="black",
            ls="--",
            label=f"{correct.param} = {wrongfid.wrong_fiducial_value}",
        ),
        plt.Line2D(
            [0],
            [0],
            color="black",
            ls=":",
            label=f"{correct.param} = {wrongfid_2.wrong_fiducial_value}",
        ),
    ]

    # HL chi2s
    file = correct.chi2_folder + f"HL_chi2_{correct.name}.npy"
    correct_HL_chi2 = np.load(file)

    file = wrongfid.chi2_folder + f"HL_chi2_{wrongfid.name}.npy"
    wrongfid_HL_chi2 = np.load(file)

    file = wrongfid_2.chi2_folder + f"HL_chi2_{wrongfid_2.name}.npy"
    wrongfid_2_HL_chi2 = np.load(file)

    combined_HL_posterior_plot(
        correct,
        correct_HL_chi2,
        wrongfid_HL_chi2,
        wrongfid_2_HL_chi2,
        "HL",
        "red",
        ax=axes[0],
    )

    # mHL chi2s
    file = correct.chi2_folder + f"mHL_chi2_{correct.name}.npy"
    correct_mHL_chi2 = np.load(file)

    file = wrongfid.chi2_folder + f"mHL_chi2_{wrongfid.name}.npy"
    wrongfid_mHL_chi2 = np.load(file)

    file = wrongfid_2.chi2_folder + f"mHL_chi2_{wrongfid_2.name}.npy"
    wrongfid_2_mHL_chi2 = np.load(file)

    combined_HL_posterior_plot(
        correct,
        correct_mHL_chi2,
        wrongfid_mHL_chi2,
        wrongfid_2_mHL_chi2,
        "mHL",
        "dodgerblue",
        ax=axes[1],
    )

    # cHL chi2s
    file = correct.chi2_folder + f"cross_chi2_{correct.name}.npy"
    correct_cHL_chi2 = np.load(file)

    file = wrongfid.chi2_folder + f"cross_chi2_{wrongfid.name}.npy"
    wrongfid_cHL_chi2 = np.load(file)

    file = wrongfid_2.chi2_folder + f"cross_chi2_{wrongfid_2.name}.npy"
    wrongfid_2_cHL_chi2 = np.load(file)

    combined_HL_posterior_plot(
        correct,
        correct_cHL_chi2,
        wrongfid_cHL_chi2,
        wrongfid_2_cHL_chi2,
        "cHL",
        "forestgreen",
        ax=axes[2],
    )

    grid = correct.grid
    N = (correct.grid_steps - 1) // 100 + 1
    ticks = np.array([round(grid[i * 100], 4) for i in range(N)])
    for ax in axes:
        ax.vlines(correct.grid[correct.fid_idx], 0, 1.1, color="black", ls="--")
        ax.set_ylabel(r"Relative Probability")
        handles, labels = ax.get_legend_handles_labels()
        handles = [mpatches.Patch(color=handles[0].get_color(), label=labels[0])]
        handles += extra_handles
        ax.legend(handles=handles, loc=loc)
        ax.set_ylim(0, 1.1)
        ax.set_xticks(
            ticks,
            labels=[ticks[i] for i in range(len(ticks))],
        )
        if "EE" in correct.name:
            N = (correct.grid_steps - 1) // 160 + 1
            ticks = np.array([round(grid[i * 160], 4) for i in range(N)])
            ax.set_xticks(
                ticks,
                labels=[ticks[i] for i in range(len(ticks))],
            )
            ax.set_xlim(0.04 - 0.0005, 0.08 + 0.0005)
        if set_xlim:
            ax.set_xlim(-0.00005, 0.002 + 0.00005)
            ax.set_xticks(
                ticks / 10,
                labels=[ticks[i] for i in range(len(ticks))],
            )

    axes[-1].set_xlabel(correct.param_latex)
    if set_xlim:
        axes[-1].set_xlabel(r"$10 \times$" + correct.param_latex)

    extra = (
        f"{correct.offset_type}_offset_fsky{correct.fsky}"
        if correct.want_offset
        else f"fsky{correct.fsky}_"
    )
    if correct.save_plots:
        plt.savefig(
            correct.plots_folder
            + f"{extra}QML_complete_{correct.extra}wrongfid_"
            + f"comparison_{correct.name}.png"
        )

    if correct.show_plots:
        plt.show()


def main():
    yaml.add_constructor("!join", join)

    parsed_args = parse_args()
    config_path = parsed_args.config_path
    field = parsed_args.field
    num_chs = parsed_args.num_chs
    want_notens = parsed_args.want_notens
    config_dir = os.path.dirname(os.path.abspath(config_path)) + "/configs/"

    # ========== Launching scripts ==========
    msg = f"PLOTTING STUFF for {field} with {num_chs} CHANNELS!".center(40)
    msg = f"°`°º¤ø,,ø¤°º¤ø,,ø¤º°`° {msg} °`°º¤ø,,ø¤°º¤ø,,ø¤º°`°".center(100)
    print(f"\n{msg}\n")

    name = f"{num_chs}ch/QML_{field}_{num_chs}ch"

    notens_name = f"{name}_notens"
    types = [
        "",
        "_mismatch",
        "_mismatch_2",
        "_wrongfid",
        "_wrongfid_2",
    ]

    names = [f"{name}{t}_config.yaml" for t in types]
    if want_notens and field == "BB":
        names += [f"{notens_name}{t}_config.yaml" for t in types]

    configs_collection = [config_dir + n for n in names]

    settings: Dict[str, Settings] = {}
    for i, config in enumerate(configs_collection):
        n = (
            config.split("/")[-1]
            .replace("QML_", "")
            .replace("_config.yaml", "")
            .replace(f"{field}_{num_chs}ch_", "")
        )
        if i == 0:
            n = "correct"
        settings[n] = Settings(config, read_theory_spectra=False)

    produce_mismatch_plot(
        settings["correct"], settings["mismatch"], settings["mismatch_2"]
    )
    produce_wrongfid_plot(
        settings["correct"], settings["wrongfid"], settings["wrongfid_2"]
    )

    if want_notens and field == "BB":
        produce_mismatch_plot(
            settings["notens"],
            settings["notens_mismatch"],
            settings["notens_mismatch_2"],
            loc="upper right",
            set_xlim=True,
        )
        produce_wrongfid_plot(
            settings["notens"],
            settings["notens_wrongfid"],
            settings["notens_wrongfid_2"],
            loc="upper right",
            set_xlim=True,
        )


if __name__ == "__main__":
    main()
