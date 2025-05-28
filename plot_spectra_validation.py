import os
import re
from typing import Dict, List

import healpy as hp
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib import gridspec
from matplotlib.axes import Axes

from charm_like.chi2_classes import compute_montecarlo_offset
from charm_like.settings_class import Settings
from utils.common_functions import configure_plt, join, parse_args


def get_quantities(settings: Settings):
    ell = settings.ell

    spectra = settings.get_all_spectra()

    settings._get_templates()
    settings._store_fiducial_spectrum()

    theo_spectra = settings.theo_spectra[:, :, settings.field_idx]

    fiducial = theo_spectra[settings.fid_idx]
    noise_spec = (settings.ch_noises[0] / 180 / 60 * np.pi) ** 2

    beam = hp.read_cl(settings.beams_folder + settings.beam_file)
    noise_spec /= beam[settings.field_idx, 2 : 32 + 1] ** 2

    return ell, spectra, fiducial, noise_spec


def spectra_comparison(cutsky_settings: Dict[str, Settings]):
    configure_plt()

    fig = plt.figure(figsize=(18, 5))
    gs_main = gridspec.GridSpec(1, 1)

    gs_grid = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs_main[0], hspace=0, wspace=0.12
    )

    grid_axes: List[Axes] = np.empty((1, 3), dtype=object)
    for i in range(1):
        for j in range(3):
            grid_axes[i, j] = fig.add_subplot(gs_grid[i, j])

    N_data = cutsky_settings["EE"].N_data

    for idx, case in enumerate(["EE", "BB", "BB_notens"]):
        ell, spectra, fiducial, noise_spec = get_quantities(cutsky_settings[case])

        std = spectra[-N_data:].std(axis=0)
        mean = spectra[-N_data:].mean(axis=0)
        off_auto = compute_montecarlo_offset(ell, spectra[-N_data:, 0])
        off_cross = compute_montecarlo_offset(ell, spectra[-N_data:, 1])

        grid_axes[0, idx].plot(ell, mean[0], color="red")
        grid_axes[0, idx].plot(ell, mean[1], color="dodgerblue")

        grid_axes[0, idx].plot(ell, off_auto, color="red", ls=":")
        grid_axes[0, idx].plot(ell, off_cross, color="dodgerblue", ls=":")

        grid_axes[0, idx].fill_between(
            ell,
            mean[0] - std[0],
            mean[0] + std[0],
            color="red",
            alpha=0.2,
            zorder=-20,
        )
        grid_axes[0, idx].fill_between(
            ell,
            mean[1] - std[1],
            mean[1] + std[1],
            color="dodgerblue",
            alpha=0.2,
            zorder=-20,
        )

        grid_axes[0, idx].plot(ell, fiducial, color="black", zorder=-10)
        grid_axes[0, idx].plot(
            ell, fiducial + noise_spec, color="black", zorder=-10, ls="--"
        )

        if idx == 1:
            grid_axes[0, idx].set_ylim(20e-7, None)
        elif idx == 2:
            grid_axes[0, idx].set_ylim(8e-7, None)

        grid_axes[0, idx].set_xlabel(r"$\ell$")
        grid_axes[0, idx].loglog()

        grid_axes[0, idx].set_xlim(2, 32)

        grid_axes[0, idx].legend(
            loc="upper center",
            title=f"{case.replace('_notens', '')}",
            title_fontsize=15,
        )

    for col in range(3):
        for row in range(1):
            if row == 0 and col == 0:
                grid_axes[row, col].set_ylabel(r"$C_\ell$")
            else:
                grid_axes[row, col].set_ylabel("")

    handles = [
        plt.Line2D([0], [0], color="red", ls="-", label=r"$\hat{C}_\ell^{\rm AxA}$"),
        plt.Line2D(
            [0], [0], color="dodgerblue", ls="-", label=r"$\hat{C}_\ell^{\rm AxB}$"
        ),
        mpatches.Patch(
            color="red",
            label=r"$\hat{C}_\ell^{\rm AxA} \pm \sigma_\ell^{\rm AxA}$",
            alpha=0.5,
        ),
        mpatches.Patch(
            color="dodgerblue",
            label=r"$\hat{C}_\ell^{\rm AxB} \pm \sigma_\ell^{\rm AxB}$",
            alpha=0.5,
        ),
        plt.Line2D([0], [0], color="red", ls=":", label=r"$O_\ell^{\rm AxA}$"),
        plt.Line2D([0], [0], color="dodgerblue", ls=":", label=r"$O_\ell^{\rm AxB}$"),
        plt.Line2D([0], [0], color="black", ls="-", label=r"$C_\ell^{\rm Fid.}$"),
        plt.Line2D(
            [0], [0], color="black", ls="--", label=r"$C_\ell^{\rm Fid.} + N_\ell$"
        ),
    ]

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.17),
        ncol=4,
        frameon=False,
    )

    off = -0.0
    fig.patches.append(
        mpatches.Rectangle(
            (0.3 + off, 0.98),
            (0.5 - (0.3 + off)) * 2,
            0.16,
            linewidth=1,
            edgecolor="black",
            facecolor="none",
            transform=fig.transFigure,
            zorder=5,
            alpha=0.2,
        )
    )

    params = cutsky_settings["EE"]
    if params.save_plots:
        plt.savefig(
            params.plots_folder
            + f"../{params.extra_plots}"
            + f"spectra_mosaic_{params.N_chs}ch.png"
        )

    if params.show_plots:
        plt.show()


if __name__ == "__main__":
    yaml.add_constructor("!join", join)

    parsed_args = parse_args()
    config_path = parsed_args.config_path
    field = parsed_args.field
    num_chs = parsed_args.num_chs

    cutsky_settings = {}

    name = f"{num_chs}ch/QML_EE_{num_chs}ch"
    names = [f"{name}_config.yaml"]

    name = f"{num_chs}ch/QML_BB_{num_chs}ch"
    names += [f"{name}_config.yaml"]

    name = f"{num_chs}ch/QML_BB_{num_chs}ch_notens"
    names += [f"{name}_config.yaml"]

    config_dir = os.path.dirname(os.path.abspath(config_path)) + "/configs/"
    configs_collection = [config_dir + n for n in names]

    keys = ["EE", "BB", "BB_notens"]

    settings: Dict[str, Settings] = {}
    for i, config in enumerate(configs_collection):
        n = (
            config.split("/")[-1]
            .replace("QML_", "")
            .replace("_config.yaml", "")
            .replace(f"{num_chs}ch_", "")
            .replace(f"_{num_chs}ch", "")
        )

        config_dict = yaml.load(open(config), Loader=yaml.FullLoader)
        fsky = 40
        config_dict["fsky"] = fsky
        for key, value in config_dict.items():
            if isinstance(value, str):
                config_dict[key] = re.sub(r"fsky\d+", f"fsky{fsky}", value)
            elif isinstance(value, list):
                if isinstance(value[0], str):
                    config_dict[key] = [
                        re.sub(r"fsky\d+", f"fsky{fsky}", v) for v in value
                    ]
        cutsky_settings[keys[i]] = Settings.from_dict(
            config_dict, read_theory_spectra=False
        )

    configure_plt()

    spectra_comparison(cutsky_settings)
