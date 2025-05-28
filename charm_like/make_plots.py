import matplotlib.pyplot as plt
import numpy as np

from charm_like.compute_statistics import plot_statistics, save_statistics
from charm_like.settings_class import Settings
from utils.common_functions import chs2idx, configure_plt, idx2chs


def make_spectra_plot(params: Settings, all_spectra):
    plt.figure()
    ch1 = 0
    ch2 = 0
    idx = chs2idx(ch1, ch2, params.N_chs)
    plt.plot(
        params.ell,
        np.mean(all_spectra[-params.N_data :, idx], axis=0),
        label=f"Data {params.ch_names[ch1]}x{params.ch_names[ch2]}",
        color="red",
    )
    plt.plot(
        params.ell,
        np.mean(all_spectra[: -params.N_data, idx], axis=0),
        label=f"Sims {params.ch_names[ch1]}x{params.ch_names[ch2]}",
        color="red",
        ls="--",
    )
    ch2 = 1
    idx = chs2idx(ch1, ch2, params.N_chs)
    plt.plot(
        params.ell,
        np.mean(all_spectra[-params.N_data :, idx], axis=0),
        label=f"Data {params.ch_names[ch1]}x{params.ch_names[ch2]}",
        color="dodgerblue",
    )
    plt.plot(
        params.ell,
        np.mean(all_spectra[: -params.N_data, idx], axis=0),
        label=f"Sims {params.ch_names[ch1]}x{params.ch_names[ch2]}",
        color="dodgerblue",
        ls="--",
    )

    plt.plot(
        params.ell,
        params.theo_spectra[params.fid_idx][:, params.field_idx],
        label=r"$C_\ell^{fid}$",
        ls="-",
        color="navy",
    )

    all_noises = params.get_all_noises_spectra()
    plt.plot(
        params.ell,
        params.theo_spectra[params.fid_idx][:, params.field_idx]
        + np.mean(all_noises[-params.N_data :], axis=0)[0],
        label=r"$C_\ell^{fid} + N_\ell / b_\ell^2$",
        ls="-",
        color="maroon",
    )

    if params.want_single_field_HL:
        spectra = params.get_single_field_combination_spectra()
        plt.plot(
            params.ell,
            np.mean(spectra[-params.N_data :], axis=0),
            label=r"$C_\ell^{comb.}$",
            ls="-",
            color="forestgreen",
        )

    if params.wrong_fiducial_value is not None:
        plt.plot(
            params.ell,
            params.theo_spectra[params.wrong_fid_idx][:, params.field_idx],
            label=r"Wrong $C_\ell^{fid}$",
            ls="--",
            color="navy",
        )

        plt.plot(
            params.ell,
            params.theo_spectra[params.wrong_fid_idx][:, params.field_idx]
            + np.mean(all_noises[: -params.N_data], axis=0)[0],
            label=r"Wrong $C_\ell^{fid} + N_\ell / b_\ell^2$",
            ls="--",
            color="maroon",
        )

    plt.legend()
    plt.loglog()

    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$C_{\ell}\ [\mu \text{K}^2]$")

    if params.save_plots:
        plt.savefig(
            params.plots_folder + f"{params.extra_plots}spectra_{params.name}.png"
        )
    if params.show_plots:
        plt.show()


def make_spectra_validation_plot(params: Settings, all_spectra):
    fiducial = params.get_empirical_fiducial()

    colors = [
        "maroon",
        "navy",
        "dodgerblue",
        "darkorange",
        "deepskyblue",
        "red",
    ]

    plt.figure()
    for ch1 in range(params.N_chs):
        for ch2 in range(ch1, params.N_chs):
            idx = chs2idx(ch1, ch2, params.N_chs)
            plt.plot(
                params.ell,
                (np.mean(all_spectra[-params.N_data :, idx], axis=0) - fiducial[idx])
                / np.std(all_spectra[-params.N_data :, idx], axis=0)
                * np.sqrt(params.N_data),
                label=f"{params.ch_names[ch1]}x{params.ch_names[ch2]}",
                color=colors[idx],
            )

    plt.plot(params.ell, np.zeros_like(fiducial[0]), ls="--", color="k")

    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$\Delta C_{\ell}/\sigma_\ell\ [\mu \text{K}^2]$")

    plt.legend()
    plt.semilogx()

    fiducial = params.fiducial_spectrum[: params.N_ell, params.field_idx]
    plt.figure()
    for ch1 in range(params.N_chs):
        for ch2 in range(ch1, params.N_chs):
            idx = chs2idx(ch1, ch2, params.N_chs)
            _fiducial = fiducial.copy()
            if ch1 == ch2:
                _fiducial += params.noise_spectra[ch1]
            plt.plot(
                params.ell,
                (np.mean(all_spectra[-params.N_data :, idx], axis=0) - _fiducial)
                / np.std(all_spectra[-params.N_data :, idx], axis=0)
                * np.sqrt(params.N_data),
                label=f"{params.ch_names[ch1]}x{params.ch_names[ch2]}",
                color=colors[idx],
            )

    plt.plot(params.ell, np.zeros_like(fiducial), ls="--", color="k")

    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$\Delta C_{\ell}/\sigma_\ell\ [\mu \text{K}^2]$")

    plt.legend()
    plt.semilogx()

    if params.save_plots:
        plt.savefig(
            params.plots_folder
            + f"{params.extra_plots}validation_spectra_{params.name}.png"
        )
    if params.show_plots:
        plt.show()


def make_single_field_posterior_plot(
    params: Settings,
    single_HL_chi2s,
    cHL_chi2s,
    single_field_HL_chi2,
    extra_label="",
):
    plt.figure()
    auto_label = "HL"
    cross_label = "cHL"

    labels = []
    combined_label = ""
    ch_names = params.ch_names
    for i in range(params.N_auto + params.N_cross):
        ch1, ch2 = idx2chs(i, params.N_chs)
        if i in params.auto_idxs:
            label = auto_label
        else:
            label = cross_label
            if combined_label == "":
                combined_label = ch_names[ch1] + "x" + ch_names[ch2]
            else:
                combined_label += " + " + ch_names[ch1] + "x" + ch_names[ch2]

        labels.append(label)

        plt.plot(
            params.grid,
            np.exp(
                -0.5
                * (
                    np.mean(single_HL_chi2s[i], axis=0)
                    - np.min(np.mean(single_HL_chi2s[i], axis=0))
                )
            ),
            label=f"{labels[i]} {params.ch_names[ch1]}x{params.ch_names[ch2]}",
        )

    plt.plot(
        params.grid,
        np.exp(
            -0.5 * (np.mean(cHL_chi2s, axis=0) - np.min(np.mean(cHL_chi2s, axis=0)))
        ),
        label=f"cHL {combined_label}",
    )

    if params.want_single_field_HL:
        plt.plot(
            params.grid,
            np.exp(
                -0.5
                * (
                    np.mean(single_field_HL_chi2, axis=0)
                    - np.min(np.mean(single_field_HL_chi2, axis=0))
                )
            ),
            label=r"$C^{\text{cross}}_\ell$-level combination",
        )

    plt.axvline(params.fiducial_value, color="k", ls="--", label="Fiducial")

    plt.xlabel(params.param_latex)
    plt.ylabel("Relative Probability")

    plt.ylim(0, None)
    plt.xlim(params.grid[0], params.grid[-1])

    if params.fiducial_value == 0:
        plt.xlim(-0.00005, 0.002)

    plt.legend()

    if params.save_plots:
        plt.savefig(
            params.plots_folder
            + f"{extra_label}"
            + f"{params.offset_type}_offset_{params.extra_plots}"
            + f"HL_vs_CHL_comparison_{params.name}.png"
        )

    if params.show_plots:
        plt.show()


def make_HL_posterior_plot(
    params: Settings,
    HL_chi2,
    mHL_chi2,
    single_field_HL_chi2,
    cHL_chi2s,
    hybrid_HL_chi2,
    oMHL=False,
    extra_label="",
):
    plt.figure()
    plt.plot(
        params.grid,
        np.exp(-0.5 * (np.mean(HL_chi2, axis=0) - np.min(np.mean(HL_chi2, axis=0)))),
        label="HL",
        color="red",
    )

    plt.plot(
        params.grid,
        np.exp(-0.5 * (np.mean(mHL_chi2, axis=0) - np.min(np.mean(mHL_chi2, axis=0)))),
        label="mHL",
        color="dodgerblue",
        ls="-",
    )

    if params.want_single_field_HL:
        plt.plot(
            params.grid,
            np.exp(
                -0.5
                * (
                    np.mean(single_field_HL_chi2, axis=0)
                    - np.min(np.mean(single_field_HL_chi2, axis=0))
                )
            ),
            label="HL from maps comb.",
            color="maroon",
            ls="--",
        )

    plt.plot(
        params.grid,
        np.exp(
            -0.5 * (np.mean(cHL_chi2s, axis=0) - np.min(np.mean(cHL_chi2s, axis=0)))
        ),
        label="cHL",
        color="forestgreen",
    )

    if hybrid_HL_chi2 is not None:
        plt.plot(
            params.grid,
            np.exp(
                -0.5
                * (
                    np.mean(hybrid_HL_chi2, axis=0)
                    - np.min(np.mean(hybrid_HL_chi2, axis=0))
                )
            ),
            label="mHL hybrid",
            color="darkmagenta",
            ls="-",
        )

    if params.want_pixel_based:
        pixel_chi2 = params.get_pixel_based_chi2()
        grid = params.get_pixel_based_grid()
        plt.plot(
            grid,
            np.exp(
                -0.5
                * (np.mean(pixel_chi2, axis=0) - np.min(np.mean(pixel_chi2, axis=0)))
            ),
            label="Pixel-based",
            color="goldenrod",
            ls="-",
        )

    plt.vlines(params.grid[params.fid_idx], 0, 1, color="black", ls="--")

    plt.xlabel(params.param_latex)
    plt.ylabel(r"Relative Probability")

    plt.legend()

    plt.ylim(0, None)

    if params.fiducial_value == 0.01:
        plt.xlim(0, 0.02)
    elif params.fiducial_value == 0.06:
        plt.xlim(0.03, 0.09)
    elif params.fiducial_value == 0.0:
        plt.xlim(0.0, 0.002)

    labeloMHL = "oMHL" if oMHL else "MHL"
    if params.save_plots:
        plt.savefig(
            params.plots_folder
            + f"{extra_label}"
            + f"{params.offset_type}_offset_{params.extra_plots}"
            + f"CHL_vs_{labeloMHL}_comparison_{params.name}.png"
        )

    if params.show_plots:
        plt.show()


def compare_stacked_HL_plot(
    params: Settings,
    cHL_chi2s,
    mHL_chi2,
    single_field_HL_chi2,
    pixel_chi2,
    oHL=False,
    extra_label="",
):
    plt.figure()
    plt.plot(
        params.grid,
        np.exp(
            -0.5 * (np.mean(cHL_chi2s, axis=0) - np.min(np.mean(cHL_chi2s, axis=0)))
        ),
        label="Stacking cross-spectra",
        color="red",
    )

    plt.plot(
        params.grid,
        np.exp(-0.5 * (np.mean(mHL_chi2, axis=0) - np.min(np.mean(mHL_chi2, axis=0)))),
        label="mHL",
        color="dodgerblue",
    )

    if params.want_single_field_HL:
        plt.plot(
            params.grid,
            np.exp(
                -0.5
                * (
                    np.mean(single_field_HL_chi2, axis=0)
                    - np.min(np.mean(single_field_HL_chi2, axis=0))
                )
            ),
            label=r"$C^{\text{cross}}_\ell$-level combination",
            color="forestgreen",
        )

    if params.want_pixel_based:
        pixel_chi2 = params.get_pixel_based_chi2()
        grid = params.get_pixel_based_grid()
        plt.plot(
            grid,
            np.exp(
                -0.5
                * (np.mean(pixel_chi2, axis=0) - np.min(np.mean(pixel_chi2, axis=0)))
            ),
            label="Pixel-based",
            color="goldenrod",
            ls="-",
        )

    plt.vlines(params.grid[params.fid_idx], 0, 1, color="black", ls="--")

    plt.xlabel(params.param_latex)
    plt.ylabel(r"Relative Probability")

    plt.legend()

    plt.ylim(0, None)

    if params.fiducial_value == 0.01:
        plt.xlim(0, 0.02)
    elif params.fiducial_value == 0.06:
        plt.xlim(0.03, 0.09)
    elif params.fiducial_value == 0.0:
        plt.xlim(0.0, 0.002)

    label = "oHL" if oHL else "HL"
    if params.save_plots:
        plt.savefig(
            params.plots_folder
            + f"{extra_label}"
            + f"{params.offset_type}_offset_{params.extra_plots}"
            + f"{label}_vs_stack_{params.name}.png"
        )

    if params.show_plots:
        plt.show()


def produce_plots(params: Settings):
    configure_plt()

    all_spectra = params.get_all_spectra()

    make_spectra_plot(params, all_spectra)
    make_spectra_validation_plot(params, all_spectra)

    single_field_chi2s = np.load(
        params.chi2_folder
        + f"{params.offset_type}_offset_single_field_chi2s_"
        + f"{params.extra_chi2s}{params.name}.npy"
    )
    cHL_chi2s = np.load(
        params.chi2_folder
        + f"{params.offset_type}_offset_cross_chi2_"
        + f"{params.extra_chi2s}{params.name}.npy"
    )

    single_field_HL_chi2 = None
    pixel_chi2 = None
    if params.want_single_field_HL:
        single_field_HL_chi2 = np.load(
            params.chi2_folder + f"SFHL_chi2_{params.extra_chi2s}{params.name}.npy"
        )

    make_single_field_posterior_plot(
        params, single_field_chi2s, cHL_chi2s, single_field_HL_chi2
    )

    HL_chi2 = np.load(
        params.chi2_folder + f"HL_chi2_{params.extra_chi2s}{params.name}.npy"
    )
    mHL_chi2 = np.load(
        params.chi2_folder + f"mHL_chi2_{params.extra_chi2s}{params.name}.npy"
    )

    hybrid_HL_chi2 = None
    if params.custom_idxs is not None:
        hybrid_HL_chi2 = np.load(
            params.chi2_folder
            + f"mHL_hybrid_chi2_{params.extra_chi2s}{params.name}.npy"
        )

    make_HL_posterior_plot(
        params,
        HL_chi2,
        mHL_chi2,
        single_field_HL_chi2,
        cHL_chi2s,
        hybrid_HL_chi2,
    )

    compare_stacked_HL_plot(
        params, cHL_chi2s, mHL_chi2, single_field_HL_chi2, pixel_chi2
    )

    plot_statistics(
        params,
        HL_chi2,
        mHL_chi2,
        cHL_chi2s,
        hybrid_HL_chi2,
    )

    if params.print_statistics or params.save_statistics:
        save_statistics(
            params,
            HL_chi2,
            mHL_chi2,
            cHL_chi2s,
            hybrid_HL_chi2,
        )

    if params.want_offset:
        HL_chi2 = np.load(
            params.chi2_folder
            + f"{params.offset_type}_offset_HL_chi2_"
            + f"{params.extra_chi2s}{params.name}.npy"
        )
        mHL_chi2 = np.load(
            params.chi2_folder
            + f"{params.offset_type}_offset_mHL_chi2_"
            + f"{params.extra_chi2s}{params.name}.npy"
        )

        hybrid_HL_chi2 = None
        if params.custom_idxs is not None:
            hybrid_HL_chi2 = np.load(
                params.chi2_folder
                + f"{params.offset_type}_offset_mHL_hybrid_chi2_"
                + f"{params.extra_chi2s}{params.name}.npy"
            )

        single_field_HL_chi2 = None
        pixel_chi2 = None
        if params.want_single_field_HL:
            single_field_HL_chi2 = np.load(
                params.chi2_folder
                + f"{params.offset_type}_offset_SFHL_chi2_"
                + f"{params.extra_chi2s}{params.name}.npy"
            )

        make_HL_posterior_plot(
            params,
            HL_chi2,
            mHL_chi2,
            single_field_HL_chi2,
            cHL_chi2s,
            hybrid_HL_chi2,
            oMHL=params.want_offset,
        )

        compare_stacked_HL_plot(
            params,
            cHL_chi2s,
            mHL_chi2,
            single_field_HL_chi2,
            pixel_chi2,
            oHL=params.want_offset,
        )

        plot_statistics(
            params,
            HL_chi2,
            mHL_chi2,
            cHL_chi2s,
            hybrid_HL_chi2,
            want_offset=params.want_offset,
        )

        if params.print_statistics or params.save_statistics:
            save_statistics(
                params,
                HL_chi2,
                mHL_chi2,
                cHL_chi2s,
                hybrid_HL_chi2,
                want_offset=params.want_offset,
            )


def produce_gaussian_plots(params: Settings):
    configure_plt()

    single_field_chi2s = np.load(
        params.chi2_folder
        + "gaussian_single_field_chi2s_"
        + f"{params.extra_chi2s}{params.name}.npy"
    )
    cHL_chi2s = np.load(
        params.chi2_folder
        + "gaussian_cross_chi2_"
        + f"{params.extra_chi2s}{params.name}.npy"
    )

    single_field_HL_chi2 = None
    pixel_chi2 = None
    make_single_field_posterior_plot(
        params,
        single_field_chi2s,
        cHL_chi2s,
        single_field_HL_chi2,
        extra_label="gaussian_",
    )

    HL_chi2 = np.load(
        params.chi2_folder + f"full_gaussian_chi2_{params.extra_chi2s}{params.name}.npy"
    )
    mHL_chi2 = np.load(
        params.chi2_folder
        + f"gaussian_without_auto_chi2_{params.extra_chi2s}{params.name}.npy"
    )

    hybrid_HL_chi2 = None
    make_HL_posterior_plot(
        params,
        HL_chi2,
        mHL_chi2,
        single_field_HL_chi2,
        cHL_chi2s,
        hybrid_HL_chi2,
        extra_label="gaussian_",
    )

    compare_stacked_HL_plot(
        params,
        cHL_chi2s,
        mHL_chi2,
        single_field_HL_chi2,
        pixel_chi2,
        extra_label="gaussian_",
    )

    plot_statistics(
        params,
        HL_chi2,
        mHL_chi2,
        cHL_chi2s,
        hybrid_HL_chi2,
        extra_label="gaussian_",
    )

    if params.print_statistics or params.save_statistics:
        save_statistics(
            params,
            HL_chi2,
            mHL_chi2,
            cHL_chi2s,
            hybrid_HL_chi2,
        )
