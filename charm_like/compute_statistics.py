import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.interpolate import UnivariateSpline

from charm_like.settings_class import Settings
from utils.common_functions import configure_plt


def upscale_posterior(posterior, grid, fine_grid):
    interp_func = UnivariateSpline(grid, posterior, s=0)
    fine_posterior = interp_func(fine_grid)
    fine_posterior = np.maximum(fine_posterior, 0)
    fine_posterior /= np.max(fine_posterior)
    return fine_posterior


def get_posterior(name, chi2, params: Settings, N=10000):
    fine_grid = np.linspace(params.grid.min(), params.grid.max(), N)
    posterior = np.exp(-0.5 * (np.mean(chi2, axis=0) - np.min(np.mean(chi2, axis=0))))
    if "pixel" in name:
        pixel_grid = params.get_pixel_based_grid()
        posterior = upscale_posterior(posterior, pixel_grid, fine_grid)
    else:
        posterior = upscale_posterior(posterior, params.grid, fine_grid)
    return posterior


def compute_peak(posterior, grid):
    peak_idx = np.argmax(posterior)
    return [peak_idx, grid[peak_idx]]


def compute_hdi(posterior, grid, cred_mass=0.95):
    posterior = posterior / np.sum(posterior)
    sorted_indices = np.argsort(posterior)[::-1]
    sorted_posterior = posterior[sorted_indices]

    cumulative_sum = np.cumsum(sorted_posterior)
    cutoff_index = np.argmax(cumulative_sum >= cred_mass)

    hdi_indices = sorted_indices[: cutoff_index + 1]
    hdi_bounds = [np.min(grid[hdi_indices]), np.max(grid[hdi_indices])]

    return hdi_bounds[0], hdi_bounds[1]


def compute_second_moment_sigma(posterior, grid):
    second_moment = np.sum(grid**2 * posterior) / np.sum(posterior)
    first_moment = np.sum(grid * posterior) / np.sum(posterior)

    return np.sqrt(second_moment - first_moment**2)


def compute_first_moment(posterior, grid):
    first_moment = np.sum(grid * posterior) / np.sum(posterior)
    return first_moment


def collect_statistics(posterior, params: Settings, N=10000):
    fine_grid = np.linspace(params.grid.min(), params.grid.max(), N)

    peak = compute_peak(posterior, fine_grid)
    hdi_low, hdi_high = compute_hdi(posterior, fine_grid, cred_mass=0.68)
    hdi_low_95, hdi_high_95 = compute_hdi(posterior, fine_grid, cred_mass=0.95)
    second_moment = compute_second_moment_sigma(posterior, fine_grid)
    first_moment = compute_first_moment(posterior, fine_grid)

    return peak, hdi_low, hdi_high, hdi_low_95, hdi_high_95, second_moment, first_moment


def save_statistics(
    params: Settings,
    HL_chi2,
    mHL_chi2,
    cHL_chi2s,
    hybrid_HL_chi2,
    want_offset=False,
    extra_label="",
):
    names = ["HL", "mHL", "cHL"]
    labels = ["HL", "mHL", "cHL"]
    chi2s = [HL_chi2, mHL_chi2, cHL_chi2s]

    if params.custom_idxs is not None:
        names.append("mHL_hybrid")
        labels.append("mHL hybrid")
        chi2s.append(hybrid_HL_chi2)

    if params.want_pixel_based:
        names.insert(0, "pixel_based")
        labels.insert(0, "Pixel-based")
        chi2s.insert(0, params.get_pixel_based_chi2())

    extra = f"{params.offset_type}_offset_" if want_offset else ""
    out_file_path = (
        params.plots_folder
        + f"{extra_label}"
        + f"{extra}{params.extra_plots}statistics_report_{params.name}.txt"
    )
    msg = f"Statistics for {params.extra_chi2s}{params.name}\n\n"
    if params.print_statistics:
        print(msg)
    if params.save_statistics:
        with open(out_file_path, "w") as f:
            f.write(msg)

    for label, name, chi2 in zip(labels, names, chi2s):
        posterior = get_posterior(name, chi2, params)
        peak, hdi_low, hdi_high, hdi_low_95, hdi_high_95, second_moment, mean = (
            collect_statistics(posterior, params)
        )

        msg = f"{label}\n"
        msg += f"Peak: {peak[1]}\n"
        msg += f"68% HDI: {hdi_low} - {hdi_high}\n"
        msg += f"95% HDI: {hdi_low_95} - {hdi_high_95}\n"
        msg += f"Second moment: {second_moment}\n"
        msg += f"Mean: {mean}\n\n"

        if params.print_statistics:
            print(msg)
        if params.save_statistics:
            with open(out_file_path, "a") as f:
                f.write(msg)


def compute_stats(names, chi2s, params):
    peaks, hdi_low, hdi_high = [], [], []
    yerr, hdi_low_95, hdi_high_95 = [], [], []
    yerr_95, second_moment, means = [], [], []
    for name, chi2 in zip(names, chi2s):
        posterior = get_posterior(name, chi2, params)
        (
            peak,
            hdi_low_,
            hdi_high_,
            hdi_low_95_,
            hdi_high_95_,
            second_moment_,
            mean_,
        ) = collect_statistics(posterior, params)
        peaks.append(peak[1])
        hdi_low.append(hdi_low_)
        hdi_high.append(hdi_high_)
        yerr.append([peak[1] - hdi_low_, hdi_high_ - peak[1]])
        hdi_low_95.append(hdi_low_95_)
        hdi_high_95.append(hdi_high_95_)
        yerr_95.append([peak[1] - hdi_low_95_, hdi_high_95_ - peak[1]])
        second_moment.append(second_moment_)
        means.append(mean_)

    peaks = np.array(peaks)
    hdi_low = np.array(hdi_low)
    hdi_high = np.array(hdi_high)
    yerr = np.array(yerr).T
    hdi_low_95 = np.array(hdi_low_95)
    hdi_high_95 = np.array(hdi_high_95)
    yerr_95 = np.array(yerr_95).T
    second_moment = np.array(second_moment)
    means = np.array(means)

    return {
        "peaks": np.array(peaks),
        "hdi_low": np.array(hdi_low),
        "hdi_high": np.array(hdi_high),
        "yerr": np.array(yerr).T,
        "hdi_low_95": np.array(hdi_low_95),
        "hdi_high_95": np.array(hdi_high_95),
        "yerr_95": np.array(yerr_95).T,
        "second_moment": np.array(second_moment),
        "means": np.array(means),
    }


def plot_bar(
    params: Settings,
    ax,
    i,
    x,
    stats,
    color,
    bar_width=0.08,
    marker="X",
    marker_size=200,
    ls="-",
):
    ax.fill_between(
        [x - bar_width, x + bar_width],
        stats["hdi_low"][i],
        stats["hdi_high"][i],
        color=color,
        alpha=0.7,
        ls=ls,
    )
    ax.fill_between(
        [x - bar_width, x + bar_width],
        stats["hdi_low_95"][i],
        stats["hdi_high_95"][i],
        color=color,
        alpha=0.4,
        ls=ls,
    )
    ax.scatter(
        x,
        stats["peaks"][i],
        marker=marker,
        s=marker_size,
        edgecolor="black",
        linewidth=1.5,
        facecolor="white",
        ls=ls,
    )

    ylim_low, ylim_high = ax.get_ylim()
    thresh = (ylim_high - ylim_low) / 20
    if stats["peaks"][i] - ylim_low < thresh:
        valign = "bottom"
    else:
        valign = "center"
    ax.text(
        x + 0.2,
        stats["peaks"][i],
        s=f"{round(stats['peaks'][i], 4)}",
        horizontalalignment="left",
        verticalalignment=valign,
        fontsize=10,
    )

    valign = "center"
    ax.text(
        x + 0.2,
        stats["hdi_high_95"][i],
        s=f"{round(stats['hdi_high_95'][i], 4)}",
        horizontalalignment="left",
        verticalalignment="center",
        fontsize=10,
    )

    if "notens" not in params.name:
        ax.text(
            x + 0.2,
            stats["hdi_low_95"][i],
            s=f"{round(stats['hdi_low_95'][i], 4)}",
            horizontalalignment="left",
            verticalalignment=valign,
            fontsize=10,
        )


def kl_divergence(p, q, epsilon=1e-12):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)

    p /= p.sum()
    q /= q.sum()

    return np.sum(p * np.log(p / q))


def get_reliabilities(params: Settings, chi2s):
    reliabilities = []
    for i in range(len(chi2s)):
        ref_posterior = np.exp(
            -0.5 * (np.mean(chi2s[0], axis=0) - np.min(np.mean(chi2s[0], axis=0)))
        )
        method_posterior = np.exp(
            -0.5 * (np.mean(chi2s[i], axis=0) - np.min(np.mean(chi2s[i], axis=0)))
        )
        pixel_grid = params.get_pixel_based_grid()

        if method_posterior.shape != ref_posterior.shape:
            method_posterior = upscale_posterior(
                method_posterior, params.grid, pixel_grid
            )

        reliabilities.append(kl_divergence(ref_posterior, method_posterior))

    return np.array(reliabilities)


def plot_statistics(
    params: Settings,
    HL_chi2,
    mHL_chi2,
    cHL_chi2s,
    hybrid_HL_chi2,
    want_offset=False,
    in_ax: Axes = None,
    extra_label="",
):
    names = ["HL", "mHL", "cHL"]
    chi2s = [HL_chi2, mHL_chi2, cHL_chi2s]
    labels = ["HL", "mHL", "cHL"]
    colors = ["red", "dodgerblue", "forestgreen"]

    if params.custom_idxs is not None:
        names.append("mHL_hybrid")
        chi2s.append(hybrid_HL_chi2)
        labels.append("mHL hybrid")
        colors.append("darkmagenta")

    if params.want_pixel_based:
        names.insert(0, "pixel-based")
        chi2s.insert(0, params.get_pixel_based_chi2())
        labels.insert(0, "PB")
        colors.insert(0, "goldenrod")

    if params.want_single_field_HL:
        harm_like = f"{params.offset_type}_offset_SFHL"
        single_field_HL_chi2 = np.load(
            params.chi2_folder
            + f"{harm_like}_chi2_{params.extra_chi2s}{params.name}.npy"
        )
        names.append("Single-field")
        chi2s.append(single_field_HL_chi2)
        labels.append("Single-field")
        colors.append("maroon")

    main_stats = compute_stats(names, chi2s, params)

    configure_plt()

    x_positions = np.array(np.arange(len(names))) * 1.5

    reliability = get_reliabilities(params, chi2s)
    print(names)
    print(reliability)
    sorted_idx = np.argsort(reliability)

    labels = [labels[i] for i in sorted_idx]
    colors = [colors[i] for i in sorted_idx]
    main_stats = {k: v[sorted_idx] for k, v in main_stats.items()}
    chi2s = [chi2s[i] for i in sorted_idx]

    if in_ax is None:
        _, ax = plt.subplots()
    else:
        ax = in_ax

    for i, x in enumerate(x_positions):
        plot_bar(
            params,
            ax,
            i,
            x,
            main_stats,
            color=colors[i],
            bar_width=0.08,
            marker="X",
            marker_size=200,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(rf"{params.param_latex}")

    ax.hlines(
        params.fiducial_value,
        -1,
        10,
        linestyles="--",
        color="black",
        alpha=0.2,
        zorder=-10,
        lw=1,
    )

    y_min = min(main_stats["hdi_low_95"][i] for i in range(len(x_positions)))
    y_max = max(main_stats["hdi_high_95"][i] for i in range(len(x_positions)))
    ax.set_ylim(y_min - 0.06 * abs(y_min), y_max + 0.06 * abs(y_max))

    offset = 0.4
    ax.set_xlim(min(x_positions) - offset, max(x_positions) + offset)

    handles = [
        Patch(facecolor="dimgray", edgecolor="black", alpha=0.7, label="68% CI"),
        Patch(facecolor="dimgray", edgecolor="black", alpha=0.4, label="95% CI"),
        Line2D(
            [0],
            [0],
            linestyle="",
            marker="X",
            label="MAP",
            markersize=10,
            markeredgecolor="black",
            markeredgewidth=1.5,
            markerfacecolor="white",
        ),
    ]

    ax.legend(handles=handles, loc="upper center", ncol=5, frameon=True)
    xlim_low, xlim_high = ax.get_xlim()
    ax.set_xlim(xlim_low, xlim_high + 1)

    xlim_low, xlim_high = ax.get_xlim()
    half_point = (xlim_low + xlim_high) / 2
    ylim_low, ylim_high = ax.get_ybound()
    padding = 0.05 * (ylim_high - ylim_low)
    ax.text(
        x=half_point,
        y=ylim_low - padding,
        s="$\\longrightarrow$\nLess reliable",
        horizontalalignment="center",
        verticalalignment="top",
        fontsize=15,
        linespacing=1.5,
    )

    if in_ax is None:
        extra = f"{params.offset_type}_offset_" if want_offset else ""
        if params.save_plots:
            plt.savefig(
                params.plots_folder
                + f"{extra_label}"
                + f"{extra}{params.extra_plots}"
                + f"statistics_report_{params.name}.png"
            )

        if params.show_plots:
            plt.show()
