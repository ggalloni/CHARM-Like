import argparse
import os
import subprocess
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import yaml


def join(loader: yaml.Loader, node: yaml.Node) -> str:
    seq = loader.construct_sequence(node)
    return "".join([str(i) for i in seq])


def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse argurments and return parser object"
    )

    parser.add_argument(
        "-C",
        "--config_path",
        type=str,
        default="/home/ggalloni/Projects/GitHub/CHARM_like/default_config.yaml",
        help="Path to the configuration file. If not provided, takes the default path.",
    )

    parser.add_argument(
        "-N",
        "--num_chs",
        type=int,
        default=3,
        help="Number of channels to use.",
    )

    parser.add_argument(
        "-F",
        "--field",
        type=str,
        default="BB",
        help="Requested field to analyse (BB or EE).",
    )

    parser.add_argument(
        "-T",
        "--want_notens",
        type=boolean_string,
        default=True,
        help=r"Want to compute $r=0$ case (work only if field is BB).",
    )

    parser.add_argument(
        "-G",
        "--want_gaussian",
        type=boolean_string,
        default=False,
        help=r"Want to compute Gaussian case.",
    )

    return parser.parse_args()


def create_all_folders(config):
    # Primary folders
    plots_folder = config["plots_folder"]
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder, exist_ok=True)
    chi2_folder = config["chi2_folder"]
    if not os.path.exists(chi2_folder):
        os.makedirs(chi2_folder, exist_ok=True)

    # Datasets subfolder
    nameproducts = config["name"]
    chi2_res_dir = chi2_folder + nameproducts + "/"
    if not os.path.exists(chi2_res_dir):
        os.makedirs(chi2_res_dir, exist_ok=True)
    plots_res_dir = plots_folder + nameproducts + "/"
    if not os.path.exists(plots_res_dir):
        os.makedirs(plots_res_dir, exist_ok=True)


def launch_collection_of_configs(configs_collection: List[str]):
    for config in configs_collection:
        command = f"python -u validation.py -C {config}"
        config_name = command.split(" ")[-1]
        config_name = config_name.split("/")[-1]
        print(f"\nLaunching script for {config_name}...\n")

        process = subprocess.Popen(
            command,
            shell=True,
            stdout=None,
            stderr=None,
        )
        process.wait()  # Wait for the process to complete


def check_params(config):
    assert config["name"] != "", "Name of the dataset not provided"
    assert config["field"].lower() in ["bb", "ee"], "Field not provided or not valid"

    N_chan = config["N_channels"]
    channel_names = config["channel_names"]
    channel_noises = config["channel_noises"]
    if config["want_noise_mismatch"]:
        mismatch_factors = config["mismatch_factors"]

    assert len(channel_names) == N_chan, "Number of channel names does not match N_chan"
    assert len(channel_noises) == N_chan, (
        "Number of channel noises does not match N_chan"
    )
    if config["want_noise_mismatch"]:
        assert len(mismatch_factors) == N_chan, (
            "Number of mismatch factors does not match N_chan"
        )

    nside = config["nside"]
    lmax = config["lmax"]
    assert lmax <= 3 * nside - 1, "lmax is too high for the given nside"

    grid = np.round(np.linspace(config["grid_start"], config["grid_end"], config["grid_steps"]), 8)
    assert config["fiducial"] in grid, "Fiducial value not in the grid"
    if config.get("wrong_fiducial", None) is not None:
        assert config["wrong_fiducial"] in grid, "Wrong fiducial value not in the grid"


def chs2idx(ch1, ch2, N_chs):
    return ch1 * (2 * N_chs - ch1 + 1) // 2 + (ch2 - ch1)


def idx2chs(idx, N_chs):
    total = N_chs * (N_chs + 1) // 2
    if idx < 0 or idx >= total:
        raise ValueError(
            f"Index {idx} out of bounds for a matrix with {N_chs} channels."
        )

    i = 0
    while (i * (2 * N_chs - i + 1)) // 2 <= idx:
        i += 1
    i -= 1
    elements_before_i = i * (2 * N_chs - i + 1) // 2
    j = i + (idx - elements_before_i)
    return i, j


def configure_plt():
    plt.rc("axes", labelsize=20, linewidth=1.5)
    plt.rc("xtick", direction="in", labelsize=15, top=True)
    plt.rc("ytick", direction="in", labelsize=15, right=True)

    plt.rc("xtick.major", width=1.1, size=5)
    plt.rc("ytick.major", width=1.1, size=5)

    plt.rc("xtick.minor", width=1.1, size=3)
    plt.rc("ytick.minor", width=1.1, size=3)

    plt.rc("lines", linewidth=2)
    plt.rc("legend", frameon=False, fontsize=15)
    plt.rc("figure", dpi=100, autolayout=True, figsize=[10, 7])
    plt.rc("savefig", dpi=300, bbox="tight")


def vec2mat(vect, N_fields):
    cross_idxs = np.array(
        [
            chs2idx(ch1, ch2, N_fields)
            for ch1 in range(N_fields)
            for ch2 in range(ch1 + 1, N_fields)
        ]
    )
    auto_idxs = np.array([chs2idx(ch, ch, N_fields) for ch in range(N_fields)])

    if len(vect.shape) == 2:
        mat = np.zeros((vect.shape[0], N_fields, N_fields))
    else:
        mat = np.zeros((1, N_fields, N_fields))
        vect = vect[None, :]

    for i in auto_idxs:
        ch1, ch2 = idx2chs(i, N_fields)
        mat[:, ch1, ch2] = vect[:, i]
    for i in cross_idxs:
        ch1, ch2 = idx2chs(i, N_fields)
        mat[:, ch1, ch2] = mat[:, ch2, ch1] = vect[:, i]
    return np.array(np.squeeze(mat))


def mat2vec(mat):
    N_chs = mat.shape[1]
    vec = []
    for idx in range(N_chs * (N_chs + 1) // 2):
        ch1, ch2 = idx2chs(idx, N_chs)
        vec.append(mat[:, ch1, ch2])
    return np.array(vec).T
