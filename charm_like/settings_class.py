import os
import sys
from dataclasses import dataclass
from functools import partialmethod

import healpy as hp
import numpy as np
import yaml
from tqdm import tqdm

from utils.common_functions import (
    check_params,
    chs2idx,
    create_all_folders,
    join,
    parse_args,
)

tqdm.__init__ = partialmethod(
    tqdm.__init__, colour="green", ncols=100, leave=True, file=sys.stdout
)


def get_params(config_path: str = None) -> "Settings":
    yaml.add_constructor("!join", join)

    if config_path is None:
        parsed_args = parse_args()
        config_path = parsed_args.config_path
        config_path = os.path.abspath(config_path)

    return Settings(config_path)


@dataclass
class Settings:
    """
    Class to store settings and relevant quantities for the main script.
    """

    config_path: str
    read_theory_spectra: bool = True

    def __post_init__(self):
        self.config = yaml.load(open(self.config_path), Loader=yaml.FullLoader)

        check_params(self.config)
        create_all_folders(self.config)

        self._store_passed_settings()
        self._set_flags()
        self._compute_relevant_quantities()
        self._specify_folders()

        self._compute_noise_spectra()
        if self.read_theory_spectra:
            self._get_templates()
            self._store_fiducial_spectrum()

    @classmethod
    def from_dict(
        cls, config_dict: dict, read_theory_spectra: bool = True
    ) -> "Settings":
        import tempfile

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".yaml", mode="w"
        ) as temp_file:
            yaml.dump(config_dict, temp_file)
            temp_config_path = temp_file.name

        return cls(
            config_path=temp_config_path, read_theory_spectra=read_theory_spectra
        )

    def __repr__(self):
        return f"Settings(config_path={self.config_path})"

    def _store_passed_settings(self):
        self.nside = self.config["nside"]

        self.lmin = self.config["lmin"]
        self.lmax = self.config["lmax"]

        self.N_sims = self.config["N_sims"]
        self.N_data = self.config["N_data"]

        self.chi2_folder = self.config["chi2_folder"]
        self.pixel_folder = self.config["pixel_folder"]
        self.plots_folder = self.config["plots_folder"]
        self.spectra_folder = self.config["spectra_folder"]
        self.templates_folder = self.config["templates_folder"]

        self.param = self.config["param"]
        self.grid_start = self.config["grid_start"]
        self.grid_end = self.config["grid_end"]
        self.grid_steps = self.config["grid_steps"]

        self.fiducial_value = self.config["fiducial"]
        self.wrong_fiducial_value = self.config.get("wrong_fiducial", None)

        self.r_fiducial = self.config["r_fiducial"]
        self.tau_fiducial = self.config["tau_fiducial"]

        self.extra = self.config["extra"]
        self.extra_plots = self.config["extra_plots"]
        self.extra_chi2s = self.config["extra_chi2s"]
        self.name = self.config["name"]

        self.spectra_filenames = self.config["spectra_filenames"]
        self.noises_filenames = self.config["noises_filenames"]

        self.field = self.config["field"]
        self.fsky = self.config["fsky"]
        self.N_chs = self.config["N_channels"]

        self.N_processes = self.config.get("N_processes", None)

        self.ch_names = self.config["channel_names"]
        self.ch_noises = self.config["channel_noises"]

        self.beams_folder = self.config["beams_folder"]
        self.beam_file = self.config["beam_file"]

        self.custom_combination = self.config.get("custom_combination", None)

    def _set_flags(self):
        self.want_noise_mismatch = self.config["want_noise_mismatch"]
        self.want_single_field_HL = self.config["want_single_field_HL"]
        self.want_offset = self.config["want_offset"]
        self.offset_type = self.config["offset_type"]
        self.want_empirical_fiducial = self.config["want_empirical_fiducial"]
        self.want_SH_approximation = self.config["want_SH_approximation"]
        self.want_pixel_based = self.config["want_pixel_based"]
        self.want_gaussian = self.config["want_gaussian"]
        self.compute_and_save_chi2 = self.config["compute_and_save_chi2"]
        self.want_parallel_computation = self.config["want_parallel_computation"]
        self.show_plots = self.config["show_plots"]
        self.save_plots = self.config["save_plots"]
        self.print_statistics = self.config["print_statistics"]
        self.save_statistics = self.config["save_statistics"]

    def _compute_relevant_quantities(self):
        self.n_pix = 12 * self.nside**2

        self.large_lmax = self.nside * 3 - 1
        self.ell = np.arange(self.lmin, self.lmax + 1)
        self.ell_factor = self.ell * (self.ell + 1) / (2 * np.pi)
        self.N_ell = len(self.ell)

        self.N_cross = self.N_chs * (self.N_chs - 1) // 2
        self.N_auto = self.N_chs

        self.N_cov = int(self.N_sims - self.N_data)

        self.all_idxs = np.arange(self.N_chs * (self.N_chs + 1) // 2)
        self.cross_idxs = np.array(
            [
                chs2idx(ch1, ch2, self.N_chs)
                for ch1 in range(self.N_chs)
                for ch2 in range(ch1 + 1, self.N_chs)
            ]
        )
        self.auto_idxs = np.array(
            [chs2idx(ch, ch, self.N_chs) for ch in range(self.N_chs)]
        )

        self.custom_idxs = None
        if self.custom_combination is not None:
            idxs = []
            for spectrum_label in self.custom_combination:
                ch1, ch2 = spectrum_label[0], spectrum_label[-1]
                ch1 = self.ch_names.index(ch1)
                ch2 = self.ch_names.index(ch2)
                idxs.append(chs2idx(ch1, ch2, self.N_chs))
            self.custom_idxs = np.array(idxs)

        if self.field == "BB":
            self.field_idx = 2
            self.param_latex = r"$r$"
        elif self.field == "EE":
            self.field_idx = 1
            self.param_latex = r"$\tau$"
        else:
            raise ValueError("Field not recognized. Choose 'EE' or 'BB'.")

        self.grid = np.round(
            np.linspace(self.grid_start, self.grid_end, self.grid_steps), 8
        )
        self.fid_idx = np.where(self.grid == self.fiducial_value)[0][0]
        if self.wrong_fiducial_value is not None:
            self.wrong_fid_idx = np.where(self.grid == self.wrong_fiducial_value)[0][0]

        self.mismatch_factors = np.ones(self.N_chs)
        if self.want_noise_mismatch:
            self.mismatch_factors = self.config["mismatch_factors"]

        self.bl = hp.read_cl(self.beams_folder + self.beam_file)[:, 0 : self.lmax + 1]

    def _specify_folders(self):
        self.chi2_folder = self.chi2_folder + self.name + "/"
        self.plots_folder = self.plots_folder + self.name + "/"
        self.spectra_folder = self.spectra_folder + self.name + "/"

    def _compute_noise_spectra(self):
        if self.noises_filenames[0].endswith(".npy"):
            self.noise_spectra = np.array(
                [
                    np.load(self.spectra_folder + filename)
                    for filename in self.noises_filenames
                ]
            )
            self.noise_spectra = np.mean(self.noise_spectra[:, : -self.N_data], axis=1)
        elif self.noises_filenames[0].endswith(".dat"):
            self.noise_spectra = np.array(
                [
                    np.loadtxt(self.spectra_folder + filename)
                    for filename in self.noises_filenames
                ]
            )
        else:
            raise ValueError("Noise spectra format not recognized.")

        self.noise_spectra = (
            self.noise_spectra[:, : self.N_ell] / self.ell_factor[None, :]
        )

    def get_fiducial(self):
        if self.want_empirical_fiducial:
            all_spectra = self.get_all_spectra()
            fiducial = np.mean(all_spectra[: -self.N_data, self.all_idxs], axis=0)
        else:
            fiducial = self.fiducial_spectrum[: self.N_ell, self.field_idx]
        return fiducial

    def _get_templates(self):
        self.theo_spectra = np.zeros((len(self.grid), self.N_ell, 4))
        for grididx in tqdm(range(len(self.grid)), desc="Loading templates".center(30)):
            file_path = (
                f"dls_{self.param}_likelihood_"
                f"{self.param}{round(self.grid[grididx], 8)}.txt"
            )
            self.theo_spectra[grididx] = (
                np.loadtxt(self.templates_folder + file_path)[
                    self.lmin - 2 : self.lmax - 1, [1, 2, 3, 4]
                ]
                / self.ell_factor[:, None]
            )

    def _store_fiducial_spectrum(self):
        ls = np.arange(self.lmin, self.large_lmax + 1)
        ls_factor = ls * (ls + 1) / (2 * np.pi)

        file_path = (
            f"dls_{self.param}_likelihood_"
            f"{self.param}{round(self.grid[self.fid_idx], 8)}.txt"
        )
        self.fiducial_spectrum = (
            np.loadtxt(self.templates_folder + file_path)[
                self.lmin - 2 : self.large_lmax - 1, [1, 2, 3, 4]
            ]
            / ls_factor[:, None]
        )

        if self.wrong_fiducial_value is not None:
            file_path = (
                f"dls_{self.param}_likelihood_"
                f"{self.param}{round(self.grid[self.wrong_fid_idx], 8)}.txt"
            )
            self.fiducial_spectrum = (
                np.loadtxt(self.templates_folder + file_path)[
                    self.lmin - 2 : self.large_lmax - 1, [1, 2, 3, 4]
                ]
                / ls_factor[:, None]
            )
            file_path = (
                f"dls_{self.param}_likelihood_"
                f"{self.param}{round(self.grid[self.fid_idx], 8)}.txt"
            )
            self.correct_fiducial_spectrum = (
                np.loadtxt(self.templates_folder + file_path)[
                    self.lmin - 2 : self.large_lmax - 1, [1, 2, 3, 4]
                ]
                / ls_factor[:, None]
            )
            self.correct_fiducial_spectrum = np.concatenate(
                [np.zeros((self.lmin, 4)), self.correct_fiducial_spectrum], axis=0
            )

        self.full_fiducial_spectrum = np.concatenate(
            [np.zeros((self.lmin, 4)), self.fiducial_spectrum], axis=0
        )

    def get_spectra_filenames(self):
        return [self.spectra_folder + filename for filename in self.spectra_filenames]

    def get_noise_spectra_filenames(self):
        return [self.spectra_folder + filename for filename in self.noises_filenames]

    def get_all_spectra(self):
        files_spectra = self.get_spectra_filenames()

        all_spectra = np.zeros(
            (self.N_sims, self.N_chs * (self.N_chs + 1) // 2, self.N_ell)
        )
        for ch1 in range(self.N_chs):
            for ch2 in range(ch1, self.N_chs):
                idx = chs2idx(ch1, ch2, self.N_chs)
                all_spectra[:, idx] = np.load(files_spectra[idx])[:, : self.N_ell]
                all_spectra[:, idx] /= self.ell_factor
        return all_spectra

    def get_all_noises_spectra(self):
        files_noises = self.get_noise_spectra_filenames()

        all_noises = np.zeros((self.N_sims, self.N_chs, self.N_ell))
        for ch in range(self.N_chs):
            if files_noises[0].endswith(".npy"):
                all_noises[:, ch] = np.load(files_noises[ch])[:, : self.N_ell]
            elif files_noises[0].endswith(".dat"):
                noise_spectrum = np.loadtxt(files_noises[ch])[: self.N_ell]
                all_noises[:, ch] = np.tile(noise_spectrum, (self.N_sims, 1))
            all_noises[:, ch] = all_noises[:, ch] / self.ell_factor[None, :]
        return all_noises

    def get_empirical_noises(self):
        all_noises = self.get_all_noises_spectra()
        return np.mean(all_noises[: -self.N_data], axis=0)

    def get_pixel_based_chi2(self):
        _name = self.name
        _name = _name.replace("mismatch_", "")
        _name = _name.replace("wrongfid_", "")
        _name = _name.replace("2_", "")

        channels = "all" if self.N_chs == 3 else "A+B"
        pixel_based_filepath = (
            self.pixel_folder
            + f"{_name}_fsky{int(self.fsky)}_{channels}_pixel_based.npy"
        )
        return -2 * np.load(pixel_based_filepath)[-self.N_data :]

    def get_pixel_based_grid(self):
        if self.field == "BB":
            start = 0.0
            end = 0.04
            steps = 3201
        elif self.field == "EE":
            start = 0.03
            end = 0.09
            steps = 961
        return np.linspace(start, end, steps)

    def get_single_field_combination_spectra(self):
        _name = self.name
        _name = _name.replace("mismatch_", "")
        _name = _name.replace("wrongfid_", "")
        _name = _name.replace("2_", "")

        _folder = self.spectra_folder
        _folder = _folder.replace("mismatch_", "")
        _folder = _folder.replace("wrongfid_", "")
        _folder = _folder.replace("2_", "")

        channels = "all" if self.N_chs == 3 else "A+B"
        filepath = _folder + f"{_name}_fsky{int(self.fsky)}_{channels}_spectra_QML.npy"
        return np.load(filepath)[:, 0, : self.N_ell] / self.ell_factor

    def get_single_field_combination_noises(self):
        _name = self.name
        _name = _name.replace("mismatch_", "")
        _name = _name.replace("wrongfid_", "")
        _name = _name.replace("2_", "")

        _folder = self.spectra_folder
        _folder = _folder.replace("mismatch_", "")
        _folder = _folder.replace("wrongfid_", "")
        _folder = _folder.replace("2_", "")

        channels = "all" if self.N_chs == 3 else "A+B"
        filepath = _folder + f"{_name}_fsky{int(self.fsky)}_{channels}_nlhat.dat"
        return np.loadtxt(filepath)[: self.N_ell, 1] / self.ell_factor
