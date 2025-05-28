import sys
import time
from dataclasses import dataclass
from functools import partialmethod

import numpy as np
from ray.util.multiprocessing import Pool
from tqdm import tqdm

from charm_like.settings_class import Settings
from utils.common_functions import chs2idx, idx2chs

tqdm.__init__ = partialmethod(
    tqdm.__init__, colour="green", ncols=100, leave=True, file=sys.stdout
)


def chi2_gaussian(data, model, invcov):
    X = data - model[None, :]
    return np.diag(X @ invcov @ X.T)


def compute_chi2(
    r_idx,
    clth,
    cldata,
    test_inv_cov,
    N_fields=3,
    *,
    exclude_auto=False,
    custom_idxs=None,
):
    x = compute_XLs(
        r_idx,
        clth,
        cldata,
        N_fields=N_fields,
        exclude_auto=exclude_auto,
        custom_idxs=custom_idxs,
    )

    return np.einsum("ni,ij,nj->n", x, test_inv_cov, x)


def parallel_chi2(
    clth,
    cldata,
    test_inv_cov,
    N_fields=3,
    *,
    exclude_auto=False,
    custom_idxs=None,
    want_parallel_computation=True,
    N_processes=4,
):
    N_rs = clth.shape[0]

    if want_parallel_computation:
        pool = Pool(processes=N_processes)
        res = np.array(
            pool.map(
                lambda x: compute_chi2(
                    x,
                    clth,
                    cldata,
                    test_inv_cov,
                    N_fields=N_fields,
                    exclude_auto=exclude_auto,
                    custom_idxs=custom_idxs,
                ),
                range(N_rs),
            )
        ).T
    else:
        res = []
        for n in tqdm(range(N_rs)):
            temp = compute_chi2(
                n,
                clth,
                cldata,
                test_inv_cov,
                N_fields=N_fields,
                exclude_auto=exclude_auto,
                custom_idxs=custom_idxs,
            )
            res.append(temp)
        res = np.array(res).T

    return res


def compute_XLs(
    r_idx,
    clth,
    cldata,
    N_fields=3,
    *,
    exclude_auto=False,
    custom_idxs=None,
):
    N_ell = cldata.shape[2]
    N_cross = N_fields * (N_fields - 1) // 2
    x = np.zeros((cldata.shape[0], N_fields + N_cross, N_ell))

    for ell_idx in range(N_ell):
        D = cldata[:, :, ell_idx]
        M = clth[r_idx][:, ell_idx]

        x[:, :, ell_idx] = D - M

    if custom_idxs is not None:
        x = x[:, custom_idxs, :]
    elif exclude_auto:
        cross_idxs = np.array(
            [
                chs2idx(ch1, ch2, N_fields)
                for ch1 in range(N_fields)
                for ch2 in range(ch1 + 1, N_fields)
            ]
        )
        x = x[:, cross_idxs, :]

    return x.reshape(cldata.shape[0], -1)


@dataclass
class Gaussian:
    params: Settings
    name: str
    spectra: np.ndarray
    is_auto: bool = False
    is_combined: bool = False
    combined_type: str = None  # "cHL" or "map-level" or "single-field HL"

    weights: np.ndarray = None

    def __post_init__(self):
        self.ell = self.params.ell.copy()
        self.models = self.params.theo_spectra.copy()
        self.noises = self.params.get_empirical_noises()
        self.models = self.models[:, :, self.params.field_idx]

        if self.is_combined:
            if self.combined_type == "cHL":
                self.ell = np.tile(self.ell, self.params.N_cross)
                self.models = np.tile(self.models, (1, self.params.N_cross))
                self.is_auto = False
            elif self.combined_type == "auto":
                self.ell = np.tile(self.ell, self.params.N_auto)
                self.models = np.tile(self.models, (1, self.params.N_auto))
                self.noises = np.concatenate([noise for noise in self.noises], axis=0)
                self.is_auto = True
            elif self.combined_type == "map-level":
                self.noises = np.array(
                    [
                        self.noises[ch] * self.weights[ch][: self.params.N_ell]
                        for ch in range(self.params.N_chs)
                    ]
                )
                self.noises = np.mean(self.noises, axis=0)
                self.is_auto = True
            elif self.combined_type == "single-field HL":
                self.noises = self.params.get_single_field_combination_noises()
                self.is_auto = True
        else:
            self.is_auto = self._get_is_auto()

    def _get_is_auto(self):
        ch1, ch2 = self.name.split("x")
        return ch1 == ch2

    def _get_idx(self):
        ch1, ch2 = self.name.split("x")
        n_ch1 = self.params.ch_names.index(ch1)
        n_ch2 = self.params.ch_names.index(ch2)
        return chs2idx(n_ch1, n_ch2, self.params.N_chs)

    def _compute_covmat(self):
        self.covmat = np.cov(self.spectra[: -self.params.N_data].T)
        self.inv_cov = np.linalg.inv(self.covmat)

    def compute_chi2(self):
        if not self.is_combined:
            self.idx = self._get_idx()

        self._compute_covmat()

        data = self.spectra[-self.params.N_data :]
        _models = self.models.copy()

        if self.is_auto:
            if self.is_combined:
                _models += self.noises
            else:
                ch1, _ = idx2chs(self.idx, self.params.N_chs)
                _models += self.noises[ch1]

        chi2 = np.zeros((self.params.N_data, self.params.grid_steps))
        for r_idx in tqdm(
            range(self.params.grid_steps),
            desc=f"Computing chi2 for {self.name}...".center(30),
        ):
            chi2_i = chi2_gaussian(
                data=data,
                model=_models[r_idx],
                invcov=self.inv_cov,
            )
            chi2[:, r_idx] = chi2_i

        return chi2


@dataclass
class MultiGaussian:
    params: Settings

    name: str
    complete_spectra: np.ndarray

    exclude_auto: bool = False
    custom_idxs: list = None

    want_offset: bool = False

    def _get_models(self):
        models = []
        for i in range(self.params.N_auto + self.params.N_cross):
            ch1, _ = idx2chs(i, self.params.N_chs)
            if i in self.params.auto_idxs:
                models.append(self.params.theo_spectra + (self.noises[ch1])[:, None])
            else:
                models.append(self.params.theo_spectra)
        models = np.array(models)
        models = np.swapaxes(models, 0, 1)
        return models

    def _get_single_field_covmats(self):
        return np.array(
            [
                np.cov(self.complete_spectra[: -self.params.N_data, i].T)
                for i in self.params.all_idxs
            ]
        )

    def _get_covmat(self):
        idxs = self.params.all_idxs
        if self.custom_idxs is not None:
            idxs = self.custom_idxs

        return np.cov(
            np.concatenate(
                [self.complete_spectra[: -self.params.N_data, idx] for idx in idxs],
                axis=1,
            ).T
        )

    def __post_init__(self):
        self.noises = self.params.get_empirical_noises()
        self.models = self._get_models()
        self.models = self.models[:, :, :, self.params.field_idx]
        self.datas = self.complete_spectra[-self.params.N_data :]
        self.covmat = self._get_covmat()

    def compute_X(self):
        X = compute_XLs(
            self.params.fid_idx,
            self.models,
            self.datas,
            N_fields=self.params.N_chs,
            exclude_auto=self.exclude_auto,
            custom_idxs=self.custom_idxs,
        )
        return X

    def compute_chi2(self):
        inv_covmat = np.linalg.pinv(self.covmat, rcond=1e-10, hermitian=True)
        start = time.time()
        chi2 = parallel_chi2(
            self.models,
            self.datas,
            inv_covmat,
            N_fields=self.params.N_chs,
            exclude_auto=self.exclude_auto,
            custom_idxs=self.custom_idxs,
            want_parallel_computation=self.params.want_parallel_computation,
            N_processes=self.params.N_processes,
        )
        print(
            "Parallel chi2 computation of "
            + f"{self.name.center(30)} took {round(time.time() - start, 2)} seconds!"
        )
        return chi2
