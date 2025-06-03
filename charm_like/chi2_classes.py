import sys
import time
from dataclasses import dataclass
from functools import partialmethod

import numpy as np
from ray.util.multiprocessing import Pool
from tqdm import tqdm

from charm_like.settings_class import Settings
from utils.common_functions import chs2idx, idx2chs, mat2vec, vec2mat

tqdm.__init__ = partialmethod(
    tqdm.__init__, colour="green", ncols=100, leave=True, file=sys.stdout
)


def compute_offsets(
    ell, varcl, clref, offset_type=None, fsky=1.0, iter=1, spectra=None
):
    if offset_type == "lillipop_like":
        return compute_lollipop_like_offset(ell, varcl, clref, fsky=fsky, iter=iter)
    elif offset_type == "montecarlo":
        return compute_montecarlo_offset(ell, spectra)
    else:
        raise ValueError(
            f"Unknown offset type: {offset_type}."
            "Please, choose one of the following: 'lillipop_like', 'montecarlo'."
        )


def compute_lollipop_like_offset(ell, varcl, clref, fsky=1.0, iter=1):
    # Original implementation in https://github.com/planck-npipe/lollipop .
    # See Tristram et al. - arXiv:2112.07961
    Nl = np.sqrt(np.abs(varcl - (2.0 / (2.0 * ell + 1) * clref**2) / fsky))
    for _ in range(iter):
        Nl = np.sqrt(
            np.abs(varcl - 2.0 / (2.0 * ell + 1) / fsky * (clref**2 + 2.0 * Nl * clref))
        )
    return Nl * np.sqrt((2.0 * ell + 1) / 2.0)


def compute_montecarlo_offset(ell, spectra):
    Nl = []
    for i in range(len(ell)):
        spec = spectra[:, i]
        negative_spectra = np.abs(spec[spec < 0])
        try:
            quantile = np.quantile(negative_spectra, 0.99)
        except IndexError:
            quantile = 0.0
        Nl.append(quantile)
    return np.array(Nl)


def ghl(x):
    return np.sign(x - 1) * np.sqrt(2.0 * (x - np.log(x) - 1))


def glolli(x):
    return np.sign(x) * ghl(np.abs(x))


def chi2_single_field_HL(data, model, fidu, offset, invcov, N_cov=None):
    fidu = fidu[None, :]
    offset = offset[None, :]
    model = model[None, :]
    X = (fidu + offset) * glolli((data + offset) / (model + offset))
    chi2 = np.diag(X @ invcov @ X.T)
    if N_cov is not None:
        chi2 = -2 * np.log((1 + chi2 / (N_cov - 1)) ** (-N_cov / 2))
    return chi2


def compute_chi2(
    r_idx,
    clth,
    cldata,
    cloff,
    clfidu,
    test_inv_cov,
    N_fields=3,
    *,
    exclude_auto=False,
    custom_idxs=None,
    N_cov=None,
):
    x = compute_XLs(
        r_idx,
        clth,
        cldata,
        cloff,
        clfidu,
        N_fields=N_fields,
        exclude_auto=exclude_auto,
        custom_idxs=custom_idxs,
    )

    chi2 = np.einsum("ni,ij,nj->n", x, test_inv_cov, x)
    if N_cov is not None:
        chi2 = -2 * np.log((1 + chi2 / (N_cov - 1)) ** (-N_cov / 2))
    return chi2


def parallel_chi2(
    clth,
    cldata,
    cloff,
    clfidu,
    test_inv_cov,
    N_fields=3,
    *,
    exclude_auto=False,
    custom_idxs=None,
    N_cov=None,
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
                    cloff,
                    clfidu,
                    test_inv_cov,
                    N_fields=N_fields,
                    exclude_auto=exclude_auto,
                    custom_idxs=custom_idxs,
                    N_cov=N_cov,
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
                cloff,
                clfidu,
                test_inv_cov,
                N_fields=N_fields,
                exclude_auto=exclude_auto,
                custom_idxs=custom_idxs,
                N_cov=N_cov,
            )
            res.append(temp)
        res = np.array(res).T

    return res


def compute_XLs(
    r_idx,
    clth,
    cldata,
    cloff,
    clfidu,
    N_fields=3,
    *,
    exclude_auto=False,
    custom_idxs=None,
):
    N_ell = cldata.shape[2]
    N_cross = N_fields * (N_fields - 1) // 2
    x = np.zeros((cldata.shape[0], N_fields + N_cross, N_ell))

    for ell_idx in range(N_ell):
        Off = vec2mat(cloff[:, ell_idx], N_fields)
        D = vec2mat(cldata[:, :, ell_idx], N_fields) + Off[None, :]
        M = vec2mat(clth[r_idx][:, ell_idx], N_fields) + Off[None, :]
        F = vec2mat(clfidu[:, ell_idx], N_fields) + Off[None, :]

        w, V = np.linalg.eigh(M[0])
        L = np.einsum("ij,j,kj->ik", V, 1 / np.sqrt(w), V)
        P = np.einsum("ji,njk,kl->nil", L, D, L)

        w, V = np.linalg.eigh(P)
        gg = np.sign(w) * ghl(np.abs(w))
        G = np.einsum("nij,nj,nkj->nik", V, gg, V)

        w, V = np.linalg.eigh(F[0])
        L = np.einsum("ij,j,kj->ik", V, np.sqrt(w), V)
        X = np.einsum("ji,njk,kl->nil", L, G, L)

        x[:, :, ell_idx] = mat2vec(X)

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
class SingleFieldHL:
    params: Settings
    name: str
    spectra: np.ndarray
    is_auto: bool = False
    is_combined: bool = False
    combined_type: str = None  # "cHL" or "single-field HL"
    want_auto_offset: bool = True

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
            elif self.combined_type == "single-field HL":
                self.noises = self.params.get_single_field_combination_noises()
                self.is_auto = True
        else:
            self.is_auto = self._get_is_auto()
        self.fiducial = self._get_fiducial()

    def _get_fiducial(self):
        _fidu = self.params.get_fiducial()
        if self.is_combined:
            if self.combined_type == "cHL":
                return np.concatenate(_fidu[self.params.cross_idxs])
            elif self.combined_type == "single-field HL":
                return np.mean(_fidu[self.params.cross_idxs], axis=0) + self.noises
        return _fidu[self._get_idx()]

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

    def _compute_offset(self, fiducial):
        spectra = None
        if self.params.offset_type == "montecarlo":
            spectra = self.spectra[: -self.params.N_data]
        self.offset = compute_offsets(
            self.ell,
            np.diag(self.covmat),
            fiducial,
            self.params.offset_type,
            fsky=self.params.fsky / 100,
            iter=30,
            spectra=spectra,
        )

    def compute_chi2(self):
        if not self.is_combined:
            self.idx = self._get_idx()

        self._compute_covmat()

        self._compute_offset(self.fiducial)

        data = self.spectra[-self.params.N_data :]
        _models = self.models.copy()
        _offset = self.offset.copy()

        if self.is_auto:
            if self.is_combined:
                _models += self.noises
            else:
                ch1, _ = idx2chs(self.idx, self.params.N_chs)
                _models += self.noises[ch1]
            _offset *= int(self.want_auto_offset)

        chi2 = np.zeros((self.params.N_data, self.params.grid_steps))
        for r_idx in tqdm(
            range(self.params.grid_steps),
            desc=f"Computing chi2 for {self.name}...".center(30),
        ):
            chi2_i = chi2_single_field_HL(
                data=data,
                model=_models[r_idx],
                fidu=self.fiducial,
                offset=_offset,
                invcov=self.inv_cov,
                N_cov=self.params.N_cov if self.params.want_SH_approximation else None,
            )
            chi2[:, r_idx] = chi2_i

        return chi2


class CHL(SingleFieldHL):
    def __post_init__(self):
        self.is_combined = True
        self.combined_type = "cHL"

        super().__post_init__()


@dataclass
class HL:
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
        elif self.exclude_auto:
            idxs = self.params.cross_idxs

        return np.cov(
            np.concatenate(
                [self.complete_spectra[: -self.params.N_data, idx] for idx in idxs],
                axis=1,
            ).T
        )

    def _get_single_fiels_offsets(self):
        covmats = self._get_single_field_covmats()

        single_field_offsets = np.zeros(
            (self.params.N_auto + self.params.N_cross, self.params.N_ell)
        )

        theory = self.params.theo_spectra[self.params.fid_idx]
        theory = theory[:, self.params.field_idx]
        spectra = [None] * (self.params.N_auto + self.params.N_cross)
        if self.params.offset_type == "montecarlo":
            spectra = self.complete_spectra[: -self.params.N_data]
        for i in range(self.params.N_auto + self.params.N_cross):
            if i in self.params.auto_idxs:
                ch1, _ = idx2chs(i, self.params.N_chs)
                single_field_offsets[i] = compute_offsets(
                    self.params.ell,
                    np.diag(covmats[i]),
                    theory + self.noises[ch1],
                    self.params.offset_type,
                    fsky=self.params.fsky / 100,
                    iter=30,
                    spectra=spectra[:, i],
                )
            else:
                single_field_offsets[i] = compute_offsets(
                    self.params.ell,
                    np.diag(covmats[i]),
                    theory,
                    self.params.offset_type,
                    fsky=self.params.fsky / 100,
                    iter=30,
                    spectra=spectra[:, i],
                )

        return single_field_offsets

    def _get_offset(self):
        single_field_offsets = self._get_single_fiels_offsets()

        single_field_offsets[self.params.cross_idxs] *= 0.0
        if not self.want_offset:
            single_field_offsets = np.zeros_like(single_field_offsets)

        return single_field_offsets

    def __post_init__(self):
        self.noises = self.params.get_empirical_noises()
        self.models = self._get_models()
        self.models = self.models[:, :, :, self.params.field_idx]
        self.offset = self._get_offset()
        self.data = self.complete_spectra[-self.params.N_data :]
        self.covmat = self._get_covmat()

    def compute_X(self):
        fiducial = self.params.get_fiducial()
        X = compute_XLs(
            self.params.fid_idx,
            self.models,
            self.data,
            self.offset,
            fiducial,
            N_fields=self.params.N_chs,
            exclude_auto=self.exclude_auto,
            custom_idxs=self.custom_idxs,
        )
        return X

    def compute_chi2(self):
        fiducial = self.params.get_fiducial()
        inv_covmat = np.linalg.pinv(self.covmat, rcond=1e-10, hermitian=True)
        start = time.time()
        chi2 = parallel_chi2(
            self.models,
            self.data,
            self.offset,
            fiducial,
            inv_covmat,
            N_fields=self.params.N_chs,
            exclude_auto=self.exclude_auto,
            custom_idxs=self.custom_idxs,
            N_cov=self.params.N_cov if self.params.want_SH_approximation else None,
            want_parallel_computation=self.params.want_parallel_computation,
            N_processes=self.params.N_processes,
        )
        print(
            "Parallel chi2 computation of "
            + f"{self.name.center(30)} took {round(time.time() - start, 2)} seconds!"
        )
        return chi2


class MHL(HL):
    def __post_init__(self):
        self.exclude_auto = True

        super().__post_init__()
