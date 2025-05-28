import sys
from functools import partialmethod

import camb
import numpy as np
from tqdm import tqdm

from charm_like.settings_class import Settings
from utils.common_functions import chs2idx

tqdm.__init__ = partialmethod(
    tqdm.__init__, colour="green", ncols=100, leave=True, file=sys.stdout
)


def compute_theoretical_spectrum(field="BB", param_value=0.01):
    pars = camb.CAMBparams()
    if field == "BB":
        pars.set_cosmology(
            H0=67.32, ombh2=0.02237, omch2=0.1201, mnu=0.06, omk=0, tau=0.06
        )
        pars.InitPower.set_params(As=2.12e-9, ns=0.9651, r=param_value)
    elif field == "EE":
        pars.set_cosmology(
            H0=67.32, ombh2=0.02237, omch2=0.1201, mnu=0.06, omk=0, tau=param_value
        )
        pars.InitPower.set_params(As=2.12e-9, ns=0.9651, r=0.0)
    pars.set_for_lmax(lmax=2500)

    pars.WantTensors = True
    pars.DoLensing = True

    results = camb.get_results(pars)
    res = results.get_cmb_power_spectra(
        CMB_unit="muK",
        lmax=2500,
        raw_cl=False,
    )
    ls = np.arange(res["total"].shape[0], dtype=np.int64)
    mapping = {"tt": 0, "ee": 1, "bb": 2, "te": 3, "et": 3}
    res_cls = {"ell": ls}
    for probe, i in mapping.items():
        res_cls[probe] = res["total"][:, i]

    return np.array(
        [res_cls["ell"], res_cls["tt"], res_cls["ee"], res_cls["bb"], res_cls["te"]]
    )


def compute_templates(params: Settings):
    for idx in range(params.grid_steps):
        template = compute_theoretical_spectrum(
            field=params.field, param_value=params.grid[idx]
        ).T

        print(
            f"Saving template {idx}/{params.grid_steps - 1} for"
            + f"{params.param} = {params.grid[idx]}..."
        )

        file_path = f"dls_{params.param}_likelihood_{str(idx).zfill(4)}.txt"
        np.savetxt(params.templates_folder + file_path, template[2:])


def transpose_templates(params: Settings):
    for idx in range(params.grid_steps):
        file_path = f"dls_{params.param}_likelihood_{str(idx).zfill(4)}.txt"
        template = np.loadtxt(params.templates_folder + file_path)
        template = template.T
        np.savetxt(params.templates_folder + file_path, template)


def get_norm(params: Settings, *, want_auto=False):
    norm = np.zeros(params.N_ell)
    for ch1 in range(params.N_chs):
        start = ch1 if want_auto else ch1 + 1
        for ch2 in range(start, params.N_chs):
            norm += (params.noise_spectra[ch1] * params.noise_spectra[ch2]) ** -1
    return norm


def get_spectra_weights(params: Settings, N_spec, idxs, *, want_auto=False):
    norm = get_norm(params, want_auto=want_auto)

    weights = np.zeros((N_spec, params.N_ell))
    for ch1 in range(params.N_chs):
        start = ch1 if want_auto else ch1 + 1
        for ch2 in range(start, params.N_chs):
            abs_idx = chs2idx(ch1, ch2, params.N_chs)
            rel_idx = np.where(idxs == abs_idx)[0][0]
            weights[rel_idx] = (
                params.noise_spectra[ch1] * params.noise_spectra[ch2]
            ) ** -1 / norm

    return weights
