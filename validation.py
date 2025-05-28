import os
import sys
import warnings
from functools import partialmethod

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from charm_like.chi2_functions import (
    compute_and_save_chi2_Gaussian,
    compute_and_save_chi2_HL,
    compute_and_save_chi2_MultiGaussian,
    compute_and_save_chi2_single_field,
)
from charm_like.make_plots import produce_gaussian_plots, produce_plots
from charm_like.settings_class import get_params

tqdm.__init__ = partialmethod(
    tqdm.__init__, colour="green", ncols=100, leave=True, file=sys.stdout
)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main():
    params = get_params()

    if params.compute_and_save_chi2:
        all_spectra = params.get_all_spectra()

        for ch in range(params.N_chs):
            params.noise_spectra[ch] *= params.mismatch_factors[ch]

        compute_and_save_chi2_single_field(params, all_spectra)
        compute_and_save_chi2_HL(params, all_spectra)

        if params.want_gaussian:
            compute_and_save_chi2_Gaussian(params, all_spectra)
            compute_and_save_chi2_MultiGaussian(params, all_spectra)

    if params.show_plots or params.save_plots:
        produce_plots(params)
        if params.want_gaussian:
            produce_gaussian_plots(params)


if __name__ == "__main__":
    main()
