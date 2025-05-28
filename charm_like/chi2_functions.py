from typing import List

import numpy as np

from charm_like.chi2_classes import CHL, HL, MHL, SingleFieldHL
from charm_like.gaussian_chi2_classes import Gaussian, MultiGaussian
from charm_like.settings_class import Settings
from charm_like.spectra_functions import get_spectra_weights
from utils.common_functions import idx2chs


def compute_and_save_chi2_single_field(params: Settings, all_spectra):
    single_field_likelihoods: List[SingleFieldHL] = []
    for i in range(params.N_auto + params.N_cross):
        ch1, ch2 = idx2chs(i, params.N_chs)
        name = f"{params.ch_names[ch1]}x{params.ch_names[ch2]}"
        single_field_likelihoods.append(
            SingleFieldHL(params=params, name=name, spectra=all_spectra[:, i])
        )

    single_field_chi2s = np.array(
        [like.compute_chi2() for like in single_field_likelihoods]
    )

    all_cross_likelihood = CHL(
        params=params,
        name="cHL",
        spectra=all_spectra[:, params.cross_idxs].reshape(
            (params.N_sims, params.N_ell * params.N_cross)
        ),
    )

    all_cross_chi2s = all_cross_likelihood.compute_chi2()

    np.save(
        params.chi2_folder
        + f"{params.offset_type}_offset_single_field_chi2s_"
        + f"{params.extra_chi2s}{params.name}.npy",
        single_field_chi2s,
    )
    np.save(
        params.chi2_folder
        + f"{params.offset_type}_offset_cross_chi2_"
        + f"{params.extra_chi2s}{params.name}.npy",
        all_cross_chi2s,
    )

    if params.want_single_field_HL:
        single_field_combined_spectra = params.get_single_field_combination_spectra()

        single_field_HL_like = SingleFieldHL(
            params=params,
            name="single-field HL",
            spectra=single_field_combined_spectra,
            is_combined=True,
            combined_type="single-field HL",
            weights=get_spectra_weights(params, params.N_cross, params.cross_idxs),
            want_auto_offset=False,
        )
        single_field_HL_chi2 = single_field_HL_like.compute_chi2()

        np.save(
            params.chi2_folder + f"SFHL_chi2_{params.extra_chi2s}{params.name}.npy",
            single_field_HL_chi2,
        )

        if params.want_offset:
            single_field_HL_like = SingleFieldHL(
                params=params,
                name="single-field HL",
                spectra=single_field_combined_spectra,
                is_combined=True,
                combined_type="single-field HL",
                weights=get_spectra_weights(params, params.N_cross, params.cross_idxs),
                want_auto_offset=True,
            )
            single_field_HL_chi2 = single_field_HL_like.compute_chi2()

            np.save(
                params.chi2_folder
                + f"{params.offset_type}_offset_SFHL_chi2_"
                + f"{params.extra_chi2s}{params.name}.npy",
                single_field_HL_chi2,
            )


def compute_and_save_chi2_HL(params: Settings, all_spectra):
    HL_like = HL(params, name="HL", complete_spectra=all_spectra)
    HL_chi2 = HL_like.compute_chi2()

    mHL_like = MHL(
        params,
        name="mHL",
        complete_spectra=all_spectra,
    )
    mHL_chi2 = mHL_like.compute_chi2()

    np.save(
        params.chi2_folder + f"HL_chi2_{params.extra_chi2s}{params.name}.npy",
        HL_chi2,
    )
    np.save(
        params.chi2_folder + f"mHL_chi2_{params.extra_chi2s}{params.name}.npy",
        mHL_chi2,
    )

    if params.custom_idxs is not None:
        HL_hybrid_like = HL(
            params,
            name="mHL_hybrid",
            complete_spectra=all_spectra,
            custom_idxs=params.custom_idxs,
        )
        HL_hybrid_chi2 = HL_hybrid_like.compute_chi2()

        np.save(
            params.chi2_folder
            + f"mHL_hybrid_chi2_{params.extra_chi2s}{params.name}.npy",
            HL_hybrid_chi2,
        )

    if params.want_offset:
        HL_like = HL(
            params,
            name="HL",
            complete_spectra=all_spectra,
            want_offset=params.want_offset,
        )
        HL_chi2 = HL_like.compute_chi2()

        mHL_like = MHL(
            params,
            name="mHL",
            complete_spectra=all_spectra,
            want_offset=params.want_offset,
        )
        mHL_chi2 = mHL_like.compute_chi2()

        np.save(
            params.chi2_folder
            + f"{params.offset_type}_offset_HL_chi2_"
            + f"{params.extra_chi2s}{params.name}.npy",
            HL_chi2,
        )
        np.save(
            params.chi2_folder
            + f"{params.offset_type}_offset_mHL_chi2_"
            + f"{params.extra_chi2s}{params.name}.npy",
            mHL_chi2,
        )

        if params.custom_idxs is not None:
            HL_hybrid_like = HL(
                params,
                name="mHL_hybrid",
                complete_spectra=all_spectra,
                custom_idxs=params.custom_idxs,
                want_offset=params.want_offset,
            )
            HL_hybrid_chi2 = HL_hybrid_like.compute_chi2()

            np.save(
                params.chi2_folder
                + f"{params.offset_type}_offset_mHL_hybrid_chi2_"
                + f"{params.extra_chi2s}{params.name}.npy",
                HL_hybrid_chi2,
            )


def compute_and_save_chi2_Gaussian(params: Settings, all_spectra):
    single_field_likelihoods: List[Gaussian] = []
    for i in range(params.N_auto + params.N_cross):
        ch1, ch2 = idx2chs(i, params.N_chs)
        name = f"{params.ch_names[ch1]}x{params.ch_names[ch2]}"
        single_field_likelihoods.append(
            Gaussian(params=params, name=name, spectra=all_spectra[:, i])
        )

    single_field_chi2s = np.array(
        [like.compute_chi2() for like in single_field_likelihoods]
    )

    all_cross_likelihood = Gaussian(
        params=params,
        name="cHL",
        spectra=all_spectra[:, params.cross_idxs].reshape(
            (params.N_sims, params.N_ell * params.N_cross)
        ),
        is_combined=True,
        combined_type="cHL",
    )

    all_cross_chi2s = all_cross_likelihood.compute_chi2()

    np.save(
        params.chi2_folder
        + "gaussian_single_field_chi2s_"
        + f"{params.extra_chi2s}{params.name}.npy",
        single_field_chi2s,
    )
    np.save(
        params.chi2_folder
        + "gaussian_cross_chi2_"
        + f"{params.extra_chi2s}{params.name}.npy",
        all_cross_chi2s,
    )


def compute_and_save_chi2_MultiGaussian(params: Settings, all_spectra):
    HL_like = MultiGaussian(params, name="HL", complete_spectra=all_spectra)
    HL_chi2 = HL_like.compute_chi2()

    mHL_like = MultiGaussian(
        params,
        name="mHL",
        complete_spectra=all_spectra,
        exclude_auto=True,
    )
    mHL_chi2 = mHL_like.compute_chi2()

    np.save(
        params.chi2_folder
        + f"full_gaussian_chi2_{params.extra_chi2s}{params.name}.npy",
        HL_chi2,
    )
    np.save(
        params.chi2_folder
        + f"gaussian_without_auto_chi2_{params.extra_chi2s}{params.name}.npy",
        mHL_chi2,
    )
