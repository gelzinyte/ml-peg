"""
Calculate BDEs of drug-like molecules on MLFF-optimised geometries.

Journal of Chemical Theory and Computation 2024 20 (1), 164-177
DOI: 10.1021/acs.jctc.3c00710
"""

from __future__ import annotations

from typing import Any

import pytest

from ml_peg.calcs.molecular.BDEs.calc_BDEs import evaluate_bde_structures
from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)


@pytest.mark.skip(reason="no geo opt for now")
@pytest.mark.parametrize("mlip", MODELS.items())
def test_bond_dissociation_energy_mlff_opt(mlip: tuple[str, Any]) -> None:
    """
    Calculate C-H BDEs of drug-like molecules on MLFF-optimised geometries.

    Geometry-optimises each structure (molecule, radical, isolated atom) using the
    MLFF before evaluating energies and forces. DFT reference energies are preserved
    from the input structures. Reference BDEs are computed from DFT energies at DFT
    geometries.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip

    bde_dir = (
        download_s3_data(
            key="inputs/molecular/BDEs/BDEs.zip",
            filename="BDEs.zip",
        )
        / "BDEs"
    )

    for input_xyz_file in bde_dir.glob("*.xyz"):
        out_filename = input_xyz_file.name.replace(".dft_opt.xyz", ".mlff_opt.xyz")

        evaluate_bde_structures(
            model_name=model_name,
            model=model,
            input_xyz_file=input_xyz_file.resolve(),
            out_filename=out_filename,
            geom_opt=True,
        )
