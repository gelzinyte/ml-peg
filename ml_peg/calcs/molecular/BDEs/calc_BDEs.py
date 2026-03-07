"""
Calculate bond dissociation energies of drug-like molecules.

Journal of Chemical Theory and Computation 2024 20 (1), 164-177
DOI: 10.1021/acs.jctc.3c00710
"""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

from ase.io import read, write
from janus_core.calculations.geom_opt import GeomOpt
import pytest

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"


def evaluate_bde_structures(
    model_name: str,
    model: Any,
    input_xyz_file: Path,
    out_filename: str,
    geom_opt: bool = False,
) -> None:
    """
    Evaluate MLFF energies on BDE structures, optionally after geometry optimisation.

    Parameters
    ----------
    model_name
        Name of the model, used to determine the output directory.
    model
        Model object providing get_calculator() and add_d3_calculator().
    input_xyz_file
        The input xyz file.
    out_filename
        Name of the output xyz file.
    geom_opt
        If True, geometry-optimise each structure with the MLFF before
        evaluating energies and forces.
    """
    calc = model.get_calculator()
    calc = model.add_d3_calculator(calc)

    mols_rads = read(input_xyz_file, ":")

    ats_out = []
    for at in mols_rads:
        at.calc = copy(calc)
        if geom_opt:
            geomopt = GeomOpt(struct=at, filter_class=None, write_results=False)
            geomopt.run()
            at = geomopt.struct
        at.info["pred_energy"] = at.get_potential_energy()
        at.arrays["pred_forces"] = at.get_forces()
        ats_out.append(at)

    write_dir = OUT_PATH / model_name
    write_dir.mkdir(parents=True, exist_ok=True)
    write(write_dir / out_filename, ats_out)


@pytest.mark.parametrize("mlip", MODELS.items())
def test_bond_dissociation_energy(mlip: tuple[str, Any]) -> None:
    """
    Calculate C-H bond dissociation energy of drug-like molecules.

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
        evaluate_bde_structures(
            model_name=model_name,
            model=model,
            input_xyz_file=input_xyz_file.resolve(),
            out_filename=input_xyz_file.name,
            geom_opt=False,
        )
