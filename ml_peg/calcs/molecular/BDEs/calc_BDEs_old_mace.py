"""
Calculate bond dissociation energies of drug-like molecules.

Journal of Chemical Theory and Computation 2024 20 (1), 164-177
DOI: 10.1021/acs.jctc.3c00710
"""

from __future__ import annotations

from copy import copy
from pathlib import Path

from ase.io import read, write
from tqdm import tqdm

DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"


def test_bond_dissociation_energy() -> None:
    """Calculate C-H bond dissociation energy of drug-like molecules."""
    calc = get_calc()
    model_name = "bde_mace"

    bde_dir = "/home/elena/.cache/ml_peg/BDEs"

    structures_filename = "cytochrome_p450_substrates.dft_opt.xyz"
    mols_rads = read(Path(bde_dir) / structures_filename, ":")

    ats_out = []
    for at in tqdm(mols_rads):
        at.calc = copy(calc)
        at.info["pred_energy"] = at.get_potential_energy()
        at.arrays["pred_forces"] = at.get_forces()
        ats_out.append(at)

    write_dir = OUT_PATH / model_name
    write_dir.mkdir(parents=True, exist_ok=True)
    write(write_dir / structures_filename, ats_out)


def get_calc():
    """
    Load the old BDE MACE calculator, forcing CPU map location.

    Returns
    -------
    MACECalculator
        ASE-compatible MACE calculator loaded on CPU.
    """
    import torch

    _orig_jit_load = torch.jit.load

    def _cpu_jit_load(f, *args, **kwargs):
        """
        Wrap torch.jit.load to default map_location to CPU.

        Parameters
        ----------
        f
            File path or object to load the TorchScript model from.
        *args
            Positional arguments passed to torch.jit.load.
        **kwargs
            Keyword arguments passed to torch.jit.load.

        Returns
        -------
        torch.jit.ScriptModule
            The loaded TorchScript module.
        """
        kwargs.setdefault("map_location", "cpu")
        return _orig_jit_load(f, *args, **kwargs)

    torch.jit.load = _cpu_jit_load

    from mace.calculators import mace

    mace_fn = "/home/elena/code/ml-peg/4.create_test/4.evaluate_old_mace/mace.model"
    mace_calc = (mace.MACECalculator, [], {"model_path": mace_fn, "device": "cpu"})
    return mace_calc[0](**mace_calc[2])


if __name__ == "__main__":
    test_bond_dissociation_energy()
