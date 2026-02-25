"""Define classes for all models."""

# ruff: noqa: D101, D102, F401

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from mlipx import GenericASECalculator as MlipxGenericASECalc
from mlipx.nodes.generic_ase import Device

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator
    from ase.calculators.mixing import SumCalculator

current_models = None


@dataclasses.dataclass(kw_only=True)
class SumCalc:
    """
    Base class that tracks whether a model already includes D3 dispersion.

    ``add_d3_calculator`` only wraps calculators with an explicit TorchDFTD3
    correction when ``trained_on_d3`` is ``False``; otherwise the original
    calculator is returned untouched.
    """

    trained_on_d3: bool = False
    d3_kwargs: dict = dataclasses.field(default_factory=dict)
    supported_elements: list[str] | None = None

    def is_elements_supported(self, required: set[str]) -> bool:
        """
        Return whether this model supports all elements in ``required``.

        Parameters
        ----------
        required
            Set of chemical symbols to check.

        Returns
        -------
        bool
            ``True`` if the model has no element restriction, or if every
            element in ``required`` is in ``supported_elements``.
        """
        if self.supported_elements is None:
            return True
        return required <= set(self.supported_elements)

    def skip_if_elements_unsupported(self, required: set[str] | None = None) -> None:
        """
        Skip the current pytest test if this model has element restrictions.

        Call with no arguments to skip unconditionally whenever
        ``supported_elements`` is set (suitable for benchmarks known to
        contain only unsupported elements).  Pass a set of chemical symbols
        to skip only when a specific structure contains unsupported elements.

        Parameters
        ----------
        required
            Set of chemical symbols to check.  If ``None``, the test is
            skipped whenever ``supported_elements`` is not ``None``.
        """
        import pytest

        if self.supported_elements is None:
            return
        if required is None:
            pytest.skip(f"Model only supports {self.supported_elements}")
        unsupported = required - set(self.supported_elements)
        if unsupported:
            pytest.skip(f"Structure contains unsupported elements: {unsupported}")

    def add_d3_calculator(self, calcs) -> Calculator | SumCalculator:
        """
        Add D3 dispersion to calculator(s).

        Parameters
        ----------
        calcs
            Calculator, or list of calculators, to add D3 dispersion to via a
            SumCalculator.

        Returns
        -------
        SumCalculator | Calculator
            Calculator(s) with D3 dispersion added, or the original calculator when
            the model is already trained with D3 corrections.
        """
        if self.trained_on_d3:
            return calcs
        from ase import units
        from ase.calculators.mixing import SumCalculator
        import torch
        from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator

        if not isinstance(calcs, list):
            calcs = [calcs]

        d3_calc = TorchDFTD3Calculator(
            device=self.d3_kwargs.get("device", "cpu"),
            damping=self.d3_kwargs.get("damping", "bj"),
            xc=self.d3_kwargs.get("xc", "pbe"),
            dtype=getattr(torch, self.d3_kwargs.get("dtype", "float32")),
            cutoff=self.d3_kwargs.get("cutoff", 40.0 * units.Bohr),
        )
        calcs.append(d3_calc)

        return SumCalculator(calcs)


@dataclasses.dataclass(kw_only=True)
class GenericASECalc(SumCalc, MlipxGenericASECalc):
    """Data class for generic ASE calculators."""

    default_dtype: str | None = None

    def get_calculator(self, **kwargs) -> Calculator:
        """
        Prepare and load the calculator.

        Parameters
        ----------
        **kwargs
            Any keyword arguments to pass to `get_calculator`.

        Returns
        -------
        Calculator
            Loaded ASE Calculator.
        """
        if self.default_dtype is not None:
            kwargs["default_dtype"] = self.default_dtype

        return MlipxGenericASECalc.get_calculator(self, **kwargs)


@dataclasses.dataclass(kw_only=True)
class PetMadCalc(GenericASECalc):
    """Dataclass for PET-MAD calculator."""

    def get_calculator(self, **kwargs) -> Calculator:
        """
        Prepare and load the calculator.

        Parameters
        ----------
        **kwargs
            Any keyword arguments to pass to `get_calculator`.

        Returns
        -------
        Calculator
            Loaded ASE Calculator.
        """
        if self.default_dtype is not None:
            kwargs["dtype"] = self.default_dtype
        else:
            kwargs["dtype"] = self.default_dtype

        return MlipxGenericASECalc.get_calculator(self, **kwargs)


# https://github.com/orbital-materials/orb-models
@dataclasses.dataclass(kw_only=True)
class OrbCalc(SumCalc):
    """Dataclass for Orb calculator."""

    name: str
    device: Device | None = None
    default_dtype: str = "float32-high"
    kwargs: dict = dataclasses.field(default_factory=dict)

    def get_calculator(self, **kwargs) -> Calculator:
        """
        Prepare and load the calculator.

        Parameters
        ----------
        **kwargs
            Any keyword arguments to pass to `get_calculator`.

        Returns
        -------
        Calculator
            Loaded ASE Orb Calculator.
        """
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator
        import torch._dynamo

        torch._dynamo.config.suppress_errors = True
        torch._dynamo.disable()
        import os

        os.environ["TORCH_DISABLE_MODULE_HIERARCHY_TRACKING"] = "1"

        method = getattr(pretrained, self.name)
        if self.device is None:
            orbff = method(precision=self.default_dtype, **self.kwargs)
            calc = ORBCalculator(orbff, **self.kwargs)
        elif self.device == Device.AUTO:
            orbff = method(
                device=Device.resolve_auto(),
                precision=self.default_dtype,
                **self.kwargs,
            )
            calc = ORBCalculator(orbff, device=Device.resolve_auto(), **self.kwargs)
        else:
            orbff = method(
                device=self.device, precision=self.default_dtype, **self.kwargs
            )
            calc = ORBCalculator(orbff, device=self.device, **self.kwargs)

        return calc

    @property
    def available(self) -> bool:
        """
        Check whether the calculator module is available.

        Returns
        -------
        bool
            Whether the calculator can be loaded.
        """
        try:
            from orb_models.forcefield import pretrained
            from orb_models.forcefield.calculator import ORBCalculator

            return True
        except ImportError:
            return False


@dataclasses.dataclass(kw_only=True)
class FairChemCalc(SumCalc):
    """Dataclass for fairchem (UMA) calculator."""

    model_name: str
    task_name: str
    device: Device | str = "cpu"
    default_dtype: str = "float32"
    overrides: dict = dataclasses.field(default_factory=dict)

    def get_calculator(self) -> Calculator:
        """
        Prepare and load the calculator.

        Returns
        -------
        Calculator
            Loaded ASE Orb Calculator.
        """
        from fairchem.core import FAIRChemCalculator, pretrained_mlip
        # torch.serialization.add_safe_globals([slice])

        predictor = pretrained_mlip.get_predict_unit(
            self.model_name, device=self.device, overrides=self.overrides
        )
        return FAIRChemCalculator(predictor, task_name=self.task_name)

    @property
    def available(self) -> bool:
        """
        Check whether the calculator module is available.

        Returns
        -------
        bool
            Whether the calculator can be loaded.
        """
        try:
            from fairchem.core import pretrained_mlip

            return self.model_name in pretrained_mlip._MODEL_CKPTS.checkpoints
        except Exception:
            return False


@dataclasses.dataclass(kw_only=True)
class ANICalc(SumCalc):
    """Dataclass for TorchANI calculators (ANI-2x etc.)."""

    model_name: str = "ANI2x"
    device: Device | None = None
    default_dtype: str = "float32"
    kwargs: dict = dataclasses.field(default_factory=dict)
    supported_elements: list[str] = dataclasses.field(
        default_factory=lambda: ["H", "C", "N", "O", "S", "F", "Cl"]
    )

    def get_calculator(self) -> Calculator:
        """
        Prepare and load the TorchANI ASE calculator.

        Returns
        -------
        Calculator
            Loaded ASE TorchANI Calculator.
        """
        import torchani

        model = getattr(torchani.models, self.model_name)()

        device = self.device
        if device == Device.AUTO or device == "auto":
            device = Device.resolve_auto()
        if device is not None:
            model = model.to(device)

        return model.ase()

    @property
    def available(self) -> bool:
        """
        Check whether the torchani package is available.

        Returns
        -------
        bool
            Whether the calculator can be loaded.
        """
        try:
            import torchani  # noqa: F401

            return True
        except ImportError:
            return False


@dataclasses.dataclass(kw_only=True)
class TBLiteCalc(SumCalc):
    """Dataclass for tblite calculators (GFN2-xTB etc.)."""

    method: str = "GFN2-xTB"

    def get_calculator(self, **kwargs) -> Calculator:
        """
        Prepare and load the tblite ASE calculator.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        Calculator
            Loaded ASE TBLite Calculator.
        """
        from tblite.ase import TBLite

        return TBLite(method=self.method)

    @property
    def available(self) -> bool:
        """
        Check whether the tblite package is available.

        Returns
        -------
        bool
            Whether the calculator can be loaded.
        """
        try:
            from tblite.ase import TBLite  # noqa: F401

            return True
        except ImportError:
            return False
