"""
Dose Calculator
===============
Implements a simplified pencil-beam dose model for RL training.

Real radiotherapy uses Monte Carlo photon transport (very slow).
For RL, we need fast approximate dose calculation.

Model:
  - Each beam is a Gaussian-profile strip through the patient body
  - Dose = sum of all beam contributions
  - Attenuation: exponential falloff for tissue depth
  - Penumbra: Gaussian lateral falloff (beam edge softening)

This is physically inspired but fast — suitable for thousands of
RL training episodes.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from .phantom import PatientPhantom, Beam


class DoseCalculator:
    """Fast approximate dose calculation for RL training."""

    # Calibrated so 7 beams converging at isocenter produce ~1.0 Gy
    BEAM_SCALE: float = 0.40

    def __init__(
        self,
        grid_size: int = 64,
        beam_width_sigma: float = 4.0,  # Tighter beam = better tumor focus
        attenuation_mu: float = 0.012,
        prescription_dose: float = 1.0,
    ):
        self.grid_size = grid_size
        self.beam_width_sigma = beam_width_sigma
        self.attenuation_mu = attenuation_mu
        self.prescription_dose = prescription_dose

        # Pre-compute coordinate grids
        y, x = np.mgrid[:grid_size, :grid_size]
        self._cx = x.astype(np.float32) - grid_size / 2
        self._cy = y.astype(np.float32) - grid_size / 2

    def compute(self, patient: PatientPhantom, beams: List[Beam]) -> np.ndarray:
        """
        Compute cumulative dose distribution for all beams.

        Returns:
            dose: (grid_size, grid_size) float32 array, values in [0, ~2.0]
                  (normalized so prescription = 1.0)
        """
        if not beams:
            return np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        dose = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        # Isocenter = tumor center (beams converge HERE, not at grid center)
        isocenter: Optional[Tuple[float, float]] = patient.tumor_center

        for beam in beams:
            dose += self._compute_single_beam(beam, patient.body_mask, isocenter)

        return dose.astype(np.float32)

    def _compute_single_beam(
        self, beam: Beam, body_mask: np.ndarray,
        isocenter: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """
        Compute dose from a single beam converging on the isocenter (tumor center).
        The beam is aimed AT the tumor, not at the grid center.
        """
        angle_rad = np.deg2rad(beam.angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        # Use isocenter if provided, else default to grid center
        if isocenter is not None:
            iso_x = isocenter[0] - self.grid_size / 2
            iso_y = isocenter[1] - self.grid_size / 2
        else:
            iso_x, iso_y = 0.0, 0.0

        # Shift coordinate system to isocenter
        cx_shifted = self._cx - iso_x
        cy_shifted = self._cy - iso_y

        # Lateral distance from beam central axis (through isocenter)
        lateral = -cx_shifted * sin_a + cy_shifted * cos_a

        # Gaussian beam profile — tighter = more conformal to tumor
        profile = np.exp(-0.5 * (lateral / self.beam_width_sigma) ** 2)

        # Depth along beam direction
        depth = cx_shifted * cos_a + cy_shifted * sin_a
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        attenuation = np.exp(-self.attenuation_mu * depth_norm * self.grid_size)

        beam_dose = profile * attenuation * beam.dose_weight * self.BEAM_SCALE

        # No dose outside patient body
        beam_dose *= body_mask.astype(np.float32)

        return beam_dose.astype(np.float32)

    def get_dvh_summary(
        self, dose: np.ndarray, patient: PatientPhantom
    ) -> Dict[str, float]:
        """
        Compute key DVH metrics (used in info dict and grading).

        Returns dict with:
          - tumor_d95: dose to 95% of tumor volume (should be >= 0.95)
          - tumor_coverage: fraction of tumor with dose >= 95% of prescription
          - oar_<name>_mean: mean dose to each OAR
          - oar_<name>_max:  max dose to each OAR
        """
        summary = {}

        # Tumor metrics
        tumor_dose = dose[patient.tumor_mask]
        if len(tumor_dose) > 0:
            summary["tumor_d95"] = float(np.percentile(tumor_dose, 5))   # D95
            summary["tumor_dmean"] = float(np.mean(tumor_dose))
            summary["tumor_dmax"] = float(np.max(tumor_dose))
            summary["tumor_coverage"] = float(
                np.mean(tumor_dose >= 0.95 * patient.prescription_dose)
            )
        else:
            summary["tumor_d95"] = 0.0
            summary["tumor_dmean"] = 0.0
            summary["tumor_dmax"] = 0.0
            summary["tumor_coverage"] = 0.0

        # OAR metrics
        for oar in patient.oars:
            oar_dose = dose[oar.mask]
            if len(oar_dose) > 0:
                key = oar.name.lower().replace(" ", "_")
                summary[f"oar_{key}_mean"] = float(np.mean(oar_dose))
                summary[f"oar_{key}_max"]  = float(np.max(oar_dose))
                summary[f"oar_{key}_limit"] = float(oar.limit)
                summary[f"oar_{key}_violation"] = float(
                    max(0.0, np.mean(oar_dose) - oar.limit)
                )

        return summary
