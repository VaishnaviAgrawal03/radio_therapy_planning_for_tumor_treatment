"""
Patient Phantom
===============
Defines the 3D patient model as a 2D grid (coronal slice) with:
  - Tumor (target volume)
  - Organs-at-risk (OARs): spinal cord, heart, lungs, etc.

All coordinates are in grid units (0 to GRID_SIZE-1).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
import numpy as np

# Grid resolution used by all patient generators
_GRID_SIZE = 64


@dataclass
class Beam:
    """Represents a single radiation beam."""
    angle: float        # degrees [0, 180)
    dose_weight: float  # relative weight [0.1, 1.0]

    def to_dict(self) -> Dict[str, float]:
        return {"angle": round(self.angle, 2), "dose_weight": round(self.dose_weight, 3)}


@dataclass
class OAR:
    """Organ-at-Risk structure."""
    name: str
    mask: np.ndarray     # boolean grid mask
    limit: float         # dose limit in Gy (normalized)
    priority: int        # 1=critical, 2=important, 3=moderate

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "limit": self.limit,
            "priority": self.priority,
            "voxels": int(np.sum(self.mask)),
        }


@dataclass
class PatientPhantom:
    """
    Complete patient model.
    All dose values are in Gy (normalized so prescription = 1.0).
    """
    case_id: str
    grid_size: int
    tumor_mask: np.ndarray       # boolean grid
    oars: List[OAR]
    prescription_dose: float     # normalized to 1.0
    body_mask: np.ndarray        # patient body outline

    # Metadata
    tumor_center: Tuple[float, float] = (32, 32)
    tumor_radius: float = 8.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "grid_size": self.grid_size,
            "prescription_dose": self.prescription_dose,
            "tumor_voxels": int(np.sum(self.tumor_mask)),
            "tumor_center": self.tumor_center,
            "oars": [oar.to_dict() for oar in self.oars],
        }


def _make_circular_mask(grid_size: int, cx: float, cy: float, r: float) -> np.ndarray:
    """Create a boolean circular mask."""
    y, x = np.ogrid[:grid_size, :grid_size]
    return ((x - cx) ** 2 + (y - cy) ** 2) <= r ** 2


def _make_elliptical_mask(
    grid_size: int, cx: float, cy: float, rx: float, ry: float, angle_deg: float = 0
) -> np.ndarray:
    """Create a boolean elliptical mask with optional rotation."""
    y, x = np.ogrid[:grid_size, :grid_size]
    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    dx = (x - cx) * cos_a + (y - cy) * sin_a
    dy = -(x - cx) * sin_a + (y - cy) * cos_a
    return (dx / rx) ** 2 + (dy / ry) ** 2 <= 1.0


def _make_rect_mask(
    grid_size: int, x1: int, y1: int, x2: int, y2: int
) -> np.ndarray:
    """Create a boolean rectangular mask."""
    mask = np.zeros((grid_size, grid_size), dtype=bool)
    mask[max(0, y1):min(grid_size, y2), max(0, x1):min(grid_size, x2)] = True
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Task-specific patient generators
# ─────────────────────────────────────────────────────────────────────────────

class ProstatePatientGenerator:
    """
    Task 1 (Easy): Prostate cancer.
    Simple spherical tumor in pelvis, 2 OARs: rectum and bladder.
    """

    def generate(self, rng: np.random.Generator) -> PatientPhantom:
        G = _GRID_SIZE

        # Tumor: prostate gland (center of pelvis)
        cx = rng.uniform(28, 36)
        cy = rng.uniform(28, 36)
        r  = rng.uniform(5, 8)
        tumor_mask = _make_circular_mask(G, cx, cy, r)

        # Body outline
        body_mask = _make_circular_mask(G, G // 2, G // 2, G // 2 - 4)

        # Rectum: posterior to prostate
        rect_cx = cx + rng.uniform(6, 10)
        rect_mask = _make_elliptical_mask(G, rect_cx, cy, 4, 6)

        # Bladder: anterior to prostate
        blad_cx = cx - rng.uniform(6, 10)
        blad_mask = _make_circular_mask(G, blad_cx, cy, rng.uniform(4, 7))

        # Ensure OARs don't overlap tumor
        rect_mask &= ~tumor_mask
        blad_mask &= ~tumor_mask

        oars = [
            OAR("Rectum",  rect_mask, limit=0.40, priority=1),
            OAR("Bladder", blad_mask, limit=0.50, priority=2),
        ]

        return PatientPhantom(
            case_id=f"prostate_{int(rng.integers(1000, 9999))}",
            grid_size=G,
            tumor_mask=tumor_mask,
            oars=oars,
            prescription_dose=1.0,
            body_mask=body_mask,
            tumor_center=(cx, cy),
            tumor_radius=r,
        )


class HeadNeckPatientGenerator:
    """
    Task 2 (Medium): Head & neck cancer.
    Irregular tumor, 7 critical OARs: spinal cord, both parotids,
    mandible, larynx, brainstem, esophagus.
    """

    def generate(self, rng: np.random.Generator) -> PatientPhantom:
        G = _GRID_SIZE

        # Tumor: oropharynx (off-center, irregular)
        cx = rng.uniform(26, 38)
        cy = rng.uniform(22, 34)
        rx = rng.uniform(5, 9)
        ry = rng.uniform(4, 7)
        angle = rng.uniform(0, 45)
        tumor_mask = _make_elliptical_mask(G, cx, cy, rx, ry, angle)

        body_mask = _make_elliptical_mask(G, G // 2, G // 2 + 4, G // 2 - 4, G // 2 - 3)

        # Spinal cord: posterior, narrow strip
        spine_mask = _make_rect_mask(G, int(G * 0.70), 10, int(G * 0.76), G - 10)

        # Brainstem: superior-posterior
        bs_cx, bs_cy = G // 2, int(G * 0.20)
        brainstem_mask = _make_elliptical_mask(G, bs_cx, bs_cy, 4, 6)

        # Right parotid
        rp_mask = _make_circular_mask(G, cx - rng.uniform(10, 14), cy - rng.uniform(2, 5), 5)

        # Left parotid
        lp_mask = _make_circular_mask(G, cx + rng.uniform(10, 14), cy - rng.uniform(2, 5), 5)

        # Mandible: anterior curve
        mand_mask = _make_elliptical_mask(G, G // 2, cy + rng.uniform(10, 14), 14, 4)

        # Larynx: inferior
        lar_mask = _make_circular_mask(G, G // 2, cy + rng.uniform(13, 17), 4)

        # Esophagus: narrow posterior-inferior
        eso_mask = _make_rect_mask(G, int(G * 0.47), int(cy + 12), int(G * 0.53), G - 8)

        # Clean overlaps
        for m in [spine_mask, brainstem_mask, rp_mask, lp_mask, mand_mask, lar_mask, eso_mask]:
            m &= ~tumor_mask

        oars = [
            OAR("Spinal cord",   spine_mask,    limit=0.45, priority=1),
            OAR("Brainstem",     brainstem_mask, limit=0.45, priority=1),
            OAR("Right parotid", rp_mask,        limit=0.26, priority=2),
            OAR("Left parotid",  lp_mask,        limit=0.26, priority=2),
            OAR("Mandible",      mand_mask,      limit=0.60, priority=3),
            OAR("Larynx",        lar_mask,       limit=0.40, priority=2),
            OAR("Esophagus",     eso_mask,       limit=0.34, priority=2),
        ]

        return PatientPhantom(
            case_id=f"head_neck_{int(rng.integers(1000, 9999))}",
            grid_size=G,
            tumor_mask=tumor_mask,
            oars=oars,
            prescription_dose=1.0,
            body_mask=body_mask,
            tumor_center=(cx, cy),
            tumor_radius=(rx + ry) / 2,
        )


class PediatricBrainPatientGenerator:
    """
    Task 3 (Hard): Pediatric brain tumor.
    Tumor is 2-3mm from critical brainstem. Maximum precision required.
    Any overdose = catastrophic penalty.
    """

    def generate(self, rng: np.random.Generator) -> PatientPhantom:
        G = _GRID_SIZE

        # Brainstem: center of the challenge
        bs_cx = G // 2 + rng.uniform(-3, 3)
        bs_cy = G // 2 + rng.uniform(-2, 4)
        bs_rx, bs_ry = 5.0, 8.0
        brainstem_mask = _make_elliptical_mask(G, bs_cx, bs_cy, bs_rx, bs_ry)

        # Tumor: adjacent to brainstem (very close!)
        offset_angle = rng.uniform(0, 360)
        offset_r = bs_rx + rng.uniform(1.5, 3.0)  # just 2-3 voxels away
        t_cx = bs_cx + offset_r * np.cos(np.deg2rad(offset_angle))
        t_cy = bs_cy + offset_r * np.sin(np.deg2rad(offset_angle))
        t_r = rng.uniform(3, 5)
        tumor_mask = _make_circular_mask(G, t_cx, t_cy, t_r)
        tumor_mask &= ~brainstem_mask  # ensure no overlap

        # Optic chiasm: critical visual structure
        oc_mask = _make_circular_mask(G, G // 2, bs_cy - rng.uniform(8, 12), 3)

        # Right cochlea
        rc_mask = _make_circular_mask(G, bs_cx - rng.uniform(8, 12), bs_cy, 2)

        # Left cochlea
        lc_mask = _make_circular_mask(G, bs_cx + rng.uniform(8, 12), bs_cy, 2)

        # Whole brain (low dose constraint)
        brain_mask = _make_circular_mask(G, G // 2, G // 2, G // 2 - 6)
        brain_mask &= ~tumor_mask

        body_mask = _make_circular_mask(G, G // 2, G // 2, G // 2 - 3)

        # Clean overlaps
        for m in [oc_mask, rc_mask, lc_mask]:
            m &= ~tumor_mask
            m &= ~brainstem_mask

        oars = [
            OAR("Brainstem",    brainstem_mask, limit=0.30, priority=1),
            OAR("Optic chiasm", oc_mask,        limit=0.25, priority=1),
            OAR("Right cochlea",rc_mask,         limit=0.20, priority=2),
            OAR("Left cochlea", lc_mask,         limit=0.20, priority=2),
            OAR("Whole brain",  brain_mask,      limit=0.60, priority=3),
        ]

        return PatientPhantom(
            case_id=f"pediatric_brain_{int(rng.integers(1000, 9999))}",
            grid_size=G,
            tumor_mask=tumor_mask,
            oars=oars,
            prescription_dose=1.0,
            body_mask=body_mask,
            tumor_center=(t_cx, t_cy),
            tumor_radius=t_r,
        )
