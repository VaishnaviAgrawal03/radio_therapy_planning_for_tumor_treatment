"""
Dose Heatmap Renderer
=====================
Renders the current treatment plan as an RGB image showing:
  - Patient body outline
  - Dose distribution (color heatmap)
  - Tumor contour (green)
  - OAR contours (orange/red by priority)
  - Beam directions (white arrows)
  - Metrics overlay
"""

import numpy as np
from typing import List, Optional

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    import io
    from PIL import Image as _PILImage
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..physics.phantom import PatientPhantom, Beam


def render_heatmap(
    dose: np.ndarray,
    patient: "PatientPhantom",
    beams: "List[Beam]",
    reward: float = 0.0,
    step: int = 0,
    size: int = 512,
) -> np.ndarray:
    """
    Render a full-color treatment plan visualization.

    Returns:
        RGB image as (H, W, 3) uint8 numpy array
    """
    if not _MATPLOTLIB_AVAILABLE:
        return _simple_render(dose, patient, size)

    G = dose.shape[0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), facecolor="#0d1117")

    # ── Left: Dose heatmap ────────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#0d1117")

    # Custom dose colormap: dark blue → cyan → yellow → red
    colors = ["#060a10", "#0a3d6b", "#1a7abf", "#00d4aa", "#f5c842", "#e84343", "#ffffff"]
    cmap = LinearSegmentedColormap.from_list("dose", colors, N=256)

    if dose.max() > 0:
        im = ax.imshow(dose, cmap=cmap, vmin=0, vmax=min(dose.max(), 2.0),
                       origin="upper", interpolation="bilinear")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="Dose (normalized)", 
                     ).ax.tick_params(colors="white")
    else:
        ax.imshow(np.zeros_like(dose), cmap=cmap, vmin=0, vmax=1, origin="upper")

    # Body outline
    body_contour = _get_contour(patient.body_mask.astype(np.uint8))
    for contour in body_contour:
        ax.plot(contour[:, 1], contour[:, 0], "w-", linewidth=0.8, alpha=0.4)

    # Tumor contour (bright green)
    tumor_contour = _get_contour(patient.tumor_mask.astype(np.uint8))
    for contour in tumor_contour:
        ax.plot(contour[:, 1], contour[:, 0], color="#00ff88", linewidth=2.0,
                label="Tumor (PTV)", zorder=5)

    # OAR contours
    oar_colors = {1: "#ff4444", 2: "#ffaa00", 3: "#aaaaff"}
    legend_patches = [mpatches.Patch(color="#00ff88", label="Tumor (PTV)")]

    for oar in patient.oars[:4]:  # show up to 4 OARs
        color = oar_colors.get(oar.priority, "#ffffff")
        oar_contour = _get_contour(oar.mask.astype(np.uint8))
        for contour in oar_contour:
            ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=1.5,
                    linestyle="--" if oar.priority > 1 else "-", zorder=4)
        legend_patches.append(mpatches.Patch(color=color, label=oar.name))

    # Beam directions — converge on tumor center (isocenter)
    # This matches dose_calculator which aims beams at tumor_center
    if hasattr(patient, 'tumor_center') and patient.tumor_center is not None:
        iso_x = float(patient.tumor_center[0])
        iso_y = float(patient.tumor_center[1])
    else:
        iso_x, iso_y = G // 2, G // 2

    for beam in beams:
        angle_rad = np.deg2rad(beam.angle)
        r = G // 2 - 2
        dx = r * np.cos(angle_rad)
        dy = r * np.sin(angle_rad)
        ax.annotate("", xy=(iso_x + dx, iso_y + dy), xytext=(iso_x - dx, iso_y - dy),
                    arrowprops=dict(arrowstyle="-", color="white",
                                    alpha=0.3 + 0.5 * beam.dose_weight, lw=1.2))

    # White crosshair at isocenter = where all beams converge = tumor center
    if beams:
        ax.plot(iso_x, iso_y, "+", color="white", markersize=12,
                markeredgewidth=2.0, alpha=0.9, zorder=10)

    ax.legend(handles=legend_patches, loc="upper right", fontsize=7,
              facecolor="#1a1a2e", edgecolor="#333", labelcolor="white")
    ax.set_title(f"Dose Distribution  |  Step {step}  |  Reward: {reward:.3f}",
                 color="white", fontsize=11, pad=10)
    ax.set_xlabel("X (voxels)", color="#aaaaaa", fontsize=9)
    ax.set_ylabel("Y (voxels)", color="#aaaaaa", fontsize=9)
    ax.tick_params(colors="#666666")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    # ── Right: DVH curves ─────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#0d1117")

    dose_range = np.linspace(0, 2.0, 100)

    # Tumor DVH (green)
    tumor_doses = dose[patient.tumor_mask]
    if len(tumor_doses) > 0:
        dvh_tumor = [np.mean(tumor_doses >= d) for d in dose_range]
        ax2.plot(dose_range, dvh_tumor, color="#00ff88", linewidth=2.5,
                 label="Tumor", zorder=5)
        # Prescription line
        ax2.axvline(x=0.95, color="#00ff88", linestyle=":", alpha=0.5, linewidth=1)

    # OAR DVHs
    for oar in patient.oars[:4]:
        oar_doses = dose[oar.mask]
        if len(oar_doses) > 0:
            color = oar_colors.get(oar.priority, "#aaaaaa")
            dvh_oar = [np.mean(oar_doses >= d) for d in dose_range]
            ax2.plot(dose_range, dvh_oar, color=color, linewidth=1.5,
                     linestyle="--" if oar.priority > 1 else "-",
                     label=oar.name, alpha=0.85)
            # Dose limit line
            ax2.axvline(x=oar.limit, color=color, linestyle=":", alpha=0.35, linewidth=1)

    ax2.set_xlim(0, 1.8)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("Dose (normalized)", color="#aaaaaa", fontsize=9)
    ax2.set_ylabel("Volume fraction", color="#aaaaaa", fontsize=9)
    ax2.set_title("Dose-Volume Histogram (DVH)", color="white", fontsize=11, pad=10)
    ax2.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#333", labelcolor="white")
    ax2.grid(True, alpha=0.1, color="#444444")
    ax2.tick_params(colors="#666666")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#333333")

    # Metrics text
    score_text = f"Beams: {len(beams)}"
    ax2.text(0.02, 0.05, score_text, transform=ax2.transAxes,
             color="#888888", fontsize=8)

    plt.tight_layout(pad=1.5)

    # Convert to numpy array
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=96, bbox_inches="tight",
                facecolor="#0d1117")
    buf.seek(0)
    img = _PILImage.open(buf).convert("RGB")
    frame = np.array(img)
    plt.close(fig)
    buf.close()

    return frame


def _get_contour(mask: np.ndarray):
    """Extract contour points from a binary mask using simple edge detection."""
    try:
        from skimage import measure
        contours = measure.find_contours(mask, 0.5)
        return contours
    except ImportError:
        # Fallback: return empty contours
        return []


def _simple_render(dose: np.ndarray, patient, size: int) -> np.ndarray:
    """Minimal fallback renderer (no matplotlib)."""
    G = dose.shape[0]
    # Normalize dose to [0, 255]
    d_norm = (np.clip(dose / (dose.max() + 1e-8), 0, 1) * 255).astype(np.uint8)
    # Create RGB: dose as blue channel, tumor as green, OAR as red
    rgb = np.zeros((G, G, 3), dtype=np.uint8)
    rgb[:, :, 2] = d_norm  # blue = dose
    rgb[patient.tumor_mask, 1] = 200  # green = tumor
    for oar in patient.oars:
        rgb[oar.mask, 0] = 180  # red = OAR
    # Resize to target size (simple repeat)
    scale = size // G
    rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
    return rgb
