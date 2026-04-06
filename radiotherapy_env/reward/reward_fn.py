"""
Reward Function
===============
Computes per-step partial reward and final plan score.

Reward = tumor_coverage * 0.55
       - oar_penalty    * 0.40
       + plan_efficiency* 0.05

Range: [0.0, 1.0] (clipped)

Design principles:
  1. Partial reward every step → agent learns faster (dense reward signal)
  2. Training reward is a smoothed approximation of compute_score — same
     metrics, same weights — so the agent optimises what it is graded on.
  3. OAR penalty for critical organs uses a steep ramp (full penalty at
     10 % above limit) rather than a wide linear slope, matching the
     near-binary behaviour of the grader while preserving gradient signal.
  4. Plan efficiency is minor (5 %) — matching the grader weight.
"""

import numpy as np
from typing import List
from ..physics.phantom import PatientPhantom, Beam


# OAR priority weights for penalty computation
PRIORITY_WEIGHTS = {1: 1.5, 2: 1.0, 3: 0.5}  # critical, important, moderate


def compute_reward(
    dose: np.ndarray,
    patient: PatientPhantom,
    beams: List[Beam],
) -> float:
    """
    Compute per-step training reward in [0.0, 1.0].

    Args:
        dose:    Current dose distribution grid
        patient: Patient phantom with tumor and OARs
        beams:   Current list of active beams

    Returns:
        reward: float in [0.0, 1.0]
    """
    if not beams:
        return 0.0

    # ── 1. Tumor Coverage (weight: 0.55) ─────────────────────────────────────
    # Matches compute_score exactly: 50% D95 + 50% coverage_95pct.
    # D95 = dose received by the coldest 5% of the tumor (np.percentile(…, 5)).
    # Previously the training used mean_dose_ratio instead of D95, so the agent
    # could score well by over-dosing the bulk while leaving cold spots — the
    # grader penalises those cold spots via D95 but training didn't.
    tumor_dose = dose[patient.tumor_mask]
    if len(tumor_dose) == 0:
        tumor_coverage = 0.0
    else:
        d95 = float(np.percentile(tumor_dose, 5))
        coverage_95pct = float(np.mean(tumor_dose >= 0.95 * patient.prescription_dose))
        tumor_coverage = (
            0.5 * min(1.0, d95 / (patient.prescription_dose + 1e-8))
            + 0.5 * coverage_95pct
        )

    # ── 2. OAR Penalty (weight: 0.40) ─────────────────────────────────────────
    oar_penalty = 0.0
    total_weight = 0.0

    for oar in patient.oars:
        oar_dose = dose[oar.mask]
        if len(oar_dose) == 0:
            continue

        w = PRIORITY_WEIGHTS.get(oar.priority, 1.0)
        mean_dose = float(np.mean(oar_dose))
        max_dose  = float(np.max(oar_dose))

        if oar.priority == 1:
            # Critical organs (e.g. brainstem, spinal cord): steep ramp.
            # Grader uses binary pass/fail at the limit; we approximate that
            # with a ramp that reaches 1.0 (full penalty) at only 10% above
            # the limit — sharp enough to deter near-misses, but still
            # differentiable so RL gradient signal survives.
            # Mirror the grader's 50/50 mean+max split.
            mean_violation = min(
                1.0,
                max(0.0, mean_dose - oar.limit) / (0.1 * oar.limit + 1e-8),
            )
            max_violation = min(
                1.0,
                max(0.0, max_dose - oar.limit * 1.05) / (0.1 * oar.limit + 1e-8),
            )
            violation = 0.5 * mean_violation + 0.5 * max_violation
        else:
            # Non-critical organs: linear gradient matching the grader's
            # normalization (reaches 1.0 at 50% above limit, same as grader).
            # Previously used / limit (twice as lenient as the grader used
            # / (limit * 0.5)), so small violations were under-penalised.
            violation = min(
                1.0,
                max(0.0, mean_dose - oar.limit) / (oar.limit * 0.5 + 1e-8),
            )

        oar_penalty += w * violation
        total_weight += w

    if total_weight > 0:
        oar_penalty /= total_weight  # normalize to [0, ~1]
    oar_penalty = min(oar_penalty, 1.0)

    # ── 3. Plan Efficiency (weight: 0.05) ────────────────────────────────────
    # Matches compute_score exactly: beam-count score only.
    # The weight_efficiency sub-term was training-only noise; dropping it
    # removes a discrepancy and halves the weight (10%→5%) to match grader.
    n_beams = len(beams)
    plan_efficiency = max(0.0, 1.0 - abs(n_beams - 6) / 7.0) if n_beams > 0 else 0.0

    # ── Final reward ─────────────────────────────────────────────────────────
    reward = (
        tumor_coverage * 0.55
        - oar_penalty  * 0.40
        + plan_efficiency * 0.05
    )

    return float(np.clip(reward, 0.0, 1.0))


def compute_score(
    dose: np.ndarray,
    patient: PatientPhantom,
    beams: List[Beam],
) -> float:
    """
    Compute FINAL plan quality score [0.0, 1.0] for the auto-grader.

    Uses stricter clinical criteria:
      - Tumor D95 >= 95% prescription (hard requirement)
      - All OAR mean doses within limits
      - Critical OAR max doses within limits

    This is what judges see — not the training reward.
    """
    if not beams:
        return 0.0

    score_components = []

    # ── Tumor coverage score ─────────────────────────────────────────────────
    tumor_dose = dose[patient.tumor_mask]
    if len(tumor_dose) > 0:
        d95 = float(np.percentile(tumor_dose, 5))  # D95
        coverage_95 = float(np.mean(tumor_dose >= 0.95 * patient.prescription_dose))
        tumor_score = 0.5 * min(1.0, d95 / patient.prescription_dose) + 0.5 * coverage_95
    else:
        tumor_score = 0.0
    score_components.append(("tumor", tumor_score, 0.55))

    # ── OAR compliance score ─────────────────────────────────────────────────
    oar_scores = []
    for oar in patient.oars:
        oar_dose = dose[oar.mask]
        if len(oar_dose) == 0:
            oar_scores.append(1.0)
            continue

        mean_dose = float(np.mean(oar_dose))
        max_dose  = float(np.max(oar_dose))

        if oar.priority == 1:
            # Critical: both mean and max must be within limits
            mean_ok = mean_dose <= oar.limit
            max_ok  = max_dose  <= oar.limit * 1.05
            individual_score = (0.5 * float(mean_ok) + 0.5 * float(max_ok))
        else:
            # Non-critical: linear gradient
            individual_score = max(0.0, 1.0 - max(0, mean_dose - oar.limit) / (oar.limit * 0.5))

        oar_scores.append(individual_score)

    mean_oar_score = float(np.mean(oar_scores)) if oar_scores else 1.0
    score_components.append(("oar", mean_oar_score, 0.40))

    # ── Plan efficiency score ─────────────────────────────────────────────────
    n = len(beams)
    eff = max(0.0, 1.0 - abs(n - 6) / 7.0)
    score_components.append(("efficiency", eff, 0.05))

    # Weighted final score
    final = sum(s * w for _, s, w in score_components)
    return float(np.clip(final, 0.0, 1.0))
