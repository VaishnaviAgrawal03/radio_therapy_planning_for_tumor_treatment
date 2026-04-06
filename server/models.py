"""
OpenEnv typed models for RadiotherapyPlanningEnv.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class RadiotherapyAction(BaseModel):
    """Action for the radiotherapy environment."""
    action: int = Field(
        ge=0, le=7,
        description="Action index: 0=add beam, 1=rotate +10, 2=rotate -10, "
        "3=increase dose, 4=decrease dose, 5=remove beam, 6=fine-tune, 7=lock plan"
    )


class RadiotherapyObservation(BaseModel):
    """Observation from the radiotherapy environment."""
    dvh_tumor: List[float] = Field(description="Cumulative DVH for tumor (50 bins)")
    dvh_oar: List[List[float]] = Field(description="DVH for top 3 OARs (3x50)")
    beams: List[List[float]] = Field(description="Beam config [angle, weight, active] (7x3)")
    constraints: List[float] = Field(description="Constraint violations [tumor, oar1, oar2, oar3]")
    step_frac: List[float] = Field(description="Episode progress fraction")
    score: float = Field(default=0.0, description="Current plan quality score")
    n_beams: int = Field(default=0, description="Number of active beams")
    task: str = Field(default="prostate", description="Current task name")
