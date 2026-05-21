"""
runner_profile.py - Runner personal profile for personalized analysis
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RunnerProfile:
    """Personal information that improves analysis accuracy."""
    height_cm: Optional[float] = None  # 150-220
    weight_kg: Optional[float] = None  # 40-150
    gender: Optional[str] = None  # 'male', 'female'
    age: Optional[int] = None  # 10-100

    @property
    def estimated_torso_cm(self) -> float:
        """
        Estimate torso length from height.
        Torso (hip-to-shoulder) ~ 30% of height for most people.
        Falls back to 50cm if height not provided.
        """
        if self.height_cm:
            return self.height_cm * 0.30
        return 50.0

    @property
    def estimated_leg_length_cm(self) -> Optional[float]:
        """Estimate leg length from height (~45% of height)."""
        if self.height_cm:
            return self.height_cm * 0.45
        return None

    @property
    def cadence_adjustment(self) -> float:
        """
        Cadence expectation adjustment.
        Taller runners naturally have lower cadence, so we're more lenient.
        Returns a factor applied to cadence scoring width.
        """
        if not self.height_cm:
            return 1.0
        # Reference height 170cm → factor 1.0
        # 190cm → 1.15 (15% wider acceptable range)
        # 150cm → 0.85 (tighter)
        return 0.5 + (self.height_cm / 170.0) * 0.5

    @property
    def vertical_oscillation_adjustment(self) -> float:
        """
        Vertical oscillation expectation adjustment.
        Taller runners naturally have slightly more vertical oscillation.
        """
        if not self.height_cm:
            return 1.0
        return self.height_cm / 170.0

    @property
    def vo_max_estimate(self) -> Optional[float]:
        """
        Rough VO2 max estimate based on age and gender (non-athlete baseline).
        Used for context, not for precise measurement.
        """
        if not self.age:
            return None
        base = 45.0 if self.gender == 'male' else 38.0
        age_factor = 1.0 - (self.age - 20) * 0.005 if self.age >= 20 else 1.0
        return round(base * age_factor, 1)

    def injury_risk_context(self) -> str:
        """Return injury risk context based on profile."""
        risks = []
        if self.gender == 'female':
            risks.append("女性跑者髂胫束综合征风险较男性约高 2 倍，注意髋部力量训练")
        if self.age and self.age > 40:
            risks.append(f"{self.age}岁以上建议加强力量训练和恢复时间")
        if not risks:
            return ""
        return "; ".join(risks)

    def to_dict(self) -> dict:
        return {
            "height_cm": self.height_cm,
            "weight_kg": self.weight_kg,
            "gender": self.gender,
            "age": self.age,
        }

    @staticmethod
    def from_dict(d: dict) -> "RunnerProfile":
        return RunnerProfile(
            height_cm=d.get("height_cm"),
            weight_kg=d.get("weight_kg"),
            gender=d.get("gender"),
            age=d.get("age"),
        )
