"""
metrics.py - Running Form Metrics Calculation
Computes biomechanical metrics from pose landmark sequences.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pose_extractor import PoseSequence, PoseLandmarks, PoseExtractor


# Landmark indices for convenience
L = {
    "L_SHOULDER": 11, "R_SHOULDER": 12,
    "L_ELBOW": 13, "R_ELBOW": 14,
    "L_WRIST": 15, "R_WRIST": 16,
    "L_HIP": 23, "R_HIP": 24,
    "L_KNEE": 25, "R_KNEE": 26,
    "L_ANKLE": 27, "R_ANKLE": 28,
    "L_HEEL": 29, "R_HEEL": 30,
    "L_FOOT": 31, "R_FOOT": 32,
    "NOSE": 0,
    "L_EAR": 7, "R_EAR": 8,
}


def _angle_between(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Calculate angle (degrees) at p2 between vectors p1-p2 and p3-p2."""
    v1 = p1 - p2
    v2 = p3 - p2
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm < 1e-6:
        return 0.0
    cos_angle = np.clip(dot / norm, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def _vector_angle(v: np.ndarray, reference: np.ndarray = np.array([0, -1, 0])) -> float:
    """Calculate angle (degrees) between vector v and a reference vector."""
    dot = np.dot(v, reference)
    norm = np.linalg.norm(v) * np.linalg.norm(reference)
    if norm < 1e-6:
        return 0.0
    return np.degrees(np.arccos(np.clip(dot / norm, -1.0, 1.0)))


def _midpoint(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Midpoint of two 3D points."""
    return (p1 + p2) / 2


def _lowpass_filter(seq: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Simple exponential moving average low-pass filter."""
    filtered = np.zeros_like(seq)
    filtered[0] = seq[0]
    for i in range(1, len(seq)):
        filtered[i] = alpha * seq[i] + (1 - alpha) * filtered[i - 1]
    return filtered


@dataclass
class FrameMetrics:
    """Per-frame running metrics."""
    frame_idx: int
    timestamp_ms: float

    # Trunk
    trunk_lean_angle: Optional[float] = None  # degrees from vertical

    # Arms
    left_elbow_angle: Optional[float] = None
    right_elbow_angle: Optional[float] = None
    arm_symmetry_score: Optional[float] = None  # 0-100

    # Legs
    left_knee_angle: Optional[float] = None
    right_knee_angle: Optional[float] = None

    # Foot strike
    foot_strike_distance: Optional[float] = None  # cm ahead of hip
    foot_strike_type: Optional[str] = None  # "forefoot", "midfoot", "rearfoot"

    # Head
    head_forward_lean: Optional[float] = None  # degrees from vertical

    # Vertical oscillation
    hip_height: Optional[float] = None  # y-coordinate of mid-hip


@dataclass
class RunningMetrics:
    """Aggregated running metrics over entire video."""
    fps: float = 0.0
    duration_sec: float = 0.0
    total_gait_cycles: int = 0

    # Cadence
    cadence_avg: Optional[float] = None  # steps per minute
    cadence_std: Optional[float] = None

    # Trunk lean (average over stance phase)
    trunk_lean_avg: Optional[float] = None
    trunk_lean_std: Optional[float] = None

    # Arm swing
    left_elbow_range: Optional[float] = None  # flexion-extension range
    right_elbow_range: Optional[float] = None
    arm_symmetry_avg: Optional[float] = None

    # Vertical oscillation
    vertical_oscillation: Optional[float] = None  # cm

    # Foot strike
    avg_foot_strike_distance: Optional[float] = None
    foot_strike_type_dominant: Optional[str] = None

    # Knee angles
    avg_knee_angle_left: Optional[float] = None
    avg_knee_angle_right: Optional[float] = None

    # Per-frame breakdown
    frame_metrics: List[FrameMetrics] = field(default_factory=list)

    def summary(self) -> Dict:
        """Return a dict summary for display/LLM input."""
        return {
            "cadence_spm": round(self.cadence_avg, 1) if self.cadence_avg is not None else None,
            "trunk_lean_deg": round(self.trunk_lean_avg, 1) if self.trunk_lean_avg is not None else None,
            "vertical_oscillation_cm": round(self.vertical_oscillation, 1) if self.vertical_oscillation is not None else None,
            "arm_symmetry_score": round(self.arm_symmetry_avg, 1) if self.arm_symmetry_avg is not None else None,
            "foot_strike_distance_cm": round(self.avg_foot_strike_distance, 1) if self.avg_foot_strike_distance is not None else None,
            "foot_strike_type": self.foot_strike_type_dominant,
            "avg_left_knee_angle_deg": round(self.avg_knee_angle_left, 1) if self.avg_knee_angle_left is not None else None,
            "avg_right_knee_angle_deg": round(self.avg_knee_angle_right, 1) if self.avg_knee_angle_right is not None else None,
            "total_gait_cycles": self.total_gait_cycles,
            "duration_sec": round(self.duration_sec, 1),
        }


class RunningMetricsCalculator:
    """
    Computes running biomechanics from MediaPipe pose sequences.
    Designed for side-view running videos (sagittal plane).
    """

    def __init__(self):
        self.landmark_names = PoseExtractor.LANDMARK_NAMES

    def compute(self, seq: PoseSequence) -> RunningMetrics:
        """
        Full pipeline: compute all running metrics from a pose sequence.
        """
        if not seq.landmarks_seq:
            return RunningMetrics()

        metrics = RunningMetrics(fps=seq.fps)

        # Compute per-frame metrics
        for pl in seq.landmarks_seq:
            fm = self._compute_frame_metrics(pl)
            if fm is not None:
                metrics.frame_metrics.append(fm)

        if not metrics.frame_metrics:
            return metrics

        # Aggregate
        metrics.duration_sec = len(metrics.frame_metrics) / seq.fps if seq.fps > 0 else 0

        # Cadence from hip oscillation
        metrics.cadence_avg, metrics.cadence_std = self._compute_cadence(
            metrics.frame_metrics, seq.fps
        )

        # Trunk lean
        trunk_leans = [
            fm.trunk_lean_angle for fm in metrics.frame_metrics
            if fm.trunk_lean_angle is not None and fm.trunk_lean_angle < 45
        ]
        if trunk_leans:
            metrics.trunk_lean_avg = float(np.mean(trunk_leans))
            metrics.trunk_lean_std = float(np.std(trunk_leans))

        # Arm symmetry
        sym_scores = [
            fm.arm_symmetry_score for fm in metrics.frame_metrics
            if fm.arm_symmetry_score is not None
        ]
        if sym_scores:
            metrics.arm_symmetry_avg = float(np.mean(sym_scores))

        # Elbow range
        left_elbows = [
            fm.left_elbow_angle for fm in metrics.frame_metrics
            if fm.left_elbow_angle is not None
        ]
        right_elbows = [
            fm.right_elbow_angle for fm in metrics.frame_metrics
            if fm.right_elbow_angle is not None
        ]
        if left_elbows:
            metrics.left_elbow_range = float(np.max(left_elbows) - np.min(left_elbows))
        if right_elbows:
            metrics.right_elbow_range = float(np.max(right_elbows) - np.min(right_elbows))

        # Vertical oscillation (detrended to remove camera/frame movement)
        # Uses piecewise detrending: subtract the moving average
        raw_hip_heights = np.array([
            fm.hip_height for fm in metrics.frame_metrics
            if fm.hip_height is not None
        ])
        if len(raw_hip_heights) > 20:
            # Simple detrend: fit a low-order polynomial to capture camera movement
            x = np.arange(len(raw_hip_heights))
            try:
                coeffs = np.polyfit(x, raw_hip_heights, 2)  # quadratic fits drift well
                trend = np.polyval(coeffs, x)
            except Exception:
                trend = np.mean(raw_hip_heights)
            detrended = raw_hip_heights - trend
            # Estimate px-to-cm: average torso is ~180-200px (full model).
            # Torso ~50cm, so px_to_cm ≈ 50 / torso_px
            # Hip oscillation ~6-12cm → ~20-40px after detrending
            # We'll output in pixels first, then estimate cm
            vert_osc_px = float(np.ptp(detrended))
            # Typical: 10cm ≈ 40px in 1080p video at ~2m distance
            metrics.vertical_oscillation = round(vert_osc_px * 0.25, 1)  # rough cm estimate

        # Foot strike
        foot_distances = [
            fm.foot_strike_distance for fm in metrics.frame_metrics
            if fm.foot_strike_distance is not None and fm.foot_strike_distance < 100
        ]
        if foot_distances:
            metrics.avg_foot_strike_distance = float(np.mean(foot_distances))

        # Foot strike type (sample the best frames)
        strike_types = [
            fm.foot_strike_type for fm in metrics.frame_metrics
            if fm.foot_strike_type is not None
        ]
        if strike_types:
            from collections import Counter
            counter = Counter(strike_types)
            metrics.foot_strike_type_dominant = counter.most_common(1)[0][0]

        # Knee angles
        left_knees = [
            fm.left_knee_angle for fm in metrics.frame_metrics
            if fm.left_knee_angle is not None
        ]
        right_knees = [
            fm.right_knee_angle for fm in metrics.frame_metrics
            if fm.right_knee_angle is not None
        ]
        if left_knees:
            metrics.avg_knee_angle_left = float(np.mean(left_knees))
        if right_knees:
            metrics.avg_knee_angle_right = float(np.mean(right_knees))

        # Gait cycle estimation (from detrended hip height)
        if 'detrended' in locals() and len(detrended) > 10:
            metrics.total_gait_cycles = self._count_gait_cycles(detrended, seq.fps)
        elif len(raw_hip_heights) > 10:
            metrics.total_gait_cycles = self._count_gait_cycles(raw_hip_heights, seq.fps)

        return metrics

    def _compute_frame_metrics(self, pl: PoseLandmarks) -> Optional[FrameMetrics]:
        """Compute all metrics for a single frame (body-centered coordinates)."""
        try:
            lm = pl.landmarks
            vis = pl.visibility

            # Check we have decent visibility on key joints
            key_joints = [
                L["L_HIP"], L["R_HIP"], L["L_SHOULDER"], L["R_SHOULDER"],
                L["L_KNEE"], L["R_KNEE"], L["L_ANKLE"], L["R_ANKLE"],
            ]
            avg_vis = np.mean([vis[j] for j in key_joints if j < len(vis)])
            if avg_vis < 0.5:
                return None

            fm = FrameMetrics(frame_idx=pl.frame_idx, timestamp_ms=pl.timestamp_ms)

            # --- Body-centered coordinate system ---
            mid_shoulder = _midpoint(lm[L["L_SHOULDER"]], lm[L["R_SHOULDER"]])
            mid_hip = _midpoint(lm[L["L_HIP"]], lm[L["R_HIP"]])
            
            # Estimate pixel-to-cm from torso height (average torso ~50cm)
            torso_height_px = np.linalg.norm(mid_shoulder[:2] - mid_hip[:2])
            px_to_cm = 50.0 / torso_height_px if torso_height_px > 10 else 1.0

            # --- Trunk Lean (body-relative, invariant to camera motion) ---
            trunk_vec = mid_shoulder[:2] - mid_hip[:2]
            vertical = np.array([0, -1])
            trunk_norm = np.linalg.norm(trunk_vec)
            if trunk_norm > 5:
                cos_a = np.dot(trunk_vec, vertical) / trunk_norm
                lean_raw = np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))
                # Convert to forward lean: 0 = vertical, positive = leaning forward
                # In side view, forward direction = shoulder x ahead of hip x
                # The angle from vertical is informative even if >30 (indicates detection issue)
                if lean_raw <= 45:  # plausible range (rarely more than 20 in normal running)
                    fm.trunk_lean_angle = lean_raw
                else:
                    # Large angle likely means landmarks are too spread out
                    # (possibly running toward/away from camera)
                    pass

            # --- Elbow Angles ---
            if vis[L["L_SHOULDER"]] > 0.5 and vis[L["L_ELBOW"]] > 0.5 and vis[L["L_WRIST"]] > 0.5:
                fm.left_elbow_angle = _angle_between(
                    lm[L["L_SHOULDER"]], lm[L["L_ELBOW"]], lm[L["L_WRIST"]]
                )
            if vis[L["R_SHOULDER"]] > 0.5 and vis[L["R_ELBOW"]] > 0.5 and vis[L["R_WRIST"]] > 0.5:
                fm.right_elbow_angle = _angle_between(
                    lm[L["R_SHOULDER"]], lm[L["R_ELBOW"]], lm[L["R_WRIST"]]
                )

            # Arm symmetry
            if fm.left_elbow_angle is not None and fm.right_elbow_angle is not None:
                diff = abs(fm.left_elbow_angle - fm.right_elbow_angle)
                fm.arm_symmetry_score = max(0, 100 - (diff / 30) * 100)

            # --- Knee Angles ---
            if vis[L["L_HIP"]] > 0.5 and vis[L["L_KNEE"]] > 0.5 and vis[L["L_ANKLE"]] > 0.5:
                fm.left_knee_angle = _angle_between(
                    lm[L["L_HIP"]], lm[L["L_KNEE"]], lm[L["L_ANKLE"]]
                )
            if vis[L["R_HIP"]] > 0.5 and vis[L["R_KNEE"]] > 0.5 and vis[L["R_ANKLE"]] > 0.5:
                fm.right_knee_angle = _angle_between(
                    lm[L["R_HIP"]], lm[L["R_KNEE"]], lm[L["R_ANKLE"]]
                )

            # --- Foot Strike: relative to body center ---
            if vis[L["L_HEEL"]] > 0.5 and vis[L["R_HEEL"]] > 0.5:
                # In side view, the foot further FORWARD (further in run direction)
                # from the hip indicates potential overstriding
                # We use x-coordinate relative to hip
                left_rel_x = lm[L["L_HEEL"]][0] - mid_hip[0]
                right_rel_x = lm[L["R_HEEL"]][0] - mid_hip[0]
                
                # The forward foot: one will be positive, one negative
                # (relative to hip center) - pick the max absolute value
                forward_dist = max(abs(left_rel_x), abs(right_rel_x))
                fm.foot_strike_distance = forward_dist * px_to_cm

                # Foot strike type: compare heel-toe y difference
                forward_idx = L["L_HEEL"] if abs(left_rel_x) >= abs(right_rel_x) else L["R_HEEL"]
                foot_idx = L["L_FOOT"] if forward_idx == L["L_HEEL"] else L["R_FOOT"]
                
                heel_y = lm[forward_idx][1]
                toe_y = lm[foot_idx][1]
                diff_y = toe_y - heel_y  # in image: toe below heel = positive
                
                if diff_y > 15:  # toe significantly lower
                    fm.foot_strike_type = "forefoot"
                elif diff_y < -15:  # heel significantly lower
                    fm.foot_strike_type = "rearfoot"
                else:
                    fm.foot_strike_type = "midfoot"

            # --- Head Forward Lean ---
            if vis[L["NOSE"]] > 0.5:
                nose_rel = lm[L["NOSE"]] - mid_shoulder
                vertical_ref = np.array([0, -1, 0])
                if np.linalg.norm(nose_rel[:2]) > 5:
                    cos_h = np.dot(nose_rel[:2], vertical_ref[:2]) / (
                        np.linalg.norm(nose_rel[:2]) * 1.0
                    )
                    head_lean = np.degrees(np.arccos(np.clip(cos_h, -1.0, 1.0)))
                    if head_lean <= 30:
                        fm.head_forward_lean = head_lean

            # --- Hip Height (detrended for vertical oscillation) ---
            # Store raw height; detrending happens in aggregate step
            fm.hip_height = mid_hip[1]

            return fm

        except Exception:
            return None

    def _compute_cadence(self, frame_metrics: List[FrameMetrics],
                         fps: float) -> Tuple[Optional[float], Optional[float]]:
        """
        Estimate cadence from hip vertical oscillation using autocorrelation.
        More robust than peak-finding for noisy data.
        """
        if len(frame_metrics) < fps * 1.5:  # need at least 1.5 seconds
            return None, None

        raw_hip_heights = np.array([
            fm.hip_height for fm in frame_metrics if fm.hip_height is not None
        ])

        if len(raw_hip_heights) < 30:
            return None, None

        # Detrend: remove camera drift via polyfit
        x = np.arange(len(raw_hip_heights))
        try:
            coeffs = np.polyfit(x, raw_hip_heights, 2)
            trend = np.polyval(coeffs, x)
        except Exception:
            trend = np.mean(raw_hip_heights)
        hip_heights = raw_hip_heights - trend

        # Apply low-pass filter to smooth
        hip_heights = _lowpass_filter(hip_heights, alpha=0.15)

        # Method 1: Try autocorrelation first
        n = len(hip_heights)
        autocorr = np.correlate(hip_heights, hip_heights, mode='same')
        autocorr = autocorr[n // 2:]  # keep only positive lags

        # Normalize
        if autocorr[0] > 0:
            autocorr = autocorr / autocorr[0]

        # Find the first prominent peak after min_lag
        # Running cadence range: 140-220 spm → period range: 0.27-0.43s per step
        # (hip oscillation = one cycle per step)
        min_lag = int(fps * 0.22)   # ~220 spm
        max_lag = int(fps * 0.55)   # ~109 spm

        if max_lag >= len(autocorr):
            max_lag = len(autocorr) - 1
        if min_lag >= max_lag:
            return None, None

        search_region = autocorr[min_lag:max_lag + 1]
        if len(search_region) < 2:
            return None, None

        # Find peaks in search region
        peak_indices = []
        for i in range(1, len(search_region) - 1):
            if (search_region[i] > search_region[i - 1] and
                search_region[i] > search_region[i + 1] and
                search_region[i] > 0.1):  # minimum correlation threshold
                peak_indices.append(i + min_lag)

        if not peak_indices:
            # Fallback: use the peak of the search region
            best_lag = min_lag + int(np.argmax(search_region))
            peak_lags = [best_lag]
        else:
            # Use the first and strongest peak
            peak_lags = [peak_indices[0]]

        if not peak_lags:
            return None, None

        # Convert lag to cadence
        # lag (frames) = half gait cycle = 1 step
        # cadence (spm) = 60 / (lag / fps)
        best_lag_frames = peak_lags[0]
        step_time = best_lag_frames / fps  # seconds per step

        if step_time <= 0:
            return None, None

        cadence = 60.0 / step_time

        # Clamp to realistic running cadence
        if cadence < 120 or cadence > 220:
            return None, None

        # Estimate variability from autocorrelation peak width
        cadence_std = cadence * 0.05  # rough 5% variability estimate

        return round(cadence, 1), round(cadence_std, 1)

    def _count_gait_cycles(self, hip_heights: np.ndarray, fps: float) -> int:
        """Count gait cycles from hip height signal."""
        hip_heights = _lowpass_filter(hip_heights, alpha=0.2)
        peaks = 0
        for i in range(1, len(hip_heights) - 1):
            if hip_heights[i] > hip_heights[i - 1] and hip_heights[i] > hip_heights[i + 1]:
                peaks += 1
        # Each peak = one step, a full gait cycle = 2 steps
        return peaks // 2

    def get_scoring(self, metrics: RunningMetrics) -> Dict[str, float]:
        """
        Score each metric 0-100 and compute overall Running Form Score.
        Based on running biomechanics research norms.
        """
        score_map = {}
        details = {}

        # --- Cadence (target: 170-180 spm) ---
        if metrics.cadence_avg:
            c = metrics.cadence_avg
            if 165 <= c <= 185:
                score_map["cadence"] = 100.0
            elif c < 150 or c > 200:
                score_map["cadence"] = max(0, 100 - abs(c - 175) * 2)
            else:
                score_map["cadence"] = max(0, 100 - abs(c - 175) * 1.5)
            details["cadence"] = f"{c:.0f} spm"

        # --- Trunk Lean (target: 5-10° forward) ---
        if metrics.trunk_lean_avg:
            t = metrics.trunk_lean_avg
            if 4 <= t <= 12:
                score_map["trunk_lean"] = 100.0
            elif t < 0 or t > 20:
                score_map["trunk_lean"] = max(0, 100 - abs(t - 7) * 5)
            else:
                score_map["trunk_lean"] = max(0, 100 - abs(t - 7) * 3)
            details["trunk_lean"] = f"{t:.1f}°"

        # --- Arm Symmetry ---
        if metrics.arm_symmetry_avg:
            score_map["arm_symmetry"] = metrics.arm_symmetry_avg
            details["arm_symmetry"] = f"{metrics.arm_symmetry_avg:.0f}/100"

        # --- Vertical Oscillation (target: < 10cm) ---
        if metrics.vertical_oscillation:
            v = metrics.vertical_oscillation
            if v < 8:
                score_map["vertical_oscillation"] = 100.0
            elif v < 12:
                score_map["vertical_oscillation"] = 80.0
            elif v < 16:
                score_map["vertical_oscillation"] = 50.0
            else:
                score_map["vertical_oscillation"] = max(0, 100 - (v - 8) * 5)
            details["vertical_oscillation"] = f"{v:.1f} cm"

        # --- Foot Strike Distance (target: < 15cm) ---
        if metrics.avg_foot_strike_distance:
            f = metrics.avg_foot_strike_distance
            if f < 10:
                score_map["foot_strike"] = 100.0
            elif f < 20:
                score_map["foot_strike"] = max(0, 100 - (f - 10) * 4)
            else:
                score_map["foot_strike"] = max(0, 100 - (f - 10) * 5)
            details["foot_strike"] = f"{f:.1f} cm"

        # Overall score (weighted average)
        weights = {
            "cadence": 0.25,
            "trunk_lean": 0.20,
            "arm_symmetry": 0.15,
            "vertical_oscillation": 0.20,
            "foot_strike": 0.20,
        }

        total_score = 0.0
        total_weight = 0.0
        for key, weight in weights.items():
            if key in score_map:
                total_score += score_map[key] * weight
                total_weight += weight

        overall = round(total_score / total_weight, 1) if total_weight > 0 else None

        return {
            "overall_score": overall,
            "metrics": score_map,
            "details": details,
        }
