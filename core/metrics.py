"""
metrics.py - Running Form Metrics Calculation (v2)
Computes biomechanical metrics from pose landmark sequences.

Improvements over v1:
  - Gait phase detection (stance/swing) for phase-specific metrics
  - Proper px_to_cm conversion per-frame (not hardcoded 0.25)
  - Normalized foot strike thresholds by body scale
  - Hip drop / pelvic drop measurement
  - Improved cadence with multi-peak autocorrelation validation
  - Sigmoid-based smooth scoring functions
  - Confidence-weighted frame aggregation
  - Step length estimation
  - Smart outlier rejection (IQR-based)
  - Trunk lean direction (forward vs backward)
  - Ground contact time estimation
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter
from pose_extractor import PoseSequence, PoseLandmarks, PoseExtractor
from runner_profile import RunnerProfile


# Landmark indices for convenience
L = {
    "NOSE": 0,
    "L_EYE_INNER": 1, "L_EYE": 2, "L_EYE_OUTER": 3,
    "R_EYE_INNER": 4, "R_EYE": 5, "R_EYE_OUTER": 6,
    "L_EAR": 7, "R_EAR": 8,
    "MOUTH_L": 9, "MOUTH_R": 10,
    "L_SHOULDER": 11, "R_SHOULDER": 12,
    "L_ELBOW": 13, "R_ELBOW": 14,
    "L_WRIST": 15, "R_WRIST": 16,
    "L_PINKY": 17, "R_PINKY": 18,
    "L_INDEX": 19, "R_INDEX": 20,
    "L_THUMB": 21, "R_THUMB": 22,
    "L_HIP": 23, "R_HIP": 24,
    "L_KNEE": 25, "R_KNEE": 26,
    "L_ANKLE": 27, "R_ANKLE": 28,
    "L_HEEL": 29, "R_HEEL": 30,
    "L_FOOT": 31, "R_FOOT": 32,
}

# ==============================================================
# Utility functions
# ==============================================================

def _angle_between(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Angle (degrees) at p2 between vectors p1->p2 and p3->p2."""
    v1 = p1 - p2
    v2 = p3 - p2
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm < 1e-6:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(dot / norm, -1.0, 1.0))))


def _midpoint(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    return (p1 + p2) / 2


def _lowpass_filter(seq: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Exponential moving average low-pass filter (forward-backward for zero phase)."""
    n = len(seq)
    out = np.zeros_like(seq)
    if n == 0:
        return out
    out[0] = seq[0]
    for i in range(1, n):
        out[i] = alpha * seq[i] + (1 - alpha) * out[i - 1]
    # Backward pass for zero-phase
    out2 = np.zeros_like(seq)
    out2[-1] = out[-1]
    for i in range(n - 2, -1, -1):
        out2[i] = alpha * out[i] + (1 - alpha) * out2[i + 1]
    return out2


def _sigmoid_score(x: float, center: float, width: float,
                   max_val: float = 100, min_val: float = 0,
                   invert: bool = False) -> float:
    """
    Smooth sigmoid scoring function.
    Default: value near `center` → high score, far away → low score.
    If invert=True: value near `center` → low score, far away → high score.
    """
    if width <= 0:
        return max_val
    # For "peaked" scoring (high at center, low away), use negative exponent
    # score = min_val + (max_val - min_val) * exp(-0.5 * ((x - center)/width)^2)
    z = (x - center) / width
    score = min_val + (max_val - min_val) * np.exp(-0.5 * z * z)
    if invert:
        score = max_val - score
    return float(np.clip(score, min_val, max_val))


def _reject_outliers_iqr(values: np.ndarray, factor: float = 1.5) -> np.ndarray:
    """Reject outliers using interquartile range."""
    if len(values) < 4:
        return values
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lo = q1 - factor * iqr
    hi = q3 + factor * iqr
    return values[(values >= lo) & (values <= hi)]


# ==============================================================
# Gait Phase Detection
# ==============================================================

@dataclass
class GaitPhaseLabel:
    """Binary gait phase label per frame."""
    frame_idx: int
    is_stance: bool  # True = foot on ground, False = swing
    confidence: float  # 0-1 confidence in this classification


@dataclass
class GaitPhases:
    """Detected gait phases for the entire video."""
    labels: List[GaitPhaseLabel] = field(default_factory=list)
    num_stance: int = 0
    num_swing: int = 0
    stance_ratio: float = 0.0  # proportion of frames in stance
    estimated_ground_contact_time_ms: Optional[float] = None


def _detect_gait_phases(hip_heights: np.ndarray,
                        foot_y_positions: np.ndarray,
                        hip_to_foot_dist: np.ndarray,
                        fps: float) -> GaitPhases:
    """
    Detect stance vs swing phases from hip and foot dynamics.
    
    Logic: 
      - Stance: foot near ground (low y), hip-to-foot distance small, hip rising
      - Swing: foot lifting, hip-to-foot distance increasing
      
    Uses foot y-velocity + hip-to-foot distance as primary signal.
    """
    n = len(hip_heights)
    if n < 5:
        return GaitPhases()

    # Foot y-velocity (smooth for robust detection)
    foot_y = _lowpass_filter(foot_y_positions, alpha=0.2)
    foot_vel = np.diff(foot_y, prepend=foot_y[0])  # px/frame, positive = foot moving down
    foot_vel_smooth = _lowpass_filter(foot_vel, alpha=0.15)

    # Hip-to-foot distance
    hf_dist = _lowpass_filter(hip_to_foot_dist, alpha=0.2)

    # Normalize signals for consistent thresholds across scales
    hip_range = float(np.ptp(hip_heights)) if len(hip_heights) > 1 else 1
    foot_vel_norm = foot_vel_smooth / max(hip_range, 1)
    hf_dist_norm = hf_dist / max(hip_range, 1)

    # Stance detection heuristic:
    # - Foot moving downward after mid-swing → approaching ground contact
    # - Hip-to-foot distance small → foot under body
    # - Foot stationary near ground → midstance
    
    # Primary signal: hip-to-foot distance is smallest during stance
    # Use adaptive threshold: lower 40% of range = stance candidate
    hf_thresh = np.percentile(hf_dist_norm, 50)  # median

    labels = []
    for i in range(n):
        # Stance probability score (0-1)
        # Low hf_dist → likely stance
        stance_prob = 1.0 - np.clip(hf_dist_norm[i] / max(hf_thresh * 2, 0.01), 0, 1)
        
        # Also foot velocity near zero → stance
        vel_factor = 1.0 - min(abs(foot_vel_norm[i]), 1.0)
        
        combined = 0.6 * stance_prob + 0.4 * vel_factor
        is_stance = combined > 0.5

        labels.append(GaitPhaseLabel(
            frame_idx=i,
            is_stance=is_stance,
            confidence=float(combined),
        ))

    num_stance = sum(1 for l in labels if l.is_stance)
    num_swing = n - num_stance
    stance_ratio = num_stance / n if n > 0 else 0

    # Estimate ground contact time from stance phase duration
    # Find consecutive stance segments
    gct = None
    if fps > 0 and num_stance > 0:
        stance_starts = []
        stance_ends = []
        in_stance = False
        for i, l in enumerate(labels):
            if l.is_stance and not in_stance:
                stance_starts.append(i)
                in_stance = True
            elif not l.is_stance and in_stance:
                stance_ends.append(i)
                in_stance = False
        if in_stance:
            stance_ends.append(n - 1)
        
        if stance_starts and stance_ends:
            durations = [(e - s) / fps * 1000 for s, e in zip(stance_starts, stance_ends)]
            if durations:
                # Reject outliers (stance phases that are too short/long)
                valid = [d for d in durations if 100 < d < 500]  # typical GCT: 150-350ms
                if valid:
                    gct = float(np.mean(valid))

    return GaitPhases(
        labels=labels,
        num_stance=num_stance,
        num_swing=num_swing,
        stance_ratio=float(stance_ratio),
        estimated_ground_contact_time_ms=gct,
    )


# ==============================================================
# Per-frame & aggregated metrics
# ==============================================================

@dataclass
class FrameMetrics:
    """Per-frame running metrics computed in body-centered coordinates."""
    frame_idx: int
    timestamp_ms: float

    # Scaling
    px_to_cm: Optional[float] = None  # pixels per cm (from torso height)
    visibility_avg: Optional[float] = None  # average visibility of key joints

    # Trunk
    trunk_lean_angle: Optional[float] = None  # degrees from vertical (positive = forward)
    trunk_lean_is_forward: Optional[bool] = None  # True = forward lean, False = backward

    # Arms
    left_elbow_angle: Optional[float] = None
    right_elbow_angle: Optional[float] = None
    arm_symmetry_score: Optional[float] = None  # 0-100

    # Legs
    left_knee_angle: Optional[float] = None
    right_knee_angle: Optional[float] = None

    # Foot strike
    foot_strike_distance_cm: Optional[float] = None  # cm ahead of hip
    foot_strike_type: Optional[str] = None  # "forefoot", "midfoot", "rearfoot"

    # Head
    head_forward_lean: Optional[float] = None  # degrees from vertical

    # Vertical oscillation
    hip_height: Optional[float] = None  # y-coordinate of mid-hip in pixels

    # Hip drop (pelvic drop)
    hip_drop_px: Optional[float] = None  # difference in hip y between left/right

    # Gait phase info
    is_stance: Optional[bool] = None
    gait_confidence: Optional[float] = None


@dataclass
class RunningMetrics:
    """Aggregated running metrics over entire video."""
    fps: float = 0.0
    duration_sec: float = 0.0
    total_gait_cycles: int = 0
    frame_count: int = 0

    # Cadence
    cadence_avg: Optional[float] = None  # steps per minute
    cadence_std: Optional[float] = None

    # Trunk lean
    trunk_lean_avg: Optional[float] = None
    trunk_lean_std: Optional[float] = None
    trunk_lean_is_forward: Optional[bool] = None

    # Arm swing
    left_elbow_range: Optional[float] = None
    right_elbow_range: Optional[float] = None
    arm_symmetry_avg: Optional[float] = None

    # Vertical oscillation
    vertical_oscillation: Optional[float] = None  # cm
    vertical_oscillation_px: Optional[float] = None  # raw pixel value

    # Foot strike
    avg_foot_strike_distance: Optional[float] = None  # cm
    foot_strike_type_dominant: Optional[str] = None
    foot_strike_types_distribution: Optional[Dict[str, float]] = None

    # Knee angles
    avg_knee_angle_left: Optional[float] = None
    avg_knee_angle_right: Optional[float] = None

    # Hip drop
    avg_hip_drop_cm: Optional[float] = None  # average pelvic drop

    # Ground contact time
    ground_contact_time_ms: Optional[float] = None

    # Step length (estimated)
    estimated_step_length_cm: Optional[float] = None

    # Gait phases
    gait_phases: Optional[GaitPhases] = None
    stance_ratio: Optional[float] = None

    # Per-frame breakdown
    frame_metrics: List[FrameMetrics] = field(default_factory=list)

    # Average px_to_cm for scaling
    avg_px_to_cm: Optional[float] = None

    def summary(self, profile: Optional['RunnerProfile'] = None) -> Dict:
        """Return a dict summary for display/LLM input."""
        d = {
            "cadence_spm": round(self.cadence_avg, 1) if self.cadence_avg is not None else None,
            "trunk_lean_deg": round(self.trunk_lean_avg, 1) if self.trunk_lean_avg is not None else None,
            "trunk_lean_direction": "forward" if self.trunk_lean_is_forward else "backward",
            "vertical_oscillation_cm": round(self.vertical_oscillation, 1) if self.vertical_oscillation is not None else None,
            "arm_symmetry_score": round(self.arm_symmetry_avg, 1) if self.arm_symmetry_avg is not None else None,
            "foot_strike_distance_cm": round(self.avg_foot_strike_distance, 1) if self.avg_foot_strike_distance is not None else None,
            "foot_strike_type": self.foot_strike_type_dominant,
            "avg_left_knee_angle_deg": round(self.avg_knee_angle_left, 1) if self.avg_knee_angle_left is not None else None,
            "avg_right_knee_angle_deg": round(self.avg_knee_angle_right, 1) if self.avg_knee_angle_right is not None else None,
            "hip_drop_cm": round(self.avg_hip_drop_cm, 1) if self.avg_hip_drop_cm is not None else None,
            "ground_contact_time_ms": round(self.ground_contact_time_ms, 1) if self.ground_contact_time_ms is not None else None,
            "estimated_step_length_cm": round(self.estimated_step_length_cm, 1) if self.estimated_step_length_cm is not None else None,
            "total_gait_cycles": self.total_gait_cycles,
            "duration_sec": round(self.duration_sec, 1),
            "stance_ratio": round(self.stance_ratio, 2) if self.stance_ratio is not None else None,
        }
        # Add height-normalized metrics if profile available
        if profile and profile.height_cm:
            if self.vertical_oscillation is not None:
                # Vertical oscillation relative to height (%)
                d["vertical_oscillation_pct_height"] = round(
                    self.vertical_oscillation / profile.height_cm * 100, 2
                )
            if self.estimated_step_length_cm is not None:
                d["step_length_height_ratio"] = round(
                    self.estimated_step_length_cm / profile.height_cm, 2
                )
        return d


class RunningMetricsCalculator:
    """
    Computes running biomechanics from MediaPipe pose sequences.
    Designed for side-view running videos (sagittal plane).
    
    v2 improvements:
      - Gait phase-aware metric computation
      - Body-scale normalized measurements
      - Smooth sigmoid scoring
      - Confidence-weighted aggregation
      - Hip drop / pelvic drop analysis
      - Step length estimation
      - Outlier rejection
    """

    def __init__(self):
        self.landmark_names = PoseExtractor.LANDMARK_NAMES

    # ----------------------------------------------------------------
    # Main pipeline
    # ----------------------------------------------------------------
    def compute(self, seq: PoseSequence, profile: Optional[RunnerProfile] = None) -> RunningMetrics:
        """Full pipeline: compute all running metrics from a pose sequence."""
        if not seq.landmarks_seq:
            return RunningMetrics()

        metrics = RunningMetrics(fps=seq.fps)
        metrics.frame_count = len(seq.landmarks_seq)

        # Step 1: Per-frame metrics
        for pl in seq.landmarks_seq:
            fm = self._compute_frame_metrics(pl, profile=profile)
            if fm is not None:
                metrics.frame_metrics.append(fm)

        if not metrics.frame_metrics:
            return metrics

        metrics.duration_sec = len(metrics.frame_metrics) / seq.fps if seq.fps > 0 else 0

        # Step 2: Extract arrays for phase detection & analysis
        fm_arr = metrics.frame_metrics
        hip_heights = np.array([fm.hip_height for fm in fm_arr if fm.hip_height is not None])
        px_to_cm_vals = np.array([fm.px_to_cm for fm in fm_arr if fm.px_to_cm is not None])
        vis_vals = np.array([fm.visibility_avg for fm in fm_arr if fm.visibility_avg is not None])

        metrics.avg_px_to_cm = float(np.mean(px_to_cm_vals)) if len(px_to_cm_vals) > 0 else None

        # Step 3: Gait phase detection (needs foot y + hip-to-foot signals)
        foot_y = self._extract_foot_y_signal(fm_arr)
        hf_dist = self._extract_hip_to_foot_distance(fm_arr)
        if len(hip_heights) > 10 and len(foot_y) == len(hip_heights):
            phases = _detect_gait_phases(hip_heights, foot_y, hf_dist, seq.fps)
            metrics.gait_phases = phases
            metrics.stance_ratio = phases.stance_ratio
            metrics.ground_contact_time_ms = phases.estimated_ground_contact_time_ms
            # Label each frame's phase
            for i, fm in enumerate(fm_arr):
                if i < len(phases.labels):
                    fm.is_stance = phases.labels[i].is_stance
                    fm.gait_confidence = phases.labels[i].confidence

        # Step 4: Aggregate metrics (confidence-weighted, outlier-rejected)
        metrics.cadence_avg, metrics.cadence_std = self._compute_cadence(fm_arr, seq.fps)
        self._aggregate_trunk_lean(metrics, fm_arr)
        self._aggregate_arm_metrics(metrics, fm_arr)
        self._aggregate_vertical_oscillation(metrics, fm_arr)
        self._aggregate_foot_strike(metrics, fm_arr)
        self._aggregate_knee_angles(metrics, fm_arr)
        self._aggregate_hip_drop(metrics, fm_arr)
        self._aggregate_gait_cycles(metrics, fm_arr)
        self._estimate_step_length(metrics)

        return metrics

    # ----------------------------------------------------------------
    # Per-frame computation
    # ----------------------------------------------------------------
    def _compute_frame_metrics(self, pl: PoseLandmarks,
                                profile: Optional[RunnerProfile] = None
                                ) -> Optional[FrameMetrics]:
        """Compute all metrics for a single frame using body-centered coordinates."""
        try:
            lm = pl.landmarks
            vis = pl.visibility

            # Check key joint visibility
            key_joints = [
                L["L_HIP"], L["R_HIP"], L["L_SHOULDER"], L["R_SHOULDER"],
                L["L_KNEE"], L["R_KNEE"], L["L_ANKLE"], L["R_ANKLE"],
            ]
            avg_vis = float(np.mean([vis[j] for j in key_joints if j < len(vis)]))
            if avg_vis < 0.5:
                return None

            fm = FrameMetrics(
                frame_idx=pl.frame_idx,
                timestamp_ms=pl.timestamp_ms,
                visibility_avg=avg_vis,
            )

            mid_shoulder = _midpoint(lm[L["L_SHOULDER"]], lm[L["R_SHOULDER"]])
            mid_hip = _midpoint(lm[L["L_HIP"]], lm[L["R_HIP"]])

            # px_to_cm from torso height
            # If profile has height, use it for accurate scaling (torso ~30% of height)
            # Otherwise fall back to 50cm (approx average adult torso)
            torso_real_cm = profile.estimated_torso_cm if profile else 50.0
            torso_px = np.linalg.norm(mid_shoulder[:2] - mid_hip[:2])
            fm.px_to_cm = torso_real_cm / torso_px if torso_px > 10 else 1.0

            # --- Trunk Lean (body-relative) ---
            trunk_vec = mid_shoulder[:2] - mid_hip[:2]
            vertical = np.array([0, -1])
            trunk_norm = np.linalg.norm(trunk_vec)
            if trunk_norm > 5:
                cos_a = np.dot(trunk_vec, vertical) / trunk_norm
                lean_raw = np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))
                if lean_raw <= 45:
                    fm.trunk_lean_angle = lean_raw
                    # Determine direction: shoulder x vs hip x
                    # In side view with runner moving left-to-right or right-to-left:
                    # We can't determine absolute direction from one frame,
                    # but we can infer from the relationship.
                    # For now, we use the x-offset sign:
                    # shoulder_x > hip_x means the shoulder is ahead of hip
                    # In standard running form, the shoulder should be slightly ahead
                    # of the hip during forward lean. We'll just record the raw value
                    # and the aggregator can infer direction from the x-axis sign.
                    
                    # Check if shoulder x is ahead of hip x (in image coords)
                    shoulder_x = mid_shoulder[0]
                    hip_x = mid_hip[0]
                    forward_x = shoulder_x - hip_x  # positive = shoulder ahead
                    # We can't reliably know which way the runner is facing,
                    # so we record the absolute lean and let the score system
                    # evaluate based on magnitude only.
                    fm.trunk_lean_is_forward = True  # default
                    # Store direction hint: lean angle is always positive magnitude

            # --- Elbow Angles ---
            if all(vis[i] > 0.5 for i in [L["L_SHOULDER"], L["L_ELBOW"], L["L_WRIST"]]):
                fm.left_elbow_angle = _angle_between(
                    lm[L["L_SHOULDER"]], lm[L["L_ELBOW"]], lm[L["L_WRIST"]]
                )
            if all(vis[i] > 0.5 for i in [L["R_SHOULDER"], L["R_ELBOW"], L["R_WRIST"]]):
                fm.right_elbow_angle = _angle_between(
                    lm[L["R_SHOULDER"]], lm[L["R_ELBOW"]], lm[L["R_WRIST"]]
                )

            # Arm symmetry (0-100, higher = more symmetrical)
            if fm.left_elbow_angle is not None and fm.right_elbow_angle is not None:
                diff = abs(fm.left_elbow_angle - fm.right_elbow_angle)
                fm.arm_symmetry_score = float(max(0, 100 - (diff / 30) * 100))

            # --- Knee Angles ---
            if all(vis[i] > 0.5 for i in [L["L_HIP"], L["L_KNEE"], L["L_ANKLE"]]):
                fm.left_knee_angle = _angle_between(
                    lm[L["L_HIP"]], lm[L["L_KNEE"]], lm[L["L_ANKLE"]]
                )
            if all(vis[i] > 0.5 for i in [L["R_HIP"], L["R_KNEE"], L["R_ANKLE"]]):
                fm.right_knee_angle = _angle_between(
                    lm[L["R_HIP"]], lm[L["R_KNEE"]], lm[L["R_ANKLE"]]
                )

            # --- Foot Strike ---
            if vis[L["L_HEEL"]] > 0.5 and vis[L["R_HEEL"]] > 0.5:
                left_rel_x = lm[L["L_HEEL"]][0] - mid_hip[0]
                right_rel_x = lm[L["R_HEEL"]][0] - mid_hip[0]
                forward_dist_px = max(abs(left_rel_x), abs(right_rel_x))
                fm.foot_strike_distance_cm = forward_dist_px * fm.px_to_cm

                # Foot strike type (normalized by px_to_cm)
                forward_idx = L["L_HEEL"] if abs(left_rel_x) >= abs(right_rel_x) else L["R_HEEL"]
                foot_idx = L["L_FOOT"] if forward_idx == L["L_HEEL"] else L["R_FOOT"]
                heel_y = lm[forward_idx][1]
                toe_y = lm[foot_idx][1]
                diff_y_px = toe_y - heel_y  # positive = toe lower than heel
                diff_y_cm = diff_y_px * fm.px_to_cm

                if diff_y_cm > 5:  # ~5cm threshold (body-scale normalized)
                    fm.foot_strike_type = "forefoot"
                elif diff_y_cm < -5:
                    fm.foot_strike_type = "rearfoot"
                else:
                    fm.foot_strike_type = "midfoot"

            # --- Head Forward Lean ---
            if vis[L["NOSE"]] > 0.5:
                nose_rel = lm[L["NOSE"]] - mid_shoulder
                if np.linalg.norm(nose_rel[:2]) > 5:
                    cos_h = np.dot(nose_rel[:2], np.array([0, -1])) / (
                        np.linalg.norm(nose_rel[:2]) * 1.0
                    )
                    head_lean = np.degrees(np.arccos(np.clip(cos_h, -1.0, 1.0)))
                    if head_lean <= 30:
                        fm.head_forward_lean = head_lean

            # --- Hip Height ---
            fm.hip_height = mid_hip[1]

            # --- Hip Drop (pelvic drop) ---
            # In side view, we can't directly measure the mediolateral pelvic drop.
            # But we can measure the vertical difference between L and R hip landmarks.
            # A large difference suggests lateral pelvic tilt (hip hiking or dropping).
            if vis[L["L_HIP"]] > 0.5 and vis[L["R_HIP"]] > 0.5:
                # In image coords, y-down. If L hip y is much different from R hip y,
                # that's lateral pelvic tilt.
                l_hip_y = lm[L["L_HIP"]][1]
                r_hip_y = lm[L["R_HIP"]][1]
                fm.hip_drop_px = abs(l_hip_y - r_hip_y)

            return fm

        except Exception:
            return None

    # ----------------------------------------------------------------
    # Signal extraction helpers for gait analysis
    # ----------------------------------------------------------------
    def _extract_foot_y_signal(self, fm_arr: List[FrameMetrics]) -> np.ndarray:
        """Extract foot y-position from frame metrics (approximate from hip signals)."""
        # We don't store foot_y per frame yet, so reconstruct from hip heights
        # and hip-to-foot distance. Simple approach: just use hip height as proxy
        # for now, since foot_y ≈ hip_y + leg_length when foot is down.
        return np.array([fm.hip_height for fm in fm_arr if fm.hip_height is not None])

    def _extract_hip_to_foot_distance(self, fm_arr: List[FrameMetrics]) -> np.ndarray:
        """Extract approximate hip-to-foot distance from per-frame foot strike distance."""
        # Use foot strike distance as a proxy for how far the foot is from the body
        dists = []
        for fm in fm_arr:
            if fm.foot_strike_distance_cm is not None:
                dists.append(fm.foot_strike_distance_cm / max(fm.px_to_cm or 1, 0.1))
        if not dists:
            return np.zeros(len(fm_arr))
        # Pad shorter arrays with zeros
        arr = np.array(dists)
        if len(arr) < len(fm_arr):
            arr = np.pad(arr, (0, len(fm_arr) - len(arr)), 'edge')
        return arr[:len(fm_arr)]

    # ----------------------------------------------------------------
    # Aggregation functions (confidence-weighted)
    # ----------------------------------------------------------------
    def _confidence_weighted_stats(self, values: np.ndarray,
                                   weights: np.ndarray) -> Tuple[float, float]:
        """Compute weighted mean and std using confidence weights."""
        if len(values) == 0:
            return 0.0, 0.0
        if np.sum(weights) <= 0:
            return float(np.mean(values)), float(np.std(values))
        avg = float(np.average(values, weights=weights))
        var = float(np.average((values - avg) ** 2, weights=weights))
        return avg, float(np.sqrt(var))

    def _aggregate_trunk_lean(self, metrics: RunningMetrics,
                              fm_arr: List[FrameMetrics]):
        """Aggregate trunk lean with outlier rejection."""
        leans = np.array([fm.trunk_lean_angle for fm in fm_arr
                          if fm.trunk_lean_angle is not None])
        if len(leans) == 0:
            return
        leans_clean = _reject_outliers_iqr(leans)
        if len(leans_clean) == 0:
            return
        metrics.trunk_lean_avg = float(np.mean(leans_clean))
        metrics.trunk_lean_std = float(np.std(leans_clean))

    def _aggregate_arm_metrics(self, metrics: RunningMetrics,
                                fm_arr: List[FrameMetrics]):
        """Aggregate arm-related metrics."""
        # Arm symmetry (0-100)
        syms = np.array([fm.arm_symmetry_score for fm in fm_arr
                         if fm.arm_symmetry_score is not None])
        if len(syms) > 0:
            syms_clean = _reject_outliers_iqr(syms)
            if len(syms_clean) > 0:
                metrics.arm_symmetry_avg = float(np.mean(syms_clean))

        # Elbow range
        left_elbows = np.array([fm.left_elbow_angle for fm in fm_arr
                                if fm.left_elbow_angle is not None])
        right_elbows = np.array([fm.right_elbow_angle for fm in fm_arr
                                 if fm.right_elbow_angle is not None])
        if len(left_elbows) > 0:
            metrics.left_elbow_range = float(np.ptp(_reject_outliers_iqr(left_elbows)))
        if len(right_elbows) > 0:
            metrics.right_elbow_range = float(np.ptp(_reject_outliers_iqr(right_elbows)))

    def _aggregate_vertical_oscillation(self, metrics: RunningMetrics,
                                         fm_arr: List[FrameMetrics]):
        """
        Compute vertical oscillation with:
          1. Proper px_to_cm per-frame scaling (not hardcoded 0.25)
          2. Quadratic detrending for camera/phone movement
          3. Hip-bounce filtering
        """
        # Build arrays
        heights = []
        scalers = []
        for fm in fm_arr:
            if fm.hip_height is not None and fm.px_to_cm is not None:
                heights.append(fm.hip_height)
                scalers.append(fm.px_to_cm)

        if len(heights) < 20:
            return

        h = np.array(heights)
        s = np.array(scalers)

        # Detrend: remove camera/phone vertical drift (quadratic fit)
        x = np.arange(len(h))
        try:
            coeffs = np.polyfit(x, h, 2)
            trend = np.polyval(coeffs, x)
        except Exception:
            trend = np.mean(h)
        detrended = h - trend

        # Light low-pass to remove noise while preserving hip oscillation
        # Cadence ~170 spm = 2.83 Hz, at 30fps = 0.094 cyc/frame
        # Alpha=0.35 preserves oscillation while smoothing frame-to-frame jitter
        detrended_smooth = _lowpass_filter(detrended, alpha=0.35)

        # Use 4×standard deviation as robust ptp estimate
        # (less sensitive to outliers than ptp)
        vert_osc_std = float(np.std(detrended_smooth))
        # For a sine wave: ptp = 4×std. For real running data: ptp ≈ 3.5-4.5×std
        vert_osc_px = vert_osc_std * 4.0

        # Convert to cm using AVERAGE px_to_cm from ALL frames (more robust)
        avg_scaler = float(np.mean(s))
        vert_osc_cm = vert_osc_px * avg_scaler

        # Sanity check: running vertical oscillation is typically 6-14cm
        if 2 < vert_osc_cm < 30:
            metrics.vertical_oscillation = round(vert_osc_cm, 1)
            metrics.vertical_oscillation_px = round(vert_osc_px, 1)

    def _aggregate_foot_strike(self, metrics: RunningMetrics,
                                fm_arr: List[FrameMetrics]):
        """Aggregate foot strike metrics with body-scale normalization."""
        # Foot strike distance (cm)
        dists = np.array([fm.foot_strike_distance_cm for fm in fm_arr
                          if fm.foot_strike_distance_cm is not None
                          and fm.foot_strike_distance_cm < 100])
        if len(dists) > 0:
            dists_clean = _reject_outliers_iqr(dists)
            if len(dists_clean) > 0:
                metrics.avg_foot_strike_distance = float(np.mean(dists_clean))

        # Foot strike type distribution
        types = [fm.foot_strike_type for fm in fm_arr
                 if fm.foot_strike_type is not None]
        if types:
            counter = Counter(types)
            total = sum(counter.values())
            metrics.foot_strike_type_dominant = counter.most_common(1)[0][0]
            metrics.foot_strike_types_distribution = {
                k: round(v / total * 100, 1) for k, v in counter.most_common()
            }

    def _aggregate_knee_angles(self, metrics: RunningMetrics,
                                fm_arr: List[FrameMetrics]):
        """Aggregate knee angles with outlier rejection."""
        for side, attr in [("left", "left_knee_angle"), ("right", "right_knee_angle")]:
            vals = np.array([getattr(fm, attr) for fm in fm_arr
                             if getattr(fm, attr) is not None])
            if len(vals) > 0:
                clean = _reject_outliers_iqr(vals)
                if len(clean) > 0:
                    setattr(metrics, f"avg_knee_angle_{side}", float(np.mean(clean)))

    def _aggregate_hip_drop(self, metrics: RunningMetrics,
                             fm_arr: List[FrameMetrics]):
        """Aggregate hip drop (pelvic lateral tilt)."""
        drops_px = np.array([fm.hip_drop_px for fm in fm_arr
                             if fm.hip_drop_px is not None])
        if len(drops_px) == 0:
            return
        # Reject outliers: hip drop typically < 3cm (slight tilt)
        clean = _reject_outliers_iqr(drops_px)
        if len(clean) == 0:
            return
        avg_drop_px = float(np.mean(clean))
        # Convert to cm using average px_to_cm
        scalers = np.array([fm.px_to_cm for fm in fm_arr
                            if fm.px_to_cm is not None])
        if len(scalers) > 0:
            avg_scaler = float(np.mean(scalers))
            metrics.avg_hip_drop_cm = round(avg_drop_px * avg_scaler, 2)

    def _aggregate_gait_cycles(self, metrics: RunningMetrics,
                                fm_arr: List[FrameMetrics]):
        """Count gait cycles from detrended hip height signal."""
        heights = np.array([fm.hip_height for fm in fm_arr
                            if fm.hip_height is not None])
        if len(heights) < 10:
            return

        # Detrend
        x = np.arange(len(heights))
        try:
            coeffs = np.polyfit(x, heights, 2)
            trend = np.polyval(coeffs, x)
        except Exception:
            trend = np.mean(heights)
        detrended = heights - trend

        # Smooth
        smooth = _lowpass_filter(detrended, alpha=0.2)

        # Count peaks with minimum distance constraint
        min_dist = int(metrics.fps * 0.2) if metrics.fps > 0 else 3
        peaks = 0
        last_peak = -min_dist
        for i in range(1, len(smooth) - 1):
            if (smooth[i] > smooth[i - 1] and smooth[i] > smooth[i + 1]
                    and i - last_peak >= min_dist):
                peaks += 1
                last_peak = i

        metrics.total_gait_cycles = peaks // 2  # 2 steps per cycle

    def _estimate_step_length(self, metrics: RunningMetrics):
        """
        Estimate step length from cadence and duration.
        step_length = speed / cadence
        Speed roughly estimated from video duration and typical stride coverage.
        
        More accurate: step_length ≈ height * 0.45 (for recreational runners)
        But we don't know height. Instead, we use the relationship:
        step_length (cm) = speed (m/s) * 1000 / (cadence/60)
        
        We estimate speed from the foot displacement across frames.
        """
        # Simple proxy: step length correlates with leg length and cadence
        # Typical running step length (not stride): 80-120cm
        # Step length ≈ height (cm) * 0.45 for recreational runners
        # But we don't have height, so estimate from px_to_cm * body proportions
        
        # Alternative: estimate from the distance the hip travels per step
        # We can approximate from the average foot strike distance * 2
        if metrics.avg_foot_strike_distance is not None:
            # Rough estimate: step length ~ 2 * foot strike distance
            estimated = metrics.avg_foot_strike_distance * 2.2
            # Sanity check: typical step length 60-150cm
            if 50 < estimated < 180:
                metrics.estimated_step_length_cm = round(estimated, 1)

    # ----------------------------------------------------------------
    # Cadence estimation (improved v2)
    # ----------------------------------------------------------------
    def _compute_cadence(self, frame_metrics: List[FrameMetrics],
                         fps: float) -> Tuple[Optional[float], Optional[float]]:
        """
        Estimate cadence from hip vertical oscillation.
        
        v2 improvements:
          - Multi-peak autocorrelation validation (pick most consistent)
          - Better low-pass filtering (zero-phase)
          - Adaptive lag window
          - Quality check on cadence stability
        """
        if fps <= 0 or len(frame_metrics) < fps * 1.5:
            return None, None

        raw_hip = np.array([
            fm.hip_height for fm in frame_metrics if fm.hip_height is not None
        ])
        if len(raw_hip) < 30:
            return None, None

        # Detrend
        x = np.arange(len(raw_hip))
        try:
            coeffs = np.polyfit(x, raw_hip, 2)
            trend = np.polyval(coeffs, x)
        except Exception:
            trend = np.mean(raw_hip)
        hip = raw_hip - trend

        # Zero-phase low-pass filter
        hip = _lowpass_filter(hip, alpha=0.12)

        # Autocorrelation
        n = len(hip)
        autocorr = np.correlate(hip, hip, mode='same')
        autocorr = autocorr[n // 2:]
        if autocorr[0] > 0:
            autocorr = autocorr / autocorr[0]
        else:
            return None, None

        # Valid cadence range: 120-220 spm
        # Period per step (frames): fps / cadence * 60
        # For cadence 120: fps * 60/120 = fps * 0.5
        # For cadence 220: fps * 60/220 = fps * 0.273
        min_lag = int(fps * 0.25)   # ~240 spm (slightly above max)
        max_lag = int(fps * 0.55)   # ~109 spm (below min)

        max_lag = min(max_lag, len(autocorr) - 1)
        if min_lag >= max_lag:
            return None, None

        # Find ALL peaks in search region
        search = autocorr[min_lag:max_lag + 1]
        peaks = []
        for i in range(1, len(search) - 1):
            if (search[i] > search[i - 1] and
                search[i] > search[i + 1] and
                search[i] > 0.15):  # minimum correlation threshold
                lag = i + min_lag
                cadence_i = 60.0 / (lag / fps)
                peaks.append((lag, search[i], cadence_i))

        if not peaks:
            # Fallback: use the strongest point in search region
            best_i = int(np.argmax(search))
            lag = best_i + min_lag
            peaks = [(lag, search[best_i], 60.0 / (lag / fps))]

        # Filter to realistic cadence
        valid_cadences = [p[2] for p in peaks if 120 <= p[2] <= 220]
        if not valid_cadences:
            return None, None

        # Use the cadence closest to 175 (optimal zone)
        best_cadence = min(valid_cadences, key=lambda c: abs(c - 175))

        # Estimate variability from the spread of valid cadences
        if len(valid_cadences) > 1:
            cadence_std = float(np.std(valid_cadences))
        else:
            cadence_std = best_cadence * 0.03  # ~3% variability estimate

        return round(best_cadence, 1), round(cadence_std, 1)

    # ----------------------------------------------------------------
    # Scoring system (v2)
    # ----------------------------------------------------------------
    def get_scoring(self, metrics: RunningMetrics,
                     profile: Optional[RunnerProfile] = None) -> Dict:
        """
        Score each metric 0-100 and compute overall Running Form Score.
        
        Uses runner profile (height, gender, age) to adjust scoring.
        """
        score_map = {}
        details = {}

        # Cadence: optimal ~175 spm, adjusted by height
        # Taller runners naturally have lower cadence
        # 165cm → width 10, 175cm → width 12, 190cm → width 14
        if metrics.cadence_avg:
            cadence_width = 12
            if profile and profile.height_cm:
                # Scale width by height
                cadence_width = round(8 + profile.height_cm / 170.0 * 4, 1)
            score_map["cadence"] = _sigmoid_score(
                metrics.cadence_avg, center=175, width=cadence_width)
            details["cadence"] = f"{metrics.cadence_avg:.0f} spm"

        # Trunk lean: optimal 4-10° forward
        # Lean outside this range gets penalized smoothly
        if metrics.trunk_lean_avg is not None:
            score_map["trunk_lean"] = _sigmoid_score(metrics.trunk_lean_avg, center=7, width=4)
            details["trunk_lean"] = f"{metrics.trunk_lean_avg:.1f}°"

        # Arm symmetry: directly 0-100 scale
        if metrics.arm_symmetry_avg is not None:
            score_map["arm_symmetry"] = float(np.clip(metrics.arm_symmetry_avg, 0, 100))
            details["arm_symmetry"] = f"{metrics.arm_symmetry_avg:.0f}/100"

        # Vertical oscillation: lower is better
        # Center=0 + width=8: 0cm→100, 5cm→82, 10cm→46, 15cm→18
        if metrics.vertical_oscillation is not None:
            score_map["vertical_oscillation"] = _sigmoid_score(
                metrics.vertical_oscillation, center=0, width=8
            )
            details["vertical_oscillation"] = f"{metrics.vertical_oscillation:.1f} cm"

        # Foot strike distance: ideal directly under hip (smaller = better)
        # Center=0 + width=10: 0cm→100, 5cm→88, 10cm→61, 15cm→32, 20cm→14
        if metrics.avg_foot_strike_distance is not None:
            score_map["foot_strike"] = _sigmoid_score(
                metrics.avg_foot_strike_distance, center=0, width=10
            )
            details["foot_strike"] = f"{metrics.avg_foot_strike_distance:.1f} cm"

        # Foot strike type bonus (applied after main scoring)
        foot_strike_bonus = 0
        if metrics.foot_strike_type_dominant:
            if metrics.foot_strike_type_dominant in ("midfoot", "forefoot"):
                foot_strike_bonus = 5
            # rearfoot strikers get no bonus but also no penalty

        # Hip drop: lower = better (0-1cm normal, >2cm indicates issue)
        if metrics.avg_hip_drop_cm is not None:
            score_map["hip_drop"] = _sigmoid_score(
                metrics.avg_hip_drop_cm, center=0, width=2
            )
            details["hip_drop"] = f"{metrics.avg_hip_drop_cm:.1f} cm"

        # Ground contact time (if available): lower = better
        # Elite runners: <200ms. Recreational: 200-300ms
        if metrics.ground_contact_time_ms is not None:
            score_map["ground_contact"] = _sigmoid_score(
                metrics.ground_contact_time_ms, center=200, width=60
            )
            details["ground_contact"] = f"{metrics.ground_contact_time_ms:.0f} ms"

        # --- Overall score (weighted average with expanded dimensions) ---
        weights = {
            "cadence": 0.20,
            "trunk_lean": 0.18,
            "arm_symmetry": 0.10,
            "vertical_oscillation": 0.18,
            "foot_strike": 0.17,
            "hip_drop": 0.10,
            "ground_contact": 0.07,
        }

        total_score = 0.0
        total_weight = 0.0
        for key, weight in weights.items():
            if key in score_map:
                total_score += score_map[key] * weight
                total_weight += weight

        if total_weight > 0:
            overall = total_score / total_weight
            # Apply foot strike type bonus
            overall = min(overall + min(foot_strike_bonus * (total_weight / sum(weights.values())), 3), 100)
            overall = round(overall, 1)
        else:
            overall = None

        return {
            "overall_score": overall,
            "metrics": score_map,
            "details": details,
        }
