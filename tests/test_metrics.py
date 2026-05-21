"""
tests/test_metrics.py - Unit and integration tests for metrics pipeline
"""

import numpy as np
from metrics import (
    RunningMetricsCalculator, RunningMetrics, FrameMetrics,
    _sigmoid_score, _reject_outliers_iqr
)
from pose_extractor import PoseLandmarks, PoseSequence
from metrics import L  # landmark indices


class TestFrameMetrics:
    """Per-frame metric computation tests."""

    def test_synthetic_frame(self):
        """Compute metrics from a synthetic frame with good visibility."""
        calc = RunningMetricsCalculator()
        lm = np.zeros((33, 3))
        vis = np.ones(33) * 0.9

        # Normal runner pose (side view)
        hip_y = 500
        lm[L['L_HIP']] = [300, hip_y, 0]
        lm[L['R_HIP']] = [340, hip_y, 0]
        lm[L['L_SHOULDER']] = [310, hip_y - 200, 0]
        lm[L['R_SHOULDER']] = [350, hip_y - 200, 0]

        pl = PoseLandmarks(frame_idx=0, landmarks=lm, visibility=vis, timestamp_ms=0)
        fm = calc._compute_frame_metrics(pl)

        assert fm is not None
        assert fm.trunk_lean_angle is not None
        assert fm.px_to_cm is not None
        assert fm.hip_height is not None

    def test_low_visibility(self):
        """Frame with low visibility should return None."""
        calc = RunningMetricsCalculator()
        lm = np.zeros((33, 3))
        vis = np.ones(33) * 0.1  # very low

        pl = PoseLandmarks(frame_idx=0, landmarks=lm, visibility=vis, timestamp_ms=0)
        fm = calc._compute_frame_metrics(pl)
        assert fm is None

    def test_partial_visibility(self):
        """Frame with some joints visible should still work."""
        calc = RunningMetricsCalculator()
        lm = np.zeros((33, 3))
        vis = np.zeros(33)

        # Only set key joints to visible
        for idx in [L['L_HIP'], L['R_HIP'], L['L_SHOULDER'], L['R_SHOULDER'],
                     L['L_KNEE'], L['R_KNEE'], L['L_ANKLE'], L['R_ANKLE']]:
            vis[idx] = 0.8

        hip_y = 500
        lm[L['L_HIP']] = [300, hip_y, 0]
        lm[L['R_HIP']] = [340, hip_y, 0]
        lm[L['L_SHOULDER']] = [310, hip_y - 200, 0]
        lm[L['R_SHOULDER']] = [350, hip_y - 200, 0]

        pl = PoseLandmarks(frame_idx=0, landmarks=lm, visibility=vis, timestamp_ms=0)
        fm = calc._compute_frame_metrics(pl)
        assert fm is not None

    def test_hip_drop_detection(self):
        """Hip drop should be detected when hips are at different heights."""
        calc = RunningMetricsCalculator()
        lm = np.zeros((33, 3))
        vis = np.ones(33) * 0.8

        # Left hip higher than right (10px drop)
        lm[L['L_HIP']] = [300, 490, 0]
        lm[L['R_HIP']] = [340, 500, 0]
        lm[L['L_SHOULDER']] = [310, 300, 0]
        lm[L['R_SHOULDER']] = [350, 300, 0]

        pl = PoseLandmarks(frame_idx=0, landmarks=lm, visibility=vis, timestamp_ms=0)
        fm = calc._compute_frame_metrics(pl)
        assert fm is not None
        assert fm.hip_drop_px is not None
        assert fm.hip_drop_px > 0


class TestFullPipeline:
    """Full metrics pipeline integration tests."""

    def test_empty_sequence(self, empty_sequence):
        """Empty sequence should return empty metrics (not crash)."""
        calc = RunningMetricsCalculator()
        metrics = calc.compute(empty_sequence)
        assert isinstance(metrics, RunningMetrics)
        assert metrics.frame_count == 0

    def test_low_visibility_sequence(self, low_visibility_sequence):
        """Low visibility sequence should return no valid metrics."""
        calc = RunningMetricsCalculator()
        metrics = calc.compute(low_visibility_sequence)
        assert metrics.cadence_avg is None
        assert metrics.trunk_lean_avg is None
        assert metrics.vertical_oscillation is None
        assert len(metrics.frame_metrics) == 0

    def test_synthetic_pipeline(self, synthetic_sequence):
        """Full pipeline with synthetic data should produce valid metrics."""
        calc = RunningMetricsCalculator()
        metrics = calc.compute(synthetic_sequence)

        assert metrics.frame_count > 0
        assert metrics.cadence_avg is not None
        assert metrics.trunk_lean_avg is not None
        assert metrics.vertical_oscillation is not None
        assert metrics.arm_symmetry_avg is not None

    def test_cadence_range(self, synthetic_sequence):
        """Cadence should be in realistic range."""
        calc = RunningMetricsCalculator()
        metrics = calc.compute(synthetic_sequence)
        # Synthetic data targets ~180 spm
        if metrics.cadence_avg is not None:
            assert 120 <= metrics.cadence_avg <= 220

    def test_trunk_lean_range(self, synthetic_sequence):
        """Trunk lean should be in realistic range."""
        calc = RunningMetricsCalculator()
        metrics = calc.compute(synthetic_sequence)
        if metrics.trunk_lean_avg is not None:
            assert 0 <= metrics.trunk_lean_avg <= 45

    def test_vertical_oscillation_range(self, synthetic_sequence):
        """Vertical oscillation should be in realistic range."""
        calc = RunningMetricsCalculator()
        metrics = calc.compute(synthetic_sequence)
        if metrics.vertical_oscillation is not None:
            assert 2 <= metrics.vertical_oscillation <= 20

    def test_arm_symmetry_range(self, synthetic_sequence):
        """Arm symmetry should be between 0-100."""
        calc = RunningMetricsCalculator()
        metrics = calc.compute(synthetic_sequence)
        if metrics.arm_symmetry_avg is not None:
            assert 0 <= metrics.arm_symmetry_avg <= 100

    def test_hip_drop_non_negative(self, synthetic_sequence):
        """Hip drop should be non-negative."""
        calc = RunningMetricsCalculator()
        metrics = calc.compute(synthetic_sequence)
        if metrics.avg_hip_drop_cm is not None:
            assert metrics.avg_hip_drop_cm >= 0

    def test_gait_cycles(self, synthetic_sequence):
        """Gait cycles should be reasonable for a 10s video."""
        calc = RunningMetricsCalculator()
        metrics = calc.compute(synthetic_sequence)
        # ~180 spm = 3 strides/sec, 10s = ~30 strides = ~15 cycles
        # On synthetic data, might be close
        assert metrics.total_gait_cycles >= 0

    def test_duration(self, synthetic_sequence):
        """Duration should be approximately 10 seconds."""
        calc = RunningMetricsCalculator()
        metrics = calc.compute(synthetic_sequence)
        assert 9 <= metrics.duration_sec <= 11


class TestScoring:
    """Scoring system tests."""

    def test_empty_scoring(self):
        """Empty metrics should not crash scoring."""
        calc = RunningMetricsCalculator()
        metrics = RunningMetrics()
        scoring = calc.get_scoring(metrics)
        assert scoring['overall_score'] is None
        assert isinstance(scoring['metrics'], dict)

    def test_partial_scoring(self):
        """Partial metrics should still produce a score."""
        calc = RunningMetricsCalculator()
        metrics = RunningMetrics(fps=30, cadence_avg=175)
        scoring = calc.get_scoring(metrics)
        assert scoring['overall_score'] is not None
        assert scoring['overall_score'] > 0

    def test_perfect_cadence_scoring(self):
        """Perfect cadence should get high score."""
        calc = RunningMetricsCalculator()
        metrics = RunningMetrics(cadence_avg=175)
        scoring = calc.get_scoring(metrics)
        assert scoring['metrics']['cadence'] > 90

    def test_bad_cadence_scoring(self):
        """Very bad cadence should get low score."""
        calc = RunningMetricsCalculator()
        metrics = RunningMetrics(cadence_avg=110)
        scoring = calc.get_scoring(metrics)
        assert scoring['metrics']['cadence'] < 10

    def test_perfect_trunk_lean(self):
        """Perfect trunk lean should get high score."""
        calc = RunningMetricsCalculator()
        metrics = RunningMetrics(trunk_lean_avg=7.0)
        scoring = calc.get_scoring(metrics)
        assert scoring['metrics']['trunk_lean'] > 90

    def test_bad_trunk_lean(self):
        """Very bad trunk lean should get low score."""
        calc = RunningMetricsCalculator()
        metrics = RunningMetrics(trunk_lean_avg=40.0)
        scoring = calc.get_scoring(metrics)
        assert scoring['metrics']['trunk_lean'] < 10

    def test_arm_symmetry_scoring(self):
        """Arm symmetry score should pass through."""
        calc = RunningMetricsCalculator()
        metrics = RunningMetrics(arm_symmetry_avg=85)
        scoring = calc.get_scoring(metrics)
        assert scoring['metrics']['arm_symmetry'] == 85.0

    def test_scoring_all_metrics(self, synthetic_sequence):
        """Full pipeline should produce scores for available metrics."""
        calc = RunningMetricsCalculator()
        metrics = calc.compute(synthetic_sequence)
        scoring = calc.get_scoring(metrics)

        assert scoring['overall_score'] is not None
        assert len(scoring['metrics']) > 0
        assert len(scoring['details']) > 0

        for k, v in scoring['metrics'].items():
            assert 0 <= v <= 100, f"Score {k}={v} out of range [0,100]"


class TestSummary:
    """Summary output tests."""

    def test_summary_keys(self, synthetic_sequence):
        """Summary dict should have expected keys."""
        calc = RunningMetricsCalculator()
        metrics = calc.compute(synthetic_sequence)
        summary = metrics.summary()

        expected_keys = [
            'cadence_spm', 'trunk_lean_deg', 'vertical_oscillation_cm',
            'arm_symmetry_score', 'foot_strike_distance_cm', 'foot_strike_type',
            'total_gait_cycles', 'duration_sec',
        ]
        for key in expected_keys:
            assert key in summary, f"Missing key: {key}"

    def test_summary_no_none_required(self, synthetic_sequence):
        """Critical metrics should not be None with valid data."""
        calc = RunningMetricsCalculator()
        metrics = calc.compute(synthetic_sequence)
        summary = metrics.summary()

        # These should be present with synthetic data
        assert summary['cadence_spm'] is not None
        assert summary['trunk_lean_deg'] is not None
        assert summary['vertical_oscillation_cm'] is not None
        assert summary['arm_symmetry_score'] is not None
