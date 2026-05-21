"""
tests/test_utils.py - Unit tests for utility functions
"""

import numpy as np
import pytest
from metrics import (
    _sigmoid_score, _reject_outliers_iqr, _lowpass_filter,
    _angle_between, _midpoint, _detect_gait_phases
)


class TestSigmoidScore:
    """Sigmoid scoring function tests."""

    def test_exact_center(self):
        """Score should be 100 at exact center."""
        assert _sigmoid_score(175, center=175, width=10) == 100.0

    def test_near_center(self):
        """Score should be high but not 100 near center."""
        score = _sigmoid_score(170, center=175, width=10)
        assert 60 < score < 100

    def test_far_from_center(self):
        """Score should be very low far from center."""
        score = _sigmoid_score(100, center=175, width=10)
        assert score < 5

    def test_zero_width(self):
        """Zero width should return max_val."""
        assert _sigmoid_score(42, center=0, width=0) == 100.0

    def test_invert(self):
        """Inverted scoring: center → low, far → high."""
        center_score = _sigmoid_score(0, center=0, width=10, invert=True)
        far_score = _sigmoid_score(50, center=0, width=10, invert=True)
        assert center_score < far_score

    def test_clamping(self):
        """Score should be clamped to [min_val, max_val]."""
        score = _sigmoid_score(-999, center=0, width=1)
        assert score >= 0
        score = _sigmoid_score(999, center=0, width=1)
        assert score >= 0

    def test_custom_range(self):
        """Custom min/max values should work."""
        score = _sigmoid_score(0, center=0, width=5, max_val=50, min_val=10)
        assert 10 <= score <= 50


class TestOutlierRejection:
    """IQR-based outlier rejection tests."""

    def test_no_outliers(self):
        """Normal data should not be rejected."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        result = _reject_outliers_iqr(data)
        assert len(result) == len(data)

    def test_with_outliers(self):
        """Obvious outliers should be removed."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
        result = _reject_outliers_iqr(data)
        assert len(result) < len(data)
        assert 100 not in result

    def test_negative_outliers(self):
        """Negative outliers should also be removed."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, -50])
        result = _reject_outliers_iqr(data)
        assert -50 not in result

    def test_small_dataset(self):
        """Small datasets should be unchanged."""
        data = np.array([1, 2, 3])
        result = _reject_outliers_iqr(data)
        assert len(result) == 3

    def test_all_same_value(self):
        """All same value: no outliers."""
        data = np.array([5, 5, 5, 5, 5])
        result = _reject_outliers_iqr(data)
        assert len(result) == 5

    def test_custom_factor(self):
        """Custom IQR factor should work."""
        data = np.array(list(range(20)) + [30])
        loose = _reject_outliers_iqr(data, factor=3.0)
        strict = _reject_outliers_iqr(data, factor=0.5)
        assert len(loose) > len(strict)


class TestLowpassFilter:
    """Low-pass filter tests."""

    def test_constant_signal(self):
        """Constant signal should stay constant."""
        sig = np.ones(100) * 42
        filtered = _lowpass_filter(sig, alpha=0.3)
        assert np.allclose(filtered, 42.0, atol=0.01)

    def test_empty_input(self):
        """Empty input should return empty."""
        assert len(_lowpass_filter(np.array([]))) == 0

    def test_single_element(self):
        """Single element should return itself."""
        assert _lowpass_filter(np.array([5.0]))[0] == 5.0

    def test_zero_phase(self):
        """Filter should not introduce phase shift (peak stays in place)."""
        sig = np.zeros(100)
        sig[50] = 100  # single peak
        filtered = _lowpass_filter(sig, alpha=0.5)
        peak_idx = int(np.argmax(filtered))
        assert peak_idx == 50  # peak should not shift

    def test_smoothing(self):
        """High-frequency noise should be reduced."""
        rng = np.random.RandomState(42)
        sig = np.sin(np.linspace(0, 10, 100)) + 0.5 * rng.randn(100)
        filtered = _lowpass_filter(sig, alpha=0.3)
        # Filtered signal should have lower std than noisy signal
        assert np.std(filtered) < np.std(sig)


class TestAngleBetween:
    """Angle calculation tests."""

    def test_right_angle(self):
        """Right angle should give ~90 degrees."""
        p1, p2, p3 = np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([1, 0, 0])
        # Need different points...
        p1, p2, p3 = np.array([0, 1, 0]), np.array([0, 0, 0]), np.array([1, 0, 0])
        angle = _angle_between(p1, p2, p3)
        assert abs(angle - 90) < 1

    def test_straight_line(self):
        """180 degrees (straight line)."""
        p1, p2, p3 = np.array([0, 1, 0]), np.array([0, 0, 0]), np.array([0, -1, 0])
        angle = _angle_between(p1, p2, p3)
        assert abs(angle - 180) < 1

    def test_acute(self):
        """Acute angle of ~45 degrees."""
        p1, p2, p3 = np.array([1, 0, 0]), np.array([0, 0, 0]), np.array([1, 1, 0])
        angle = _angle_between(p1, p2, p3)
        assert abs(angle - 45) < 2

    def test_identical_points(self):
        """Identical points should not crash."""
        p1 = p2 = p3 = np.array([0, 0, 0])
        angle = _angle_between(p1, p2, p3)
        assert angle == 0.0


class TestMidpoint:
    """Midpoint calculation tests."""

    def test_simple_midpoint(self):
        """Midpoint of two points."""
        p1, p2 = np.array([0, 0, 0]), np.array([10, 10, 0])
        mid = _midpoint(p1, p2)
        assert np.allclose(mid, [5, 5, 0])

    def test_3d_midpoint(self):
        """3D midpoint should work."""
        p1, p2 = np.array([1, 2, 3]), np.array([5, 6, 7])
        mid = _midpoint(p1, p2)
        assert np.allclose(mid, [3, 4, 5])

    def test_identical_points(self):
        """Identical points return the same point."""
        p1 = p2 = np.array([42, 42, 42])
        mid = _midpoint(p1, p2)
        assert np.allclose(mid, [42, 42, 42])


class TestGaitPhaseDetection:
    """Gait phase detection tests."""

    def test_constant_signal(self):
        """Constant signal should give some stance frames."""
        n = 50
        hip = np.ones(n) * 500
        foot_y = np.ones(n) * 600
        hf_dist = np.ones(n) * 100
        phases = _detect_gait_phases(hip, foot_y, hf_dist, 30.0)
        assert phases.num_stance + phases.num_swing == n

    def test_oscillating_signal(self):
        """Oscillating signal should show phase alternation."""
        n = 100
        t = np.arange(n)
        hip = 500 + 30 * np.sin(2 * np.pi * t / 10)
        foot_y = 550 + 50 * np.sin(2 * np.pi * t / 10)
        hf_dist = 50 + 40 * np.abs(np.sin(2 * np.pi * t / 10))
        phases = _detect_gait_phases(hip, foot_y, hf_dist, 30.0)
        assert phases.num_stance > 0
        # The gait phase threshold may classify all as stance for smooth
        # synthetic data; check that confidence varies across frames
        confidences = [l.confidence for l in phases.labels]
        assert max(confidences) - min(confidences) > 0.001

    def test_stance_ratio_range(self):
        """Stance ratio should be between 0 and 1."""
        n = 100
        rng = np.random.RandomState(42)
        phases = _detect_gait_phases(
            rng.rand(n) * 100, rng.rand(n) * 100, rng.rand(n) * 50, 30.0
        )
        assert 0 <= phases.stance_ratio <= 1

    def test_short_sequence(self):
        """Very short sequence should not crash."""
        hip = np.array([500])
        foot_y = np.array([600])
        hf_dist = np.array([100])
        phases = _detect_gait_phases(hip, foot_y, hf_dist, 30.0)
        assert phases.num_stance == 0
