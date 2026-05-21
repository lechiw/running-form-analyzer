"""
tests/test_comparison.py - Tests for cross-video comparison module
"""

import json
from comparison import (
    compare_analyses, format_comparison_report, comparison_to_dict,
    _is_improvement, _is_significant
)


# Sample data for testing
BEFORE = {
    'cadence_spm': 162.0,
    'trunk_lean_deg': 3.2,
    'vertical_oscillation_cm': 11.5,
    'arm_symmetry_score': 55.0,
    'foot_strike_distance_cm': 18.0,
    'hip_drop_cm': 2.8,
    'overall_score': 52.0,
}

AFTER = {
    'cadence_spm': 171.0,
    'trunk_lean_deg': 6.5,
    'vertical_oscillation_cm': 9.2,
    'arm_symmetry_score': 72.0,
    'foot_strike_distance_cm': 12.0,
    'hip_drop_cm': 1.5,
    'overall_score': 72.0,
}


class TestImprovementDetection:
    """Test the _is_improvement helper."""

    def test_higher_is_better(self):
        """Cadence: higher → improvement."""
        assert _is_improvement("cadence_spm", 5.0, 160) == True

    def test_lower_is_better(self):
        """Vertical oscillation: lower → improvement."""
        assert _is_improvement("vertical_oscillation_cm", -2.0, 10) == True

    def test_higher_worsens(self):
        """Vertical oscillation increasing → regression."""
        assert _is_improvement("vertical_oscillation_cm", 2.0, 10) == False


class TestSignificanceDetection:
    """Test the _is_significant helper."""

    def test_large_delta_is_significant(self):
        """Big change → significant."""
        assert _is_significant("cadence_spm", 9.0) == True

    def test_small_delta_not_significant(self):
        """Tiny change → not significant."""
        assert _is_significant("cadence_spm", 0.5) == False

    def test_unknown_metric_not_significant(self):
        """Unknown metric → not significant."""
        assert _is_significant("unknown_metric", 100.0) == False


class TestCompareAnalyses:
    """Main comparison function tests."""

    def test_basic_comparison(self):
        """Basic comparison should work with typical data."""
        cr = compare_analyses(BEFORE, AFTER)
        assert cr.score_before == 52.0
        assert cr.score_after == 72.0
        assert cr.score_delta == 20.0

    def test_delta_count(self):
        """Should produce deltas for all defined metrics that have data."""
        cr = compare_analyses(BEFORE, AFTER)
        # 6 metrics have data in both
        assert len(cr.deltas) > 0
        # All deltas should have values
        for d in cr.deltas:
            assert d.before_value is not None or d.after_value is not None

    def test_improvements_detected(self):
        """All metrics should show improvement (synthetic data)."""
        cr = compare_analyses(BEFORE, AFTER)
        for d in cr.deltas:
            if d.delta is not None and abs(d.delta) > 0.01:
                assert d.is_improvement, f"{d.label}: delta={d.delta} not improvement"

    def test_insights_generated(self):
        """Comparisons should generate insights."""
        cr = compare_analyses(BEFORE, AFTER)
        assert len(cr.insights) > 0

    def test_empty_after(self):
        """Missing 'after' data should not crash."""
        cr = compare_analyses(BEFORE, {})
        assert cr.score_before == 52.0
        assert cr.score_after is None

    def test_empty_before(self):
        """Missing 'before' data should not crash."""
        cr = compare_analyses({}, AFTER)
        assert cr.score_before is None
        assert cr.score_after == 72.0

    def test_empty_both(self):
        """Both empty should not crash."""
        cr = compare_analyses({}, {})
        assert len(cr.deltas) == 0  # no data = no deltas
        assert cr.score_before is None
        assert cr.score_after is None

    def test_custom_labels(self):
        """Custom labels should appear in output."""
        cr = compare_analyses(BEFORE, AFTER,
                               label_before="月初", label_after="月末")
        assert cr.before_label == "月初"
        assert cr.after_label == "月末"

    def test_no_change(self):
        """Identical data should show no change."""
        cr = compare_analyses(BEFORE, BEFORE)
        for d in cr.deltas:
            if d.delta is not None:
                assert abs(d.delta) < 0.01

    def test_worsening_score(self):
        """Worsening data should show regressions."""
        worse = {k: v - 10 for k, v in AFTER.items() if isinstance(v, (int, float))}
        cr = compare_analyses(AFTER, worse)
        assert cr.score_delta < 0


class TestOutputFormats:
    """Test different output formats."""

    def test_report_string(self):
        """Report should be a non-empty string."""
        cr = compare_analyses(BEFORE, AFTER)
        report = format_comparison_report(cr)
        assert isinstance(report, str)
        assert len(report) > 100
        assert "跨视频跑姿对比报告" in report

    def test_json_serialization(self):
        """JSON output should be valid and complete."""
        cr = compare_analyses(BEFORE, AFTER)
        d = comparison_to_dict(cr)
        required = ['before_label', 'after_label', 'deltas', 'insights']
        for field in required:
            assert field in d, f"Missing field: {field}"
        # Round-trip JSON
        json_str = json.dumps(d, ensure_ascii=False)
        restored = json.loads(json_str)
        assert restored['score_before'] == 52.0

    def test_delta_values_round_trip(self):
        """Delta values should survive JSON serialization."""
        cr = compare_analyses(BEFORE, AFTER)
        d = comparison_to_dict(cr)
        for item in d['deltas']:
            assert 'name' in item
            assert 'label' in item
            assert 'delta' in item
