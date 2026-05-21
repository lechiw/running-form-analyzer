"""
comparison.py - Cross-video running form comparison
Compares two analysis results to track improvement over time.
"""

import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class MetricDelta:
    """Change in a single metric between two analyses."""
    name: str  # metric key (e.g. 'cadence_spm')
    label: str  # Chinese name
    unit: str  # 'spm', '°', 'cm', '/100', 'ms', etc.
    
    before_value: Optional[float]
    after_value: Optional[float]
    delta: Optional[float]  # after - before
    delta_pct: Optional[float]  # percentage change
    
    is_improvement: bool  # True if the change is positive
    is_significant: bool  # True if change exceeds meaningful threshold
    significance_reason: str  # why it matters


@dataclass
class ComparisonResult:
    """Full comparison between two running analyses."""
    # Metadata
    before_label: str  # e.g. "训练前 (2026-05-01)"
    after_label: str   # e.g. "训练后 (2026-05-21)"
    
    # Overall
    score_before: Optional[float] = None
    score_after: Optional[float] = None
    score_delta: Optional[float] = None
    
    # Metric-level
    deltas: List[MetricDelta] = field(default_factory=list)
    
    # Summary
    improvements: List[str] = field(default_factory=list)  # what got better
    regressions: List[str] = field(default_factory=list)   # what got worse
    insights: List[str] = field(default_factory=list)       # auto-generated advice
    
    # If fatigue analysis available
    fatigue_comparison: Optional[Dict] = None


# ── Metric definitions for display ──
METRIC_INFO = {
    "cadence_spm": {
        "label": "步频",
        "unit": "spm",
        "higher_is_better": True,
        "ideal": "170-180",
        "significant_delta": 3,  # ±3 spm is meaningful
    },
    "trunk_lean_deg": {
        "label": "躯干前倾",
        "unit": "°",
        "higher_is_better": None,  # optimal range
        "ideal_range": (4, 10),
        "significant_delta": 2,
    },
    "vertical_oscillation_cm": {
        "label": "垂直振幅",
        "unit": "cm",
        "higher_is_better": False,
        "ideal": "<10",
        "significant_delta": 1,
    },
    "arm_symmetry_score": {
        "label": "手臂对称",
        "unit": "/100",
        "higher_is_better": True,
        "ideal": ">80",
        "significant_delta": 5,
    },
    "foot_strike_distance_cm": {
        "label": "触地距离",
        "unit": "cm",
        "higher_is_better": False,
        "ideal": "<15",
        "significant_delta": 2,
    },
    "hip_drop_cm": {
        "label": "髋部倾斜",
        "unit": "cm",
        "higher_is_better": False,
        "ideal": "<2",
        "significant_delta": 0.5,
    },
    "ground_contact_time_ms": {
        "label": "触地时间",
        "unit": "ms",
        "higher_is_better": False,
        "ideal": "<250",
        "significant_delta": 15,
    },
    "estimated_step_length_cm": {
        "label": "步幅",
        "unit": "cm",
        "higher_is_better": True,
        "ideal": "因人而异",
        "significant_delta": 5,
    },
}


def _is_improvement(key: str, delta: float, before: float) -> bool:
    """Determine if a delta represents improvement."""
    info = METRIC_INFO.get(key)
    if not info:
        return None
    
    hb = info.get("higher_is_better")
    
    if hb is True:
        return delta > 0  # bigger = better
    elif hb is False:
        return delta < 0  # smaller = better
    else:
        # Optimal range: check if moving toward it
        lo, hi = info.get("ideal_range", (0, 100))
        # Moving toward range is improvement
        if before < lo:
            return delta > 0  # moving up = improvement
        elif before > hi:
            return delta < 0  # moving down = improvement
        else:
            return abs(delta) < 0.5  # already in range, small change = fine


def _is_significant(key: str, delta: float) -> bool:
    """Check if delta exceeds meaningful threshold."""
    info = METRIC_INFO.get(key)
    if not info:
        return False
    threshold = info.get("significant_delta", 999)
    return abs(delta) >= threshold


def compare_analyses(result_before: Dict, result_after: Dict,
                     label_before: str = "之前",
                     label_after: str = "之后") -> ComparisonResult:
    """
    Compare two analysis results.
    
    Args:
        result_before: Dict from metrics.summary() or analysis JSON
        result_after: Dict from metrics.summary() or analysis JSON
        label_before: Display label for first run
        label_after: Display label for second run
    
    Returns:
        ComparisonResult with all deltas and insights
    """
    cr = ComparisonResult(
        before_label=label_before,
        after_label=label_after,
    )
    
    # Handle both full result dicts and summary dicts
    def _get(result, key):
        # Try direct first, then drill into nested
        if key in result:
            return result.get(key)
        # Check nested: result["result"]["metrics"][key]
        if "result" in result and isinstance(result["result"], dict):
            r = result["result"]
            if "metrics" in r and key in r["metrics"]:
                return r["metrics"][key]
            if key in r:
                return r[key]
        return None
    
    # Scores (from get_scoring output)
    score_before = _get(result_before, "overall_score")
    score_after = _get(result_after, "overall_score")
    if score_before is not None:
        cr.score_before = float(score_before)
    if score_after is not None:
        cr.score_after = float(score_after)
    if cr.score_before is not None and cr.score_after is not None:
        cr.score_delta = round(cr.score_after - cr.score_before, 1)
    
    # Per-metric deltas
    for key, info in METRIC_INFO.items():
        bv = _get(result_before, key)
        av = _get(result_after, key)
        
        if bv is None and av is None:
            continue
        
        delta = None
        delta_pct = None
        if bv is not None and av is not None:
            delta = round(float(av) - float(bv), 2)
            if float(bv) != 0:
                delta_pct = round(delta / float(bv) * 100, 1)
        
        is_imp = _is_improvement(key, delta or 0, float(bv or 0))
        is_sig = _is_significant(key, delta or 0)
        
        reason = ""
        if is_sig and is_imp:
            reason = f"✅ 显著改善"
        elif is_sig and not is_imp and is_imp is not None:
            reason = f"⚠️ 明显退步"
        elif is_sig:
            reason = "变化明显"
        
        cr.deltas.append(MetricDelta(
            name=key,
            label=info["label"],
            unit=info["unit"],
            before_value=float(bv) if bv is not None else None,
            after_value=float(av) if av is not None else None,
            delta=delta,
            delta_pct=delta_pct,
            is_improvement=is_imp if is_imp is not None else True,
            is_significant=is_sig,
            significance_reason=reason,
        ))
    
    # Generate insights
    cr.insights = _generate_insights(cr)
    
    return cr


def _generate_insights(cr: ComparisonResult) -> List[str]:
    """Auto-generate insights from comparison data."""
    insights = []
    
    # Overall score change
    if cr.score_delta is not None:
        if cr.score_delta > 5:
            insights.append(f"🎉 总体跑姿评分提升 {cr.score_delta:.0f} 分！训练效果明显。")
        elif cr.score_delta > 0:
            insights.append(f"👍 总体评分提升 {cr.score_delta:.0f} 分，稳定进步中。")
        elif cr.score_delta < -5:
            insights.append(f"⚠️ 总体评分下降 {abs(cr.score_delta):.0f} 分，可能需要调整训练计划。")
        elif cr.score_delta < 0:
            insights.append(f"📉 总体评分略有下降（{cr.score_delta:.0f} 分），疲劳或状态波动？")
        else:
            insights.append("⏸️ 总体评分不变，维持稳定。")
    
    # Significant changes
    improvements = [d for d in cr.deltas if d.is_improvement and d.is_significant]
    regressions = [d for d in cr.deltas if not d.is_improvement and d.is_significant]
    
    if improvements:
        imp_names = "、".join([d.label for d in improvements[:3]])
        insights.append(f"✅ {imp_names}有显著改善，继续保持！")
    
    if regressions:
        reg_names = "、".join([d.label for d in regressions[:3]])
        insight_hints = {
            "垂直振幅": "试试加入弹跳训练和核心稳定性练习",
            "髋部倾斜": "加强臀中肌训练（蚌式开合、侧抬腿）",
            "触地距离": "主动缩小步幅，让脚落在身体正下方",
            "步频": "用180BPM节拍器跑步热身",
            "手臂对称": "照镜子练习摆臂，肘部90°前后摆动",
        }
        for d in regressions[:3]:
            hint = insight_hints.get(d.label, "注意这个指标，下次训练重点调整")
            insights.append(f"⚠️ {d.label}下降（{d.delta:+.1f}{d.unit}）。{hint}")
    
    # Fatigue comparison
    # (skipped for now - would need fatigue results from both runs)
    
    return insights


def format_comparison_report(cr: ComparisonResult) -> str:
    """Format comparison as printable text."""
    lines = []
    lines.append("=" * 60)
    lines.append("📊 跨视频跑姿对比报告")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"📅 {cr.before_label}  →  {cr.after_label}")
    lines.append("")
    
    # Score
    if cr.score_delta is not None:
        arrow = "▲" if cr.score_delta > 0 else "▼"
        lines.append(f"🏆 总体评分：{cr.score_before} → {cr.score_after}  ({arrow} {abs(cr.score_delta):.1f})")
        lines.append("")
    
    # Metric table
    lines.append(f"{'指标':<14} {'之前':>8} {'之后':>8} {'变化':>10} {'评价':>12}")
    lines.append("-" * 56)
    
    for d in cr.deltas:
        if d.before_value is None and d.after_value is None:
            continue
        
        b_str = f"{d.before_value:.1f}" if d.before_value is not None else "N/A"
        a_str = f"{d.after_value:.1f}" if d.after_value is not None else "N/A"
        
        if d.delta is not None:
            arrow = "▲" if d.delta > 0 else "▼"
            delta_str = f"{arrow} {abs(d.delta):.1f}"
        else:
            delta_str = "N/A"
        
        if d.is_improvement and d.is_significant:
            status = "✅ 改善"
        elif d.is_significant:
            status = "⚠️ 退步"
        else:
            status = "→ 稳定"
        
        lines.append(f"{d.label:<14} {b_str:>8} {a_str:>8} {delta_str:>10} {status:>12}")
    
    lines.append("")
    
    # Insights
    if cr.insights:
        lines.append("💡 分析结论：")
        for insight in cr.insights:
            lines.append(f"  {insight}")
        lines.append("")
    
    lines.append("=" * 60)
    return "\n".join(lines)


# ── JSON serialization ──

def comparison_to_dict(cr: ComparisonResult) -> Dict:
    """Serialize comparison result for API / storage."""
    return {
        "before_label": cr.before_label,
        "after_label": cr.after_label,
        "score_before": cr.score_before,
        "score_after": cr.score_after,
        "score_delta": cr.score_delta,
        "deltas": [
            {
                "name": d.name,
                "label": d.label,
                "unit": d.unit,
                "before": d.before_value,
                "after": d.after_value,
                "delta": d.delta,
                "delta_pct": d.delta_pct,
                "is_improvement": d.is_improvement,
                "is_significant": d.is_significant,
            }
            for d in cr.deltas
        ],
        "insights": cr.insights,
    }
