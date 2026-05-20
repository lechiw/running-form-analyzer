"""
fatigue_analyzer.py - Running Fatigue Analysis
Compares running form between early and late segments of a run
to detect fatigue-induced form deterioration.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from metrics import RunningMetrics, RunningMetricsCalculator, FrameMetrics
from pose_extractor import PoseSequence


# Fatigue thresholds for key metrics
FATIGUE_THRESHOLDS = {
    "cadence_spm": {"drop": -5, "label": "步频"},
    "trunk_lean_deg": {"drop": -3, "label": "躯干前倾角"},
    "vertical_oscillation_cm": {"rise": 3, "label": "垂直振幅"},
    "foot_strike_distance_cm": {"rise": 5, "label": "触地距离"},
    "arm_symmetry_score": {"drop": -15, "label": "手臂对称性"},
}

# Detailed risk descriptions
FATIGUE_SIGNALS = {
    "cadence_spm": {
        "direction": "drop",
        "message": "步频显著下降，说明肌肉力量不足以维持高步频，开始依赖骨骼和韧带支撑",
        "advice": "加强小腿和踝关节力量训练（跳绳、提踵）。长跑中注意主动维持步频。",
    },
    "trunk_lean_deg": {
        "direction": "drop",
        "message": "躯干前倾角减小（上身直立），说明核心力量下降，跑步经济性降低",
        "advice": "加强核心训练（平板支撑、鸟狗式）。跑步时有意识地保持微微前倾。",
    },
    "vertical_oscillation_cm": {
        "direction": "rise",
        "message": "垂直振幅增大，说明核心失稳，能量浪费在上下弹跳而不是前进",
        "advice": "想象头顶有天花板，减少弹跳。加强臀部和核心力量。",
    },
    "foot_strike_distance_cm": {
        "direction": "rise",
        "message": "触地距离增大（过度跨步加重），疲劳时步幅失控，损伤风险升高",
        "advice": "疲劳时主动缩小步幅，用节拍器稳定步频。避免在疲劳时追求速度。",
    },
    "arm_symmetry_score": {
        "direction": "drop",
        "message": "手臂对称性下降，躯干旋转代偿增加，跑姿开始不对称",
        "advice": "放松肩部，有意注意摆臂节奏。交叉训练改善左右侧力量平衡。",
    },
}


@dataclass
class FatigueDelta:
    """Difference (delta) for a single metric between baseline and fatigue."""
    name: str  # Metric key
    label: str  # Chinese name
    baseline: Optional[float]
    fatigue: Optional[float]
    delta: Optional[float]  # fatigue - baseline
    threshold: Optional[float]  # threshold for significance
    is_significant: bool  # True if delta exceeds threshold
    is_degradation: bool  # True if change is bad (direction matters)


@dataclass
class FatigueReport:
    """Complete fatigue analysis results."""
    deltas: List[FatigueDelta] = field(default_factory=list)

    # Fatigue level
    fatigue_level: str = "normal"  # normal / mild / moderate / severe
    fatigue_score: float = 0.0  # 0-100

    # Segment info
    baseline_duration: float = 0.0
    fatigue_duration: float = 0.0

    # Significant changes (filtered)
    significant_deltas: List[FatigueDelta] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Serialize to dict for LLM analysis."""
        return {
            "fatigue_level": self.fatigue_level,
            "fatigue_score": round(self.fatigue_score, 1),
            "baseline_duration_sec": round(self.baseline_duration, 1),
            "fatigue_duration_sec": round(self.fatigue_duration, 1),
            "deltas": [
                {
                    "metric": d.label,
                    "baseline": d.baseline,
                    "fatigue": d.fatigue,
                    "change": round(d.delta, 1) if d.delta is not None else None,
                    "significant": d.is_significant,
                }
                for d in self.deltas
            ],
        }


class FatigueAnalyzer:
    """
    Compares running form between early and late segments of a run.
    """

    def __init__(self, baseline_ratio: float = 0.25, fatigue_ratio: float = 0.25):
        """
        Args:
            baseline_ratio: Portion at start to use as baseline (e.g., 0.25 = first 25%)
            fatigue_ratio: Portion at end to use as fatigue comparison
        """
        self.baseline_ratio = baseline_ratio
        self.fatigue_ratio = fatigue_ratio

    def analyze(self, seq: PoseSequence) -> FatigueReport:
        """
        Full fatigue analysis pipeline.

        Args:
            seq: Full pose sequence from the video

        Returns:
            FatigueReport with deltas and fatigue level
        """
        n = len(seq.landmarks_seq)
        if n < 60:  # Need at least 2 seconds of data
            return FatigueReport(fatigue_level="insufficient_data")

        # Split sequence
        baseline_end = int(n * self.baseline_ratio)
        fatigue_start = int(n * (1 - self.fatigue_ratio))

        baseline_seq = self._slice_sequence(seq, 0, baseline_end)
        fatigue_seq = self._slice_sequence(seq, fatigue_start, n)

        # Compute metrics for each segment
        calculator = RunningMetricsCalculator()
        baseline_metrics = calculator.compute(baseline_seq)
        fatigue_metrics = calculator.compute(fatigue_seq)

        # Compute deltas
        report = self._compute_deltas(baseline_metrics, fatigue_metrics)
        report.baseline_duration = baseline_metrics.duration_sec
        report.fatigue_duration = fatigue_metrics.duration_sec

        # Classify fatigue level
        report.fatigue_level, report.fatigue_score = self._classify_fatigue(report)

        return report

    def _slice_sequence(self, seq: PoseSequence, start: int, end: int) -> PoseSequence:
        """Extract a slice of the pose sequence."""
        sliced = PoseSequence(
            fps=seq.fps,
            total_frames=end - start,
            video_path=seq.video_path,
        )
        sliced.landmarks_seq = seq.landmarks_seq[start:end]
        return sliced

    def _compute_deltas(self, baseline: RunningMetrics,
                        fatigue: RunningMetrics) -> FatigueReport:
        """Compute deltas between baseline and fatigue metrics."""
        report = FatigueReport()

        # Map metric names to their accessors
        metrics_map = {
            "cadence_spm": (baseline.cadence_avg, fatigue.cadence_avg),
            "trunk_lean_deg": (baseline.trunk_lean_avg, fatigue.trunk_lean_avg),
            "vertical_oscillation_cm": (baseline.vertical_oscillation, fatigue.vertical_oscillation),
            "foot_strike_distance_cm": (baseline.avg_foot_strike_distance, fatigue.avg_foot_strike_distance),
            "arm_symmetry_score": (baseline.arm_symmetry_avg, fatigue.arm_symmetry_avg),
        }

        for key, (base_val, fatigue_val) in metrics_map.items():
            info = FATIGUE_THRESHOLDS.get(key)
            if not info:
                continue

            label = info["label"]
            delta = None
            is_significant = False

            if base_val is not None and fatigue_val is not None:
                delta = fatigue_val - base_val

                # Check if change exceeds threshold
                threshold = info.get("drop") or info.get("rise", 0)
                if "drop" in info:
                    is_significant = delta < threshold  # negative change
                elif "rise" in info:
                    is_significant = delta > threshold  # positive change

            is_degradation = self._is_degradation(key, delta)

            report.deltas.append(FatigueDelta(
                name=key,
                label=label,
                baseline=base_val,
                fatigue=fatigue_val,
                delta=delta,
                threshold=info.get("drop") or info.get("rise"),
                is_significant=is_significant,
                is_degradation=is_degradation,
            ))

        # Find significant deltas
        report.significant_deltas = [d for d in report.deltas if d.is_significant]

        return report

    def _is_degradation(self, key: str, delta: Optional[float]) -> bool:
        """Determine if a delta represents degradation."""
        if delta is None:
            return False

        signal = FATIGUE_SIGNALS.get(key)
        if not signal:
            return False

        if signal["direction"] == "drop":
            return delta < 0  # negative change = bad (cadence dropping, etc.)
        elif signal["direction"] == "rise":
            return delta > 0  # positive change = bad (oscillation increasing)
        return False

    def _classify_fatigue(self, report: FatigueReport) -> Tuple[str, float]:
        """Classify fatigue level based on significant deltas."""
        if not report.deltas:
            return "normal", 0.0

        n_significant = len(report.significant_deltas)
        n_total = len([d for d in report.deltas if d.baseline is not None])

        if n_total == 0:
            return "normal", 0.0

        # Calculate fatigue score: 0-100 based on how many metrics degraded
        severity = n_significant / n_total

        # Also factor in the magnitude of changes
        magnitude_score = 0.0
        for d in report.significant_deltas:
            if d.delta is not None and d.threshold:
                # How far past the threshold (capped at 3x)
                excess = min(abs(d.delta) / abs(d.threshold), 3.0)
                magnitude_score += excess

        avg_magnitude = magnitude_score / max(n_total, 1)

        # Combined score
        score = (severity * 50 + min(avg_magnitude / 3.0, 1.0) * 50)

        if score < 15:
            return "normal", score
        elif score < 35:
            return "mild", score
        elif score < 60:
            return "moderate", score
        else:
            return "severe", score


def format_fatigue_report(report: FatigueReport) -> str:
    """Format fatigue analysis as a printable string."""
    lines = []
    lines.append("=" * 60)
    lines.append("🔄 疲劳对比分析")
    lines.append("=" * 60)
    lines.append("")

    if report.fatigue_level == "insufficient_data":
        lines.append("⚠️ 视频太短（<2秒），无法进行疲劳分析")
        lines.append("   建议拍摄至少 20 秒以上的连续跑步视频")
        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    # Fatigue level
    level_icons = {
        "normal": "✅",
        "mild": "⚡",
        "moderate": "⚠️",
        "severe": "🔴",
    }
    level_names = {
        "normal": "正常（无明显疲劳）",
        "mild": "轻度疲劳",
        "moderate": "中度疲劳",
        "severe": "重度疲劳",
    }

    icon = level_icons.get(report.fatigue_level, "?")
    name = level_names.get(report.fatigue_level, "未知")
    lines.append(f"{icon} 疲劳等级：{name}（评分：{report.fatigue_score:.0f}/100）")
    lines.append("")

    # Segment info
    lines.append(f"📐 对比区间：前段 {report.baseline_duration:.1f}s → 后段 {report.fatigue_duration:.1f}s")
    lines.append("")

    # Delta table
    if report.deltas:
        lines.append(f"{'指标':<16} {'前段':>8} {'后段':>8} {'变化':>8} {'状态':>8}")
        lines.append("-" * 52)

        for d in report.deltas:
            if d.baseline is None and d.fatigue is None:
                continue

            label = d.label
            base_str = f"{d.baseline:.1f}" if d.baseline is not None else "N/A"
            fatigue_str = f"{d.fatigue:.1f}" if d.fatigue is not None else "N/A"

            if d.delta is not None:
                arrow = "▼" if d.delta < 0 else "▲"
                delta_str = f"{arrow} {abs(d.delta):.1f}"
            else:
                delta_str = "N/A"

            if d.is_significant and d.is_degradation:
                status = "⚠️ 退化"
            elif d.is_significant and not d.is_degradation:
                status = "✅ 改善"
            else:
                status = "→ 正常"

            lines.append(f"{label:<16} {base_str:>8} {fatigue_str:>8} {delta_str:>8} {status:>8}")

        lines.append("")

    # Significant findings
    if report.significant_deltas:
        lines.append("📋 显著变化：")
        for d in report.significant_deltas:
            signal = FATIGUE_SIGNALS.get(d.name)
            if signal:
                lines.append(f"\n  ⚡ {d.label} 变化 {d.delta:+.1f}")
                lines.append(f"     {signal['message']}")
                lines.append(f"     💡 {signal['advice']}")
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


# Additional LLM prompt for fatigue analysis
FATIGUE_LLM_PROMPT = """
## 疲劳分析数据
- 疲劳等级：{fatigue_level}
- 疲劳评分：{fatigue_score}/100
- 前段时长：{baseline_duration}秒
- 后段时长：{fatigue_duration}秒

### 各指标变化
{metric_changes}

请结合上述疲劳数据，在"🎯 本周训练建议"部分加入针对疲劳管理的训练建议。
"""
