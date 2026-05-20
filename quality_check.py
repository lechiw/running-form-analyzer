"""
quality_check.py - Video Quality Assessment for Running Form Analysis
Checks if the video is suitable for accurate biomechanical analysis.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pose_extractor import PoseSequence, PoseLandmarks
from metrics import L as LM, _midpoint


@dataclass
class QualityReport:
    """Assessment of video quality for running form analysis."""
    passed: bool
    score: float  # 0-100

    # Per-check results
    is_side_view: bool
    torso_orientation_ratio: Optional[float]
    has_good_visibility: bool
    runner_in_frame: bool
    enough_frames: bool

    # Human-readable messages
    summary: str  # One-line summary
    issues: List[str]  # List of issues to fix
    tips: List[str]  # Shooting tips

    # Guidance
    shooting_guide: str  # Full guidance text


# Standard shooting guide (ASCII art included)
SHOOTING_GUIDE = """
📸 **标准拍摄指南**

想获得准确的跑姿分析，请这样拍：

1️⃣ **拍摄位置**
```
       手机（横屏，固定在三脚架上）
           📱
    [侧面]  ← 关键！
           
       🏃 →  跑  → 🏃
     跑步机或跑道

  距离：2-3 米
  高度：手机与腰部齐平
```

2️⃣ **拍摄要求**
   ✅ 横屏拍摄（16:9）
   ✅ 手机固定（三脚架或靠墙）
   ✅ 侧面视角（能看到完整侧身轮廓）
   ✅ 跑步者居中，全身入画
   ✅ 拍摄 20-30 秒即可

3️⃣ **着装建议**
   👕 穿浅色/亮色紧身运动服
   ❌ 不要穿宽松深色衣服（会降低检测精度）

4️⃣ **常见错误**
   ❌ 正对/背对镜头 → 看不到跑姿侧面特征
   ❌ 手持拍摄 → 画面抖动影响数据
   ❌ 距离太远 → 骨架关键点太小
   ❌ 距离太近 → 手脚出画
"""


class VideoQualityChecker:
    """Check if a video is suitable for running form analysis."""

    def __init__(self):
        pass

    def check(self, seq: PoseSequence, sample_frames: int = 10) -> QualityReport:
        """
        Analyze a pose sequence and assess video quality.

        Args:
            seq: Extracted pose sequence
            sample_frames: Number of frames to sample for quality check

        Returns:
            QualityReport with pass/fail and guidance
        """
        issues = []
        tips = []
        checks = {}

        # Need at least some pose data
        if len(seq.landmarks_seq) < 3:
            return QualityReport(
                passed=False, score=0,
                is_side_view=False, torso_orientation_ratio=None,
                has_good_visibility=False, runner_in_frame=False,
                enough_frames=False,
                summary="❌ 未检测到人体骨架",
                issues=["视频中未检测到人体，请确保画面中有人"],
                tips=[],
                shooting_guide=SHOOTING_GUIDE,
            )

        # Sample frames evenly from the sequence
        step = max(1, len(seq.landmarks_seq) // sample_frames)
        sampled = seq.landmarks_seq[::step][:sample_frames]

        # Check 1: Torso orientation (side view check)
        orientation_ratios = []
        for pl in sampled:
            ratio = self._torso_orientation_ratio(pl)
            if ratio is not None:
                orientation_ratios.append(ratio)

        avg_ratio = float(np.mean(orientation_ratios)) if orientation_ratios else None
        is_side_view = avg_ratio is not None and 0.2 < avg_ratio < 2.0
        # ratio < 0.2 = torso near-vertical (might be front/back view)
        # ratio 0.2-2.0 = reasonable side view
        # ratio > 2.0 = torso horizontal, likely facing camera

        if avg_ratio is not None:
            if avg_ratio > 2.0:
                issues.append(f"躯干在画面中接近水平（比率: {avg_ratio:.1f}），可能是正对或背对镜头拍摄")
                tips.append("请从侧面拍摄，手机与跑步者成 90 度角")
            elif avg_ratio < 0.2:
                issues.append(f"躯干在画面中接近垂直（比率: {avg_ratio:.1f}），可能拍摄角度有偏差")
                tips.append("确保画面中能看到跑步者的完整侧身轮廓")

        # Check 2: Landmark visibility
        visibility_scores = []
        for pl in sampled:
            vis = pl.visibility
            key_joints = [LM["NOSE"], LM["L_SHOULDER"], LM["R_SHOULDER"],
                         LM["L_HIP"], LM["R_HIP"],
                         LM["L_KNEE"], LM["R_KNEE"],
                         LM["L_ANKLE"], LM["R_ANKLE"]]
            joint_vis = [vis[j] for j in key_joints if j < len(vis)]
            visibility_scores.append(float(np.mean(joint_vis)))

        avg_visibility = float(np.mean(visibility_scores)) if visibility_scores else 0
        has_good_visibility = avg_visibility > 0.7

        if not has_good_visibility:
            issues.append(f"骨架检测可信度偏低（平均: {avg_visibility:.0%}）")
            tips.append("建议穿浅色紧身运动服，避免深色宽松衣物")
            tips.append("拍摄距离不要过远，确保身体占画面高度的 1/3 以上")

        # Check 3: Is runner in frame (not too close/too far)
        # Rough check: torso should be 100-500 pixels in a 1080p video
        in_frame = True
        torso_sizes = []
        for pl in sampled:
            lm = pl.landmarks
            if (pl.visibility[LM["L_SHOULDER"]] > 0.5 and
                pl.visibility[LM["R_SHOULDER"]] > 0.5 and
                pl.visibility[LM["L_HIP"]] > 0.5 and
                pl.visibility[LM["R_HIP"]] > 0.5):
                ms = _midpoint(lm[LM["L_SHOULDER"]], lm[LM["R_SHOULDER"]])
                mh = _midpoint(lm[LM["L_HIP"]], lm[LM["R_HIP"]])
                size = float(np.linalg.norm(ms[:2] - mh[:2]))
                torso_sizes.append(size)

        if torso_sizes:
            avg_torso = float(np.mean(torso_sizes))
            if avg_torso < 50:
                issues.append(f"画面中人物太小（躯干仅 {avg_torso:.0f}px），关键点精度不够")
                tips.append("请靠近拍摄，或使用更高分辨率的视频")
                in_frame = False
            elif avg_torso > 500:
                issues.append(f"画面中人物太大（躯干 {avg_torso:.0f}px），可能部分身体出画")
                tips.append("请拉远距离，确保全身都在画面内")
                in_frame = False

        # Check 4: Enough frames with pose data
        enough_frames = len(seq.landmarks_seq) >= 30  # at least 1 second

        # Calculate overall score
        score_parts = []
        if is_side_view:
            score_parts.append(40)
        elif avg_ratio is not None:
            score_parts.append(10)
        else:
            score_parts.append(0)

        score_parts.append(30 if has_good_visibility else 5)
        score_parts.append(20 if in_frame else 0)
        score_parts.append(10 if enough_frames else 0)

        score = sum(score_parts)

        # Pass threshold
        passed = (is_side_view and has_good_visibility and in_frame and
                  score >= 70)

        # Summary
        if passed:
            summary = "✅ 视频质量合格，适合跑姿分析"
        elif is_side_view and not has_good_visibility:
            summary = "⚠️ 拍摄角度正确，但骨架可见度偏低"
        elif not is_side_view:
            summary = "❌ 非侧面视角，跑姿分析结果将不准确"
        else:
            summary = "⚠️ 视频质量有待改善"

        return QualityReport(
            passed=passed,
            score=score,
            is_side_view=is_side_view,
            torso_orientation_ratio=avg_ratio,
            has_good_visibility=has_good_visibility,
            runner_in_frame=in_frame,
            enough_frames=enough_frames,
            summary=summary,
            issues=issues,
            tips=tips,
            shooting_guide=SHOOTING_GUIDE,
        )

    def _torso_orientation_ratio(self, pl: PoseLandmarks) -> Optional[float]:
        """
        Compute torso aspect ratio.
        ratio = |shoulder_x - hip_x| / |shoulder_y - hip_y|
        
        Side view: ratio ~ 0.5-1.5 (torso is vertical)
        Front/back view: ratio > 3.0 (torso appears horizontal)
        """
        lm = pl.landmarks
        vis = pl.visibility

        if (vis[LM["L_SHOULDER"]] < 0.5 or vis[LM["R_SHOULDER"]] < 0.5 or
            vis[LM["L_HIP"]] < 0.5 or vis[LM["R_HIP"]] < 0.5):
            return None

        ms = _midpoint(lm[LM["L_SHOULDER"]], lm[LM["R_SHOULDER"]])
        mh = _midpoint(lm[LM["L_HIP"]], lm[LM["R_HIP"]])

        dx = abs(ms[0] - mh[0])
        dy = abs(ms[1] - mh[1])

        if dy < 1:
            return 100.0  # nearly horizontal = definitely not side view

        return dx / dy


def print_quality_report(report: QualityReport) -> str:
    """Format a QualityReport as a printable string."""
    lines = []
    lines.append("=" * 60)
    lines.append("🎥 拍摄质量检查")
    lines.append("=" * 60)
    lines.append("")
    lines.append(report.summary)
    lines.append(f"   综合评分：{report.score}/100")
    lines.append("")

    if report.issues:
        lines.append("📋 发现的问题：")
        for issue in report.issues:
            lines.append(f"  ❌ {issue}")
        lines.append("")

    if report.tips:
        lines.append("💡 改善建议：")
        for tip in report.tips:
            lines.append(f"  • {tip}")
        lines.append("")

    if not report.passed:
        lines.append(report.shooting_guide)

    lines.append("=" * 60)

    return "\n".join(lines)
