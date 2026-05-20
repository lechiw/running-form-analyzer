"""
analyzer.py - AI Running Form Report Generator
Uses an LLM to generate natural language feedback from running metrics.
"""

from typing import Dict, Optional
from metrics import RunningMetrics, RunningMetricsCalculator


# System prompt for the running form AI coach
RUNNING_COACH_SYSTEM_PROMPT = """你是一名跑步生物力学专家和资深跑步教练。你的任务是根据跑姿数据，为跑者提供专业、具体、可行的改进建议。

分析原则：
1. **不要泛泛而谈**：每条建议必须基于具体数据
2. **优先级排序**：先解决受伤风险最高的问题，再谈效率优化
3. **可执行**：每个问题都要配一个具体的练习方法
4. **正向鼓励**：先说做得好的，再指出改进空间
5. **专业但不晦涩**：用跑者能听懂的语言

输出格式：
## 📊 跑姿评分
总体评分：XX/100
各维度得分：

## ✅ 做得好的地方
- ...

## ⚠️ 需要改进
- **问题描述**：[具体问题]
- **数据支撑**：[你的数据显示...]
- **改进方法**：[具体练习]
- **优先级**：高/中/低

## 🎯 本周训练建议
- [可执行的1-3条建议]
"""


class AIRunningCoach:
    """Generates AI-powered running form analysis reports."""

    def __init__(self, llm_api_func=None):
        """
        Args:
            llm_api_func: Function that takes (system_prompt, user_prompt) and returns text.
                          Falls back to a template-based report if None.
        """
        self.llm = llm_api_func

    def generate_report(self, metrics: RunningMetrics,
                        scoring: Dict) -> str:
        """Generate a full running form analysis report."""
        summary = metrics.summary()
        score_info = scoring.get("details", {})

        if self.llm:
            return self._generate_llm_report(summary, score_info, scoring)
        else:
            return self._generate_template_report(summary, score_info, scoring)

    def _generate_llm_report(self, summary: Dict, details: Dict,
                              scoring: Dict) -> str:
        """Generate report using LLM."""
        overall = scoring.get("overall_score", "N/A")

        user_prompt = f"""请分析以下跑姿数据，给出专业反馈：

## 跑者数据
- 步频：{summary.get('cadence_spm', 'N/A')} spm
- 躯干前倾角：{summary.get('trunk_lean_deg', 'N/A')}°
- 垂直振幅：{summary.get('vertical_oscillation_cm', 'N/A')} cm
- 手臂对称性评分：{summary.get('arm_symmetry_score', 'N/A')}/100
- 触地距离：{summary.get('foot_strike_distance_cm', 'N/A')} cm（超过15cm为过度跨步）
- 着地方式：{summary.get('foot_strike_type', 'N/A')}
- 左膝角度：{summary.get('avg_left_knee_angle_deg', 'N/A')}°
- 右膝角度：{summary.get('avg_right_knee_angle_deg', 'N/A')}°
- 总步态周期数：{summary.get('total_gait_cycles', 'N/A')}
- 视频时长：{summary.get('duration_sec', 'N/A')} 秒

## 各维度评分
{chr(10).join(f'- {k}: {v}' for k, v in details.items())}

## 总体跑姿评分：{overall}/100

请按照系统提示的格式，输出完整的分析报告。"""

        try:
            response = self.llm(RUNNING_COACH_SYSTEM_PROMPT, user_prompt)
            return response
        except Exception as e:
            return f"⚠️ AI分析生成失败：{e}\n\n" + self._generate_template_report(summary, details, scoring)

    def _generate_template_report(self, summary: Dict, details: Dict,
                                   scoring: Dict) -> str:
        """Generate a template-based report (fallback when no LLM)."""
        overall = scoring.get("overall_score", "N/A")
        metric_scores = scoring.get("metrics", {})

        lines = []
        lines.append("📊 **跑姿分析报告**")
        lines.append(f"总体评分：**{overall}/100**")
        lines.append("")

        # Find top weaknesses
        weaknesses = sorted(
            [(k, v) for k, v in metric_scores.items()],
            key=lambda x: x[1]
        ) if metric_scores else []

        # Good points (score >= 80)
        goods = [(k, v) for k, v in weaknesses if v >= 80]
        needs = [(k, v) for k, v in weaknesses if v < 80]

        if goods:
            lines.append("✅ **做得好的地方**")
            for k, v in goods:
                name_map = {
                    "cadence": "步频",
                    "trunk_lean": "躯干姿态",
                    "arm_symmetry": "手臂对称性",
                    "vertical_oscillation": "垂直振幅",
                    "foot_strike": "触地控制",
                }
                lines.append(f"  • **{name_map.get(k, k)}**: {v:.0f}/100 — 保持！")
            lines.append("")

        if needs:
            lines.append("⚠️ **需要改进**")
            for k, v in needs:
                detail = details.get(k, "")
                if k == "cadence":
                    lines.append(f"  • **步频** ({v:.0f}/100)：{detail}")
                    lines.append(f"    → 建议：使用180 BPM节拍器跑步，逐步提升步频")
                elif k == "trunk_lean":
                    lines.append(f"  • **躯干前倾** ({v:.0f}/100)：{detail}")
                    lines.append(f"    → 建议：保持核心收紧，想象胸口有束光射向正前方10米地面")
                elif k == "arm_symmetry":
                    lines.append(f"  • **手臂对称性** ({v:.0f}/100)：{detail}")
                    lines.append(f"    → 建议：照镜子练习摆臂，肘部90°前后摆动，不交叉过中线")
                elif k == "vertical_oscillation":
                    lines.append(f"  • **垂直振幅** ({v:.0f}/100)：{detail}")
                    lines.append(f"    → 建议：想象头顶有天花板，尽量减少弹跳")
                elif k == "foot_strike":
                    lines.append(f"  • **触地控制** ({v:.0f}/100)：{detail}")
                    lines.append(f"    → 建议：缩短步幅，加快步频，让脚落在身体正下方")
                lines.append("")

        lines.append("🎯 **本周训练建议**")
        lines.append("  1. 如果步频偏低：跑前热身时用180BPM节拍器跑2分钟适应")
        lines.append("  2. 注意恢复：每周增加跑量不超过10%")
        lines.append("  3. 力量训练：每周2次核心+腿部力量，预防跑步损伤")

        return "\n".join(lines)
