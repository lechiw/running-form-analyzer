"""
analyzer.py - AI Running Form Report Generator
Generates natural language reports from running metrics.
Supports LLM-powered reports (DeepSeek/OpenAI) with template fallback.
"""

import os
from typing import Dict, Optional, Callable, Any
from metrics import RunningMetrics


# System prompt for the running form AI coach
RUNNING_COACH_SYSTEM_PROMPT = """你是一名跑步生物力学专家和资深跑步教练。你的任务是根据跑姿数据，为跑者提供专业、具体、可行的改进建议。

分析原则：
1. **不要泛泛而谈**：每条建议必须基于具体数据
2. **优先级排序**：先解决受伤风险最高的问题，再谈效率优化
3. **可执行**：每个问题都要配一个具体的练习方法
4. **正向鼓励**：先说做得好的，再指出改进空间
5. **专业但不晦涩**：用跑者能听懂的语言
6. **使用中文**：全文用中文输出

输出格式：
## 📊 跑姿评分
总体评分：XX/100
各维度得分：（列出各维度及分数）

## ✅ 做得好的地方
- [具体说明，至少一条]

## ⚠️ 需要改进
按优先级排列，每条包含：
- **问题描述**：[具体问题]
- **数据支撑**：[你的数据显示...]
- **改进方法**：[具体可执行的练习或调整]
- **优先级**：高/中/低

## 🎯 本周训练建议
- 给出 2-3 条具体的、可执行的训练建议

如果提供了疲劳分析数据，必须在"训练建议"部分参考疲劳分析结果给出针对性建议。

注意：如果某些数据标记为"N/A"或明显异常（如垂直振幅>50cm），说明该指标因拍摄角度问题未获取到。请在报告中说明这一点，不要强行分析不可靠的数据。"""


class AIRunningCoach:
    """Generates AI-powered running form analysis reports."""

    def __init__(self, llm_api_func: Optional[Callable] = None,
                 fatigue_report: Optional[Dict] = None):
        """
        Args:
            llm_api_func: Function (system_prompt, user_prompt) -> str.
            fatigue_report: Optional dict from FatigueReport.to_dict()
        """
        self.llm = llm_api_func
        self.fatigue_report = fatigue_report

    def generate_report(self, metrics: RunningMetrics, scoring: Dict) -> str:
        """Generate a full running form analysis report."""
        summary = metrics.summary()
        score_info = scoring.get("details", {})

        if self.llm:
            try:
                return self._generate_llm_report(summary, score_info, scoring)
            except Exception as e:
                error_msg = f"⚠️ LLM 报告生成失败：{e}，使用模板报告。\n\n"
                return error_msg + self._generate_template_report(summary, score_info, scoring)
        else:
            return self._generate_template_report(summary, score_info, scoring)

    def _generate_llm_report(self, summary: Dict, details: Dict,
                              scoring: Dict) -> str:
        """Generate report using LLM."""
        overall = scoring.get("overall_score", "N/A")

        # Filter out unreliable metrics
        notes = []
        unreliable = []
        if summary.get("vertical_oscillation_cm") and summary["vertical_oscillation_cm"] > 25:
            unreliable.append("垂直振幅（数据异常，>25cm超出正常范围，可能为拍摄角度导致）")
        if not summary.get("cadence_spm"):
            unreliable.append("步频（未获取到，建议用标准侧面视角重拍）")
        if not summary.get("trunk_lean_deg"):
            unreliable.append("躯干前倾角（未获取到，建议用标准侧面视角重拍）")

        if unreliable:
            notes.append("以下指标因拍摄条件限制数据不可靠：" + "、".join(unreliable))

        # Fatigue section
        fatigue_section = ""
        if self.fatigue_report and self.fatigue_report.get("fatigue_level"):
            fr = self.fatigue_report
            fatigue_section = f"""
## 疲劳分析数据
- 疲劳等级：{fr.get('fatigue_level', 'N/A')}
- 疲劳评分：{fr.get('fatigue_score', 'N/A')}/100
- 前段时长：{fr.get('baseline_duration_sec', 'N/A')}秒
- 后段时长：{fr.get('fatigue_duration_sec', 'N/A')}秒

### 各指标前后段变化
"""
            for d in fr.get("deltas", []):
                change = d.get("change")
                if change is not None:
                    arrow = "▼" if change < 0 else "▲"
                    fatigue_section += f"- {d['metric']}: {d['baseline']} → {d['fatigue']} ({arrow} {abs(change):.1f})\n"

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
- 骨盆倾斜/髋部下坠：{summary.get('hip_drop_cm', 'N/A')} cm（>2cm需关注）
- 触地时间：{summary.get('ground_contact_time_ms', 'N/A')} ms
- 步幅估计：{summary.get('estimated_step_length_cm', 'N/A')} cm
- 支撑相比例：{summary.get('stance_ratio', 'N/A')}

## 各维度评分
{chr(10).join(f'- {k}: {v}' for k, v in details.items())}

## 总体跑姿评分：{overall}/100

## 备注
{' | '.join(notes) if notes else '无'}
{fatigue_section}
请按照系统提示的输出格式，输出完整的分析报告。"""

        response = self.llm(RUNNING_COACH_SYSTEM_PROMPT, user_prompt)
        return response.strip()

    def _generate_template_report(self, summary: Dict, details: Dict,
                                   scoring: Dict) -> str:
        """Generate a template-based report (fallback when no LLM)."""
        overall = scoring.get("overall_score", "N/A")
        metric_scores = scoring.get("metrics", {})

        lines = []
        lines.append("📊 **跑姿分析报告（模板版）**")
        lines.append(f"总体评分：**{overall}/100**")
        lines.append("")
        lines.append("> 💡 设置 DEEPSEEK_API_KEY 环境变量可开启 AI 智能分析报告")
        lines.append("")

        # Fatigue section
        if self.fatigue_report and self.fatigue_report.get("fatigue_level"):
            fr = self.fatigue_report
            lines.append("🔄 **疲劳分析**")
            level_map = {"normal": "✅ 正常", "mild": "⚡ 轻度",
                         "moderate": "⚠️ 中度", "severe": "🔴 重度"}
            level = level_map.get(fr.get("fatigue_level", ""), fr.get("fatigue_level"))
            lines.append(f"   等级：{level}（评分：{fr.get('fatigue_score', 0):.0f}/100）")
            for d in fr.get("deltas", []):
                if d.get("significant") and d.get("change") is not None:
                    arrow = "▼" if d["change"] < 0 else "▲"
                    lines.append(f"   {d['metric']}: {d['baseline']} → {d['fatigue']} ({arrow}{abs(d['change']):.1f})")
            lines.append("")

        # Sort metrics by score (ascending)
        weaknesses = sorted(
            [(k, v) for k, v in metric_scores.items()],
            key=lambda x: x[1]
        ) if metric_scores else []

        goods = [(k, v) for k, v in weaknesses if v >= 80]
        needs = [(k, v) for k, v in weaknesses if v < 80]

        name_map = {
            "cadence": "步频",
            "trunk_lean": "躯干姿态",
            "arm_symmetry": "手臂对称性",
            "vertical_oscillation": "垂直振幅",
            "foot_strike": "触地控制",
            "hip_drop": "骨盆倾斜",
            "ground_contact": "触地时间",
        }

        if goods:
            lines.append("✅ **做得好的地方**")
            for k, v in goods:
                lines.append(f"  • **{name_map.get(k, k)}**: {v:.0f}/100 — 保持！")
            lines.append("")

        if needs:
            lines.append("⚠️ **需要改进**")
            tips = {
                "cadence": ("步频偏低" if needs else "步频",
                    "使用 180 BPM 节拍器跑步热身，逐步提升步频"),
                "trunk_lean": ("躯干姿态",
                    "保持核心收紧，想象胸口有束光射向正前方 10 米地面"),
                "arm_symmetry": ("手臂对称性",
                    "照镜子练习摆臂，肘部 90° 前后摆动，不交叉过中线"),
                "vertical_oscillation": ("垂直振幅",
                    "想象头顶有天花板，尽量减少弹跳"),
                "foot_strike": ("触地控制",
                    "缩短步幅，加快步频，让脚落在身体正下方"),
                "hip_drop": ("骨盆倾斜",
                    "加强臀中肌训练（蚌式开合、侧抬腿）。跑步时注意骨盆保持水平"),
                "ground_contact": ("触地时间",
                    "加快步频可减少触地时间。练习快速抬腿和弹性落地"),
            }
            for k, v in needs:
                detail = details.get(k, "")
                label, tip = tips.get(k, (k, "参考教练指导调整"))
                lines.append(f"  • **{label}** ({v:.0f}/100)：{detail}")
                lines.append(f"    → 建议：{tip}")
            lines.append("")

        lines.append("🎯 **本周训练建议**")
        lines.append("  1. 跑前热身：动态拉伸 + 高抬腿 3 组 x 20 秒")
        lines.append("  2. 注意恢复：每周增加跑量不超过 10%")
        lines.append("  3. 力量训练：每周 2 次核心 + 腿部力量")

        if self.fatigue_report and self.fatigue_report.get("fatigue_level") in ("moderate", "severe"):
            lines.append("")
            lines.append("🔄 **疲劳管理补充**")
            lines.append("  检测到明显疲劳退化，建议：")
            lines.append("  • 本周减少 20% 跑量，优先恢复")
            lines.append("  • 跑后冰敷 + 拉伸，警惕跑步损伤")

        return "\n".join(lines)
