"""
app.py - Web UI for Running Form Analyzer
Streamlit-based interface for video upload, analysis, and results.
"""

import streamlit as st
import sys
import os
import tempfile
import json
from pathlib import Path

# Ensure project root is on path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pose_extractor import PoseExtractor
from metrics import RunningMetricsCalculator
from visualizer import RunningFormVisualizer
from analyzer import AIRunningCoach
from llm_client import create_llm_client
from quality_check import VideoQualityChecker, print_quality_report
from fatigue_analyzer import FatigueAnalyzer, format_fatigue_report
from main import analyze_video


# Page config
st.set_page_config(
    page_title="🏃 Running Form Analyzer",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .metric-good { color: #00cc66; font-weight: bold; }
    .metric-bad { color: #ff4444; font-weight: bold; }
    .metric-normal { color: #ffaa00; font-weight: bold; }
    .stProgress > div > div > div > div { background-color: #00cc66; }
    .report-box {
        background-color: #1e1e1e;
        border-radius: 8px;
        padding: 20px;
        border: 1px solid #333;
        font-family: monospace;
        white-space: pre-wrap;
        line-height: 1.6;
    }
    .st-emotion-cache-1y4p8pa { padding: 2rem 1rem; }
</style>
""", unsafe_allow_html=True)


def main():
    st.title("🏃 Running Form Analyzer")
    st.markdown("上传跑步视频，AI 自动分析跑姿并生成报告")

    # Sidebar config
    with st.sidebar:
        st.header("⚙️ 分析设置")

        stride = st.slider("采样步长", min_value=1, max_value=5, value=2,
                           help="每 N 帧处理一帧。越大越快但精度降低")
        max_frames = st.number_input("最大处理帧数", min_value=0, max_value=3000,
                                      value=0, step=100,
                                      help="0 = 处理全部")
        render_video = st.checkbox("🎬 生成可视化视频", value=True,
                                   help="带骨架叠加的标注视频（耗时较长）")
        do_fatigue = st.checkbox("🔄 疲劳对比分析", value=True,
                                 help="对比前段和后段跑姿变化")

        llm_provider = st.selectbox("AI 报告引擎",
                                     ["deepseek", "openai", "template"],
                                     index=0,
                                     help="template = 模板报告（无需 API Key）")

        st.divider()
        st.markdown("### 📸 拍摄指南")
        st.info(
            "为获得最佳分析结果：\n\n"
            "1. **侧面拍摄**，手机固定\n"
            "2. 横屏，全身入画\n"
            "3. 跑步机最佳，拍 20-30 秒\n"
            "4. 穿浅色紧身运动服"
        )

        # Show API status
        if llm_provider != "template":
            client = create_llm_client(provider=llm_provider, auto_fallback=True)
            if client:
                st.success(f"✅ {llm_provider} API 已连接")
            else:
                st.warning(f"⚠️ {llm_provider} API 未配置，将使用模板报告")

    # Main area
    uploaded_file = st.file_uploader(
        "选择跑步视频",
        type=["mp4", "mov", "avi", "mkv", "webm"],
        help="支持 MP4、MOV、AVI 等常见格式",
    )

    if uploaded_file is not None:
        # Save uploaded file
        video_ext = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=video_ext) as tmp:
            tmp.write(uploaded_file.getbuffer())
            video_path = tmp.name

        st.video(video_path, format="video/mp4")

        # Analysis button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_btn = st.button("🚀 开始分析", type="primary", use_container_width=True)

        if analyze_btn:
            with st.spinner("分析中，请稍候..."):
                results = run_analysis(
                    video_path=video_path,
                    stride=stride,
                    max_frames=max_frames if max_frames > 0 else None,
                    render=render_video,
                    do_fatigue=do_fatigue,
                    llm_provider=llm_provider if llm_provider != "template" else None,
                )

            display_results(results)

        # Cleanup temp file
        try:
            os.unlink(video_path)
        except Exception:
            pass

    else:
        # Welcome / demo
        st.markdown("""
        ### 👈 上传视频开始分析

        **支持的格式**：MP4、MOV、AVI、MKV、WebM

        ---

        #### 分析完成后你将得到：

        | 功能 | 说明 |
        |------|------|
        | 🎥 拍摄质量检测 | 检查视频角度、可见度、人物大小 |
        | 📊 7项跑姿指标 | 步频、躯干前倾角、垂直振幅、手臂对称性等 |
        | 🏆 综合评分 | 0-100 跑姿评分，各维度独立打分 |
        | 🎬 可视化视频 | 骨架叠加 + 实时数据面板 |
        | 🔄 疲劳分析 | 前后段对比，检测疲劳退化 |
        | 📝 AI 分析报告 | DeepSeek/OpenAI 智能分析 |
        """)


def run_analysis(video_path: str, stride: int, max_frames, render: bool,
                 do_fatigue: bool, llm_provider: str) -> dict:
    """Run the full analysis pipeline and return results."""
    progress_bar = st.progress(0, text="初始化...")
    results = {}

    # Step 1: Extract pose
    progress_bar.progress(10, text="📐 提取骨架...")
    extractor = PoseExtractor(model_complexity=1)
    seq = extractor.extract_from_video(video_path, max_frames=max_frames, stride=stride)

    if len(seq.landmarks_seq) == 0:
        st.error("❌ 未检测到人体骨架，请检查视频是否有人")
        return {"error": "No pose detected"}

    results["total_frames"] = len(seq.landmarks_seq)

    # Step 1.5: Quality check
    progress_bar.progress(30, text="🎥 检查拍摄质量...")
    checker = VideoQualityChecker()
    quality_report = checker.check(seq)
    results["quality"] = {
        "passed": quality_report.passed,
        "score": quality_report.score,
        "summary": quality_report.summary,
        "issues": quality_report.issues,
        "tips": quality_report.tips,
        "guide": quality_report.shooting_guide,
    }

    # Step 2: Metrics
    progress_bar.progress(50, text="📊 计算跑姿指标...")
    calculator = RunningMetricsCalculator()
    metrics = calculator.compute(seq)
    scoring = calculator.get_scoring(metrics)
    results["metrics"] = metrics.summary()
    results["scoring"] = scoring

    # Step 3: Render
    output_video_path = None
    if render:
        progress_bar.progress(65, text="🎬 渲染可视化视频...")
        output_dir = Path(video_path).parent / "output"
        output_dir.mkdir(exist_ok=True)
        output_video = str(output_dir / "analysis_result.mp4")
        visualizer = RunningFormVisualizer()
        visualizer.render_video(video_path, output_video, seq, metrics)
        output_video_path = output_video

    # Step 3.5: Fatigue
    fatigue_text = None
    if do_fatigue:
        progress_bar.progress(80, text="🔄 疲劳分析...")
        fatigue_analyzer = FatigueAnalyzer()
        fatigue_report = fatigue_analyzer.analyze(seq)
        fatigue_text = format_fatigue_report(fatigue_report)
        results["fatigue"] = fatigue_report.to_dict()

    # Step 4: Report
    progress_bar.progress(90, text="📝 生成分析报告...")
    provider = llm_provider or os.environ.get("LLM_PROVIDER", "deepseek")
    llm_client = create_llm_client(provider=provider, auto_fallback=True)
    fatigue_dict = results.get("fatigue")
    coach = AIRunningCoach(llm_api_func=llm_client, fatigue_report=fatigue_dict)
    report = coach.generate_report(metrics, scoring)
    results["report"] = report
    results["llm_active"] = llm_client is not None

    # Save report to file
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    report_path = output_dir / "analysis_report.txt"
    report_path.write_text(report, encoding="utf-8")
    results["report_path"] = str(report_path)

    progress_bar.progress(100, text="✅ 分析完成！")
    return results


def display_results(results: dict):
    """Display analysis results in the Streamlit UI."""
    if "error" in results:
        return

    st.divider()
    st.header("📊 分析结果")

    cols = st.columns(3)
    cols[0].metric("处理帧数", results.get("total_frames", "N/A"))
    score = results.get("scoring", {}).get("overall_score")
    if score is not None:
        score_color = ":green" if score >= 70 else ":orange" if score >= 40 else ":red"
        cols[1].metric("🏆 跑姿评分", f"{score:.0f}/100")
    else:
        cols[1].metric("🏆 跑姿评分", "N/A")

    cols[2].metric("AI 报告",
                    "✅ 已启用" if results.get("llm_active") else "📝 模板模式")

    # Quality check
    quality = results.get("quality", {})
    if quality:
        with st.expander("🎥 拍摄质量检查", expanded=not quality.get("passed", True)):
            if quality.get("passed"):
                st.success(quality.get("summary", "通过"))
            else:
                st.warning(quality.get("summary", "有问题"))
            st.write(f"评分：{quality.get('score', 0)}/100")

            if quality.get("issues"):
                st.markdown("**发现的问题：**")
                for issue in quality["issues"]:
                    st.markdown(f"- ❌ {issue}")

            if quality.get("tips"):
                st.markdown("**改善建议：**")
                for tip in quality["tips"]:
                    st.markdown(f"- 💡 {tip}")

            if quality.get("guide") and not quality.get("passed"):
                with st.popover("📸 查看拍摄指南"):
                    st.markdown(quality["guide"])

    # Metrics
    with st.expander("📊 跑姿指标", expanded=True):
        metrics = results.get("metrics", {})
        if metrics:
            metric_cols = st.columns(3)

            metric_items = [
                ("步频", f"{metrics.get('cadence_spm', 'N/A')} spm", "cadence_spm"),
                ("躯干前倾角", f"{metrics.get('trunk_lean_deg', 'N/A')}°", "trunk_lean_deg"),
                ("垂直振幅", f"{metrics.get('vertical_oscillation_cm', 'N/A')} cm",
                 "vertical_oscillation_cm"),
                ("手臂对称性", f"{metrics.get('arm_symmetry_score', 'N/A')}/100",
                 "arm_symmetry_score"),
                ("触地距离", f"{metrics.get('foot_strike_distance_cm', 'N/A')} cm",
                 "foot_strike_distance_cm"),
                ("着地方式", metrics.get("foot_strike_type", "N/A"), None),
            ]

            for i, (label, value, key) in enumerate(metric_items):
                col = metric_cols[i % 3]
                col.metric(label, value)

            # Score breakdown
            scoring = results.get("scoring", {})
            details = scoring.get("details", {})
            if details:
                st.markdown("**各维度评分：**")
                score_cols = st.columns(len(details))
                for i, (key, val) in enumerate(details.items()):
                    name_map = {
                        "cadence": "步频", "trunk_lean": "躯干",
                        "arm_symmetry": "手臂", "vertical_oscillation": "振幅",
                        "foot_strike": "触地",
                    }
                    score_cols[i].metric(name_map.get(key, key), val)

    # Fatigue
    fatigue = results.get("fatigue")
    if fatigue:
        with st.expander("🔄 疲劳分析", expanded=True):
            level_map = {"normal": "✅ 正常", "mild": "⚡ 轻度",
                         "moderate": "⚠️ 中度", "severe": "🔴 重度"}
            level = level_map.get(fatigue.get("fatigue_level", ""))
            level_score = fatigue.get("fatigue_score", 0)
            st.metric("疲劳等级", f"{level}（{level_score:.0f}/100）")

            deltas = fatigue.get("deltas", [])
            if deltas:
                delta_data = []
                for d in deltas:
                    if d.get("baseline") is not None or d.get("fatigue") is not None:
                        change = d.get("change")
                        if change is not None:
                            change_str = f"{'▼' if change < 0 else '▲'} {abs(change):.1f}"
                        else:
                            change_str = "N/A"
                        delta_data.append({
                            "指标": d["metric"],
                            "前段": d.get("baseline", "N/A"),
                            "后段": d.get("fatigue", "N/A"),
                            "变化": change_str,
                            "状态": "⚠️ 退化" if d.get("significant") and d.get("change", 0) < 0
                                   else "✅ 改善" if d.get("significant")
                                   else "→ 正常",
                        })

                if delta_data:
                    st.table(delta_data)

    # Report
    report = results.get("report", "")
    if report:
        with st.expander("📝 分析报告", expanded=True):
            st.markdown(f'<div class="report-box">{report}</div>',
                       unsafe_allow_html=True)

            report_path = results.get("report_path")
            if report_path and os.path.exists(report_path):
                with open(report_path, "r", encoding="utf-8") as f:
                    st.download_button(
                        label="📥 下载报告",
                        data=f.read(),
                        file_name="running_analysis_report.txt",
                        mime="text/plain",
                    )

    # Rendered video
    video_output = results.get("video_output")
    if video_output and os.path.exists(video_output):
        with st.expander("🎬 可视化视频", expanded=True):
            st.video(video_output, format="video/mp4")
            with open(video_output, "rb") as f:
                st.download_button(
                    label="📥 下载可视化视频",
                    data=f,
                    file_name="running_analysis_annotated.mp4",
                    mime="video/mp4",
                )


if __name__ == "__main__":
    main()
