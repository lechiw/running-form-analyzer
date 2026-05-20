"""
app.py - Web UI for Running Form Analyzer
Streamlit-based interface for video upload, analysis, and results.
"""

import streamlit as st
import sys
import os
import tempfile
import json
import base64
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pose_extractor import PoseExtractor
from metrics import RunningMetricsCalculator
from visualizer import RunningFormVisualizer
from analyzer import AIRunningCoach
from llm_client import create_llm_client
from quality_check import VideoQualityChecker
from fatigue_analyzer import FatigueAnalyzer, format_fatigue_report

st.set_page_config(page_title="🏃 Running Form Analyzer", page_icon="🏃",
                   layout="wide", initial_sidebar_state="expanded")

# ── CSS ────────────────────────────────────────
st.markdown("""
<style>
    .stProgress > div > div > div > div { background-color: #00cc66; }
    .report-box {
        background-color: #f8f9fa; color: #1a1a1a;
        border-radius: 8px; padding: 20px;
        border: 1px solid #dee2e6;
        font-family: -apple-system,'Microsoft YaHei',monospace;
        white-space: pre-wrap; line-height: 1.6; font-size: 14px;
    }
    .report-box strong { color: #1a1a1a; }
    .report-box em { color: #333; }
    video { max-height: 400px !important; width: 100% !important; border-radius: 8px; }
    .st-emotion-cache-1y4p8pa { padding: 2rem 1rem; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

def _generate_html_report(text: str) -> str:
    """Wrap plain-text report into a styled HTML page."""
    html_text = text.replace("\n", "<br>")
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>跑姿分析报告</title>
<style>
body {{ font-family: -apple-system,'Microsoft YaHei',sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; background: #fff; color: #1a1a1a; line-height: 1.8; }}
pre {{ background: #f8f9fa; padding: 20px; border-radius: 8px; border: 1px solid #dee2e6; white-space: pre-wrap; word-wrap: break-word; font-family: inherit; font-size: 14px; }}
</style></head><body><pre>{html_text}</pre></body></html>"""


def main():
    st.title("🏃 Running Form Analyzer")
    st.markdown("上传跑步视频，AI 自动分析跑姿并生成报告")

    # ── Sidebar ──
    with st.sidebar:
        st.header("⚙️ 分析设置")
        stride = st.slider("采样步长", 1, 5, 2, help="每 N 帧处理一帧")
        max_frames = st.number_input("最大帧数", 0, 3000, 0, step=100,
                                     help="0 = 全部")
        render_video = st.checkbox("🎬 生成可视化视频", True)
        do_fatigue = st.checkbox("🔄 疲劳对比分析", True)
        llm_provider = st.selectbox("AI 报告引擎",
                                    ["deepseek", "openai", "template"], 0)
        st.divider()
        st.markdown("### 📸 拍摄指南")
        st.info("侧面拍摄 · 手机固定 · 横屏 · 全身入画 · 浅色紧身衣")
        if llm_provider != "template":
            client = create_llm_client(provider=llm_provider, auto_fallback=True)
            if client:
                st.success(f"✅ {llm_provider} API 已连接")
            else:
                st.warning("⚠️ API 未配置，使用模板报告")

    # ── File upload ──
    uploaded_file = st.file_uploader("选择跑步视频",
        type=["mp4", "mov", "avi", "mkv", "webm"])

    # ── Init session state ──
    if "results" not in st.session_state:
        st.session_state.results = None
    if "video_uploaded" not in st.session_state:
        st.session_state.video_uploaded = None
    if "video_path" not in st.session_state:
        st.session_state.video_path = None

    if uploaded_file is not None:
        # Save uploaded file (only when new file)
        if (st.session_state.video_uploaded != uploaded_file.name or
            st.session_state.results is None):
            video_ext = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=video_ext) as tmp:
                tmp.write(uploaded_file.getbuffer())
                st.session_state.video_path = tmp.name
            st.session_state.video_uploaded = uploaded_file.name
            st.session_state.results = None  # Clear old results on new upload

        # Preview video (centered)
        _, col_vid, _ = st.columns([1, 2, 1])
        with col_vid:
            st.video(st.session_state.video_path, format="video/mp4")

        # Analyze button
        _, col_btn, _ = st.columns([1, 2, 1])
        with col_btn:
            analyze_btn = st.button("🚀 开始分析", type="primary",
                                     use_container_width=True)

        if analyze_btn:
            with st.spinner("分析中，请稍候..."):
                st.session_state.results = _run_analysis(
                    video_path=st.session_state.video_path,
                    stride=stride,
                    max_frames=max_frames if max_frames > 0 else None,
                    render=render_video,
                    do_fatigue=do_fatigue,
                    llm_provider=llm_provider if llm_provider != "template" else None,
                )

    # ── Display results from session state ──
    if st.session_state.results:
        _display_results(st.session_state.results)


def _run_analysis(video_path: str, stride: int, max_frames, render: bool,
                  do_fatigue: bool, llm_provider: str) -> dict:
    """Run the full analysis pipeline."""
    progress = st.progress(0, text="初始化...")
    results = {}

    # Step 1
    progress.progress(10, text="📐 提取骨架...")
    extractor = PoseExtractor(model_complexity=1)
    seq = extractor.extract_from_video(video_path, max_frames=max_frames, stride=stride)
    if len(seq.landmarks_seq) == 0:
        st.error("❌ 未检测到人体骨架")
        return {"error": "No pose detected"}
    results["total_frames"] = len(seq.landmarks_seq)

    # Step 1.5
    progress.progress(30, text="🎥 检查拍摄质量...")
    qc = VideoQualityChecker()
    qr = qc.check(seq)
    results["quality"] = {
        "passed": qr.passed, "score": qr.score, "summary": qr.summary,
        "issues": qr.issues, "tips": qr.tips, "guide": qr.shooting_guide,
    }

    # Step 2
    progress.progress(50, text="📊 计算跑姿指标...")
    calc = RunningMetricsCalculator()
    metrics = calc.compute(seq)
    scoring = calc.get_scoring(metrics)
    results["metrics"] = metrics.summary()
    results["scoring"] = scoring

    # Step 3: Render video
    output_video = None
    if render:
        progress.progress(65, text="🎬 渲染可视化视频...")
        out_path = str(OUTPUT_DIR / "analysis_result.mp4")
        v = RunningFormVisualizer()
        v.render_video(video_path, out_path, seq, metrics)
        output_video = out_path
    results["video_output"] = output_video

    # Step 3.5
    if do_fatigue:
        progress.progress(80, text="🔄 疲劳分析...")
        fa = FatigueAnalyzer()
        fr = fa.analyze(seq)
        results["fatigue"] = fr.to_dict()

    # Step 4
    progress.progress(90, text="📝 生成分析报告...")
    provider = llm_provider or os.environ.get("LLM_PROVIDER", "deepseek")
    client = create_llm_client(provider=provider, auto_fallback=True)
    coach = AIRunningCoach(llm_api_func=client, fatigue_report=results.get("fatigue"))
    report = coach.generate_report(metrics, scoring)
    results["report"] = report
    results["llm_active"] = client is not None

    # Save HTML report
    html = _generate_html_report(report)
    report_html = OUTPUT_DIR / "analysis_report.html"
    report_html.write_text(html, encoding="utf-8")
    results["report_html_path"] = str(report_html)

    progress.progress(100, text="✅ 分析完成！")
    return results


def _display_results(r: dict):
    """Display analysis results (uses session_state, won't disappear)."""
    if "error" in r:
        return

    st.divider()
    st.header("📊 分析结果")

    # Summary cards
    a, b, c = st.columns(3)
    a.metric("处理帧数", r.get("total_frames", "N/A"))
    score = r.get("scoring", {}).get("overall_score")
    b.metric("🏆 跑姿评分", f"{score:.0f}/100" if score is not None else "N/A")
    c.metric("AI 报告", "✅ 已启用" if r.get("llm_active") else "📝 模板")

    # 1. Quality check
    q = r.get("quality", {})
    if q:
        with st.expander("🎥 拍摄质量检查",
                          expanded=not q.get("passed", True)):
            if q.get("passed"):
                st.success(q.get("summary", "通过"))
            else:
                st.warning(q.get("summary", "有问题"))
            st.write(f"评分：{q.get('score', 0)}/100")
            for issue in (q.get("issues") or []):
                st.markdown(f"- ❌ {issue}")
            for tip in (q.get("tips") or []):
                st.markdown(f"- 💡 {tip}")

    # 2. Metrics
    metrics = r.get("metrics", {})
    with st.expander("📊 跑姿指标", expanded=True):
        if metrics:
            cols = st.columns(3)
            items = [
                ("步频", f"{metrics.get('cadence_spm','N/A')} spm"),
                ("躯干前倾角", f"{metrics.get('trunk_lean_deg','N/A')}°"),
                ("垂直振幅", f"{metrics.get('vertical_oscillation_cm','N/A')} cm"),
                ("手臂对称性", f"{metrics.get('arm_symmetry_score','N/A')}/100"),
                ("触地距离", f"{metrics.get('foot_strike_distance_cm','N/A')} cm"),
                ("着地方式", metrics.get("foot_strike_type","N/A")),
            ]
            for i, (label, val) in enumerate(items):
                cols[i % 3].metric(label, val)

            details = r.get("scoring", {}).get("details", {})
            if details:
                st.markdown("**各维度评分：**")
                sc = st.columns(len(details))
                name_map = {"cadence":"步频","trunk_lean":"躯干",
                            "arm_symmetry":"手臂","vertical_oscillation":"振幅",
                            "foot_strike":"触地"}
                for i, (k, v) in enumerate(details.items()):
                    sc[i].metric(name_map.get(k, k), v)

    # 3. Fatigue
    fatigue = r.get("fatigue")
    if fatigue:
        with st.expander("🔄 疲劳分析", expanded=True):
            lv = {"normal":"✅ 正常","mild":"⚡ 轻度",
                  "moderate":"⚠️ 中度","severe":"🔴 重度"}
            level = lv.get(fatigue.get("fatigue_level",""), fatigue.get("fatigue_level"))
            st.metric("疲劳等级", f"{level}（{fatigue.get('fatigue_score',0):.0f}/100）")
            rows = []
            for d in (fatigue.get("deltas") or []):
                if d.get("baseline") is not None or d.get("fatigue") is not None:
                    ch = d.get("change")
                    cs = f"{'▼' if ch and ch<0 else '▲'} {abs(ch):.1f}" if ch else "N/A"
                    rows.append({"指标":d["metric"],"前段":d.get("baseline","N/A"),
                                 "后段":d.get("fatigue","N/A"),"变化":cs})
            if rows:
                st.table(rows)

    # 4. Report (inline preview + HTML download)
    report_text = r.get("report", "")
    if report_text:
        with st.expander("📝 分析报告", expanded=True):
            st.markdown(f'<div class="report-box">{report_text}</div>',
                       unsafe_allow_html=True)
            html_path = r.get("report_html_path")
            if html_path and os.path.exists(html_path):
                with open(html_path, "r", encoding="utf-8") as f:
                    st.download_button("📥 下载 HTML 报告",
                        data=f.read(),
                        file_name="running_analysis_report.html",
                        mime="text/html",
                        use_container_width=True)

    # 5. Rendered video preview + download
    video_path = r.get("video_output")
    if video_path and os.path.exists(video_path):
        with st.expander("🎬 可视化视频", expanded=True):
            st.video(video_path, format="video/mp4")
            with open(video_path, "rb") as f:
                st.download_button("📥 下载可视化视频",
                    data=f.read(),
                    file_name="running_analysis_annotated.mp4",
                    mime="video/mp4",
                    use_container_width=True)


if __name__ == "__main__":
    main()
