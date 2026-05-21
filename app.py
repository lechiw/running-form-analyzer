"""
app.py - Web UI for Running Form Analyzer
Streamlit-based interface for video upload, analysis, and results.
"""

import streamlit as st
import sys, os, tempfile, json, re, shutil, subprocess
from pathlib import Path

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

from comparison import compare_analyses, format_comparison_report, comparison_to_dict
from runner_profile import RunnerProfile

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

st.markdown("""<style>video{max-height:200px!important;width:100%!important;border-radius:8px}</style>""", unsafe_allow_html=True)

# ── Markdown → HTML ──
def _md_to_html(text: str) -> str:
    html = text
    html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.M)
    html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.M)
    html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.M)
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)
    html = re.sub(r'^- (.+)$', r'<li>\1</li>', html, flags=re.M)
    html = re.sub(r'^(\d+)\.\s+(.+)$', r'<li>\2</li>', html, flags=re.M)
    return html

def _generate_html_report(text: str) -> str:
    body = _md_to_html(text)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>跑姿分析报告</title>
<style>
body {{ font-family: -apple-system,'Microsoft YaHei',sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; background: #fff; color: #1a1a1a; line-height: 1.8; }}
h1,h2,h3 {{ color: #1a1a1a; }}
h2 {{ border-bottom: 2px solid #00cc66; padding-bottom: 4px; }}
ul,ol {{ padding-left: 20px; }}
li {{ margin: 4px 0; }}
</style></head><body>{body}</body></html>"""


# ── Sidebar ─────────────────────────────────────
def _sidebar():
    with st.sidebar:
        st.header("⚙️ 分析设置")
        
        mode = st.radio("工作模式", ["单次分析", "前后对比"], horizontal=True)
        
        stride = st.slider("采样步长", 1, 5, 2)
        max_frames = st.number_input("最大帧数", 0, 3000, 0, step=100, help="0=全部")
        render_video = st.checkbox("🎬 生成可视化视频", True)
        do_fatigue = st.checkbox("🔄 疲劳对比分析", True)
        llm_provider = st.selectbox("AI 报告引擎", ["deepseek", "openai", "template"], 0)
        
        # ── Personal Profile ──
        st.divider()
        st.markdown("### 📐 个人信息（选填）")
        st.caption("提供后分析更准确")
        profile_height = st.number_input("身高 (cm)", 120, 220, 175, step=1)
        profile_gender = st.selectbox("性别", ["未设置", "男性", "女性"], 0)
        profile_age = st.number_input("年龄", 10, 100, 30, step=1)
        
        profile = RunnerProfile(
            height_cm=profile_height if profile_height > 0 else None,
            gender={"男性": "male", "女性": "female"}.get(profile_gender),
            age=profile_age if profile_age > 0 else None,
        )
        
        st.divider()
        st.markdown("### 📸 拍摄指南")
        st.info("侧面拍摄 · 手机固定 · 横屏 · 全身入画 · 浅色紧身衣")
        if llm_provider != "template":
            client = create_llm_client(provider=llm_provider, auto_fallback=True)
            if client:
                st.success(f"✅ {llm_provider} API 已连接")
            else:
                st.warning("⚠️ API 未配置，使用模板报告")
    return mode, stride, max_frames, render_video, do_fatigue, llm_provider, profile


# ── Analysis pipeline ──────────────────────────
def _run_analysis(video_path, stride, max_frames, render, do_fatigue, llm_provider,
                  profile=None):
    progress = st.progress(0, text="初始化...")
    res = {}

    progress.progress(10, text="📐 提取骨架...")
    extractor = PoseExtractor(model_complexity=1)
    seq = extractor.extract_from_video(video_path, max_frames=max_frames, stride=stride)
    if not seq.landmarks_seq:
        st.error("❌ 未检测到人体骨架")
        return {"error": "No pose detected"}
    res["total_frames"] = len(seq.landmarks_seq)

    progress.progress(30, text="🎥 检查拍摄质量...")
    qc = VideoQualityChecker()
    qr = qc.check(seq)
    res["quality"] = {"passed": qr.passed, "score": qr.score, "summary": qr.summary,
                      "issues": qr.issues, "tips": qr.tips, "guide": qr.shooting_guide}

    progress.progress(50, text="📊 计算跑姿指标...")
    calc = RunningMetricsCalculator()
    metrics = calc.compute(seq, profile=profile)
    scoring = calc.get_scoring(metrics, profile=profile)
    res["metrics"] = metrics.summary(profile=profile)
    res["scoring"] = scoring

    output_video = None
    if render:
        progress.progress(65, text="🎬 渲染可视化视频...")
        raw_path = str(OUTPUT_DIR / "_raw.mp4")
        RunningFormVisualizer().render_video(video_path, raw_path, seq, metrics)
        # Re-encode to H.264 for browser
        browser_path = str(OUTPUT_DIR / "analysis_preview.mp4")
        subprocess.run(["ffmpeg", "-y", "-i", raw_path, "-c:v", "libx264",
                        "-preset", "fast", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", browser_path],
                       capture_output=True, timeout=120)
        if os.path.exists(browser_path) and os.path.getsize(browser_path) > 1000:
            output_video = browser_path
        else:
            output_video = raw_path
    res["video_output"] = output_video

    if do_fatigue:
        progress.progress(80, text="🔄 疲劳分析...")
        fr = FatigueAnalyzer().analyze(seq)
        res["fatigue"] = fr.to_dict()

    progress.progress(90, text="📝 生成分析报告...")
    provider = llm_provider or os.environ.get("LLM_PROVIDER", "deepseek")
    client = create_llm_client(provider=provider, auto_fallback=True)
    coach = AIRunningCoach(llm_api_func=client, fatigue_report=res.get("fatigue"))
    report = coach.generate_report(metrics, scoring)
    res["report"] = report
    res["llm_active"] = client is not None

    html_path = str(OUTPUT_DIR / "analysis_report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(_generate_html_report(report))
    res["report_html_path"] = html_path

    progress.progress(100, text="✅ 分析完成！")
    return res


# ── Display results ────────────────────────────
def _display(r, show_header=True):
    if "error" in r:
        return
    if show_header:
        st.divider()
        st.header("📊 分析结果")
    a, b, c = st.columns(3)
    a.metric("处理帧数", r.get("total_frames", "N/A"))
    score = r.get("scoring", {}).get("overall_score")
    b.metric("🏆 跑姿评分", f"{score:.0f}/100" if score is not None else "N/A")
    c.metric("AI 报告", "✅ 已启用" if r.get("llm_active") else "📝 模板")

    # Download buttons at top
    report_html = r.get("report_html_path")
    video_out = r.get("video_output")
    dl_cols = st.columns(2)
    with dl_cols[0]:
        if report_html and os.path.exists(report_html):
            with open(report_html, encoding="utf-8") as f:
                st.download_button("📥 下载 HTML 报告", data=f.read(),
                    file_name="running_analysis_report.html", mime="text/html",
                    use_container_width=True)
    with dl_cols[1]:
        if video_out and os.path.exists(video_out):
            with open(video_out, "rb") as f:
                st.download_button("📥 下载可视化视频", data=f.read(),
                    file_name="running_analysis_annotated.mp4", mime="video/mp4",
                    use_container_width=True)

    q = r.get("quality", {})
    if q:
        with st.expander("🎥 拍摄质量检查", expanded=not q.get("passed", True)):
            (st.success if q.get("passed") else st.warning)(q.get("summary"))
            st.write(f"评分：{q.get('score',0)}/100")
            for i in (q.get("issues") or []): st.markdown(f"- ❌ {i}")
            for t in (q.get("tips") or []): st.markdown(f"- 💡 {t}")

    m = r.get("metrics", {})
    if m:
        with st.expander("📊 跑姿指标", expanded=True):
            cols = st.columns(3)
            for i, (label, key, unit) in enumerate([
                ("步频","cadence_spm","spm"), ("躯干前倾","trunk_lean_deg","°"),
                ("垂直振幅","vertical_oscillation_cm","cm"), ("手臂对称","arm_symmetry_score","/100"),
                ("触地距离","foot_strike_distance_cm","cm"), ("着地方式","foot_strike_type","")]):
                v = m.get(key, "N/A")
                cols[i % 3].metric(label, f"{v} {unit}" if unit else str(v))
            details = r.get("scoring", {}).get("details", {})
            if details:
                st.markdown("**各维度评分：**")
                sc = st.columns(len(details))
                nm = {"cadence":"步频","trunk_lean":"躯干","arm_symmetry":"手臂",
                      "vertical_oscillation":"振幅","foot_strike":"触地"}
                for i, (k, v) in enumerate(details.items()):
                    sc[i].metric(nm.get(k, k), v)

    ft = r.get("fatigue")
    if ft:
        with st.expander("🔄 疲劳分析", expanded=True):
            lv = {"normal":"✅ 正常","mild":"⚡ 轻度","moderate":"⚠️ 中度","severe":"🔴 重度"}
            st.metric("疲劳等级", f"{lv.get(ft.get('fatigue_level',''),'?')}（{ft.get('fatigue_score',0):.0f}/100）")
            rows = []
            for d in (ft.get("deltas") or []):
                if d.get("baseline") is not None or d.get("fatigue") is not None:
                    ch = d.get("change")
                    cs = f"{'▼' if ch and ch<0 else '▲'} {abs(ch):.1f}" if ch else "N/A"
                    rows.append({"指标":d["metric"],"前段":d.get("baseline","N/A"),
                                 "后段":d.get("fatigue","N/A"),"变化":cs})
            if rows: st.table(rows)

    report = r.get("report", "")
    if report:
        with st.expander("📝 分析报告", expanded=True):
            st.markdown(report)

    vp = r.get("video_output")
    if vp and os.path.exists(vp):
        with st.expander("🎬 可视化视频", expanded=True):
            st.video(vp, format="video/mp4")


# ── Comparison display ─────────────────────────
def _display_comparison(cr):
    """Display comparison results."""
    st.divider()
    st.header("📊 前后对比结果")
    
    st.caption(f"{cr.before_label}  →  {cr.after_label}")
    
    if cr.score_delta is not None:
        delta_str = f"▲ {cr.score_delta:+.1f}" if cr.score_delta > 0 else f"▼ {cr.score_delta:.1f}"
        col1, col2, col3 = st.columns(3)
        col1.metric("训练前评分", f"{cr.score_before:.0f}")
        col3.metric("训练后评分", f"{cr.score_after:.0f}", delta=delta_str)
    
    # Metric table
    st.subheader("各指标变化")
    cols = st.columns(len(cr.deltas))
    for i, d in enumerate(cr.deltas):
        if d.before_value is None and d.after_value is None:
            continue
        delta_display = None
        if d.delta is not None and abs(d.delta) > 0.01:
            delta_display = f"{'▲' if d.is_improvement else '▼'} {abs(d.delta):.1f}"
        cols[i].metric(
            d.label,
            f"{d.after_value:.1f} →" if d.after_value is not None else "N/A",
            delta=delta_display,
        )
    
    # Insights
    if cr.insights:
        st.subheader("💡 分析结论")
        for insight in cr.insights:
            st.markdown(insight)
    
    # Full report
    with st.expander("📋 完整对比报告（可下载）"):
        report_text = format_comparison_report(cr)
        st.text(report_text)
        st.download_button(
            "📥 下载对比报告",
            data=report_text,
            file_name="running_form_comparison.txt",
            mime="text/plain",
            use_container_width=True,
        )


# ── Main ───────────────────────────────────────
def main():
    st.title("🏃 Running Form Analyzer")
    st.markdown("上传跑步视频，AI 自动分析跑姿并生成报告")

    mode, stride, max_frames, render_video, do_fatigue, llm_provider, profile = _sidebar()

    if "results" not in st.session_state:
        st.session_state.results = None
    if "video_file" not in st.session_state:
        st.session_state.video_file = None
    if "compare_before" not in st.session_state:
        st.session_state.compare_before = None
    if "compare_after" not in st.session_state:
        st.session_state.compare_after = None
    if "comparison_result" not in st.session_state:
        st.session_state.comparison_result = None

    provider = llm_provider if llm_provider != "template" else None

    if mode == "前后对比":
        st.subheader("📹 上传跑姿视频")
        st.caption("两侧使用相同的拍摄设置（同一机位、相同距离），结果最准确")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**🏋️ 训练前**")
            before_file = st.file_uploader(
                "选择训练前视频", type=["mp4", "mov", "avi", "mkv", "webm"],
                key="before_upload")
            if before_file is not None:
                ext = Path(before_file.name).suffix
                bpath = str(OUTPUT_DIR / f"before{ext}")
                with open(bpath, "wb") as f:
                    f.write(before_file.getbuffer())
                st.session_state.compare_before = bpath
                st.video(bpath)
        
        with c2:
            st.markdown("**🏃 训练后**")
            after_file = st.file_uploader(
                "选择训练后视频", type=["mp4", "mov", "avi", "mkv", "webm"],
                key="after_upload")
            if after_file is not None:
                ext = Path(after_file.name).suffix
                apath = str(OUTPUT_DIR / f"after{ext}")
                with open(apath, "wb") as f:
                    f.write(after_file.getbuffer())
                st.session_state.compare_after = apath
                st.video(apath)
        
        # Analyze buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("🚀 分析训练前", type="primary", use_container_width=True):
                if st.session_state.compare_before and os.path.exists(st.session_state.compare_before):
                    with st.spinner("分析训练前视频..."):
                        st.session_state.before_result = _run_analysis(
                            st.session_state.compare_before, stride,
                            max_frames if max_frames > 0 else None,
                            render_video, do_fatigue, provider,
                            profile=profile)
        with col2:
            if st.button("🚀 分析训练后", type="primary", use_container_width=True):
                if st.session_state.compare_after and os.path.exists(st.session_state.compare_after):
                    with st.spinner("分析训练后视频..."):
                        st.session_state.after_result = _run_analysis(
                            st.session_state.compare_after, stride,
                            max_frames if max_frames > 0 else None,
                            render_video, do_fatigue, provider,
                            profile=profile)
        with col3:
            has_both = (st.session_state.get("before_result") 
                        and st.session_state.get("after_result"))
            if st.button("📊 生成对比", type="secondary",
                         use_container_width=True, disabled=not has_both):
                if has_both:
                    b = st.session_state.before_result
                    a = st.session_state.after_result
                    b_metrics = {**b.get("metrics", {}), "overall_score": b.get("scoring", {}).get("overall_score")}
                    a_metrics = {**a.get("metrics", {}), "overall_score": a.get("scoring", {}).get("overall_score")}
                    st.session_state.comparison_result = compare_analyses(
                        b_metrics, a_metrics,
                        label_before="训练前",
                        label_after="训练后",
                    )
        
        # Show individual results
        if st.session_state.get("before_result"):
            with st.expander("训练前分析结果", expanded=False):
                _display(st.session_state.before_result, show_header=False)
        if st.session_state.get("after_result"):
            with st.expander("训练后分析结果", expanded=False):
                _display(st.session_state.after_result, show_header=False)
        
        # Show comparison
        if st.session_state.comparison_result:
            _display_comparison(st.session_state.comparison_result)
    
    else:
        # Single analysis mode (original)
        uploaded = st.file_uploader("选择跑步视频",
                                    type=["mp4", "mov", "avi", "mkv", "webm"])

        if uploaded is not None:
            ext = Path(uploaded.name).suffix
            stable = str(OUTPUT_DIR / f"upload{ext}")
            with open(stable, "wb") as f:
                f.write(uploaded.getbuffer())
            st.session_state.video_file = stable

            if st.session_state.results is None:
                st.session_state.results = None

            _, col, _ = st.columns([1, 2, 1])
            with col:
                if os.path.exists(stable):
                    st.video(stable)

            _, btn, _ = st.columns([1, 2, 1])
            with btn:
                if st.button("🚀 开始分析", type="primary", use_container_width=True):
                    with st.spinner("分析中，请稍候..."):
                        st.session_state.results = _run_analysis(
                            stable, stride,
                            max_frames if max_frames > 0 else None,
                            render_video, do_fatigue, provider,
                            profile=profile)

        if st.session_state.results:
            _display(st.session_state.results)


if __name__ == "__main__":
    main()
