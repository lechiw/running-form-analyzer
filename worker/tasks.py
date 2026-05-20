"""
Celery tasks for running form analysis.
Each task runs in a separate worker process.
"""
import os
import json
import uuid
import subprocess
from pathlib import Path
from datetime import datetime

from worker.celery_app import celery_app

# Ensure core modules are importable
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.pose_extractor import PoseExtractor
from core.metrics import RunningMetricsCalculator
from core.visualizer import RunningFormVisualizer
from core.analyzer import AIRunningCoach
from core.llm_client import create_llm_client
from core.quality_check import VideoQualityChecker
from core.fatigue_analyzer import FatigueAnalyzer


# Fix import paths: worker.py imports are relative to project root
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Storage paths
_default_storage = str(Path(__file__).parent.parent / "data")
STORAGE_DIR = Path(os.environ.get("STORAGE_DIR", _default_storage))
VIDEOS_DIR = STORAGE_DIR / "videos"
RESULTS_DIR = STORAGE_DIR / "results"
REPORTS_DIR = STORAGE_DIR / "reports"
os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


@celery_app.task(bind=True, name="analyze_video")
def analyze_video_task(self, task_id: str, video_path: str,
                       stride: int = 2, render: bool = True,
                       do_fatigue: bool = True,
                       llm_provider: str = "deepseek"):
    """
    Full analysis pipeline as a Celery task.
    Updates self state so the API can poll progress.
    """
    result = {"task_id": task_id, "status": "processing", "progress": 0}
    _update_state(self, result)

    try:
        # Step 1: Extract pose
        _update_state(self, {**result, "progress": 10,
                             "message": "📐 提取骨架..."})
        extractor = PoseExtractor(model_complexity=1)
        seq = extractor.extract_from_video(video_path, stride=stride)

        if not seq.landmarks_seq:
            return {"task_id": task_id, "status": "error",
                    "error": "未检测到人体骨架"}

        result["total_frames"] = len(seq.landmarks_seq)

        # Step 1.5: Quality check
        _update_state(self, {**result, "progress": 30,
                             "message": "🎥 检查拍摄质量..."})
        qc = VideoQualityChecker()
        qr = qc.check(seq)
        result["quality"] = {
            "passed": qr.passed, "score": qr.score, "summary": qr.summary,
            "issues": qr.issues, "tips": qr.tips,
        }

        # Step 2: Metrics
        _update_state(self, {**result, "progress": 50,
                             "message": "📊 计算跑姿指标..."})
        calc = RunningMetricsCalculator()
        metrics = calc.compute(seq)
        scoring = calc.get_scoring(metrics)
        result["metrics"] = metrics.summary()
        result["scoring"] = scoring

        # Step 3: Render video
        preview_video = None
        if render:
            _update_state(self, {**result, "progress": 65,
                                 "message": "🎬 渲染可视化视频..."})
            raw_path = str(REPORTS_DIR / f"{task_id}_raw.mp4")
            preview_path = str(REPORTS_DIR / f"{task_id}_preview.mp4")
            RunningFormVisualizer().render_video(video_path, raw_path, seq, metrics)
            # Re-encode for browser
            subprocess.run([
                "ffmpeg", "-y", "-i", raw_path,
                "-c:v", "libx264", "-preset", "fast",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                preview_path,
            ], capture_output=True, timeout=120)
            if os.path.exists(preview_path):
                preview_video = f"/api/videos/{task_id}_preview.mp4"
            if os.path.exists(raw_path):
                os.remove(raw_path)

        result["preview_video"] = preview_video

        # Step 3.5: Fatigue
        if do_fatigue:
            _update_state(self, {**result, "progress": 80,
                                 "message": "🔄 疲劳分析..."})
            fa = FatigueAnalyzer()
            fr = fa.analyze(seq)
            result["fatigue"] = fr.to_dict()

        # Step 4: Report
        _update_state(self, {**result, "progress": 90,
                             "message": "📝 生成分析报告..."})
        provider = llm_provider or os.environ.get("LLM_PROVIDER", "deepseek")
        client = create_llm_client(provider=provider, auto_fallback=True)
        coach = AIRunningCoach(llm_api_func=client,
                                fatigue_report=result.get("fatigue"))
        report = coach.generate_report(metrics, scoring)
        result["report"] = report
        result["llm_active"] = client is not None

        # Save report as JSON + HTML
        result_path = RESULTS_DIR / f"{task_id}.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        result["status"] = "completed"
        result["progress"] = 100
        result["message"] = "✅ 分析完成"
        _update_state(self, result)

        return result

    except Exception as e:
        error_result = {**result, "status": "error",
                        "error": str(e), "progress": 0}
        _update_state(self, error_result)
        return error_result


def _update_state(task, meta: dict):
    """Update Celery task state with progress info."""
    try:
        task.update_state(state="PROGRESS", meta=meta)
    except Exception:
        pass
