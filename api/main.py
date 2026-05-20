"""
FastAPI application for Running Form Analyzer.
Handles video upload, sync/async analysis, and result retrieval.
"""
import os, uuid, json, subprocess, re
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# Celery is optional
try:
    from worker.celery_app import celery_app
    from worker.tasks import analyze_video_task
    # Check if Redis is actually reachable
    import socket
    s = socket.socket()
    s.settimeout(1)
    s.connect(('localhost', 6379))
    s.close()
    CELERY_OK = True
except Exception:
    CELERY_OK = False
    celery_app = None
    print("  ⚠️  Redis/Celery unavailable, using sync mode")

# Core modules (for sync fallback)
from core.pose_extractor import PoseExtractor
from core.metrics import RunningMetricsCalculator
from core.visualizer import RunningFormVisualizer
from core.analyzer import AIRunningCoach
from core.llm_client import create_llm_client
from core.quality_check import VideoQualityChecker
from core.fatigue_analyzer import FatigueAnalyzer

# Storage
_default_data = str(Path(__file__).parent.parent / "data")
STORAGE = Path(os.environ.get("STORAGE_DIR", _default_data))
VIDEOS = STORAGE / "videos"
RESULTS = STORAGE / "results"
REPORTS = STORAGE / "reports"
FRONTEND = Path(__file__).parent.parent / "frontend"
for d in [VIDEOS, RESULTS, REPORTS]:
    d.mkdir(parents=True, exist_ok=True)

ALLOWED = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

app = FastAPI(title="Running Form Analyzer API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


# ── Routes ─────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "mode": "async" if CELERY_OK else "sync"}


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...),
                       stride: int = 2, render: bool = True,
                       do_fatigue: bool = True,
                       llm_provider: str = "deepseek"):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED:
        raise HTTPException(400, f"Unsupported: {ext}")

    task_id = str(uuid.uuid4())
    video_path = VIDEOS / f"{task_id}{ext}"
    content = await file.read()
    with open(video_path, "wb") as f:
        f.write(content)

    if CELERY_OK:
        task = analyze_video_task.delay(
            task_id=task_id, video_path=str(video_path),
            stride=stride, render=render,
            do_fatigue=do_fatigue, llm_provider=llm_provider)
        return {"task_id": task_id, "celery_task_id": task.id,
                "status": "queued",
                "video_size_mb": round(len(content) / 1024 / 1024, 1)}
    else:
        result = _run_sync(task_id, str(video_path), stride, render,
                           do_fatigue, llm_provider)
        return {"task_id": task_id, "celery_task_id": None,
                "status": "completed",
                "video_size_mb": round(len(content) / 1024 / 1024, 1),
                "result": result}


@app.get("/api/status/{task_id}")
def get_status(task_id: str):
    result_path = RESULTS / f"{task_id}.json"
    if result_path.exists():
        with open(result_path, encoding="utf-8") as f:
            data = json.load(f)
        return {"task_id": task_id, "status": "completed",
                "progress": 100, "result": data}
    return {"task_id": task_id, "status": "processing", "progress": 0}


@app.get("/api/videos/{filename}")
def get_video(filename: str):
    path = REPORTS / filename
    if not path.exists():
        raise HTTPException(404, "Not found")
    mt = "video/webm" if filename.endswith(".webm") else "video/mp4"
    return FileResponse(str(path), media_type=mt)


@app.get("/api/reports/{task_id}")
def get_report(task_id: str):
    result_path = RESULTS / f"{task_id}.json"
    if not result_path.exists():
        raise HTTPException(404, "Report not found")
    with open(result_path, encoding="utf-8") as f:
        data = json.load(f)
    html = _md_to_html(data.get("report", ""))
    return HTMLResponse(content=html, media_type="text/html",
                        headers={
                            "Content-Disposition":
                            f'attachment; filename="report_{task_id[:8]}.html"'
                        })


# ── Frontend ───────────────────────────────────

@app.get("/")
def index():
    p = FRONTEND / "index.html"
    return HTMLResponse(p.read_text(encoding="utf-8") if p.exists()
                        else "<h1>Frontend not found</h1>")


# ── Sync fallback ──────────────────────────────

def _run_sync(task_id, video_path, stride, render, do_fatigue, llm_provider):
    """Run analysis directly (no Celery/Redis needed)."""
    result = {"task_id": task_id, "status": "processing", "total_frames": 0}

    # Step 1
    extractor = PoseExtractor(model_complexity=1)
    seq = extractor.extract_from_video(video_path, stride=stride)
    if not seq.landmarks_seq:
        return {**result, "status": "error", "error": "No pose detected"}
    result["total_frames"] = len(seq.landmarks_seq)

    # Step 1.5
    qc = VideoQualityChecker()
    qr = qc.check(seq)
    result["quality"] = {"passed": qr.passed, "score": qr.score,
                         "summary": qr.summary, "issues": qr.issues,
                         "tips": qr.tips}

    # Step 2
    calc = RunningMetricsCalculator()
    metrics = calc.compute(seq)
    scoring = calc.get_scoring(metrics)
    result["metrics"] = metrics.summary()
    result["scoring"] = scoring

    # Step 3
    if render:
        raw = str(REPORTS / f"{task_id}_raw.mp4")
        preview = str(REPORTS / f"{task_id}_preview.mp4")
        RunningFormVisualizer().render_video(video_path, raw, seq, metrics)
        subprocess.run(["ffmpeg", "-y", "-i", raw, "-c:v", "libx264",
                        "-preset", "fast", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", preview],
                       capture_output=True, timeout=120)
        if os.path.exists(preview):
            result["preview_video"] = f"/api/videos/{task_id}_preview.mp4"
        if os.path.exists(raw):
            os.remove(raw)

    # Step 3.5
    if do_fatigue:
        fa = FatigueAnalyzer()
        fr = fa.analyze(seq)
        result["fatigue"] = fr.to_dict()

    # Step 4
    provider = llm_provider or os.environ.get("LLM_PROVIDER", "deepseek")
    client = create_llm_client(provider=provider, auto_fallback=True)
    coach = AIRunningCoach(llm_api_func=client,
                            fatigue_report=result.get("fatigue"))
    report = coach.generate_report(metrics, scoring)
    result["report"] = report
    result["llm_active"] = client is not None

    # Save
    with open(RESULTS / f"{task_id}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    result["status"] = "completed"
    return result


def _md_to_html(text: str) -> str:
    html = text
    html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.M)
    html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.M)
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)
    html = re.sub(r'^- (.+)$', r'<li>\1</li>', html, flags=re.M)
    return f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>跑姿分析报告</title><style>
body{{font-family:-apple-system,'Microsoft YaHei',sans-serif;max-width:800px;margin:40px auto;padding:20px;line-height:1.8}}
h2{{border-bottom:2px solid #00cc66;padding-bottom:4px}}
ul,p{{padding-left:0}}li{{margin:4px 0}}
</style></head><body>{html}</body></html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000)
