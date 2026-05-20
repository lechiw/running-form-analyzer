"""
FastAPI application for Running Form Analyzer.
Handles video upload, async task management, and result retrieval.
"""
import os
import uuid
import json
import shutil
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from worker.celery_app import celery_app
from worker.tasks import analyze_video_task

_default_storage = str(Path(__file__).parent.parent / "data")
STORAGE_DIR = Path(os.environ.get("STORAGE_DIR", _default_storage))
VIDEOS_DIR = STORAGE_DIR / "videos"
RESULTS_DIR = STORAGE_DIR / "results"
REPORTS_DIR = STORAGE_DIR / "reports"
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"  📂 Videos: {VIDEOS_DIR}")
    print(f"  📂 Results: {RESULTS_DIR}")
    print(f"  🖥️  Frontend: {FRONTEND_DIR}")
    yield


app = FastAPI(
    title="Running Form Analyzer API",
    description="上传跑步视频，异步分析跑姿并生成报告",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── API Routes ─────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...),
                       stride: int = 2,
                       render: bool = True,
                       do_fatigue: bool = True,
                       llm_provider: str = "deepseek"):
    """Upload a video and start async analysis."""
    # Validate file
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported format: {ext}")

    # Generate task ID and save video
    task_id = str(uuid.uuid4())
    video_path = VIDEOS_DIR / f"{task_id}{ext}"

    with open(video_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Start async task
    task = analyze_video_task.delay(
        task_id=task_id,
        video_path=str(video_path),
        stride=stride,
        render=render,
        do_fatigue=do_fatigue,
        llm_provider=llm_provider,
    )

    return {
        "task_id": task_id,
        "celery_task_id": task.id,
        "status": "queued",
        "video_size_mb": round(len(content) / 1024 / 1024, 1),
    }


@app.get("/api/status/{task_id}")
def get_status(task_id: str):
    """Get analysis progress and results."""
    # Check Celery async result
    from celery.result import AsyncResult
    from worker.celery_app import celery_app as app

    # Search for result in stored results
    result_path = RESULTS_DIR / f"{task_id}.json"
    if result_path.exists():
        with open(result_path, encoding="utf-8") as f:
            data = json.load(f)
        return {
            "task_id": task_id,
            "status": data.get("status", "completed"),
            "progress": 100,
            "result": data,
        }

    return {"task_id": task_id, "status": "processing", "progress": 0}


@app.get("/api/videos/{filename}")
def get_video(filename: str):
    """Serve rendered preview videos."""
    video_path = REPORTS_DIR / filename
    if not video_path.exists():
        raise HTTPException(404, "Video not found")

    media_type = "video/mp4"
    if filename.endswith(".webm"):
        media_type = "video/webm"

    return FileResponse(str(video_path), media_type=media_type)


@app.get("/api/reports/{task_id}")
def get_report(task_id: str):
    """Download analysis report as HTML."""
    result_path = RESULTS_DIR / f"{task_id}.json"
    if not result_path.exists():
        raise HTTPException(404, "Report not found")

    with open(result_path, encoding="utf-8") as f:
        data = json.load(f)

    report_text = data.get("report", "")
    html = _generate_html_report(report_text)

    return HTMLResponse(content=html, media_type="text/html",
                        headers={
                            "Content-Disposition":
                            f'attachment; filename="report_{task_id[:8]}.html"'
                        })


# ── Frontend ───────────────────────────────────

@app.get("/")
def index():
    """Serve the SPA frontend."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Running Form Analyzer</h1><p>Frontend not found</p>")


@app.get("/app/{path:path}")
def serve_frontend(path: str):
    """Serve static frontend files."""
    file_path = FRONTEND_DIR / path
    if file_path.exists() and file_path.is_file():
        return FileResponse(str(file_path))
    # SPA fallback
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text(encoding="utf-8"))
    raise HTTPException(404)


# ── Helper ────────────────────────────────────

def _generate_html_report(text: str) -> str:
    """Convert markdown report to styled HTML."""
    import re
    html = text
    html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.M)
    html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.M)
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)
    html = re.sub(r'^- (.+)$', r'<li>\1</li>', html, flags=re.M)
    html = re.sub(r'^\d+\.\s+(.+)$', r'<li>\1</li>', html, flags=re.M)
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>跑姿分析报告</title>
<style>
body{{font-family:-apple-system,'Microsoft YaHei',sans-serif;max-width:800px;margin:40px auto;padding:20px;line-height:1.8;background:#fff;color:#1a1a1a}}
h2{{border-bottom:2px solid #00cc66;padding-bottom:4px}}
ul,ol{{padding-left:20px}}
li{{margin:4px 0}}
</style></head><body>{html}</body></html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
