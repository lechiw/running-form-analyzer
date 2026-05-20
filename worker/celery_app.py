"""
Celery app configuration for Running Form Analyzer.
"""
import os
from celery import Celery

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "running_analyzer",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["worker.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Shanghai",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    # Limits
    task_soft_time_limit=600,   # 10 minutes
    task_time_limit=900,        # 15 minutes
    result_expires=86400,       # results expire after 1 day
)
