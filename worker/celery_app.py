"""Shared Celery app. Imported by both processes: the FastAPI process only ever calls
celery_app.send_task("run_scan_job", args=[job_id]) to publish work (never runs a
worker, never imports worker.tasks or any tool library); the worker process runs it
as `celery -A worker.celery_app worker`, which pulls in worker.tasks via `include`."""

from __future__ import annotations

from celery import Celery

from core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "tool_scan_worker",
    broker=settings.rabbitmq_url,
    backend=settings.valkey_url,
    include=["worker.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    task_time_limit=3600,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    # Bound how long a single .send_task() publish can hang when the broker is
    # unreachable (verified empirically: with Kombu's defaults, publishing against a
    # dead broker blocks for well over 20s with no exception - unacceptable for the
    # FastAPI request path in core/api.py, which needs to fail fast and return a
    # clean error rather than hang the request). Does not affect the worker's own
    # broker *consumption* connection, which should still reconnect/retry normally.
    broker_connection_timeout=3,
    task_publish_retry=True,
    task_publish_retry_policy={"max_retries": 1, "interval_start": 0, "interval_step": 0.5, "interval_max": 1},
)
