# Celery worker for the Tool Scan pipeline. Deliberately minimal - garak/pyrit/deepteam
# are NEVER installed into this image. They live in tool-venvs/ on the host and are
# bind-mounted in, then invoked only via subprocess (see engine/tool_scan.py's
# _get_tool_python / GARAK_PYTHON, PYRIT_PYTHON, DEEPTEAM_PYTHON), never imported here.
FROM python:3.12-slim

WORKDIR /app

COPY worker/requirements.txt /app/worker/requirements.txt
RUN pip install --no-cache-dir -r /app/worker/requirements.txt

COPY core /app/core
COPY database /app/database
COPY engine /app/engine
COPY worker /app/worker

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["celery", "-A", "worker.celery_app", "worker", "--loglevel=INFO"]
