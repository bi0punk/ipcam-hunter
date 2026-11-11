FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin hunter

WORKDIR /app

COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r /app/requirements.txt

COPY --chown=hunter:hunter app/ /app/
COPY docker/app/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && chown hunter:hunter /entrypoint.sh

ENV CONFIG_PATH=/app/config.yaml \
    EVENTS_DIR=/data/events \
    HOME=/home/hunter

RUN mkdir -p /data/events /home/hunter/.cache /home/hunter/.config/Ultralytics \
    && chown -R hunter:hunter /data/events /home/hunter

USER hunter

VOLUME ["/data/events", "/home/hunter/.cache", "/home/hunter/.config/Ultralytics"]

HEALTHCHECK --interval=60s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import os,sys,time; p='/data/events/status.json'; sys.exit(0 if os.path.exists(p) and time.time()-os.path.getmtime(p) < 120 else 1)"

ENTRYPOINT ["/entrypoint.sh"]
