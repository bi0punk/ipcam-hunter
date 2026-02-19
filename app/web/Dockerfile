FROM python:3.11-slim

WORKDIR /srv
COPY requirements.txt /srv/requirements.txt
RUN pip install --no-cache-dir -r /srv/requirements.txt

COPY webapp /srv/webapp

ENV EVENTS_DIR=/data/events
EXPOSE 8081
CMD ["uvicorn", "webapp.main:app", "--host", "0.0.0.0", "--port", "8081"]