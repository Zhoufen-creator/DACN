FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

COPY requirements.txt .

RUN python -m pip install --no-cache-dir -r requirements.txt

CMD ["python", "--version"]