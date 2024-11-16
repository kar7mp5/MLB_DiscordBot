FROM ubuntu:22.04
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod 700 bot.py
CMD ["python3", "bot.py"]
