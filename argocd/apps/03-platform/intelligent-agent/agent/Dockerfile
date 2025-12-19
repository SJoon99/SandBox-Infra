FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# 로그 출력이 바로 보이게 설정
ENV PYTHONUNBUFFERED=1

# 실행 (8000번 포트)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]