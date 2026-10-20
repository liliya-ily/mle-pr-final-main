FROM python:3.10-slim

# libgomp1 нужна implicit и CatBoost
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# один поток BLAS быстрее и стабильнее для ALS
ENV OPENBLAS_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt ./requirements.txt
COPY services/requirements.txt ./services/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# код проекта и данные подключаются томом (-v), в образ не копируются
CMD ["bash"]
