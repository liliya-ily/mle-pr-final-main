# Рекомендательная система банковских продуктов

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)
![implicit](https://img.shields.io/badge/ALS-implicit-8A2BE2?style=flat-square)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2.5-FFCC00?style=flat-square)
![MLflow](https://img.shields.io/badge/MLflow-2.7.1-0194E2?style=flat-square&logo=mlflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)
![Prometheus](https://img.shields.io/badge/Prometheus-E6522C?style=flat-square&logo=prometheus&logoColor=white)

Учебный проект курса повышения квалификации «Инженер машинного обучения» (Яндекс Практикум, итоговый проект, кейс «Рекомендации банковских продуктов»).

Система предсказывает, какие банковские продукты клиент подключит в ближайшем месяце, и возвращает топ-K продуктов. Решение двухэтапное: **ALS** отбирает кандидатов, **CatBoost** их ранжирует. Эксперименты логируются в **MLflow** (хранилище артефактов S3, Model Registry), модель обёрнута в **FastAPI**-сервис в **Docker** с метриками для **Prometheus**.

## Структура репозитория

```text
.
├── notebooks/
│   ├── 01_eda.ipynb            # исследование данных
│   ├── 02_modeling.ipynb       # признаки, ALS, ранкер, оценка, сохранение артефактов
│   └── final_notebook.ipynb    # оба ноутбука одним файлом
├── model_improvement/assets/   # графики EDA (png)
├── services/
│   ├── ml_service/             # FastAPI-сервис (fast_api_handler.py, main_monitoring.py)
│   ├── models/                 # файл модели CatBoost (model.cbm)
│   ├── assets/                 # артефакты для сервиса (parquet, json), в git не хранятся
│   ├── prometheus/             # конфиг Prometheus
│   ├── Dockerfile_ml_service
│   ├── docker-compose.yaml
│   ├── requirements.txt
│   ├── run_mlflow.sh           # запуск MLflow Tracking Server и Model Registry
│   ├── test_load.py            # скрипт нагрузки на сервис
│   └── .env.example            # шаблон переменных окружения
├── mlflow_server/
├── Monitoring.md               # описание метрик мониторинга
└── README.md
```

## Данные

Данные о клиентах банка за 1,5 года: ежемесячные записи с 28 января 2015 года. Колонки: демография и тип клиента, а также 24 флага владения продуктами `ind_*_ult1`; предсказывается подключение новых продуктов в 2016 году.

Файл `train_ver2.csv` в репозиторий не входит. Скачать архив из Яндекс.Диска и распаковать:

```bash
sudo apt-get update
sudo apt-get install -y jq unzip
wget -O train_ver2.csv.zip $(curl -s "https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key=https://disk.yandex.com/d/Io0siOESo2RAaA" | jq -r '.href')
unzip train_ver2.csv.zip
```

Ноутбуки читают `../train_ver2.csv`, поэтому файл кладётся в корень репозитория (файлы `*.csv` указаны в `.gitignore`).

## Установка

```bash
git clone <ссылка-на-репозиторий>
cd <папка-проекта>
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r services/requirements.txt
```

Переменные окружения (доступы к S3 и PostgreSQL для MLflow, имя бакета) задаются в файле `services/.env`:

```bash
cp services/.env.example services/.env
```

После копирования заполните значения. Файл `.env` в репозиторий не попадает.

## Руководство по проекту

### 1. Бизнес-задача и метрики

**Бизнес-задача:** предложить клиенту банка продукт (кредитную карту, депозит, пенсионный план и т. д.), который он с наибольшей вероятностью подключит, чтобы увеличить число продуктов на клиента.

**Техническая постановка:** для пары (клиент, месяц) ранжировать 24 продукта и вернуть топ-K (K = 5). Целевое событие: переход флага владения продуктом из 0 в 1 в текущем месяце (новое подключение).

**Схема валидации:** по времени. Модель обучается на данных 2015 года, метки берутся из 2016 года, качество оценивается на последнем месяце (май 2016).

**Метрики** (для топ-K рекомендаций каждого клиента):

- Precision@K и Recall@K по всем клиентам и по клиентам, у которых в месяце было хотя бы одно новое подключение (`pos`). Подключения редкие (около 0,4–0,7 % пар клиент-продукт), поэтому значения «по всем» малы по определению, а метрики `pos` показывают качество ранжирования там, где есть что предсказывать;
- Coverage по клиентам и по продуктам, Novelty@K, Diversity@K.

### 2. Инфраструктура обучения (MLflow)

MLflow Tracking Server с Model Registry использует PostgreSQL как backend store и S3-совместимое хранилище для артефактов. Параметры подключения читаются из `services/.env` (шаблон: `services/.env.example`). Запуск одной командой:

```bash
bash services/run_mlflow.sh
```

Веб-интерфейс: http://localhost:5000. В ноутбуках используется эксперимент `bank_products_recsys`.

### 3. EDA

Ноутбук `notebooks/01_eda.ipynb`, графики в `model_improvement/assets/` (они же логируются в MLflow как артефакты запуска `eda_artifacts`). Основные выводы:

- Таргет стабилен во времени: доля клиентов с новым продуктом в месяце около 3–5 % без выраженного тренда, есть локальные пики (середина 2015 и начало 2016 года).
- Каталог из 24 продуктов крайне неравномерен: один продукт (`ind_cco_fin_ult1`, текущий счёт) доминирует по владению, большинство клиентов имеют 1–2 продукта. Присутствует popularity bias, который учитывается при выборе метрик.
- Самый сильный разделяющий признак: `ind_actividad_cliente` (активность клиента). Заметны различия по `segmento`, `tiprel_1mes`, `canal_entrada` (требует агрегации top-N + other); `tipodom` и `indrel` малоинформативны.
- Линейные корреляции признаков с таргетом слабые, поэтому выбраны модели на градиентном бустинге.

### 4. Признаки и обучение модели

Ноутбук `notebooks/02_modeling.ipynb`:

1. **Baseline**: кандидаты по популярности, CatBoost-ранкер с признаками `score_pop`, `month_str`, `product_id`.
2. **ALS + CatBoost**: ALS (`implicit`) на матрице владения продуктами даёт топ-10 кандидатов на клиента (признак `score_als`), CatBoost ранжирует их.
3. **Дополнительные признаки**: демография и тип клиента (`age`, `log_renta`, `antiguedad`, `sexo`, `ind_nuevo`, `ind_actividad_cliente`, `segmento`, `tiprel_1mes`, `canal_entrada` и др.) и история клиента на предыдущий месяц: число продуктов, наличие нового подключения, число месяцев с подключением за последние 3 месяца, давность последнего подключения. Месяц используется как категориальный признак (сезонность).

Метрики на валидационном месяце (топ-5):

| Модель | Precision@5 (все) | Recall@5 (все) | Precision@5 (pos) | Recall@5 (pos) |
|---|---:|---:|---:|---:|
| Популярность + CatBoost (baseline) | 0,0066 | 0,0252 | 0,216 | 0,827 |
| ALS + CatBoost | 0,0128 | 0,0482 | 0,250 | 0,941 |
| ALS + CatBoost + демография и история | 0,0130 | 0,0490 | 0,253 | 0,957 |

Во всех вариантах охват клиентов равен 1,0; Novelty@5 и Diversity@5 равны 1,0 (уже подключённые продукты не рекомендуются, каталог небольшой).

**Запуски в MLflow.** Для каждой версии модели логируются параметры, метрики, сигнатура и пример входных данных, графики. Финальная модель зарегистрирована в Model Registry как `bank_products_ranker_als`, версии отличаются набором признаков (`als_candidates_catboost_ranker`, `als_ranker_with_demo_history`).

После обучения ноутбук сохраняет в S3-бакет (папка `bank_recsys_artifacts/`) артефакты для сервиса: `model.cbm`, `personal_als.parquet` (кандидаты ALS), `user_features.parquet` (признаки клиентов), `inference_config.json` (список признаков, параметры top-K).

### 5. Инфраструктура применения модели

Сервис при старте загружает артефакты из S3 в память и по запросу собирает признаки, ранжирует кандидатов и возвращает топ-K. Запуск (из папки `services`, после заполнения `.env`):

```bash
cd services
docker compose up --build
```

Адреса:

- API: http://localhost:4601/recommend
- Swagger UI: http://localhost:4601/docs
- Метрики Prometheus (сервис): http://localhost:4601/metrics
- Prometheus: http://localhost:9090

Пример запроса:

```bash
curl -s -X POST http://localhost:4601/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 15896, "month": "2016-01", "k": 5}'
```

Ответ содержит список продуктов `product_id` с оценкой модели `score` и оценкой ALS `score_als`. Запрос с неверными параметрами (например, `user_id` строкой) возвращает код 400.

Нагрузочный скрипт (при запущенном сервисе):

```bash
python services/test_load.py
```

Он отправляет корректный и некорректный запросы и серию из 20 запросов, чтобы счётчики были видны в метриках. Описание метрик мониторинга: [Monitoring.md](Monitoring.md).
