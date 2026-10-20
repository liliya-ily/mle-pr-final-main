# Рекомендательная система банковских продуктов

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)
![implicit](https://img.shields.io/badge/ALS-implicit-8A2BE2?style=flat-square)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2.5-FFCC00?style=flat-square)
![MLflow](https://img.shields.io/badge/MLflow-2.7.1-0194E2?style=flat-square&logo=mlflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)
![Prometheus](https://img.shields.io/badge/Prometheus-E6522C?style=flat-square&logo=prometheus&logoColor=white)
![Grafana](https://img.shields.io/badge/Grafana-F46800?style=flat-square&logo=grafana&logoColor=white)

Учебный проект курса повышения квалификации «Инженер машинного обучения» (Яндекс Практикум, итоговый проект, кейс «Рекомендации банковских продуктов»).

Система предсказывает, какие банковские продукты клиент подключит в ближайшем месяце, и возвращает топ-K продуктов. Решение двухэтапное: **ALS** отбирает кандидатов, **CatBoost** их ранжирует. Эксперименты логируются в **MLflow** (SQLite и локальное хранилище артефактов, Model Registry), модель обёрнута в **FastAPI**-сервис в **Docker** с метриками для **Prometheus** и дашбордом **Grafana**. Весь проект запускается локально, без облачных сервисов.

> Версия проекта с хранилищем S3 и облачной БД для MLflow, как она сдавалась на проверку курса, находится в ветке `course/s3`.

## Структура репозитория

```text
.
├── notebooks/
│   ├── 01_eda.ipynb            # исследование данных
│   ├── 02_modeling.ipynb       # признаки, ALS, ранкер, оценка, сохранение артефактов
│   └── recsys_utils.py         # функции: выборка клиентов, метрики топ-K, логирование в MLflow
├── model_improvement/assets/   # графики EDA (png)
├── services/
│   ├── ml_service/             # FastAPI-сервис (fast_api_handler.py, main_monitoring.py)
│   ├── models/                 # model.cbm: модель CatBoost (создаёт 02_modeling)
│   ├── assets/                 # таблицы и конфиг для сервиса (parquet, json), в git не хранятся
│   ├── prometheus/             # конфиг Prometheus
│   ├── grafana/                # источник данных и дашборд Grafana (provisioning)
│   ├── Dockerfile_ml_service
│   ├── docker-compose.yaml
│   ├── requirements.txt        # зависимости сервиса
│   ├── run_mlflow.sh           # запуск MLflow Tracking Server и Model Registry
│   ├── test_load.py            # скрипт нагрузки на сервис
│   └── .env.example            # необязательные переменные окружения
├── scripts/                    # train.sh, run_stack.sh, stop_stack.sh, smoke_test.sh
├── tests/                      # тесты pytest: метрики, выборка клиентов, обработчик и API сервиса
├── docker/Dockerfile           # окружение для обучения и тестов
├── mlflow_server/              # база и артефакты MLflow (создаются при запуске, в git не хранятся)
├── data/                       # исходный файл train_ver2.csv (в git не хранится)
├── requirements.txt            # зависимости ноутбуков, обучения и тестов
├── Monitoring.md               # описание метрик мониторинга
└── README.md
```

## Данные

Данные о клиентах банка за 1,5 года: ежемесячные записи с 28 января 2015 года (13,6 млн строк, около 957 тысяч клиентов). Колонки: демография и тип клиента, а также 24 флага владения продуктами `ind_*_ult1`; предсказывается подключение новых продуктов в 2016 году.

Файл `train_ver2.csv` в репозиторий не входит. Скачать архив из Яндекс.Диска и распаковать в папку `data/`:

```bash
sudo apt-get update
sudo apt-get install -y jq unzip
mkdir -p data && cd data
wget -O train_ver2.csv.zip $(curl -s "https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key=https://disk.yandex.com/d/Io0siOESo2RAaA" | jq -r '.href')
unzip train_ver2.csv.zip
cd ..
```

Файлы `*.csv` и папка `data/` указаны в `.gitignore`.

## Установка

```bash
git clone <ссылка-на-репозиторий>
cd <папка-проекта>
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Все настройки имеют значения по умолчанию. При необходимости переменные окружения (адрес MLflow, папка с данными, учётные данные Grafana) задаются в файле `services/.env`, шаблон: `services/.env.example`.

Окружение можно собрать и в Docker:

```bash
docker build -f docker/Dockerfile -t bank-recsys .
```

## Руководство по проекту

### 1. Бизнес-задача и метрики

**Бизнес-задача:** предложить клиенту банка продукт (кредитную карту, депозит, пенсионный план и т. д.), который он с наибольшей вероятностью подключит, чтобы увеличить число продуктов на клиента.

**Техническая постановка:** для пары (клиент, месяц) ранжировать 24 продукта и вернуть топ-K (K = 5). Целевое событие: переход флага владения продуктом из 0 в 1 в текущем месяце (новое подключение).

**Схема валидации:** по времени. Модель обучается на данных 2015 года, метки берутся из 2016 года, качество оценивается на последнем месяце (май 2016).

**Метрики** (для топ-K рекомендаций каждого клиента):

- Precision@K, Recall@K, MAP@K и NDCG@K (две последние учитывают позицию верного продукта в списке) по всем парам (клиент, месяц) и только по тем парам, где у клиента появился хотя бы один новый продукт (`pos`). Подключения редкие (около 0,4 % пар клиент-продукт), поэтому значения «по всем» малы по определению, а метрики `pos` показывают качество ранжирования там, где есть что предсказывать;
- Coverage по клиентам и по продуктам, Novelty@K, Diversity@K.

### 2. Инфраструктура обучения (MLflow)

MLflow Tracking Server с Model Registry использует SQLite (`mlflow_server/mlflow.db`) и локальную папку артефактов (`mlflow_server/mlartifacts`), которые сервер отдаёт клиентам сам. Все ноутбуки логируют запуски в эксперимент `bank_products_recsys`.

**Обучение.** Скрипт сам поднимает MLflow, выполняет один ноутбук и останавливает сервер. Ноутбуки запускаются отдельно:

```bash
bash scripts/train.sh 01_eda
bash scripts/train.sh 02_modeling
```

В Docker:

```bash
docker run --rm --cpus=4 --memory=12g -v "$(pwd)":/app -w /app bank-recsys bash scripts/train.sh 02_modeling
```

**Просмотр экспериментов.** После обучения запустить сервер на сохранённых данных (переобучение не нужно):

```bash
bash services/run_mlflow.sh
```

Интерфейс: http://localhost:5000. Тот же сервер в Docker:

```bash
docker run --rm -p 5000:5000 -v "$(pwd)":/app -w /app bank-recsys bash services/run_mlflow.sh
```

Сервер, запущенный для просмотра, нужно остановить перед следующим обучением: `train.sh` использует тот же порт 5000.

Что смотреть в интерфейсе:

- **Experiments → `bank_products_recsys`**: запуски `baseline_popularity_catboost`, `als_candidates_catboost_ranker`, `als_ranker_with_demo_history` и `eda_artifacts`. Отметив три первых и нажав Compare, можно сравнить метрики и параметры моделей.
- **Артефакты запуска** (вкладка Artifacts): модель CatBoost с сигнатурой и примером входных данных; у `eda_artifacts` графики EDA.
- **Models → `bank_products_ranker_als`**: версии модели в Model Registry.

### 3. EDA

Ноутбук `notebooks/01_eda.ipynb`, графики в `model_improvement/assets/` (они же логируются в MLflow как артефакты запуска `eda_artifacts`). Анализ выполняется на выборке из 21 % клиентов (около 200 тысяч): история каждого клиента сохраняется целиком, поэтому доли и распределения совпадают с полной таблицей. Основные выводы:

- Таргет стабилен во времени: доля клиентов с новым продуктом в месяце около 3–5 % без выраженного тренда, есть локальные пики (середина 2015 и начало 2016 года).
- Каталог из 24 продуктов крайне неравномерен: один продукт (`ind_cco_fin_ult1`, текущий счёт) доминирует по владению, большинство клиентов имеют 1–2 продукта. Присутствует popularity bias, который учитывается при выборе метрик.
- Самый сильный разделяющий признак: `ind_actividad_cliente` (активность клиента). Заметны различия по `segmento`, `tiprel_1mes`, `canal_entrada` (требует агрегации top-N + other); `tipodom` и `indrel` малоинформативны.
- Линейные корреляции признаков с таргетом слабые, поэтому выбраны модели на градиентном бустинге.

### 4. Признаки и обучение модели

Ноутбук `notebooks/02_modeling.ipynb`. Файл с данными читается порциями, в расчёт берётся выборка из 21 % клиентов (около 200 тысяч) с полной историей каждого, и только нужные колонки.

1. **Baseline**: кандидаты по популярности, CatBoost-ранкер с признаками `score_pop`, `month_str`, `product_id`.
2. **ALS + CatBoost**: ALS (`implicit`) на матрице владения продуктами даёт топ-10 кандидатов на клиента (признак `score_als`), CatBoost ранжирует их.
3. **Дополнительные признаки**: демография и тип клиента (`age`, `log_renta`, `antiguedad`, `sexo`, `ind_nuevo`, `ind_actividad_cliente`, `segmento`, `tiprel_1mes`, `canal_entrada` и др.) и история клиента на предыдущий месяц: число продуктов, наличие нового подключения, число месяцев с подключением за последние 3 месяца, давность последнего подключения. Месяц используется как категориальный признак (сезонность).

Метрики на валидационном месяце (топ-5):

| Модель | Precision@5 (все) | Recall@5 (все) | MAP@5 (все) | NDCG@5 (все) | Precision@5 (pos) | Recall@5 (pos) | MAP@5 (pos) | NDCG@5 (pos) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Популярность + CatBoost (baseline) | 0,0066 | 0,0252 | 0,0165 | 0,0189 | 0,216 | 0,828 | 0,542 | 0,621 |
| ALS + CatBoost | 0,0138 | 0,0498 | 0,0381 | 0,0411 | 0,265 | 0,957 | 0,734 | 0,792 |
| ALS + CatBoost + демография и история | 0,0137 | 0,0498 | 0,0405 | 0,0430 | 0,265 | 0,958 | 0,780 | 0,828 |

Охват клиентов равен 1,0 во всех вариантах; охват каталога продуктов вырос с 0,74 у baseline до 1,0 у итоговой модели; Novelty@5 и Diversity@5 равны 1,0 (уже подключённые продукты не рекомендуются, каталог небольшой).

**Запуски в MLflow.** Для каждой версии модели логируются параметры, метрики, сигнатура и пример входных данных. Финальная модель зарегистрирована в Model Registry как `bank_products_ranker_als`; запуски `baseline_popularity_catboost`, `als_candidates_catboost_ranker`, `als_ranker_with_demo_history` можно сравнивать между собой в интерфейсе MLflow.

После обучения ноутбук сохраняет артефакты для сервиса: `services/models/model.cbm`, `services/assets/personal_als.parquet` (кандидаты ALS), `services/assets/user_features.parquet` (признаки клиентов), `services/assets/inference_config.json` (список признаков, параметры top-K).

### 5. Инфраструктура применения модели

Сервис при старте загружает артефакты с диска в память и по запросу собирает признаки, ранжирует кандидатов и возвращает топ-K. Запуск сервиса, Prometheus и Grafana (после обучения):

```bash
bash scripts/run_stack.sh
```

Остановка:

```bash
bash scripts/stop_stack.sh
```

Адреса:

- API: http://localhost:4601/recommend
- Swagger UI: http://localhost:4601/docs
- Метрики Prometheus (сервис): http://localhost:4601/metrics
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin / admin), дашборд подключается автоматически

Пример запроса:

```bash
curl -s -X POST http://localhost:4601/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 15889, "month": "2016-01", "k": 5}'
```

Ответ содержит список продуктов `product_id` с оценкой модели `score` и оценкой ALS `score_als`. Запрос с неверными параметрами (например, `user_id` строкой) возвращает код 400.

Быстрая проверка запущенного сервиса (корректный запрос, запрос с ошибкой, метрики):

```bash
bash scripts/smoke_test.sh
```

Нагрузочный скрипт (при запущенном сервисе):

```bash
python services/test_load.py
```

Он отправляет корректный и некорректный запросы и серию из 20 запросов, чтобы счётчики были видны в метриках и на дашборде. Описание метрик и панелей: [Monitoring.md](Monitoring.md).

### 6. Тесты

Тесты проверяют метрики топ-K (на примере с ручным расчётом), выборку клиентов из CSV, валидацию входа и формирование рекомендаций в обработчике, а также HTTP-сервис вместе с метриками Prometheus (на небольших искусственных артефактах):

```bash
pytest
```

В Docker:

```bash
docker run --rm --cpus=2 --memory=2g -v "$(pwd)":/app -w /app bank-recsys python -m pytest -q
```
