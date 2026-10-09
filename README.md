# Property Due-Diligence Advisor: app

Photo in, ranked properties out, chatbot on top. Four containers: `api` (damage classifier), `app` (Streamlit UI + chatbot), `prometheus` (scrapes the API) and `grafana` (dashboard). SQLite lives in `./storage`.

DSML Batch 42 final project (dibimbing.id) by Brian Hernawan. The damage model is trained and evaluated in [`model/`](model/) in this same repo (xBD dataset, CC BY-NC-SA 4.0).

## What is in here

| Path | Purpose |
|---|---|
| `compose.yaml` | `api` and `prometheus` on `backend`; `app` and `grafana` on `frontend` + `backend` |
| `.env.example` | Copy to `.env` and fill in the two API keys and the Grafana admin password |
| `monitoring/` | Prometheus scrape config, Grafana datasource and the provisioned dashboard |
| `models/serving.pt` | You add this: the ResNet-18 checkpoint (see step 1) |
| `storage/advisor.db` | Created on first start |
| `data/` | Build context: Dockerfile and all Python code |
| `model/` | Training, evaluation and Maps-test scripts for the damage CNN (see `model/README.md`) |

## Run it

1. Copy the model: `mkdir -p models && cp model/checkpoints/C1_resnet18/best.pt models/serving.pt`
   (Same weights the `serving` alias points to. The MLflow registry cannot be used inside the container because `mlflow.db` stores absolute paths from your Mac.)
2. Keys: `cp .env.example .env`, then edit `.env`. Without `GOOGLE_API_KEY` the chat is off and everything else works.
3. One-time networks: `docker network create frontend` and `docker network create backend` (skip any that exist).
4. `docker compose up -d --build` (first build downloads PyTorch, a few minutes)
5. Open http://localhost:8501. API docs: http://localhost:8000/docs. Grafana: http://localhost:3000 (user `admin`, password from `.env`). Prometheus: http://localhost:9090
6. Stop: `docker compose down`. The database stays in `./storage`.

## Recommender

Two recommenders work side by side (tab 2):

1. **Weighted ranking** (`recommender.py`): `0.40 condition + 0.35 budget fit + 0.25 location`. Condition is `1 - p_damaged` from the CNN.
2. **Content-based, TF-IDF** (`content.py`): each listing's description, type, city, district and bedroom count become one text profile, turned into a TF-IDF vector.
   - *Similar properties*: cosine similarity between profiles, blended with closeness on price, land, building size and bedrooms (`0.6 text + 0.4 numbers`). The table shows the shared keywords behind each match.
   - *Describe what you want*: the free text is matched against every profile; when given, the ranking becomes `0.35 condition + 0.30 budget + 0.20 location + 0.15 text match`.

The chatbot can call the content-based recommender too (`find_similar`). All weights are design choices, not fitted values. TF-IDF needs listing text: the demo descriptions are synthetic, and real listings should carry a `description` column in the CSV.

## Monitoring

The API exposes Prometheus metrics at `GET /metrics`. Prometheus scrapes it every 15 s and keeps 15 days. Grafana opens on the provisioned dashboard **Property Advisor: damage API**.

| Metric | What it shows |
|---|---|
| `up{job="fp-advisor-api"}` | Prometheus can reach the API |
| `dd_model_loaded` | 1 when the classifier is loaded |
| `dd_predictions_total{label,damaged}` | Images classified, by 4-way label and by the binary damaged flag |
| `dd_p_damaged` (histogram) | Spread of `p_damaged`; a shift here is a cheap drift signal |
| `dd_inference_seconds` (histogram) | Preprocess + forward-pass time |
| `dd_rejected_total{reason}` | Uploads refused (too large, not an image) |
| `http_requests_total`, `http_request_duration_seconds` | Traffic, status codes and latency per endpoint |

Quick check: `curl -s http://127.0.0.1:8000/metrics | grep ^dd_`. The API (8000) and Prometheus (9090) are open on the lab network and have no login: do not publish them through a public reverse proxy. Grafana (3000) needs a password.

## Without Docker (development)

```bash
pip install -r data/requirements.txt torch torchvision
cd data
CHECKPOINT=../model/checkpoints/C1_resnet18/best.pt uvicorn serve_api:app --port 8000 &
API_URL=http://localhost:8000 DB_PATH=../storage/advisor.db streamlit run app.py
```

## Tests

`cd data && python3 test_core.py && python3 test_content.py && python3 test_agent.py` (database, both recommenders, agent wiring; no API keys needed).

## Data

- The 10 properties shown at first start are SYNTHETIC. Load real listings: `python3 -c "import db; db.load_properties_csv(db.connect('../storage/advisor.db'), 'listings.csv')"` (required columns `title,city,price_idr`; add `description` for the TF-IDF recommender).
- The `rppi` table is empty until you load the Bank Indonesia RPPI CSV with `db.load_rppi_csv` (columns `city,year,quarter,house_type,index_value,yoy_growth_pct`). Nothing reads it yet.

## Known limits

- The classifier was trained on top-down satellite crops of buildings (xBD). Use satellite crops of one building, not street photos.
- The ranking weights (0.40 condition, 0.35 budget fit, 0.25 location) are design choices, shown in the app, not fitted. The BNPB InaRISK flood hazard index is shown per property but is not part of the score.
- Chat answers depend on live web search results; the agent is told to cite URLs and never guess a price.
