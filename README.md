# Property Due-Diligence Advisor: app

Photo in, ranked properties out, chatbot on top. Two containers: `api` (damage classifier) and `app` (Streamlit UI + chatbot). SQLite lives in `./storage`.

DSML Batch 42 final project (dibimbing.id) by Brian Hernawan. The damage model is trained and evaluated in [`property-damage-model`](https://github.com/brianhernawan/property-damage-model) (xBD dataset, CC BY-NC-SA 4.0).

## What is in here

| Path | Purpose |
|---|---|
| `compose.yaml` | `api` on the `backend` network, `app` on `frontend` + `backend` |
| `.env.example` | Copy to `.env` and fill in the two keys |
| `models/serving.pt` | You add this: the ResNet-18 checkpoint (see step 1) |
| `storage/advisor.db` | Created on first start |
| `data/` | Build context: Dockerfile and all Python code |
| `sync_from_root.sh` | Copies `inference.py` and `serve_api.py` from the project root into `data/` |

## Run it

1. Copy the model: `mkdir -p models && cp ../checkpoints/C1_resnet18/best.pt models/serving.pt`
   (Same weights the `serving` alias points to. The MLflow registry cannot be used inside the container because `mlflow.db` stores absolute paths from your Mac.)
2. Keys: `cp .env.example .env`, then edit `.env`. Without `GOOGLE_API_KEY` the chat is off and everything else works.
3. One-time networks: `docker network create frontend` and `docker network create backend` (skip any that exist).
4. `./sync_from_root.sh`
5. `docker compose up -d --build` (first build downloads PyTorch, a few minutes)
6. Open http://localhost:8501. API docs: http://localhost:8000/docs
7. Stop: `docker compose down`. The database stays in `./storage`.

## Without Docker (development)

```bash
pip install -r data/requirements.txt torch torchvision
cd data
CHECKPOINT=../../checkpoints/C1_resnet18/best.pt uvicorn serve_api:app --port 8000 &
API_URL=http://localhost:8000 DB_PATH=../storage/advisor.db streamlit run app.py
```

## Tests

`cd data && python3 test_core.py && python3 test_agent.py` (database, recommender, agent wiring; no API keys needed).

## Data

- The 10 properties shown at first start are SYNTHETIC. Load real listings: `python3 -c "import db; db.load_properties_csv(db.connect('../storage/advisor.db'), 'listings.csv')"` (required columns `title,city,price_idr`).
- The `rppi` table is empty until you load the Bank Indonesia RPPI CSV with `db.load_rppi_csv` (columns `city,year,quarter,house_type,index_value,yoy_growth_pct`). Nothing reads it yet.

## Known limits

- The classifier was trained on top-down satellite crops of buildings (xBD). Use satellite crops of one building, not street photos.
- The ranking weights (0.40 condition, 0.35 budget fit, 0.25 location) are design choices, shown in the app, not fitted. The BNPB InaRISK flood hazard index is shown per property but is not part of the score.
- Chat answers depend on live web search results; the agent is told to cite URLs and never guess a price.
