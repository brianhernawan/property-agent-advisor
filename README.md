# Property Due-Diligence & Investment Advisor

Photo in, ranked properties out, chatbot on top. A CNN reads a top-down satellite crop of one building and estimates the chance it is damaged; BNPB flood-hazard data shows whether a location is flood-prone; a recommender ranks listings by condition, budget, location and (optionally) flood safety; an advisor chatbot answers price, condition and flood questions with sources.

DSML Batch 42 final project (dibimbing.id) by Brian Hernawan. Pilot scope: Greater Jakarta (the demo data also includes Bandung).

Live demo: <https://property-agent.katawarna.id/>

## Contents

1. [Results](#results)
2. [How it fits together](#how-it-fits-together)
3. [Run the app](#run-the-app)
4. [Monitoring](#monitoring)
5. [Recommender](#recommender)
6. [Advisor chatbot](#advisor-chatbot)
7. [Train and evaluate the model](#train-and-evaluate-the-model)
8. [Experiment log](#experiment-log)
9. [Real-world check on Google Earth images](#real-world-check-on-google-earth-images)
10. [Background: the xBD paper](#background-the-xbd-paper)
11. [Tests](#tests)
12. [CI/CD](#cicd)
13. [Data, licence and limits](#data-licence-and-limits)

## Results

- 159,794 building patches from xBD, 4 damage classes, scene-level 70/15/15 split (no leakage between near-duplicate crops of the same scene)
- Staged tuning: Stage A unfreeze depth → Stage B batch size and class imbalance → Stage C architecture (11 runs, tracked in MLflow)
- Held-out test set, ResNet-50 champion (n = 24,080, used once): **85.2% accuracy, 0.727 macro-F1**
- Served model: **ResNet-18**, validation macro-F1 0.6925 vs 0.6934 for ResNet-50 (a tie at about a third of the compute). ResNet-18 has no separate test-set score yet.
- Weakest class: minor-damage (F1 0.524), mostly confused with major-damage (21.8% / 17.7% cross-over)
- Real-world check on 53 Google Earth screenshots: ResNet-18 50 / 53 correct (28 / 28 damaged caught, 22 / 25 normal correct)

## How it fits together

| Layer | What runs | Where |
|---|---|---|
| Vision | xBD patches → CNN (ResNet-18 served) → label + `p_damaged` | `model/` (training), `data/serve_api.py` (serving) |
| Knowledge | BNPB InaRISK flood index, live web search, Gemini agent, recommender | `data/floodrisk.py`, `data/agent.py`, `data/recommender.py`, `data/content.py` |
| Serving | Streamlit UI, FastAPI, SQLite, Docker Compose, Prometheus + Grafana | `data/app.py`, `compose.yaml`, `monitoring/` |

| Path | Purpose |
|---|---|
| `compose.yaml` | `api` and `prometheus` on `backend`; `app` and `grafana` on `frontend` + `backend` |
| `.env.example` | Copy to `.env`: Gemini and Tavily keys, Gemini model, Grafana admin password |
| `data/` | Build context: Dockerfile, app code and tests |
| `model/` | Training, evaluation and the Google Earth test script (run from inside `model/`) |
| `monitoring/` | Prometheus scrape config, Grafana datasource and dashboard |
| `models/serving.pt` | You add this: the ResNet-18 checkpoint (step 1 below) |
| `storage/advisor.db` | SQLite, created on first start |

## Run the app

1. Copy the model: `mkdir -p models && cp /path/to/checkpoints/C1_resnet18/best.pt models/serving.pt` (checkpoints are not in this repo; the container loads the file directly because `mlflow.db` stores absolute paths from the training machine).
2. Keys: `cp .env.example .env`, then edit `.env`. Without `GOOGLE_API_KEY` the chat is off and everything else works.
3. Networks, once: `docker network create frontend` and `docker network create backend` (skip any that exist).
4. `docker compose up -d --build` (the first build downloads PyTorch, a few minutes).
5. Open the services:

| Service | Address | Notes |
|---|---|---|
| Streamlit app | http://localhost:8501 | Classify, rank, ask the advisor |
| FastAPI | http://localhost:8000/docs | `POST /classify`, `GET /health`, `/metrics` |
| Prometheus | http://localhost:9090 | No login: keep it on the lab network |
| Grafana | http://localhost:3000 | User `admin`, password from `.env` |

6. Stop: `docker compose down`. The database stays in `./storage`.

Without Docker (development):

```bash
pip install -r data/requirements.txt torch torchvision
cd data
CHECKPOINT=/path/to/C1_resnet18/best.pt uvicorn serve_api:app --port 8000 &
API_URL=http://localhost:8000 DB_PATH=../storage/advisor.db streamlit run app.py
```

## Monitoring

The API exposes Prometheus metrics at `GET /metrics`. Prometheus scrapes it every 15 s and keeps 15 days. Grafana opens on the provisioned dashboard **Property Advisor: damage API**.

| Metric | What it shows |
|---|---|
| `up{job="fp-advisor-api"}` | Prometheus can reach the API |
| `dd_model_loaded` | 1 when the classifier is loaded |
| `dd_predictions_total{label,damaged}` | Images classified, by 4-way label and by the damaged flag |
| `dd_p_damaged` (histogram) | Spread of `p_damaged`; a shift is a cheap drift signal |
| `dd_inference_seconds` (histogram) | Preprocess + forward-pass time |
| `dd_rejected_total{reason}` | Uploads refused (too large, not an image) |
| `http_requests_total`, `http_request_duration_seconds` | Traffic, status codes and latency per endpoint |

Quick check: `curl -s http://127.0.0.1:8000/metrics | grep ^dd_`. The API (8000) and Prometheus (9090) have no login: do not publish them through a public reverse proxy.

## Recommender

One recommender, two ways to use it (tab 2):

- **Ranked shortlist** (`recommender.py`): `0.40 condition + 0.35 budget fit + 0.25 location`, where condition is `1 - p_damaged` from the CNN. Add a free-text wish and a TF-IDF text match joins the score: `0.35 condition + 0.30 budget + 0.20 location + 0.15 text match`.
- **More like this** (`content.py`): content-based similarity. Each listing's description, type, city, district and bedrooms become a TF-IDF vector; similarity = `0.6 × TF-IDF cosine + 0.4 × closeness on price, land, building size and bedrooms`, with the shared keywords shown.

**Avoid flood-prone areas** (checkbox): flood safety (`1 - BNPB flood index`) takes 25% of the score: `0.75 × score above + 0.25 × flood safety`. Points outside BNPB's mapped area are scored neutral (0.5), not safe.

**Flood check for any area** (tab 2): type an area such as "Kelapa Gading, Jakarta Utara"; it is located with OpenStreetMap Nominatim and the BNPB index at that point is shown.

All weights are design choices, not fitted values. The 10 demo listings and their descriptions are synthetic; load real listings with `db.load_properties_csv` (required columns `title,city,price_idr`, add `description` for TF-IDF).

## Advisor chatbot

A LangChain tool-calling agent on Gemini (`GEMINI_MODEL`, default `gemini-3.5-flash-lite`) with five tools:

| Tool | What it does |
|---|---|
| `search_prices` | Live web search for Indonesian property prices (Tavily, DuckDuckGo fallback); every price is cited with its URL; sale and rent are kept apart |
| `get_condition` | The latest CNN result stored for a property |
| `get_flood_risk` | BNPB InaRISK flood-hazard index at the property's coordinates |
| `find_similar` | The content-based recommender: listings most like a given property |
| `flood_risk_for_area` | BNPB flood-hazard index for any named area in Indonesia (for example "Is Kelapa Gading flood-prone?") |

Flood index: the raw value (0 to 1) of BNPB's InaRISK flood-hazard layer ("Indeks Bahaya Banjir") at that point; higher means more flood-prone. BNPB computes the index from the likelihood and impact of flooding; this app only reads it, and no low/medium/high cut-offs are invented. It joins the ranking only when "Avoid flood-prone areas" is ticked. For named areas the value is read at the single point OpenStreetMap returns for that name, not averaged over the district. `no data` means the point is outside BNPB's mapped area, not zero risk. Demo properties use approximate district centres. Successful lookups are cached; failed lookups are retried on the next request.

## Train and evaluate the model

Run from `model/`. The raw xBD dataset is not in this repo; download it from the [xView2 challenge](https://xview2.org/).

```bash
cd model
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

| Step | Command | Check |
|---|---|---|
| 1. EDA | `python3 eda.py --data-root /path/to/xbd --output-dir eda_output` | Class distribution, per-disaster charts, pairing/corruption report |
| 2. Patches + split | `python3 make_patches.py --data-root /path/to/xbd --out-dir patches --workers 4` | `patches/split_report.txt`: similar class mix per split. Do not regenerate after training starts |
| 3. Speed test | `python3 train.py --manifest patches/manifest.csv --patch-root patches --model resnet18 --benchmark 30` | Minutes per epoch, to size the stages |
| 4. Stage A | `./run_experiments.sh A` | Unfreeze depth (ResNet-18, data subset) |
| 5. Stage B | `BEST_UNFREEZE=<x> BEST_LR=<x> ./run_experiments.sh B` | Batch size + imbalance handling |
| 6. Stage C | `BEST_UNFREEZE=<x> BEST_LR=<x> BEST_BS=<x> BEST_BALANCE=<x> BEST_LOSS=<x> ./run_experiments.sh C` | 3 architectures, full data |
| 7. Register | `python3 register_best.py` | Best validation macro-F1 → `property-dd-damage-cnn`, alias `champion` |
| 8. Test once | `python3 evaluate.py --checkpoint checkpoints/<best run>/best.pt --manifest patches/manifest.csv --patch-root patches --mlflow-run-id <run id>` | `eval_out/<run>/`: metrics, report, confusion matrix, per-disaster CSV |

Or run everything: `python3 run_pipeline.py --data-root /path/to/xbd --stage all --dry-run`, then without `--dry-run`. Read runs with `mlflow ui --backend-store-uri sqlite:///mlflow.db`. A crashed run resumes with the same `train.py` command plus `--resume`. Optional: `python3 promote_serving_alias.py` adds a `serving` alias for ResNet-18 in the registry (the deployed app loads the checkpoint file instead).

## Experiment log

| Stage | Run | Model | Unfreeze | LR | Batch | Balance | Val macro-F1 |
|---|---|---|---|---|---|---|---|
| A | A3_r18_full | resnet18 | full | 1e-4 | 64 | weights | 0.6456 |
| A | A2_r18_last | resnet18 | last | 1e-4 | 64 | weights | 0.6307 |
| A | A1_r18_head | resnet18 | head | 1e-3 | 64 | weights | 0.5290 |
| B | B3_sqrtw | resnet18 | last | 1e-4 | 64 | sqrt_weights | 0.6738 |
| B | B4_sampler | resnet18 | last | 1e-4 | 64 | sampler | 0.6312 |
| B | B1_bs32 | resnet18 | last | 1e-4 | 32 | weights | 0.6292 |
| B | B2_bs128 | resnet18 | last | 1e-4 | 128 | weights | 0.6249 |
| B | B5_focal | resnet18 | last | 1e-4 | 64 | weights (focal loss) | 0.6217 |
| C | C2_resnet50 ★ | resnet50 | last | 1e-4 | 64 | sqrt_weights | 0.6934 |
| C | C1_resnet18 (served) | resnet18 | last | 1e-4 | 64 | sqrt_weights | 0.6925 |
| C | C3_effb0 | efficientnet_b0 | last | 1e-4 | 64 | sqrt_weights | 0.6695 |

★ Registered as `property-dd-damage-cnn` v1, alias `champion` (MLflow run `5f8212a0eb574cbf834734feeaf4b22a`).

Test set, ResNet-50 (n = 24,080): accuracy 0.8515, macro-F1 0.7274.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| no-damage | 0.954 | 0.908 | 0.931 | 17,725 |
| minor-damage | 0.469 | 0.593 | 0.524 | 2,247 |
| major-damage | 0.626 | 0.648 | 0.637 | 2,124 |
| destroyed | 0.787 | 0.853 | 0.818 | 1,984 |

Reading it: no-damage and destroyed are visually distinct and score well; minor and major are the two rarest classes and sit next to each other on the severity scale, so they are mixed up most. Next levers: full unfreeze on the final architecture, sampler + sqrt-weights together, an ordinal loss, and more epochs or targeted augmentation at the minor/major boundary.

Glossary: **Unfreeze** = how much of the pretrained network retrains (head, last block, full). **Balance** = how the 73% no-damage imbalance was handled (class weights, square-root weights, oversampling sampler, focal loss). **Macro-F1** = F1 averaged equally over the 4 classes, so a model cannot win on the majority class alone. **Precision** = of the predictions for a class, how many were right; **recall** = of the real cases, how many were caught.

## Real-world check on Google Earth images

`model/test_maps_imagery.py` runs one or two checkpoints on your own top-down screenshots (one building per crop) and writes `predictions.csv` plus a review gallery. There is no ground truth, so it computes no accuracy; labels are set by eye.

```bash
python3 test_maps_imagery.py --images maps_input/dirty \
    --model checkpoints/C2_resnet50/best.pt checkpoints/C1_resnet18/best.pt --out-dir maps_review/raw54
```

Result on 53 screenshots (Cianjur, Lombok, Semeru, Plumpang, Palu), damaged = `p_damaged ≥ 0.5`:

| | ResNet-18 (served) | ResNet-50 |
|---|---|---|
| Damaged caught (28) | 28 / 28 | 24 / 28 |
| Normal correct (25) | 22 / 25 | 22 / 25 |
| Overall (53) | 50 / 53 | 46 / 53 |

The same images uploaded through the deployed app gave identical scores. Limits: crops under about 230 px are missed, and wide neighbourhood shots give confident wrong answers, so use one building per crop. Palu may overlap xBD's own training event. Screenshots are kept out of the repo (`maps_input/` is git-ignored).

## Background: the xBD paper

Gupta et al., "xBD: A Dataset for Assessing Building Damage from Satellite Imagery" (2019). [arXiv PDF](https://arxiv.org/pdf/1911.09296) · [CVPR Workshops version](https://openaccess.thecvf.com/content_CVPRW_2019/papers/cv4gc/Gupta_Creating_xBD_A_Dataset_for_Assessing_Building_Damage_from_Satellite_CVPRW_2019_paper.pdf)

- **What it is:** the dataset behind the xView2 challenge: 850,736 building polygons over 45,361.79 km² of Maxar/DigitalGlobe Open Data imagery (below 0.8 m ground sample distance), 22,068 images from 19 natural disasters (hurricanes, floods and monsoons, wildfires, volcanic eruptions, tsunamis, earthquakes, tornadoes).
- **How it was labelled:** building footprints were drawn on the pre-disaster image, overlaid on the post-disaster image, and each building was given a level on the Joint Damage Scale: no damage, minor damage, major damage, destroyed. The scale draws on FEMA's HAZUS, FEMA's damage assessment manual, the Kelman scale and EMS-98. Expert review found about 2–3% of labels wrong, and minor vs major damage is visually hard to separate.
- **Their baseline:** an altered U-Net finds the buildings (IoU 0.66 on buildings), then a ResNet-50 plus a small CNN with an ordinal loss grades the damage: weighted F1 0.2654 overall; per class no-damage 0.663, minor 0.144, major 0.009, destroyed 0.466.

How this project differs:

| | xBD paper | This project |
|---|---|---|
| Goal | Disaster response: map damage across a whole area | Property due diligence: screen one listing before a site visit |
| Input | Pre- and post-disaster image pair, full scene | One post-disaster crop of one building |
| Building finding | Learned (U-Net localisation) | Given: xBD polygons for training, the user's crop in the app |
| Classifier | ResNet-50 + small CNN, ordinal loss | Fine-tuned ResNet-18 / ResNet-50 / EfficientNet-B0, class-weighted loss, staged tuning |
| Output used | 4-class label | 4-class label + `p_damaged` (damaged vs not) for the recommender |
| Extra context | (none) | BNPB flood hazard, live prices, recommender, chatbot, monitoring |

The scores are not directly comparable: the paper's baseline also has to find the buildings and was evaluated on its own split, while this project classifies given building crops on a scene-level split of the 2,799 scenes used here.

## Tests

`cd data && python3 test_core.py && python3 test_content.py && python3 test_agent.py && python3 test_floodrisk.py` (database, recommender incl. flood-aware ranking, agent wiring incl. the area flood tool, flood and place lookup; no API keys or network needed). The tests are excluded from the Docker image by `.dockerignore`.

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`) runs on every push and pull request to `main`:

1. **Tests**: installs `data/requirements.txt` on Python 3.12 and runs the four test scripts above.
2. **Build and publish image**: builds the app image from `data/Dockerfile` once the tests pass. On a push to `main` it is published to GitHub Container Registry as `ghcr.io/brianhernawan/property-agent-advisor` (tags `latest` and the commit SHA). Pull requests only build.

The lab server is on a private network that GitHub's runners cannot reach, so the server is updated by hand: `git pull && docker compose up -d --build`.

## Data, licence and limits

- **xBD** (Gupta et al., 2019, "xBD: A Dataset for Assessing Building Damage from Satellite Imagery", [arXiv:1911.09296](https://arxiv.org/abs/1911.09296)) is distributed under CC BY-NC-SA 4.0. The dataset, the trained checkpoints and the MLflow database are not redistributed in this repo.
- **BNPB InaRISK** flood-hazard layer is queried live; nothing is stored beyond an in-memory cache.
- **OpenStreetMap Nominatim** turns area names into coordinates (© OpenStreetMap contributors); one request per new place, cached.
- **Google Earth** screenshots were used only for testing and are never committed.
- The classifier was trained on top-down satellite crops of single buildings. Street photos and wide neighbourhood views are out of scope.
- The 10 demo listings are synthetic. The ranking weights are design choices. Chat answers depend on live search results; the agent cites URLs and never guesses a price.
- Code: MIT licence (see `LICENSE`).
