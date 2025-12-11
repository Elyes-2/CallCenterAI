# CallCenterAI
CallCenterAI — Intelligent Ticket classification and MLOps demo

CallCenterAI is an experimental project that demonstrates an end-to-end pipeline
for classifying customer support tickets using both TF-IDF and transformer-based
models, together with lightweight MLOps practices (model tracking, artifacts,
and simple APIs). The repository includes training scripts, REST APIs, Docker
configs and example data to reproduce experiments locally.

**Quick links**
- **Code:** `src/` — training scripts and API endpoints
- **Data:** `data/` — datasets used for training (CSV + DVC metadata)
- **Models:** `models/` — saved model artifacts
- **Experiments:** `mlruns/` — MLflow run artifacts
- **Docker:** `docker/` and `docker-compose.yaml` — service images and compose

**Highlights**
- Train TF-IDF and transformer models for ticket classification
- Serve models via simple FastAPI endpoints (`agent_api.py`, `tfidf_api.py`, `transformer_api.py`)
- Track runs and artifacts using MLflow (local `mlruns/`)
- Example Dockerfiles to build isolated runtime images for experiments

---

## Repository layout

- `src/` — Python source: training scripts and API servers
- `data/` — input datasets and DVC metadata
- `models/` — exported models (e.g. `enhanced_multilingual_model/`)
- `mlruns/` — MLflow experiment runs and artifacts
- `docker/` — Dockerfiles for different service images
- `tests/` — unit and integration tests
- `requirements.txt` — runtime dependencies

---

## Quickstart (Windows PowerShell)

1. Create and activate a virtual environment

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies

```
pip install -r requirements.txt
```

3. Run tests

```
pytest -q
```

4. Run an API server locally (examples)

- TF-IDF API (default port 8002)

```
python -m uvicorn src.tfidf_api:app --reload --port 8002
```

- Transformer API (default port 8003)

```
python -m uvicorn src.transformer_api:app --reload --port 8003
```

- Agent API (combines or proxies models)

```
python -m uvicorn src.agent_api:app --reload --port 8001
```

5. (Optional) Use Docker Compose to build and run service images

```
docker-compose up --build
```

---

## Training

- Lightweight training scripts live in `src/`. Example commands:

```
python src/train.py                # older / simple training flow
python src/train_transformer.py    # transformer training helpers
python train_and_log.py            # example training that logs to MLflow
```

Check the script docstrings and `src/` files for available arguments and
configuration options.

## Models & artifacts

- Trained models are stored in `models/` (example: `enhanced_multilingual_model/`).
- MLflow run artifacts are stored under `mlruns/` — run `mlflow ui` to inspect
	experiments locally:

```
mlflow ui --backend-store-uri ./mlruns
```

---

## Development notes

- Code style and tests: run `pytest` to validate changes.
- When changing training/data code, keep experiment reproducibility in mind
	(seed values, data splits, logging of hyperparameters).
- Large models (safetensors) are excluded from version control; use `models/`
	directory to place local artifacts.

---

## Troubleshooting

- If an import fails when running module-style uvicorn commands, run the
	script directly (e.g. `python src/tfidf_api.py`) or adjust `PYTHONPATH` so
	the `src` package is importable.
- Ensure the virtual environment is activated and `requirements.txt` has been
	installed.

---

## Contributing

If you'd like to contribute, open an issue or submit a pull request. Small,
focused PRs that include tests are easiest to review.

---

## License & Contact

This repository does not include an explicit license file. Contact the
maintainer for reuse or collaboration: Elyes (repository owner).

---

## Detailed Overview

This repository is intended as a compact reference for building and operating
an experimental ticket classification system with a small MLOps workflow.

- Goals:
	- Provide reproducible training flows for both classical (TF-IDF + classifier)
		and transformer-based models.
	- Demonstrate how to track experiments and artifacts using MLflow.
	- Offer minimal, container-friendly APIs to serve models for inference.

## Architecture

- Data ingestion: CSVs live in `data/` and are versioned with DVC metadata where
	available.
- Training: scripts under `src/` produce model artifacts and log metrics to
	MLflow (local `mlruns/`).
- Serving: small FastAPI apps (`src/*.py`) expose HTTP endpoints for
	inference; Dockerfiles in `docker/` package those servers for deployment.

## Detailed Setup

- Prerequisites:
	- Python 3.9+ (3.10+ recommended)
	- `pip`, `virtualenv` or `venv`
	- Optional: Docker & docker-compose for container runs

- Create and activate the venv (PowerShell example):

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

- Install dependencies:

```
pip install -r requirements.txt
```

## Data preparation

- The `data/` folder contains `all_tickets_processed_improved_v3.csv`. If you
	update or replace datasets, keep a copy locally and update any DVC configs
	(if used) to point to external storage.
- Basic preprocessing is implemented in the training scripts — inspect
	`src/train.py` and `src/train_transformer.py` for tokenization and split
	logic.

## Training examples

Run a basic training run (examples — check script options):

```
python src/train.py --data data/all_tickets_processed_improved_v3.csv --output models/tfidf_model

python src/train_transformer.py --data data/all_tickets_processed_improved_v3.csv --model-out models/enhanced_multilingual_model

python train_and_log.py   # example script that demonstrates MLflow logging
```

- Recommended practice: run training through MLflow so hyperparameters,
	metrics and artifacts are recorded. Example (after adjusting scripts to
	accept CLI args or config files):

```
mlflow run . -P data=data/all_tickets_processed_improved_v3.csv
```

## Evaluation and metrics

- Evaluation code may be present inline in training scripts or in tests. Look
	for classification metrics such as precision, recall, F1 and confusion
	matrices in the MLflow run outputs under `mlruns/`.

## Serving & API examples

- The repo contains simple FastAPI apps in `src/`:
	- `src.tfidf_api` — serve TF-IDF + classifier model
	- `src.transformer_api` — serve the transformer-based model
	- `src.agent_api` — a higher-level endpoint that can use available models

- Example `curl` for a predict endpoint (adjust path/port to match service):

```
curl -X POST "http://localhost:8002/predict" -H "Content-Type: application/json" -d '{"text":"My internet is down since last night"}'

# Expected JSON (example):
# {"label": "network_issue", "score": 0.92}
```

If the endpoints in `src/` use a different route or request schema, adapt the
JSON body accordingly. Inspect `src/*.py` for the exact request/response shape.

## Docker / Deployment

- Build an image for a service (example):

```
docker build -f docker/Dockerfile.transformer -t callcenter/transformer:local .
```

- Start services with `docker-compose`:

```
docker-compose up --build
```

Environment variables and ports are configured in `docker-compose.yaml` —
adjust to your environment when deploying.

## MLflow

- To inspect experiments locally:

```
mlflow ui --backend-store-uri ./mlruns
```

- When running training scripts, ensure they call MLflow APIs (check
	`train_and_log.py`) or wrap runs with `mlflow.start_run()` to capture
	parameters, metrics and artifacts.

## Testing & CI

- Run the test suite locally:

```
pytest -q
```

- Tests are under `tests/` and include unit and integration-style tests that
	exercise the APIs and training scripts. Use them as examples for writing new
	tests when you modify core logic.

## Code organization (file map)

- `src/train.py` — simple training pipeline (TF-IDF + classifier)
- `src/train_transformer.py` — transformer training helpers and orchestration
- `train_and_log.py` — example training + MLflow logging
- `src/tfidf_api.py`, `src/transformer_api.py`, `src/agent_api.py` — FastAPI
	servers for inference
- `docker/` — Dockerfiles for packaging model servers
- `mlruns/` — local MLflow experiment store (generated at runtime)

## Contribution & Development workflow

- Preferred workflow:
	1. Create a branch for your change.
	2. Add or update tests in `tests/` that cover your change.
	3. Run `pytest` locally and fix failures.
	4. Open a pull request describing the change and the testing performed.

- When adding or updating models, avoid committing large binary artifacts to
	Git — prefer storing them in `models/` locally and reference them in DVC
	or artifact storage if using CI/CD.

## Troubleshooting & FAQs

- Problem: `ImportError` when running `python -m uvicorn src.tfidf_api:app`.
	- Solution: run `python src/tfidf_api.py` or ensure the repository root is in
		`PYTHONPATH` (or install the package in editable mode `pip install -e .`).
- Problem: MLflow UI shows no runs.
	- Solution: check that scripts call MLflow APIs and that `mlruns/` is
		writable and present in the repository root.

---

If you want, I can also:
- Add `CONTRIBUTING.md` and `LICENSE` files,
- Add concrete curl examples using the exact request/response models from the
	code, or
- Run the test suite now and report results.

