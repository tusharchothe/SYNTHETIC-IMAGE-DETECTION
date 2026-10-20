<p align="center">
  <img src="assets/banner.png" alt="AI Image Detector" width="100%">
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.14-red.svg" alt="PyTorch"></a>
  <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-0.100+-green.svg" alt="FastAPI"></a>
  <img src="https://img.shields.io/badge/license-Research-lightgrey.svg" alt="License">
</p>

<p align="center">
  <b>Detect AI-generated images with two independent models, then let humans decide what becomes training data.</b>
</p>

**Contents:** [How it works](#how-it-works) · [Features](#features) · [Quick start](#quick-start) · [Using the UI](#using-the-ui) · [Architecture](#architecture) · [Models](#models) · [API](#api) · [Human-in-the-loop](#human-in-the-loop) · [Datasets](#datasets) · [Training](#training) · [Performance](#performance) · [Limitations](#limitations-and-security) · [Troubleshooting](#troubleshooting)

---

## How it works

<p align="center">
  <img src="assets/pipeline.png" alt="Pipeline: upload, validate, ResNet-50 and CLIP, compare, review queue" width="100%">
</p>

1. **Upload**: the image is validated, hashed (SHA-256 + perceptual hash) and de-duplicated.
2. **Score**: a fine-tuned **ResNet-50** and a **CLIP ViT-L/14 + linear probe** each return an AI probability.
3. **Triage**: disagreement or low confidence puts the image at the front of the review queue.
4. **Label**: a reviewer records the verified class. Labels are stored apart from predictions.
5. **Improve**: approved labels become versioned datasets, which feed retraining, guardrail evaluation and model promotion.

### Flowchart

<p align="center">
  <img src="assets/flowchart.png" alt="Flowchart from image upload to model promotion" width="100%">
</p>

<details>
<summary><b>Mermaid source (editable, renders natively on GitHub)</b></summary>

```mermaid
flowchart TD
    A([Upload image]) --> B{Valid image?}
    B -- No --> X1[Reject with error]
    B -- Yes --> C{Duplicate?}
    C -- Yes --> X2[Flag duplicate]
    C -- No --> D["Run ResNet-50 + CLIP"]
    D --> E{Models agree?}
    E -- Yes --> F([Show result])
    E -- No --> G[Add to review queue]
    G --> H[Human labels image]
    H --> I{Approved and eligible?}
    I -- No --> X3[Excluded from dataset]
    I -- Yes --> J[Build versioned dataset]
    J --> K[Retrain and evaluate]
    K --> L[Promote model]
```

</details>

---

## Features

| Area | What you get |
|------|--------------|
| **Detection** | Dual-model inference on upload, percentage scores with High / Medium / Low confidence, disagreement highlighting |
| **Human review** | Pending queue with previews, structured labels (`real` / `ai` / `uncertain`), generator attribution, multi-reviewer support |
| **Datasets** | Versioned manifests and splits, automatic exclusion of duplicates, uncertain labels and missing files, class and generator balance reports |
| **Model lifecycle** | Early-stopping training, evaluation guardrails, model registry with promotion and rollback |
| **Analytics** | Prediction statistics, model agreement rate, side-by-side ResNet-50 vs CLIP breakdown |

> **Planned:** group-aware splitting (keeping images from the same prompt or session in one split). It needs prompt metadata the app does not collect yet.

---

## Quick start

```bash
git clone <repository-url> && cd version1

python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env                # defaults work for a local setup

# Model files must be in the project root
ls resnet50_sd14_best.pt clip_probe.joblib

python -m v1.simple_app
```

| Service | URL |
|---------|-----|
| API | http://localhost:8100 |
| Web UI | http://localhost:7900 |
| Swagger docs | http://localhost:8100/docs |

Databases and upload folders are created on first run. Verify with `curl http://localhost:8100/health`.

**Requirements:** Python 3.11+, 4 GB RAM (8 GB+ recommended). A CUDA GPU is optional but speeds up CLIP.

<details>
<summary><b>Dependencies and install problems</b></summary>

Main packages: `torch==2.14.1`, `torchvision==0.29.1`, `open_clip_torch>=2.20.0`, `fastapi`, `uvicorn`, `gradio>=4.0.0`, `sqlalchemy>=2.0.0`, `scikit-learn`, `joblib`, `Pillow`, `numpy`. `mlflow` is optional.

```bash
# CPU-only PyTorch (smaller download)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# open_clip fails to install
pip install --upgrade pip setuptools wheel && pip install open_clip_torch --no-cache-dir

# Gradio version conflicts
pip install gradio==4.44.0
```

Run the API and UI separately:

```bash
python -m uvicorn v1.simple_app.api:app --host localhost --port 8100
python -c "from v1.simple_app.ui import launch; launch(api_base='http://localhost:8100')"
```

</details>

### Configuration

Set these in `.env`:

| Variable | Default | Purpose |
|----------|---------|---------|
| `SIMPLE_HOST` / `SIMPLE_PORT` | `localhost` / `8100` | API address |
| `UI_HOST` / `UI_PORT` | `localhost` / `7900` | Web UI address |
| `SIMPLE_DATABASE_URL` | `sqlite:///./data/simple_app.db` | Simple workflow database |
| `AIDETECT_DATABASE_URL` | `sqlite:///./ai_detector.db` | Main storage database |
| `AIDETECT_UPLOAD_DIR` | `./data/uploads` | Uploaded image files |
| `AIDETECT_MAX_IMAGE_SIZE_BYTES` | `8388608` (8 MB) | Upload size limit |
| `AIDETECT_MODEL_PATH` | `./resnet50_sd14_best.pt` | ResNet-50 checkpoint |
| `AIDETECT_CLIP_MODEL_PATH` | `./clip_probe.joblib` | CLIP probe |
| `AIDETECT_DATASET_ROOT` | `./datasets` | Versioned datasets |
| `AIDETECT_TRAINING_DIR` | `./artifacts/training` | Training outputs |
| `AIDETECT_MODEL_REGISTRY_DIR` | `./artifacts/registry` | Model registry |

For multi-user setups, `AIDETECT_ALLOWED_REVIEWERS`, `AIDETECT_REVIEWER_TOKEN`, `AIDETECT_ALLOWED_ADMINS` and `AIDETECT_ADMIN_TOKEN` are available. The `simple_app` workflow ships with authentication **disabled**, so add proper auth middleware before exposing it.

---

## Using the UI

<p align="center">
  <img src="assets/ui-upload.png" alt="Mockup of the Upload and Detect tab" width="100%">
</p>
<!-- The image above is an illustrative mockup. Replace assets/ui-upload.png with a real screenshot when available. -->

| Tab | Purpose |
|-----|---------|
| **Upload and Detect** | Drop PNG, JPEG or WebP files (8 MB max each). Results show status, both model scores and warnings. |
| **Dashboard** | Totals, real / AI split and average confidence per model, plus the agreement rate between models. |
| **Pending Review** | Paginated queue with image preview. Statuses: ⏳ pending, ✓ approved, ✗ rejected. |
| **Review and Label** | Load an image by ID, pick `real` / `ai` / `uncertain`, set reviewer confidence (0 to 1), and add generator, evidence source and notes. |
| **Datasets** | Build and list versioned datasets. Options: dry run, require two reviews. |
| **Training and Models** | Start training (`resnet50`), set epochs, batch size and learning rate, and browse past runs. |

**Reading the scores**

| Score | Level | Meaning |
|-------|-------|---------|
| ≥ 80% | High | Strong prediction |
| 60 to 79% | Medium | Moderate certainty |
| < 60% | Low | Uncertain, review recommended |

Scores from the two models are **not directly comparable**: they differ in architecture, training and calibration, and both use uncalibrated outputs. When ResNet-50 says `real` and CLIP says `ai` (or the reverse), treat the image as a review candidate.

**Review tips:** use `1.0` confidence only for definitive cases, use `uncertain` for ambiguous images, and record reasoning in the notes for the audit trail.

---

## Architecture

<p align="center">
  <img src="assets/architecture.png" alt="System architecture" width="100%">
</p>

| Component | Technology | Role |
|-----------|-----------|------|
| Backend | FastAPI | REST API, validation, routing |
| UI | Gradio 4.0+ | Six-tab web interface |
| Database | SQLite + SQLAlchemy | Image metadata, predictions, human reviews |
| Storage | File system | Original images with UUID-prefixed safe names |
| Models | PyTorch 2.14, open_clip | ResNet-50 (trainable), CLIP ViT-L/14 (frozen) |
| Validation | Pillow, hashlib | Format check, SHA-256, perceptual hash |

### Lifecycle

<p align="center">
  <img src="assets/lifecycle.png" alt="Human-in-the-loop lifecycle" width="80%">
</p>

---

## Models

| | ResNet-50 | CLIP + linear probe |
|---|-----------|---------------------|
| **Role** | Primary, trainable detector | Comparison baseline |
| **Backbone** | ImageNet (`IMAGENET1K_V2`), head 2048 → 1 + sigmoid | OpenAI ViT-L/14, frozen, 768-d embedding |
| **Classifier** | End-to-end fine-tuned | scikit-learn logistic regression |
| **Input** | `fit_crop` 256 × 256, ImageNet normalisation | 224 × 224, CLIP preprocessing |
| **Output** | Score in [0, 1]; `ai` if ≥ 0.5 | Probability in [0, 1]; `ai` if ≥ 0.5 |
| **Version tag** | `resnet50_sd14_best.pt` | `ViT-L-14/openai` |
| **Speed (CPU)** | ~50 ms / image | ~200 ms / image |
| **GPU** | Optional | Recommended for production |

<details>
<summary><b>Preprocessing and training augmentation</b></summary>

**ResNet-50 `fit_crop`:** resize the shortest edge to 256 px, centre-crop to 256 × 256, convert to RGB tensor, normalise with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.

**Augmentation when retraining:** random horizontal flip (p = 0.5), random vertical flip (p = 0.05), standard ImageNet normalisation.

**CLIP:** uses the `open_clip` built-in pipeline. Only the probe is trained, offline; retraining CLIP itself is not supported by this pipeline.

</details>

---

## API

Base URL: `http://localhost:8100`. Interactive docs with try-it-out are at `/docs`.

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/health` | Service status |
| POST | `/upload` | Upload images and run inference |
| GET | `/queue?page=1&limit=20` | Pending review queue (limit max 100) |
| GET | `/images/{id}/file` | Serve the image file |
| GET | `/images/{id}/predictions` | Prediction history |
| POST | `/reviews/submit` | Add a human review |
| GET | `/datasets` | List dataset versions |
| POST | `/datasets/build` | Create a dataset version |
| GET | `/training/runs` | Training history |
| POST | `/training/run` | Start a training job |

**Upload**

```bash
curl -X POST http://localhost:8100/upload -F "files=@image1.png" -F "files=@image2.jpg"
```

```json
{
  "results": [{
    "filename": "image1.png",
    "status": "ok",
    "image_id": "a1b2c3d4-...",
    "predictions": [
      {"model_name": "resnet50", "predicted_class": "ai", "raw_score": 0.8742, "threshold": 0.5},
      {"model_name": "clip",     "predicted_class": "ai", "raw_score": 0.6234, "threshold": 0.5}
    ],
    "warnings": []
  }]
}
```

**Submit a review**: `image_id`, `verified_class` (`real` / `ai` / `uncertain`) and `confidence` (0.0 to 1.0) are required; `known_generator`, `evidence_source` and `source_notes` are optional.

```bash
curl -X POST http://localhost:8100/reviews/submit -H "Content-Type: application/json" -d '{
  "image_id": "a1b2c3d4-...",
  "verified_class": "ai",
  "confidence": 1.0,
  "known_generator": "Midjourney",
  "source_notes": "Clear artifacts in background"
}'
```

**Build a dataset and train**

```bash
curl -X POST http://localhost:8100/datasets/build -H "Content-Type: application/json" \
  -d '{"version": "v0002", "dry_run": false, "require_second_review": true}'

curl -X POST http://localhost:8100/training/run -H "Content-Type: application/json" \
  -d '{"dataset_version": "v0002", "model_name": "resnet50", "epochs": 10, "batch_size": 32, "learning_rate": 0.0001, "dry_run": false}'
```

---

## Human-in-the-loop

```mermaid
sequenceDiagram
    participant U as User
    participant UI as Gradio UI
    participant API as FastAPI
    participant M as Models
    participant DB as SQLite + files

    U->>UI: Upload image
    UI->>API: POST /upload
    API->>API: Validate, hash, dedup
    API->>M: ResNet-50 + CLIP inference
    M-->>API: Scores
    API->>DB: Store image and predictions
    API-->>UI: Results and warnings

    U->>UI: Open review queue
    UI->>API: GET /queue
    U->>UI: Submit label
    UI->>API: POST /reviews/submit
    API->>DB: Store review (separate table)

    U->>UI: Build dataset
    UI->>API: POST /datasets/build
    API->>DB: Query approved reviews
    API->>DB: Write manifest, splits, report
    API-->>UI: Dataset report
```

### Data separation

| Data | Table | Written by |
|------|-------|-----------|
| Image records | `image_records` | Every upload |
| Model predictions | `predictions` | Every upload |
| Verified labels | `human_reviews` | Reviewers only |

Labels never overwrite predictions, so you keep an audit trail, clean label provenance, and the ability to measure model agreement with ground truth over time.

### Dataset eligibility

```mermaid
flowchart LR
    A[Image] --> B{Approved human review?}
    B -- no --> X1[Excluded: unreviewed]
    B -- yes --> C{Class real or ai?}
    C -- uncertain --> X2[Excluded: uncertain_label]
    C -- yes --> D{File on disk?}
    D -- no --> X3[Excluded: missing_file]
    D -- yes --> E{Duplicate by SHA-256 or pHash?}
    E -- yes --> X4[Excluded: duplicate]
    E -- no --> F{Second review required?}
    F -- missing --> X5[Excluded: needs_second_review]
    F -- conflict --> X6[Excluded: disputed_label]
    F -- ok --> G[Included in dataset]
```

A rejected or otherwise ineligible review is excluded as `rejected_or_ineligible`.

---

## Datasets

Each version is immutable:

```
datasets/v0001/
├── manifest.jsonl          # one JSON object per image
├── split_definitions.json  # image IDs per split
└── dataset_report.json     # balance, duplicates, exclusions
```

```json
{"image_id": "a1b2...", "path": "data/uploads/image_a1b2.png", "label": 1, "split": "train", "generator": "stable-diffusion", "phash": "0111...", "sha256": "e396..."}
```

- `label`: `0` = real, `1` = AI
- `split`: `train`, `val`, `test`, `calibration` or `benchmark_test`
- `dataset_report.json`: `eligible_total`, `class_balance`, `generator_balance`, `split_counts`, `split_ratios`, `duplicate_counts`, `excluded_counts`

| Check | Method | Catches |
|-------|--------|---------|
| Exact duplicates | SHA-256 | Identical re-uploads |
| Near duplicates | Perceptual hash, Hamming distance < 10 | Crops, re-encodes, small edits |

The first upload is kept; later copies are marked `duplicate_of_id` and excluded.

---

## Training

```bash
curl -X POST http://localhost:8100/training/run -H "Content-Type: application/json" -d '{"dataset_version": "v0001", "model_name": "resnet50"}'
```

Or use **Training and Models** in the UI.

| Parameter | Default | Range |
|-----------|---------|-------|
| `dataset_version` | required | Must exist in `datasets/` |
| `model_name` | required | `resnet50` (CLIP is not trainable here) |
| `epochs` | 5 | 1 to 100 |
| `batch_size` | 32 | 1 to 256 |
| `learning_rate` | 0.0001 | 1e-6 to 1.0 |
| `early_stopping_patience` | 3 | 1 to 10 |
| `monitor_metric` | `val_auc` | `val_auc`, `val_acc`, `val_loss` |
| `dry_run` | false | Validate the config without training |

**Process:** validate the dataset (train and val splits, at least 2 examples per class in train) → load pretrained ResNet-50 and replace the head (2048 → 1) → Adam with weight decay and BCE loss → validate each epoch (AUC, accuracy, FPR) → save the best checkpoint and stop after `patience` epochs without improvement.

**Output:** `artifacts/training/train-resnet50-v0001-{timestamp}.pt`, a `.json` metadata file with the same name, and the training history. A failed run reports `status: "failed"` with `last_error`.

### Promotion guardrails

| Metric | Threshold |
|--------|-----------|
| ROC-AUC | ≥ 0.90 |
| False positive rate on real images | ≤ 5% |
| TPR at 1% FPR | ≥ 70% |
| Evaluation samples | ≥ 100 |

```mermaid
flowchart LR
    C[Candidate checkpoint] --> G{All guardrails pass?}
    G -- no --> R[Rejected]
    G -- yes --> A{Admin confirms?}
    A -- no --> R
    A -- yes --> B[Back up current model]
    B --> P[Promote to active]
    P -. problem found .-> RB[Rollback to backup]
```

---

## Project structure

```
.
├── resnet50_sd14_best.pt        # ResNet-50 checkpoint
├── clip_probe.joblib            # CLIP linear probe
├── .env.example · requirements.txt · pyproject.toml · view_db.py
├── data/                        # simple_app.db + uploads/
├── datasets/                    # versioned datasets (v0001, ...)
├── artifacts/                   # training outputs and model registry
├── tests/                       # pytest suite
├── doc/                         # reference paper
└── src/v1/
    ├── models.py                # ResNet-50 architecture
    ├── inference.py             # prediction logic
    ├── dataset_builder.py       # versioned dataset creation
    ├── training_pipeline.py     # training orchestration
    ├── evaluation_pipeline.py   # guardrail evaluation
    ├── model_registry.py        # promotion and rollback
    ├── services/inference_service.py   # ResNet + CLIP inference
    ├── storage/                 # SQLAlchemy engine, ORM models, storage service
    └── simple_app/              # api.py · ui.py · service.py · config.py
```

`src/v1/api.py` and `src/v1/training.py` are legacy and unused by the current workflow.

---

## Testing

```bash
pytest -q                                          # all tests
pytest tests/test_simple_workflow.py -v            # one module
pytest --cov=src/v1 --cov-report=html              # coverage report
```

| Module | Covers |
|--------|--------|
| `test_simple_workflow.py` | Upload → predict → queue, validation, dedup |
| `test_dataset_builder.py` | Exclusions, deduplication, splits |
| `test_storage_layer.py` | Records, hash uniqueness, invalid files |
| `test_review_dashboard_api.py` | Queue, label submission, approval status |
| `test_model_promotion.py` | Guardrails, promotion, rollback |
| `test_training_pipeline.py` | Config validation, dataset requirements |
| `test_api_upload_and_history.py` | Upload and history API contracts |
| `test_dashboard_analytics.py` | Metrics and filtering |

Tests use isolated temporary databases and upload folders, and mock model inference so no checkpoints are loaded.

---

## Performance

### Detection quality

ResNet-50 fine-tuned on Stable Diffusion 1.4, evaluated on an SD 1.4 validation set:

| Accuracy | ROC-AUC | TPR @ 1% FPR | Real-image FPR |
|----------|---------|--------------|----------------|
| 99.8% | 0.9997 | 98.7% | 0.3% |

> **Read these numbers with care.** They come from a single generator and may reflect shortcut learning (resolution or compression differences) rather than real generalisation. Cross-generator results for Midjourney, DALL-E and Stable Diffusion 2.x have **not** been measured, and CLIP + probe has no systematic evaluation in this repository.

### Inference speed

Intel i7-10700K, 32 GB RAM; file I/O and preprocessing excluded.

| Model | Device | Batch 1 | Batch 32 |
|-------|--------|---------|----------|
| ResNet-50 | CPU | ~50 ms | ~800 ms |
| CLIP | CPU | ~200 ms | ~2000 ms |
| ResNet-50 | CUDA | ~8 ms | n/a |
| CLIP | CUDA | ~25 ms | n/a |

---

## Limitations and security

**Model limits**

- **Not proof.** Outputs are probabilistic estimates. False positives can harm photographers and artists, so keep a human in the loop for consequential decisions.
- **False positives** come from heavy post-processing, CG or 3D-looking scenes, unusual lighting and compression artifacts.
- **False negatives** come from JPEG compression, blur or scaling, unseen generators, and AI-edited real photos.
- **Generalisation.** A detector trained on one generator may miss others. Collect diverse data, tag the generator in reviews, and use CLIP as a second opinion.
- **Adversarial robustness** is not addressed.

**Dataset shortcut checklist**

- [ ] Test generators are disjoint from train generators
- [ ] No duplicate SHA-256 hashes across splits
- [ ] Resolution and file-format distributions are balanced across classes
- [ ] Results hold after degradations (JPEG 90, blur, resize)
- [ ] Per-generator false positive rates are monitored

**Security**

| Protection | Implementation |
|------------|----------------|
| File validation | Pillow format check, 8 MB limit (configurable) |
| Path safety | UUID-prefixed safe filenames |
| Duplicate handling | SHA-256 and perceptual hash |
| SQL injection | SQLAlchemy parameterised queries |
| SSRF | No external URL fetching |

**Not implemented:** authentication and authorisation (local-only design), rate limiting, sanitisation of review notes (XSS risk if rendered in a web context), and encrypted storage.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `FileNotFoundError: ResNet checkpoint not found` | Check `ls -lh resnet50_sd14_best.pt clip_probe.joblib`, or point `AIDETECT_MODEL_PATH` and `AIDETECT_CLIP_MODEL_PATH` in `.env` at the real files. |
| Cannot reach `http://localhost:8100` | Run `curl localhost:8100/health`. Find a port conflict with `lsof -i :8100`, free it with `kill $(lsof -t -i:8100)`, then restart. |
| `CUDA out of memory` | Lower `batch_size` (for example 16), or run on CPU with `export CUDA_VISIBLE_DEVICES=""`. |
| `psutil is required for memory guardrails` | `pip install psutil` |
| `sqlite3.OperationalError: database is locked` | Several processes share one SQLite file. Run `pkill -f "python -m v1"` and start a single instance. |
| `Address already in use` (Gradio) | Set `UI_PORT=7901` in `.env`, or `kill $(lsof -t -i:7900)`. |

**Debugging:** `export AIDETECT_DATABASE_ECHO=true FASTAPI_LOG_LEVEL=DEBUG`. To inspect data, run `python view_db.py [path/to.db]`.

**Reset (deletes everything):**

```bash
rm -f ai_detector.db data/simple_app.db && rm -rf data/uploads/* datasets/v* artifacts/training/*
```

---

## License

Released under the **Research License** (see `LICENSE`): academic and research use, modification and distribution are allowed; commercial use needs permission.
