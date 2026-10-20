# Simple App Complete Workflow Guide

## Overview

The simple app provides a **complete human-in-the-loop workflow** with no authentication required. It's designed for local development and experimentation.

**Complete Workflow:**
1. **Upload** → Images are validated, stored, and automatically processed by ResNet-50 and CLIP
2. **Review** → Browse pending queue and submit human labels
3. **Dataset** → Build versioned training datasets from reviewed images
4. **Train** → Run explicit model training on dataset versions
5. **Deploy** → Promote trained models (via model registry in main app)

## Architecture

```
Simple App Components:
├── API (FastAPI)        → Endpoints for all operations
├── UI (Gradio)          → 5-tab interface for complete workflow
├── Service Layer        → Business logic for upload, review, dataset, training
└── Storage (SQLite)     → Persistent queue and metadata
```

## Requirements

All requirements met by your existing setup:
- Python 3.11+
- Dependencies in `requirements.txt`
- Virtual environment at `venv/`

## Installation

```bash
cd /home/tushar/Desktop/DSBD_project/version1

# Activate virtual environment
source venv/bin/activate

# Install/update dependencies
pip install -r requirements.txt

# Optional: Install project in editable mode
pip install -e .
```

## Running the Application

### Option 1: Both API + UI (Recommended)

Start everything with one command:

```bash
cd /home/tushar/Desktop/DSBD_project/version1
source venv/bin/activate
python -m v1.simple_app
```

This launches:
- **API** at http://localhost:8100 (docs at http://localhost:8100/docs)
- **UI** at http://localhost:7900

### Option 2: API Only

```bash
python -m v1.simple_app --api-only
```

### Option 3: UI Only

```bash
# Assumes API is already running
python -m v1.simple_app --ui-only
```

### Custom Ports

```bash
# Via command line
python -m v1.simple_app --api-port 8200 --ui-port 7901

# Or via environment variables
SIMPLE_API_PORT=8200 SIMPLE_UI_PORT=7901 python -m v1.simple_app
```

## User Interface Tabs

### Tab 1: Upload & Detect

**Purpose:** Upload images and run automatic inference

**Features:**
- Multi-file upload (PNG, JPEG, WebP)
- Automatic ResNet-50 and CLIP detection
- Duplicate detection by SHA-256 and perceptual hash
- Per-file results table with predictions

**Workflow:**
1. Select one or more images
2. Click "Upload & Detect"
3. View results: image ID, predictions, scores, warnings

### Tab 2: Pending Review Queue

**Purpose:** Browse images awaiting human review

**Features:**
- Paginated queue of pending images
- Model predictions displayed alongside each image
- Image preview by selecting from dropdown
- Refresh to see new uploads

**Workflow:**
1. Click "Refresh queue" to load pending images
2. View predictions from both models
3. Select an image ID to preview
4. Copy image ID for review submission

### Tab 3: Review & Label

**Purpose:** Submit human reviews and labels

**Features:**
- Load image by ID for review
- Submit labels: `real`, `ai`, or `uncertain`
- Optional metadata:
  - Known generator (e.g., "Midjourney", "Stable Diffusion")
  - Evidence source (e.g., original URL)
  - Free-text notes
- Confidence slider (0.0 - 1.0)

**Workflow:**
1. Paste image ID from queue
2. Click "Load Image" to preview
3. Select label and confidence
4. Optionally add generator, evidence, notes
5. Click "Submit Review"

**Label Guidelines:**
- `real`: Verified natural/real image
- `ai`: Verified AI-generated
- `uncertain`: Unsure or requires additional review

**Important:** Human labels are stored **separately** from model predictions to maintain audit trails and prevent data leakage.

### Tab 4: Dataset Management

**Purpose:** Build versioned datasets from reviewed images

**Features:**
- List existing dataset versions
- Build new dataset versions
- Dry run mode for validation
- Require second review option
- Automatic train/val/calibration/test splits
- Deduplication and exclusion reporting

**Workflow:**
1. Click "Refresh Dataset List" to see existing versions
2. Enter version ID (e.g., `v0001`)
3. Optionally enable:
   - "Dry run" to validate without writing files
   - "Require 2+ reviews" for stricter quality
4. Click "Build New Dataset"
5. Review the manifest count and report

**What Gets Included:**
- ✓ Approved reviews with binary labels (`real` or `ai`)
- ✓ Non-duplicate images (SHA-256 + perceptual hash)
- ✓ Files that exist on disk

**What Gets Excluded:**
- ✗ Uncertain labels
- ✗ Rejected or disputed records
- ✗ Missing files
- ✗ Duplicates
- ✗ Records without approved reviews

**Dataset Structure:**
```
datasets/v0001/
├── manifest.jsonl          # Complete dataset manifest
├── split_definitions.json  # Train/val/calibration/test splits
└── dataset_report.json     # Quality metrics and exclusions
```

### Tab 5: Training & Models

**Purpose:** Run explicit model training

**Features:**
- Start training runs on dataset versions
- List recent training runs and status
- Dry run mode for validation
- Configurable hyperparameters:
  - Epochs (1-100)
  - Batch size (1-256)
  - Learning rate (0.000001-1.0)
- Support for ResNet-50 and CLIP

**Workflow:**
1. Click "Refresh Training Runs" to see existing runs
2. Enter dataset version (e.g., `v0001`)
3. Select model: `resnet50` or `clip`
4. Optionally enable "Dry run" for validation
5. Configure hyperparameters (defaults are sensible)
6. Click "Start Training"
7. Wait for completion (may take several minutes)
8. Review job ID and artifact path

**Important:**
- Training runs in the current process (blocking)
- For production use, consider background workers
- Training requires explicit action (no automatic retraining)
- Artifacts are saved to `artifacts/training/`

**Training Validation:**
- Dataset version must exist
- Must have train/val/test splits
- Must contain both `real` and `ai` labels
- Files must exist on disk
- No benchmark test IDs in training split

## API Endpoints

All endpoints available at http://localhost:8100/docs (interactive docs)

### Images

- `POST /upload` — Upload and process images
- `GET /queue` — Paginated pending review queue
- `GET /images/{image_id}/file` — Serve image file
- `GET /images/{image_id}/predictions` — Prediction history

### Reviews

- `POST /reviews/submit` — Submit human review
  - Query params: `image_id`, `verified_class`, `confidence`, etc.
  - Returns: review ID and status

### Datasets

- `POST /datasets/build` — Build versioned dataset
  - Query params: `version`, `dry_run`, `require_second_review`
  - Returns: manifest count and report
- `GET /datasets` — List available dataset versions

### Training

- `POST /training/run` — Start training run
  - Query params: `dataset_version`, `model_name`, `epochs`, etc.
  - Returns: job ID and artifact path
- `GET /training/runs` — List training runs with status

### Health

- `GET /health` — Service status and configuration

## Storage Locations

Default paths (configurable via environment variables):

```
data/
├── simple_app.db          # SQLite database
└── uploads/               # Uploaded images

datasets/
├── v0001/                 # Dataset version 1
├── v0002/                 # Dataset version 2
└── versions.json          # Dataset index

artifacts/
└── training/
    ├── train-*.json       # Training status files
    └── resnet50_*.pt      # Model checkpoints
```

## Configuration

Environment variables (create `.env` file or set in shell):

```bash
# Database
DATABASE_URL=sqlite:///data/simple_app.db

# Storage
UPLOAD_DIR=data/uploads
MAX_FILE_SIZE_BYTES=8388608  # 8 MB

# API
SIMPLE_API_HOST=0.0.0.0
SIMPLE_API_PORT=8100

# UI
SIMPLE_UI_HOST=0.0.0.0
SIMPLE_UI_PORT=7900

# Models
AIDETECT_MODEL_PATH=./resnet50_sd14_best.pt
AIDETECT_CLIP_MODEL_PATH=./clip_probe.joblib
```

## Complete Workflow Example

### 1. Upload Images

```bash
# Start the app
python -m v1.simple_app
```

In the UI:
1. Go to "Upload & Detect" tab
2. Select images (PNG/JPEG/WebP, max 8 MB each)
3. Click "Upload & Detect"
4. Note the image IDs from the results table

### 2. Review Queue

1. Go to "Pending Review Queue" tab
2. Click "Refresh queue"
3. Review predictions from both models
4. Copy image IDs for labeling

### 3. Submit Reviews

1. Go to "Review & Label" tab
2. Paste image ID
3. Click "Load Image"
4. Select label: `real`, `ai`, or `uncertain`
5. Set confidence (default: 1.0)
6. Optionally add generator/evidence/notes
7. Click "Submit Review"
8. Repeat for all images

### 4. Build Dataset

1. Go to "Dataset Management" tab
2. Enter version ID: `v0001`
3. Optionally enable "Dry run" to validate first
4. Click "Build New Dataset"
5. Review the report:
   - Manifest count
   - Class balance (real vs ai)
   - Exclusion reasons
   - Split counts

### 5. Train Model

1. Go to "Training & Models" tab
2. Enter dataset version: `v0001`
3. Select model: `resnet50`
4. Configure hyperparameters (or use defaults)
5. Optionally enable "Dry run" to validate
6. Click "Start Training"
7. Wait for completion (progress shown in status)
8. Note the artifact path for the trained model

### 6. Verify Results

```bash
# Check dataset files
ls -la datasets/v0001/

# Check training artifacts
ls -la artifacts/training/

# Check database records
sqlite3 data/simple_app.db "SELECT COUNT(*) FROM image_records;"
sqlite3 data/simple_app.db "SELECT COUNT(*) FROM human_reviews;"
```

## Database Schema

### Tables

**image_records**
- Primary storage for uploaded images
- Fields: id, storage_path, sha256, perceptual_hash, review_status, etc.
- One-to-many with predictions and reviews

**predictions**
- Model predictions (ResNet-50, CLIP)
- Fields: id, image_id, model_name, predicted_class, raw_score, etc.
- **Stored separately from human labels**

**human_reviews**
- Human-submitted labels and metadata
- Fields: id, image_id, verified_class, confidence, reviewer_id, etc.
- **Independent from model predictions**

## Key Design Principles

✓ **No Authentication** — Local workflow, no login/signup/admin panel

✓ **Single Interface** — All operations in one unified UI

✓ **SQLite for Queue** — Persistent, lightweight, file-based

✓ **Filesystem Storage** — Simple, portable image storage

✓ **Separate Labels** — Human labels ≠ model predictions (audit trail)

✓ **No Auto-Retraining** — Explicit user-initiated training only

✓ **Unchanged Pipelines** — ResNet and CLIP inference preserved

✓ **Versioned Datasets** — Immutable, reproducible training data

✓ **Explicit Promotion** — Model lifecycle requires deliberate action

## Troubleshooting

### API Won't Start

```bash
# Check if port is in use
lsof -i :8100

# Use different port
python -m v1.simple_app --api-port 8200
```

### UI Can't Connect to API

```bash
# Start API first in separate terminal
python -m v1.simple_app --api-only

# Then start UI
python -m v1.simple_app --ui-only
```

### Database Locked

SQLite doesn't handle concurrent writes well. If you see "database is locked":
- Close other connections
- Restart the app
- Consider PostgreSQL for production

### Missing Dependencies

```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Import Errors

```bash
# Install project in editable mode
pip install -e .
```

## Testing

Run the test suite:

```bash
cd /home/tushar/Desktop/DSBD_project/version1
source venv/bin/activate
pytest tests/ -v
```

Specific test categories:
```bash
pytest tests/test_api_upload_and_history.py -v
pytest tests/test_dataset_builder.py -v
pytest tests/test_model_promotion.py -v
```

## Next Steps

### Model Promotion (Not in Simple App)

The simple app focuses on the data workflow. For model promotion:
1. Use the main app (`src/v1/api.py`)
2. Or implement promotion in the model registry directly

### Production Deployment

For production use:
- Add authentication/authorization
- Use PostgreSQL instead of SQLite
- Run training as background jobs (Celery, RQ)
- Add monitoring and logging
- Implement rate limiting
- Use cloud storage (S3, GCS)

### Advanced Features

Consider adding:
- Batch review interface
- Advanced filtering in queue
- Review statistics dashboard
- Model comparison metrics
- Active learning suggestions
- Export to common formats (COCO, YOLO)

## Summary

The simple app provides a **complete human-in-the-loop workflow** from upload to trained models:

1. ✅ Upload images → automatic inference
2. ✅ Browse pending queue → view predictions
3. ✅ Submit reviews → human labels
4. ✅ Build datasets → versioned, deduplicated
5. ✅ Train models → explicit, controlled

All with **no authentication**, in a **single interface**, using **SQLite + filesystem** storage.

The system maintains **strict separation** between model predictions and human labels, supports **explicit training** only, and preserves the existing **ResNet and CLIP pipelines** unchanged.

Ready to use! 🚀
