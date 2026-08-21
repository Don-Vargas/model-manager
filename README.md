# Model Manager

A FastAPI microservice that discovers Hugging Face model repositories, normalizes their
structure into a consistent data model, downloads selected artifacts into MinIO/S3-compatible
storage, and verifies them by size and checksum. It is designed to sit between Hugging Face
Hub and downstream inference services (e.g. ComfyUI, vLLM, PyTorch-based services) running on
a separate machine — those services pull ready artifacts out of MinIO via a `model-agent`
component rather than talking to Hugging Face directly.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Workflow](#workflow)
- [Installation](#installation)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Data Model](#data-model)
- [Local Development](#local-development)
- [Testing](#testing)
- [Known Limitations](#known-limitations)
- [Future Improvements](#future-improvements)

## Overview

Model Manager does four things:

1. **Discovers** a Hugging Face repository's structure (`POST /repositories`) — file listing,
   `model_index.json` (if present), and derives logical `ModelUnit`s, `Component`s, and
   `Artifact`s from them. No files are downloaded at this stage.
2. **Normalizes** that structure into pipelines (Diffusers-style, multi-component),
   monolithic sharded checkpoints (large LLMs), single independent checkpoints, LoRA
   adapters, and GGUF quantizations.
3. **Persists** the normalized structure to a relational database (SQLite by default,
   Postgres-compatible via `DATABASE_URL`).
4. **Downloads and verifies** artifacts on demand (`POST /artifacts/{id}/download`,
   `POST /artifacts/{id}/verify`) — streaming from Hugging Face into MinIO with multipart
   upload, tracking per-artifact status, and comparing checksums computed at download time
   against what's actually in MinIO.

Discovery and download are deliberately separate steps: importing a repository tells you
what it contains; downloading is a separate, per-artifact decision (e.g. "only pull the
fp16 UNet, not fp8").

### Use Cases

- **Model Registry** — a catalog of models with their internal component structure.
- **Selective Artifact Management** — download only the components/precisions you need.
- **Pipeline Orchestration** — know a model's structure (e.g. which Diffusers components
  exist) before ComfyUI/vLLM/PyTorch tries to load it.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client (API)                             │
└───────────────────────────┬───────────────────────────────────────┘
                             │
┌───────────────────────────▼───────────────────────────────────────┐
│                      FastAPI Application                          │
│  Endpoints: POST /repositories · GET /models · GET /artifacts/... │
│             POST /artifacts/{id}/download · .../verify            │
└───────────────────────────┬───────────────────────────────────────┘
                             │
┌───────────────────────────▼───────────────────────────────────────┐
│                     Repository Service                            │
│         process_and_persist() · process_huggingface_repository()  │
└───────────────────────────┬───────────────────────────────────────┘
                             │
┌───────────────────────────▼───────────────────────────────────────┐
│                       Normalizer Layer                            │
│   Discoverer (load JSON) → Detector (structure) → Classifier      │
│                          (file taxonomy)                          │
└───────────────────────────┬───────────────────────────────────────┘
                             │
┌───────────────────────────▼───────────────────────────────────────┐
│                       Providers Layer                             │
│              HuggingFaceProvider (Hub API access)                 │
└───────────────────────────┬───────────────────────────────────────┘
                             │
┌───────────────────────────▼───────────────────────────────────────┐
│                        Storage Layer                              │
│   SQLite / Postgres (metadata)   │   MinIO / S3 (artifacts)       │
└─────────────────────────────────────────────────────────────────┘
```

Downstream, a separate `model-agent` process (running on the inference VM, alongside
ComfyUI/vLLM/PyTorch) is expected to poll this API for artifacts with
`status: "available"`, and pull the corresponding objects out of MinIO using the
`minio_bucket` / `minio_prefix` / `files[].path` fields returned by the API.

## Workflow

```
 1. Discover                2. Decide                3. Download            4. Serve
 POST /repositories    →    inspect units/       →    POST /artifacts/  →   model-agent reads
 (per repo_id)              components/artifacts       {id}/download        status/minio_*
                             returned by the API                            fields via
                                                                             GET /artifacts/{id}
```

Discovery never touches MinIO — it only reads Hugging Face metadata. MinIO location and
download status only exist on an artifact after `POST /artifacts/{id}/download` succeeds.

## Installation

### Prerequisites

- Docker & Docker Compose (`docker compose`, not the standalone `docker-compose` binary)
- A running MinIO (or S3-compatible) instance, reachable from the `model-manager`
  container's Docker network
- Hugging Face Hub access (a token is required for gated repositories)

### Quick Start with Docker

```bash
# Clone the repository
git clone <repository-url>
cd model-manager

# Create .env file
cp .env.example .env
# Edit .env with your credentials

# Build and run
docker compose up -d --build

# Run tests
docker compose exec model-manager pytest -v
```

> **Note on volumes:** `compose.yml` currently bind-mounts the whole project root into the
> container (`.:/app`, `.:/tests`) for local development convenience, which shadows whatever
> the image was built with. If you deploy this without the host working tree present, switch
> those to `./app:/app/app` and `./tests:/app/tests`, or drop the bind mounts entirely and
> rely on the image build.

### Manual Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run the application
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Configuration

Create a `.env` file in the project root:

```
# Application
APP_NAME=model-manager

# Database
DATABASE_URL=sqlite:////data/model-manager.db
# For PostgreSQL:
# DATABASE_URL=postgresql://user:pass@postgres:5432/modelmanager

# Hugging Face
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# MinIO / S3
MINIO_ENDPOINT=http://minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=models
```

`MINIO_ENDPOINT` must resolve from inside the `model-manager` container — use the Docker
service/container name on the shared network (e.g. `http://minio:9000`), not `localhost`.

### MinIO Setup

```bash
docker run -d \
  --name minio \
  -p 9000:9000 \
  -p 9001:9001 \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin \
  minio/minio server /data --console-address ":9001"

docker exec minio mc alias set myminio http://localhost:9000 minioadmin minioadmin
docker exec minio mc mb myminio/models
```

`compose.yml` expects an external Docker network named `minio_default` shared between the
`model-manager` and `minio` containers — make sure both are attached to it.

## API Reference

### `GET /health`

```json
{ "status": "ok", "service": "model-manager" }
```

### `POST /repositories`

Discovers and normalizes a Hugging Face repository. Does **not** download any files.

Request:
```json
{ "repo_id": "Qwen/Qwen-Image" }
```

Response (abridged — a pipeline repository, showing both `components` and `artifacts`):
```json
{
  "id": "Qwen/Qwen-Image",
  "commit_sha": "abc123...",
  "gated": false,
  "updated_at": "2026-01-01T00:00:00Z",
  "units": [
    {
      "id": "Qwen--Qwen-Image--pipeline",
      "name": "Qwen-Image-pipeline",
      "unit_type": "pipeline",
      "task_type": "text-to-image",
      "framework": "diffusers",
      "precision": "fp16",
      "quantization": null,
      "artifacts": [],
      "components": [
        {
          "id": "Qwen--Qwen-Image--pipeline--vae",
          "name": "vae",
          "component_type": "AutoencoderKLQwenImage",
          "artifacts": [
            {
              "id": "Qwen--Qwen-Image--pipeline--vae--art--0",
              "name": "diffusion_pytorch_model.safetensors",
              "artifact_type": "weights",
              "is_sharded": false,
              "index_file": null,
              "files": ["vae/diffusion_pytorch_model.safetensors"],
              "status": "discovered",
              "minio_bucket": null,
              "minio_prefix": null
            }
          ]
        }
      ]
    }
  ]
}
```

For non-pipeline repositories (single checkpoints, monolithic sharded LLMs, LoRA adapters,
GGUF quantizations), the unit's `artifacts` array is populated directly and `components` is
empty.

Unit, component, and artifact `id`s are built from the repository id with `/` and `#`
replaced by `--`, so every id is a single, URL-safe path segment (no slashes or URL
fragments), e.g. `Qwen--Qwen-Image--pipeline--vae--art--0`.

### `GET /models`

Returns a list of all previously imported repositories, in the same shape as the
`POST /repositories` response.

### `GET /artifacts/{artifact_id}`

Returns a single artifact:
```json
{
  "id": "Qwen--Qwen-Image--pipeline--vae--art--0",
  "name": "diffusion_pytorch_model.safetensors",
  "artifact_type": "weights",
  "is_sharded": false,
  "index_file": null,
  "files": ["vae/diffusion_pytorch_model.safetensors"],
  "status": "discovered",
  "minio_bucket": null,
  "minio_prefix": null
}
```

`status` is one of: `discovered`, `downloading`, `available`, `failed` (verification-related
states `verifying` / `verified` / `corrupted` exist as values but are not currently written
by any endpoint — `/verify` reports status in its own response body instead of persisting it
back onto the artifact).

### `POST /artifacts/{artifact_id}/download`

Streams every file belonging to the artifact from Hugging Face into MinIO via multipart
upload, computing a SHA-256 checksum per file as it streams. On success, sets
`status: "available"` and records `minio_bucket` / `minio_prefix` on the artifact; on
failure, sets `status: "failed"` and rolls back.

Response:
```json
{
  "status": "available",
  "artifact_id": "Qwen--Qwen-Image--pipeline--vae--art--0",
  "bucket": "models",
  "prefix": "Qwen/Qwen-Image/Qwen--Qwen-Image--pipeline--vae--art--0/"
}
```

Works for both unit-level and component-level artifacts (an artifact belongs to either a
`ModelUnit` directly, or to a `Component` of a unit — the download endpoint resolves the
owning repository either way).

### `POST /artifacts/{artifact_id}/verify`

Re-checks every file in MinIO against the size/checksum captured at download time.

Response:
```json
{
  "artifact_id": "Qwen--Qwen-Image--pipeline--vae--art--0",
  "status": "verified",
  "files": [
    {
      "path": "vae/diffusion_pytorch_model.safetensors",
      "status": "verified",
      "size": 167411776,
      "checksum": "3f2a9c..."
    }
  ]
}
```

#### Per-file verification status codes

| Status | Description |
|---|---|
| `verified` | File exists in MinIO and checksum matches |
| `size_verified` | File exists with matching size (no checksum was recorded, e.g. downloaded before this check existed) |
| `missing` | File not found in MinIO — most commonly because it hasn't been downloaded yet |
| `size_mismatch` | File exists but size differs |
| `checksum_mismatch` | File exists but checksum differs |

Verifying an artifact that was never downloaded is safe and simply returns `missing` for
each file (falls back to the configured default bucket rather than erroring).

## Data Model

### Database Schema

```sql
-- Repositories table
repositories (
    id VARCHAR(255) PRIMARY KEY,      -- Hugging Face repo ID, e.g. "Qwen/Qwen-Image"
    commit_sha VARCHAR(64) NOT NULL,
    gated BOOLEAN DEFAULT FALSE,
    updated_at DATETIME
)

-- Model Units (logical model instances: pipeline | monolithic | single_checkpoint | lora | gguf)
model_units (
    id VARCHAR(255) PRIMARY KEY,       -- URL-safe: "<repo_id with / -> -->--<suffix>"
    repo_id VARCHAR(255) REFERENCES repositories(id),
    name VARCHAR(255) NOT NULL,
    unit_type VARCHAR(50) NOT NULL,
    task_type VARCHAR(100),
    framework VARCHAR(100),
    precision VARCHAR(50),
    quantization VARCHAR(50),
    base_model_name_or_path VARCHAR(255),
    unit_metadata JSON
)

-- Components (Diffusers pipeline components — only present for unit_type = 'pipeline')
components (
    id VARCHAR(255) PRIMARY KEY,       -- "<unit_id>--<component_name>"
    unit_id VARCHAR(255) REFERENCES model_units(id),
    name VARCHAR(255) NOT NULL,
    component_type VARCHAR(100) NOT NULL
)

-- Artifacts (files and file groups) — belongs to EITHER a unit directly OR a component
artifacts (
    id VARCHAR(255) PRIMARY KEY,
    unit_id VARCHAR(255) REFERENCES model_units(id),        -- nullable
    component_id VARCHAR(255) REFERENCES components(id),    -- nullable
    name VARCHAR(255) NOT NULL,
    artifact_type VARCHAR(50) NOT NULL,
    files JSON NOT NULL,               -- list of file paths, or {"path": ...} dicts for sharded artifacts
    is_sharded BOOLEAN DEFAULT FALSE,
    index_file VARCHAR(255),
    status VARCHAR(50) DEFAULT 'discovered',
    minio_bucket VARCHAR(255),
    minio_prefix VARCHAR(255)
)
```

### Artifact Types

| Artifact Type | Description |
|---------------|-------------|
| `weights` | Model weights (`.safetensors`, `.bin`) |
| `weights_sharded` | Sharded weights grouped with their `.index.json` |
| `config` | Configuration files (`config.json`) |
| `tokenizer` | Tokenizer files |
| `processor` | Processor/feature-extractor files |
| `runtime_code` | Python scripts, Jinja chat templates |
| `documentation` | README, LICENSE, etc. |
| `metadata` | `.gitattributes`, `model_info.json` |
| `lora_adapter` | PEFT/LoRA adapter files |
| `gguf` | GGUF quantized models |
| `unknown` | Anything not matched by the classifier |

### Detection Logic (in order)

1. **LoRA** — presence of `adapter_config.json` or LoRA-named `.safetensors` files
2. **GGUF** — any file ending in `.gguf`, one `ModelUnit` per file, quantization scheme
   parsed from the filename (e.g. `Q4_K_M`)
3. **Diffusers pipeline** — `model_index.json` present; components come from its
   top-level keys (keys whose value is a `[library, class_name]` pair; scalar/`None`-first
   entries like `boundary_ratio` are treated as pipeline metadata, not components)
4. **Single checkpoints** — root-level `.safetensors`/`.bin` files not claimed by the
   pipeline step, one `ModelUnit` per file (e.g. FLUX.1-dev's `ae` and `flux1-dev`)
5. **Monolithic sharded** — root-level `*.index.json` + matching shards, only when no
   other unit type was already found (typical for a plain sharded LLM checkpoint)

## Local Development

### Running Tests

```bash
# All tests
docker compose exec model-manager pytest -v

# Specific test file
docker compose exec model-manager pytest tests/test_huggingface_metadata.py -v

# With coverage
docker compose exec model-manager pytest --cov=app tests/
```

Test discovery/imports are configured via `pytest.ini` (`pythonpath = .`), so `pytest` works
the same whether invoked directly or via `python -m pytest`.

### Inspecting Metadata

```bash
# Collect metadata from Hugging Face into ./data/huggingface (used as test fixtures)
python scripts/collect_huggingface_metadata.py

# Inspect collected metadata
python scripts/inspect_huggingface_metadata.py
```

### Debugging

```bash
uvicorn app.main:app --reload --log-level debug
```

## Testing

```
tests/
├── test_huggingface_metadata.py   # Normalization tests (fixture-backed, /app/data/huggingface)
├── test_artifacts.py              # Artifact classification (app/providers/artifacts.py)
├── test_downloader.py             # Download/verification unit tests (mocked S3)
├── test_repository_service.py     # Service layer persistence tests (fixture-backed)
└── test_e2e.py                    # Live HTTP API tests against Hugging Face + MinIO
```

`test_e2e.py` makes real calls to the Hugging Face Hub (`hf-internal-testing/tiny-random-bert`,
`Qwen/Qwen-Image`, `black-forest-labs/FLUX.1-dev`) and, for the download/verify flow tests,
real calls to MinIO — set `MINIO_ENDPOINT` / `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` in the
test environment or those specific tests will skip with "MinIO not configured".

Fixture-backed tests (`test_huggingface_metadata.py`, parts of `test_repository_service.py`)
read from `/app/data/huggingface/<category>/<org>__<repo>/`, populated by
`scripts/collect_huggingface_metadata.py`. Tests for fixtures that aren't present skip
individually rather than failing the run.

## Known Limitations

- **No download progress tracking exposed via API** — download is a single blocking request
  until completion or failure.
- **No retry logic for failed downloads** — a `failed` artifact can be retried by calling
  `/download` again, but nothing does this automatically.
- **SQLite by default** — fine for a single-instance deployment; switch `DATABASE_URL` to
  Postgres for concurrent/multi-writer scenarios.
- **No schema migrations** — the app calls `Base.metadata.create_all()` at startup, which
  creates missing tables but does not alter existing ones. Schema changes on an existing
  database currently require a manual migration or a fresh database.
- **No authentication** — the API is unauthenticated; keep it on a private network (this is
  the intended deployment: internal to the `services` VM / Docker network, called by
  `model-agent` on a separate, trusted inference VM).
- **`artifacts.files` shape is inconsistent** — plain filename strings for most artifacts,
  `{"path": ...}` dicts for sharded artifacts. Consumers should handle both.

### Removed dead code

The following were removed as unused/orphaned during cleanup and are **not** part of the
codebase anymore, in case you find references to them in old notes or commit history:

- `app/importer.py` — pre-refactor import path, referenced model classes that no longer exist
- `app/providers/artifacts.py` — a second, unused file-classification engine; the live
  classification logic is `app/normalizer/classification.py::FileClassifier`
- `scripts/validate_huggingface_metadata.py` — a frozen, out-of-sync copy of the normalizer;
  use `app/normalizer/normalizer.py::HuggingFaceMetadataNormalizer` directly instead
- `app/schemas.py` was trimmed to only `HuggingFaceImportRequest` (the request body for
  `POST /repositories`); the response-shaped models it used to contain
  (`ArtifactResponse`, `ModelResponse`, etc.) described a data model that no longer matches
  the actual schema — see [API Reference](#api-reference) for the real response shapes

## Future Improvements

1. **Multi-Provider Support** — AWS SageMaker registry, custom S3 discovery, Git LFS.
2. **Enhanced Classification** — GPTQ/AWQ/EXL2 quantization detection, more adapter types.
3. **Performance** — batch imports, parallel artifact downloads, DB connection pooling,
   Redis caching for frequently accessed metadata.
4. **Additional Features** — webhook support for repo updates, ONNX/OpenVINO detection,
   model card indexing.
5. **Observability** — Prometheus metrics, OpenTelemetry tracing, structured JSON logging.
6. **Security** — API authentication (JWT/API keys), audit logging.
7. **Migrations** — introduce Alembic instead of relying on `create_all()`.