from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session
from typing import Any, Dict
from pydantic import BaseModel

from .config import settings
from .database import (
    ArtifactModel,
    Base,
    RepositoryModel,
    engine,
    get_db,
)
from .downloader import download_file_to_minio
from .models import ArtifactStatus
from .providers.huggingface import HuggingFaceProvider
from .services.repository_service import process_huggingface_repository
from .storage import (
    build_artifact_prefix,
    calculate_object_sha256,
    s3,
    settings as storage_settings,
)

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Model Manager",
    version="0.1.0",
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": settings.app_name,
    }


def get_artifact_or_404(
    artifact_id: str,
    db: Session,
) -> ArtifactModel:
    artifact = (
        db.query(ArtifactModel)
        .filter(ArtifactModel.id == artifact_id)
        .first()
    )

    if not artifact:
        raise HTTPException(
            status_code=404,
            detail="Artifact not found",
        )

    return artifact


def build_artifact_object_key(
    artifact: ArtifactModel,
    filename: str,
) -> str:
    prefix = getattr(artifact, "minio_prefix", "") or ""
    return f"{prefix}{filename}"


def build_artifact_object_key_from_prefix(prefix: str, filename: str) -> str:
    return f"{prefix}{filename}"


def artifact_to_response(artifact: ArtifactModel):
    return {
        "id": artifact.id,
        "name": artifact.name,
        "artifact_type": artifact.artifact_type,
        "is_sharded": artifact.is_sharded,
        "index_file": artifact.index_file,
        "files": artifact.files or [],
        "status": artifact.status,
        "minio_bucket": artifact.minio_bucket,
        "minio_prefix": artifact.minio_prefix,
        "size_bytes": artifact.size_bytes,
    }


def download_progress_response(artifact: ArtifactModel) -> Dict[str, Any]:
    """Return download progress and support both legacy and new file metadata."""
    raw_files = artifact.files or []
    files = [
        file_info if isinstance(file_info, dict) else {"path": str(file_info)}
        for file_info in raw_files
    ]

    completed_files = sum(
        file_info.get("download_status") == "available"
        or (
            artifact.status in {
                ArtifactStatus.AVAILABLE.value,
                ArtifactStatus.VERIFIED.value,
            }
            and "download_status" not in file_info
        )
        for file_info in files
    )

    total_bytes = sum(
        file_info.get("size_bytes") or file_info.get("size") or 0
        for file_info in files
    )

    downloaded_bytes = sum(
        file_info.get("downloaded_bytes")
        or (
            (file_info.get("size_bytes") or file_info.get("size") or 0)
            if artifact.status
            in {ArtifactStatus.AVAILABLE.value, ArtifactStatus.VERIFIED.value}
            else 0
        )
        for file_info in files
    )

    unit = artifact.unit or (
        artifact.component.unit if artifact.component else None
    )

    return {
        "artifact_id": artifact.id,
        "repo_id": unit.repo_id if unit else "unknown",
        "name": artifact.name,
        "status": artifact.status,
        "completed_files": completed_files,
        "total_files": len(files),
        "downloaded_bytes": downloaded_bytes,
        "total_bytes": total_bytes,
        "progress_percent": (
            downloaded_bytes / total_bytes * 100 if total_bytes else 0
        ),
        "error": next(
            (
                file_info.get("download_error")
                for file_info in files
                if file_info.get("download_error")
            ),
            None,
        ),
    }


def component_to_response(component):
    return {
        "id": component.id,
        "name": component.name,
        "component_type": component.component_type,
        "artifacts": [artifact_to_response(art) for art in component.artifacts],
    }


def repository_to_response(repo: RepositoryModel):
    return {
        "id": repo.id,
        "commit_sha": repo.commit_sha,
        "gated": repo.gated,
        "updated_at": repo.updated_at.isoformat() if repo.updated_at else None,
        "units": [
            {
                "id": unit.id,
                "name": unit.name,
                "unit_type": unit.unit_type,
                "task_type": unit.task_type,
                "framework": unit.framework,
                "precision": unit.precision,
                "quantization": unit.quantization,
                "artifacts": [
                    artifact_to_response(art) for art in unit.direct_artifacts
                ],
                "components": [
                    component_to_response(comp) for comp in unit.components
                ],
            }
            for unit in repo.units
        ],
    }


class HuggingFaceImportRequest(BaseModel):
    repo_id: str


@app.post("/repositories")
def create_repository(
    request: HuggingFaceImportRequest,
    db: Session = Depends(get_db),
):
    try:
        repo = process_huggingface_repository(
            repo_id=request.repo_id,
            db=db,
            hf_token=settings.hf_token,
        )
        return repository_to_response(repo)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing repository: {exc}",
        )


@app.get("/models")
def list_models(
    db: Session = Depends(get_db),
):
    repositories = db.query(RepositoryModel).all()
    return [repository_to_response(repo) for repo in repositories]


def download_artifact_by_id(artifact_id: str, db: Session) -> Dict[str, Any]:
    artifact = get_artifact_or_404(artifact_id, db)
    unit = artifact.unit or (artifact.component.unit if artifact.component else None)
    repo = unit.repository if unit else None

    if not repo:
        return {
            "artifact_id": artifact_id,
            "status": "error",
            "detail": "Artifact is not linked to a valid repository",
        }

    provider = HuggingFaceProvider(token=settings.hf_token)
    prefix = f"{repo.id}/{artifact.id}/"

    artifact.status = ArtifactStatus.DOWNLOADING.value
    # Persist progress state before the first network request, so another API request
    # can immediately show it in the UI.
    updated_files = []
    for file_info in artifact.files or []:
        entry = dict(file_info) if isinstance(file_info, dict) else {"path": str(file_info)}
        entry.update({"download_status": "pending", "downloaded_bytes": 0, "download_error": None})
        updated_files.append(entry)
    artifact.files = updated_files

    db.commit()

    try:
        files = artifact.files or []

        for index, file_info in enumerate(files):
            if isinstance(file_info, dict):
                filename = file_info.get("path")
                expected_size = file_info.get("size_bytes") or file_info.get("size")
                expected_checksum = file_info.get("checksum")
            else:
                filename = str(file_info)
                expected_size = None
                expected_checksum = None

            object_key = build_artifact_object_key_from_prefix(prefix, filename)
            updated_files[index]["download_status"] = "downloading"
            artifact.files = list(updated_files)  # assign a new list so SQLAlchemy tracks JSON changes
            db.commit()

            def save_progress(downloaded: int, current_index: int = index) -> None:
                updated_files[current_index]["downloaded_bytes"] = downloaded
                artifact.files = list(updated_files)
                db.commit()

            result = download_file_to_minio(
                provider=provider,
                repo_id=repo.id,
                revision=repo.commit_sha,
                filename=filename,
                object_key=object_key,
                expected_size=expected_size,
                expected_checksum=expected_checksum,
                progress_callback=save_progress,
            )

            updated_files[index].update({
                "path": filename,
                "size_bytes": result.get("size", expected_size),
                "checksum": result.get("checksum", expected_checksum),
                "downloaded_bytes": result.get("size", expected_size) or 0,
                "download_status": "available",
                "download_error": None,
            })
            artifact.files = list(updated_files)
            db.commit()

        artifact.minio_bucket = storage_settings.minio_bucket
        artifact.minio_prefix = prefix
        artifact.files = updated_files
        artifact.status = ArtifactStatus.AVAILABLE.value
        db.commit()

        return {
            "artifact_id": artifact.id,
            "status": "available",
            "bucket": storage_settings.minio_bucket,
            "prefix": prefix,
        }

    except Exception as exc:
        db.rollback()
        artifact.status = ArtifactStatus.FAILED.value
        for entry in updated_files:
            if entry.get("download_status") in {"pending", "downloading"}:
                entry["download_status"] = "failed"
                entry["download_error"] = str(exc)
                break
        artifact.files = list(updated_files)

        db.commit()
        return {
            "artifact_id": artifact.id,
            "status": "failed",
            "detail": str(exc),
        }


@app.post("/artifacts/{artifact_id}/download")
def download_artifact(
    artifact_id: str,
    db: Session = Depends(get_db),
):
    result = download_artifact_by_id(artifact_id, db)
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["detail"])
    if result["status"] == "failed":
        raise HTTPException(status_code=500, detail=f"Failed to download artifact files: {result['detail']}")
    return result


class BatchDownloadRequest(BaseModel):
    artifact_ids: list[str]


@app.post("/artifacts/download-batch")
def download_artifacts_batch(
    request: BatchDownloadRequest,
    db: Session = Depends(get_db),
):
    results = []
    for artifact_id in request.artifact_ids:
        try:
            result = download_artifact_by_id(artifact_id, db)
        except HTTPException as exc:
            result = {
                "artifact_id": artifact_id,
                "status": "error",
                "detail": exc.detail,
            }
        results.append(result)

    succeeded = sum(1 for r in results if r["status"] == "available")
    failed = len(results) - succeeded

    return {
        "total": len(results),
        "succeeded": succeeded,
        "failed": failed,
        "results": results,
    }


def verify_artifact_file(
    artifact: ArtifactModel,
    artifact_file: Any,  # <--- Se cambia 'any' por 'Any'
) -> Dict[str, Any]:
    if isinstance(artifact_file, dict):
        filename = artifact_file.get("path")
        expected_size = artifact_file.get("size_bytes") or artifact_file.get("size")
        expected_checksum = artifact_file.get("checksum")
    elif hasattr(artifact_file, "path"):
        filename = getattr(artifact_file, "path")
        expected_size = getattr(artifact_file, "size_bytes", None) or getattr(artifact_file, "size", None)
        expected_checksum = getattr(artifact_file, "checksum", None)
    else:
        filename = str(artifact_file)
        expected_size = None
        expected_checksum = None

    if not filename:
        return {"path": "unknown", "status": "invalid_file_entry"}

    object_key = build_artifact_object_key(artifact, filename)
    bucket = artifact.minio_bucket or storage_settings.minio_bucket

    try:
        response = s3.head_object(
            Bucket=bucket,
            Key=object_key,
        )
    except s3.exceptions.ClientError:
        return {
            "path": filename,
            "status": "missing",
        }

    actual_size = response["ContentLength"]

    if expected_size is not None and actual_size != expected_size:
        return {
            "path": filename,
            "status": "size_mismatch",
            "expected_size": expected_size,
            "actual_size": actual_size,
        }

    if expected_checksum:
        actual_checksum = calculate_object_sha256(
            bucket=bucket,
            key=object_key,
        )

        if actual_checksum != expected_checksum:
            return {
                "path": filename,
                "status": "checksum_mismatch",
                "expected_checksum": expected_checksum,
                "actual_checksum": actual_checksum,
            }

        return {
            "path": filename,
            "status": "verified",
            "size": actual_size,
            "checksum": actual_checksum,
        }

    return {
        "path": filename,
        "status": "size_verified",
        "size": actual_size,
    }



@app.post("/artifacts/{artifact_id}/verify")
def verify_artifact(
    artifact_id: str,
    db: Session = Depends(get_db),
):
    artifact = get_artifact_or_404(
        artifact_id,
        db,
    )

    results = []
    files = artifact.files or []

    for file_info in files:
        result = verify_artifact_file(
            artifact=artifact,
            artifact_file=file_info,
        )
        results.append(result)

    all_verified = all(
        result["status"] in {"verified", "size_verified"} for result in results
    )

    return {
        "artifact_id": artifact.id,
        "status": "verified" if all_verified else "corrupted",
        "files": results,
    }


@app.get("/downloads/status")
def download_status(db: Session = Depends(get_db)):
    """Progress for every known artifact; intended for polling by the operator UI."""
    artifacts = db.query(ArtifactModel).order_by(ArtifactModel.name).all()
    return {"downloads": [download_progress_response(artifact) for artifact in artifacts]}


@app.get("/artifacts/{artifact_id}")
def get_artifact(
    artifact_id: str,
    db: Session = Depends(get_db),
):
    artifact = get_artifact_or_404(
        artifact_id,
        db,
    )
    return artifact_to_response(artifact)
