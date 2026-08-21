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


@app.post("/artifacts/{artifact_id}/download")
def download_artifact(
    artifact_id: str,
    db: Session = Depends(get_db),
):
    artifact = get_artifact_or_404(artifact_id, db)
    unit = artifact.unit or (artifact.component.unit if artifact.component else None)
    repo = unit.repository if unit else None

    if not repo:
        raise HTTPException(status_code=400, detail="Artifact is not linked to a valid repository")

    provider = HuggingFaceProvider(token=settings.hf_token)
    prefix = f"{repo.id}/{artifact.id}/"

    artifact.status = ArtifactStatus.DOWNLOADING.value
    db.commit()

    try:
        files = artifact.files or []
        updated_files = []

        for file_info in files:
            if isinstance(file_info, dict):
                filename = file_info.get("path")
                expected_size = file_info.get("size_bytes") or file_info.get("size")
                expected_checksum = file_info.get("checksum")
            else:
                filename = str(file_info)
                expected_size = None
                expected_checksum = None

            object_key = build_artifact_object_key_from_prefix(prefix, filename)

            result = download_file_to_minio(
                provider=provider,
                repo_id=repo.id,
                revision=repo.commit_sha,
                filename=filename,
                object_key=object_key,
                expected_size=expected_size,
                expected_checksum=expected_checksum,
            )

            updated_files.append({
                "path": filename,
                "size_bytes": result.get("size", expected_size),
                "checksum": result.get("checksum", expected_checksum),
            })

        artifact.minio_bucket = storage_settings.minio_bucket
        artifact.minio_prefix = prefix
        artifact.files = updated_files
        artifact.status = ArtifactStatus.AVAILABLE.value
        db.commit()

        return {
            "status": "available",
            "artifact_id": artifact.id,
            "bucket": storage_settings.minio_bucket,
            "prefix": prefix,
        }

    except Exception as exc:
        db.rollback()
        artifact.status = ArtifactStatus.FAILED.value
        db.commit()
        raise HTTPException(status_code=500, detail=f"Failed to download artifact files: {exc}")


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
