import json
import tempfile
from pathlib import Path
from typing import Optional
from sqlalchemy.orm import Session

from app.database import (
    ArtifactModel,
    ComponentModel,
    ModelUnitModel,
    RepositoryModel,
)
from app.normalizer.models import RepositoryNormalized
from app.normalizer.normalizer import HuggingFaceMetadataNormalizer
from app.providers.huggingface import HuggingFaceProvider


class RepositoryService:
    """Gestiona la normalización de metadatos y la sincronización con la BD."""

    def __init__(self, db_session: Session):
        self.db = db_session

    def process_and_persist(self, repo_dir: Path) -> Optional[RepositoryModel]:
        normalizer = HuggingFaceMetadataNormalizer(repo_dir)
        normalized: Optional[RepositoryNormalized] = normalizer.normalize()

        if not normalized:
            return None

        # Reemplazar o actualizar registro existente
        existing_repo = self.db.query(RepositoryModel).filter_by(id=normalized.repo_id).first()
        if existing_repo:
            self.db.delete(existing_repo)
            self.db.flush()

        is_gated = bool(normalized.gated) if isinstance(normalized.gated, bool) else (
            str(normalized.gated).lower() in ("true", "1", "auto") if normalized.gated else False
        )

        repo_record = RepositoryModel(
            id=normalized.repo_id,
            commit_sha=normalized.commit_sha,
            gated=is_gated,
        )

        for unit in normalized.units:
            unit_record = ModelUnitModel(
                id=unit.id,
                repo_id=repo_record.id,
                name=unit.name,
                unit_type=unit.unit_type,
                task_type=unit.task_type,
                framework=unit.framework,
                precision=unit.precision,
                quantization=unit.quantization,
                base_model_name_or_path=unit.base_model_name_or_path,
                unit_metadata=unit.metadata,
            )

            # Artefactos directos
            for idx, art in enumerate(unit.direct_artifacts):
                art_record = ArtifactModel(
                    id=f"{unit.id}--art--{idx}",
                    name=art.name,
                    artifact_type=art.artifact_type.value if hasattr(art.artifact_type, "value") else str(art.artifact_type),
                    files=art.files,
                    is_sharded=art.is_sharded,
                    index_file=art.index_file,
                    size_bytes=art.size_bytes,
                )
                unit_record.direct_artifacts.append(art_record)

            # Componentes
            for comp in unit.components:
                comp_id = f"{unit.id}--{comp.name}"
                comp_record = ComponentModel(
                    id=comp_id,
                    name=comp.name,
                    component_type=comp.component_type,
                )

                for idx, art in enumerate(comp.artifacts):
                    art_record = ArtifactModel(
                        id=f"{comp_id}--art--{idx}",
                        name=art.name,
                        artifact_type=art.artifact_type.value if hasattr(art.artifact_type, "value") else str(art.artifact_type),
                        files=art.files,
                        is_sharded=art.is_sharded,
                        index_file=art.index_file,
                        size_bytes=art.size_bytes,
                    )
                    comp_record.artifacts.append(art_record)

                unit_record.components.append(comp_record)

            repo_record.units.append(unit_record)

        self.db.add(repo_record)
        self.db.commit()
        self.db.refresh(repo_record)
        return repo_record


def process_huggingface_repository(
    repo_id: str,
    db: Session,
    hf_token: str | None = None,
) -> RepositoryModel:
    provider = HuggingFaceProvider(token=hf_token)
    model_info = provider.get_model(repo_id)

    with tempfile.TemporaryDirectory() as tmp_dir:
        repo_path = Path(tmp_dir) / repo_id.replace("/", "_")
        repo_path.mkdir(parents=True, exist_ok=True)

        if hasattr(model_info, "__dict__"):
            info_dict = {
                "id": getattr(model_info, "id", repo_id),
                "sha": getattr(model_info, "sha", "unknown"),
                "gated": getattr(model_info, "gated", False),
                "pipeline_tag": getattr(model_info, "pipeline_tag", "unknown"),
                "library_name": getattr(model_info, "library_name", "unknown"),
                "siblings": [
                    {
                        "rfilename": getattr(s, "rfilename", str(s)),
                        "size": getattr(s, "size", None),
                    }
                    for s in getattr(model_info, "siblings", [])
                ],
            }
        elif isinstance(model_info, dict):
            info_dict = model_info
        else:
            info_dict = {"id": repo_id, "sha": "unknown"}

        with open(repo_path / "model_info.json", "w", encoding="utf-8") as f:
            json.dump(info_dict, f)

        # NEW: fetch and write model_index.json if this repo has one
        # (Diffusers pipelines only — returns None safely otherwise)
        model_index = provider.get_model_index(
            repo_id=repo_id,
            revision=info_dict.get("sha"),
        )

        if model_index is not None:
            with open(repo_path / "model_index.json", "w", encoding="utf-8") as f:
                json.dump(model_index, f)

        service = RepositoryService(db_session=db)
        repo_model = service.process_and_persist(repo_path)

        if not repo_model:
            raise ValueError(f"No se pudieron extraer metadatos válidos para {repo_id}")

        return repo_model
