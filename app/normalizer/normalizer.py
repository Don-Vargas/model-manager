from pathlib import Path
from typing import Optional, Set

from app.normalizer.classification import FileClassifier
from app.normalizer.detection import StructureDetector
from app.normalizer.discovery import RepositoryDiscoverer
from app.normalizer.models import (
    Artifact,
    ArtifactType,
    Component,
    ModelUnit,
    RepositoryNormalized,
)



def _file_path(entry) -> str:
    return entry["path"] if isinstance(entry, dict) else entry


def _sum_sizes(files, file_sizes: dict) -> Optional[int]:
    sizes = [file_sizes.get(_file_path(f)) for f in files]
    if not any(s is not None for s in sizes):
        return None
    return sum(s or 0 for s in sizes)


class HuggingFaceMetadataNormalizer:
    """Orquesta las fases de discovery, detection, classification y normalization."""

    def __init__(self, repo_dir: Path):
        self.repo_dir = repo_dir
        self.discoverer = RepositoryDiscoverer(repo_dir)

    def normalize(self) -> Optional[RepositoryNormalized]:
        if not self.discoverer.model_info and not (self.discoverer.repo_dir / "model_info.json").exists():
            return None

        model_info = self.discoverer.model_info or {}
        repo_id = model_info.get("id", self.discoverer.repo_dir.name)
        safe_repo_id = repo_id.replace("/", "--")
        commit_sha = model_info.get("sha", "unknown")
        gated = model_info.get("gated", False)
        pipeline_tag = model_info.get("pipeline_tag", "unknown")
        library_name = model_info.get("library_name", "unknown")

        all_files = self.discoverer.get_all_files()
        file_sizes = self.discoverer.get_file_sizes()  # NEW
        consumed_files: Set[str] = set()
        
        repo_norm = RepositoryNormalized(
            repo_id=repo_id, commit_sha=commit_sha, gated=gated
        )

        # ----------------------------------------------------------------------
        # PASO 1: ADAPTADOR LORA (adapter_config.json)
        # ----------------------------------------------------------------------
        lora_config = StructureDetector.detect_lora_config(self.repo_dir)
        if lora_config or "adapter_config.json" in all_files:
            base_model = lora_config.get("base_model_name_or_path") if lora_config else None
            unit = ModelUnit(
                id=f"{safe_repo_id}--lora",
                name=f"{repo_id.split('/')[-1]}-lora",
                unit_type="lora",
                task_type=pipeline_tag,
                framework="peft",
                base_model_name_or_path=base_model,
                metadata=lora_config or {},
            )

            lora_files = [
                f for f in all_files
                if f in {"adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"} or "lora" in f.lower()
            ]
            for lf in lora_files:
                art_type = FileClassifier.classify_filename(lf)
                unit.direct_artifacts.append(
                    Artifact(
                        name=Path(lf).name,
                        artifact_type=art_type,
                        files=[lf],
                        size_bytes=_sum_sizes([lf], file_sizes),
                    )
                )
                consumed_files.add(lf)

            repo_norm.units.append(unit)

        # ----------------------------------------------------------------------
        # PASO 2: ARCHIVOS GGUF (Cuantizaciones)
        # ----------------------------------------------------------------------
        gguf_files = [f for f in all_files if f.endswith(".gguf")]
        if gguf_files:
            for g_file in gguf_files:
                quant = FileClassifier.extract_gguf_quantization(g_file)
                unit_id = Path(g_file).stem
                unit = ModelUnit(
                    id=f"{safe_repo_id}--{unit_id}",
                    name=unit_id,
                    unit_type="gguf",
                    task_type=pipeline_tag,
                    framework="llama.cpp",
                    quantization=quant,
                )
                unit.direct_artifacts.append(
                    Artifact(
                        name=Path(g_file).name,
                        artifact_type=ArtifactType.GGUF,
                        files=[g_file],
                        size_bytes=_sum_sizes([g_file], file_sizes),
                    )
                )
                repo_norm.units.append(unit)
                consumed_files.add(g_file)

        # ----------------------------------------------------------------------
        # PASO 3: PIPELINE PRINCIPAL (Diffusers / Composite)
        # ----------------------------------------------------------------------
        if self.discoverer.model_index:
            comp_map, pipeline_meta = StructureDetector.parse_model_index(self.discoverer.model_index)
            
            unit = ModelUnit(
                id=f"{safe_repo_id}--pipeline",
                name=f"{repo_id.split('/')[-1]}-pipeline",
                unit_type="pipeline",
                task_type=pipeline_tag,
                framework="diffusers",
                precision=FileClassifier.derive_precision(all_files),
                metadata=pipeline_meta
            )

            for comp_name, comp_type in comp_map.items():
                comp_files = [f for f in all_files if f.startswith(f"{comp_name}/")]
                for cf in comp_files:
                    consumed_files.add(cf)

                sharded_art, local_consumed = StructureDetector.detect_sharded_artifacts(comp_files)
                for art in sharded_art:
                    art.size_bytes = _sum_sizes(art.files, file_sizes)

                remaining_files = [f for f in comp_files if f not in local_consumed]

                comp_artifacts = list(sharded_art)
                for rf in remaining_files:
                    art_type = FileClassifier.classify_filename(rf)
                    comp_artifacts.append(
                        Artifact(
                            name=Path(rf).name,
                            artifact_type=art_type,
                            files=[rf],
                            size_bytes=_sum_sizes([rf], file_sizes),
                        )
                    )

                unit.components.append(
                    Component(name=comp_name, component_type=comp_type, artifacts=comp_artifacts)
                )

            repo_norm.units.append(unit)
            consumed_files.add("model_index.json")

        unconsumed = [f for f in all_files if f not in consumed_files]

        # ----------------------------------------------------------------------
        # PASO 4: CHECKPOINTS INDEPENDIENTES EN LA RAÍZ
        # ----------------------------------------------------------------------
        root_weights = [
            f for f in unconsumed
            if "/" not in f
            and FileClassifier.classify_filename(f) in {ArtifactType.WEIGHTS, ArtifactType.LORA_ADAPTER}
            and not StructureDetector.SHARD_PATTERN.match(f)
        ]

        # A single root weight belongs to the complete repository model; leave it
        # unconsumed so the monolithic unit also includes config/tokenizer files.
        defer_to_monolithic = len(root_weights) == 1 and not repo_norm.units

        if not defer_to_monolithic:
            for w_file in root_weights:
                unit_id = Path(w_file).stem
                unit = ModelUnit(
                    id=f"{safe_repo_id}--{unit_id}",
                    name=unit_id,
                    unit_type="single_checkpoint",
                    task_type=pipeline_tag,
                    framework=library_name,
                    precision=FileClassifier.derive_precision([w_file]),
                )
                unit.direct_artifacts.append(
                    Artifact(
                        name=w_file,
                        artifact_type=FileClassifier.classify_filename(w_file),
                        files=[w_file],
                        size_bytes=_sum_sizes([w_file], file_sizes),
                    )
                )
                repo_norm.units.append(unit)
                consumed_files.add(w_file)

        unconsumed = [f for f in all_files if f not in consumed_files]

        # ----------------------------------------------------------------------
        # PASO 5: MODELO MONOLÍTICO EN RAÍZ (LLMs)
        # ----------------------------------------------------------------------
        if unconsumed and not repo_norm.units:
            sharded_art, sharded_consumed = StructureDetector.detect_sharded_artifacts(unconsumed, root_only=True)
            for art in sharded_art:
                art.size_bytes = _sum_sizes(art.files, file_sizes)

            unit = ModelUnit(
                id=f"{safe_repo_id}--main",
                name=repo_id.split("/")[-1],
                unit_type="monolithic",
                task_type=pipeline_tag,
                framework=library_name,
                precision=FileClassifier.derive_precision(unconsumed),
            )
            unit.direct_artifacts.extend(sharded_art)

            for f in unconsumed:
                if f in sharded_consumed:
                    continue
                art_type = FileClassifier.classify_filename(f)
                unit.direct_artifacts.append(
                    Artifact(
                        name=Path(f).name,
                        artifact_type=art_type,
                        files=[f],
                        size_bytes=_sum_sizes([f], file_sizes),
                    )
                )

            repo_norm.units.append(unit)

        return repo_norm
