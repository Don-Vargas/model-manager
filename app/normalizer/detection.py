# app/normalizer/detection.py

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from app.normalizer.models import Artifact, ArtifactType



class StructureDetector:
    """Detecta agrupaciones complejas: Shards, Checkpoints raíz, LoRA y GGUF."""

    SHARD_PATTERN = re.compile(r".*-\d{5}-of-\d{5}\.(safetensors|bin)$")

    @classmethod
    def detect_sharded_artifacts(
        cls, files: List[str], root_only: bool = False
    ) -> Tuple[List[Artifact], Set[str]]:
        artifacts = []
        consumed = set()

        index_files = [f for f in files if f.endswith(".index.json")]
        if root_only:
            index_files = [f for f in index_files if "/" not in f]

        for idx_file in index_files:
            idx_path = Path(idx_file)
            base_dir = str(idx_path.parent).replace("\\", "/")
            if base_dir == ".":
                base_dir = ""

            # Filtramos los shards garantizando exactamente el mismo directorio que el index
            shards = []
            for f in files:
                if not cls.SHARD_PATTERN.match(f):
                    continue

                f_parent = str(Path(f).parent).replace("\\", "/")
                if f_parent == ".":
                    f_parent = ""

                # Comprobación estricta de pertenencia al mismo directorio
                if f_parent == base_dir:
                    shards.append(f)

            if not shards:
                continue

            consumed.add(idx_file)
            for s in shards:
                consumed.add(s)

            shards_data = [{"path": s} for s in shards]

            artifacts.append(
                Artifact(
                    name=idx_path.name.replace(".index.json", ""),
                    artifact_type=ArtifactType.WEIGHTS_SHARDED,
                    is_sharded=True,
                    files=shards_data,
                    index_file=idx_file,
                )
            )

        return artifacts, consumed

    @classmethod
    def parse_model_index(
        cls, model_index: Dict[str, Any]
    ) -> Tuple[Dict[str, str], Dict[str, Any]]:
        components = {}
        metadata = {}

        for key, val in model_index.items():
            if key.startswith("_"):
                continue

            is_valid_component = (
                isinstance(val, list)
                and len(val) >= 2
                and val[0] is not None
            )

            if is_valid_component:
                components[key] = val[1]
            else:
                metadata[key] = val

        return components, metadata

    @classmethod
    def detect_lora_config(cls, repo_dir: Path) -> Optional[Dict[str, Any]]:
        """Lee adapter_config.json si existe para extraer metadatos de LoRA."""
        config_path = repo_dir / "adapter_config.json"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None
