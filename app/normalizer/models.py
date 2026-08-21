from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ArtifactType(str, Enum):
    WEIGHTS = "weights"
    WEIGHTS_SHARDED = "weights_sharded"
    GGUF = "gguf"
    LORA_ADAPTER = "lora_adapter"
    CONFIG = "config"
    TOKENIZER = "tokenizer"
    PROCESSOR = "processor"
    RUNTIME_CODE = "runtime_code"
    SCHEDULER = "scheduler"
    METADATA = "metadata"
    DOCUMENTATION = "documentation"
    UNKNOWN = "unknown"


@dataclass
class Artifact:
    name: str
    artifact_type: ArtifactType
    files: List[str]
    is_sharded: bool = False
    index_file: Optional[str] = None
    size_bytes: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["artifact_type"] = self.artifact_type.value
        return data


@dataclass
class Component:
    name: str
    component_type: str
    artifacts: List[Artifact] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "component_type": self.component_type,
            "artifacts": [art.to_dict() for art in self.artifacts],
        }


@dataclass
class ModelUnit:
    id: str
    name: str
    unit_type: str  # 'pipeline', 'monolithic', 'single_checkpoint', 'lora', 'gguf'
    task_type: str = "unknown"
    framework: str = "unknown"
    precision: str = "unknown"
    quantization: Optional[str] = None
    base_model_name_or_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    components: List[Component] = field(default_factory=list)
    direct_artifacts: List[Artifact] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "unit_type": self.unit_type,
            "task_type": self.task_type,
            "framework": self.framework,
            "precision": self.precision,
            "quantization": self.quantization,
            "base_model_name_or_path": self.base_model_name_or_path,
            "metadata": self.metadata,
            "components": [comp.to_dict() for comp in self.components],
            "direct_artifacts": [art.to_dict() for art in self.direct_artifacts],
        }


@dataclass
class RepositoryNormalized:
    repo_id: str
    commit_sha: str
    gated: bool = False
    units: List[ModelUnit] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "commit_sha": self.commit_sha,
            "gated": self.gated,
            "units": [unit.to_dict() for unit in self.units],
        }
