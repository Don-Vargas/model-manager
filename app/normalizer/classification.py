import re
from pathlib import Path
from typing import List, Optional
from app.normalizer.models import ArtifactType


class FileClassifier:
    """Aplica la taxonomía explícita a los archivos del repositorio."""

    DOC_EXTENSIONS = {".md", ".txt", ".rst"}
    DOC_NAMES = {"license", "notice", "changelog", "code_of_conduct"}
    DIFFUSERS_COMPONENT_DIRS = {
        "unet",
        "vae",
        "text_encoder",
        "text_encoder_2",
        "transformer",
        "image_encoder",
        "safety_checker",
    }
    QUANT_PATTERN = re.compile(r"(Q\d+_[KMAZ_0-9]+|IQ\d+_[A-Z0-9]+|FP\d+_\d+|BF16|FP16|Q\d+_\d+)", re.IGNORECASE)

    @classmethod
    def classify_filename(cls, filename: str) -> ArtifactType:
        path = Path(filename)
        name_lower = path.name.lower()
        ext_lower = path.suffix.lower()

        # Archivos GGUF
        if ext_lower == ".gguf":
            return ArtifactType.GGUF

        # Adaptadores LoRA
        if name_lower in {"adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"} or "lora" in name_lower:
            if ext_lower in {".safetensors", ".bin", ".pt"}:
                return ArtifactType.LORA_ADAPTER

        # Documentación
        if name_lower.startswith("readme") or path.stem.lower() in cls.DOC_NAMES:
            return ArtifactType.DOCUMENTATION
        if ext_lower in cls.DOC_EXTENSIONS and "tokenizer" not in name_lower:
            return ArtifactType.DOCUMENTATION

        # Metadata de repositorio / Git
        if name_lower in {".gitattributes", ".gitignore", "model_info.json"}:
            return ArtifactType.METADATA

        # Código ejecutable / Plantillas
        if ext_lower in {".py", ".jinja", ".sh"}:
            return ArtifactType.RUNTIME_CODE

        # Tokenizer / Processor
        if "tokenizer" in name_lower or name_lower in {"vocab.json", "merges.txt", "spiece.model"}:
            return ArtifactType.TOKENIZER
        if "processor" in name_lower:
            return ArtifactType.PROCESSOR

        # Alternate runtime formats in Diffusers components. These are
        # recognized formats, but not artifacts used by this service's stack.
        is_diffusers_component = any(
            part.lower() in cls.DIFFUSERS_COMPONENT_DIRS
            for part in path.parts[:-1]
        )
        if is_diffusers_component and ext_lower in {
            ".msgpack",
            ".xml",
            ".onnx_data",
        }:
            return ArtifactType.ALTERNATE_RUNTIME

        # Pesos de modelos
        if ext_lower in {".safetensors", ".bin", ".pt", ".pth", ".onnx"}:
            return ArtifactType.WEIGHTS

        # Configuraciones
        if ext_lower == ".json":
            return ArtifactType.CONFIG

        return ArtifactType.UNKNOWN

    @classmethod
    def extract_gguf_quantization(cls, filename: str) -> Optional[str]:
        """Extrae el esquema de cuantización del nombre del archivo GGUF (ej: Q4_K_M, Q8_0)."""
        match = cls.QUANT_PATTERN.search(Path(filename).name)
        if match:
            return match.group(1).upper()
        return None

    @classmethod
    def derive_precision(cls, filenames: List[str]) -> str:
        text_lower = " ".join(filenames).lower()
        if "fp8" in text_lower:
            return "fp8"
        if "fp16" in text_lower:
            return "fp16"
        if "bf16" in text_lower:
            return "bf16"
        return "unknown"
