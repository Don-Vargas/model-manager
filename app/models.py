from enum import Enum


class ArtifactStatus(str, Enum):
    DISCOVERED = "discovered"
    DOWNLOADING = "downloading"
    AVAILABLE = "available"
    VERIFYING = "verifying"
    VERIFIED = "verified"
    FAILED = "failed"
    CORRUPTED = "corrupted"


class ArtifactFormat(str, Enum):
    TRANSFORMERS = "transformers"
    DIFFUSERS = "diffusers"
    GGUF = "gguf"
    PYTORCH = "pytorch"
    CHECKPOINT = "checkpoint"
    LORA = "lora"
