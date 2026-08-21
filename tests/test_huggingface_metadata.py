# tests/test_huggingface_metadata.py
import pytest
import json
from pathlib import Path
from typing import Optional

from app.normalizer.classification import FileClassifier
from app.normalizer.models import ArtifactType
from app.normalizer.normalizer import HuggingFaceMetadataNormalizer


# Ruta base hacia los datos dentro del contenedor
DATA_DIR = Path("/app/data/huggingface")


@pytest.fixture(scope="module")
def assert_data_dir_exists():
    assert DATA_DIR.exists(), f"El directorio de datos {DATA_DIR} no existe."


def test_detect_sharded_transformers_model(assert_data_dir_exists):
    """
    Caso: Qwen2.5-Coder-14B-Instruct
    Verifica que un modelo monolítico sharded se agrupe correctamente en 1 solo ModelUnit
    con su artefacto sharded de 6 archivos y su índice.
    """
    repo_path = DATA_DIR / "code/Qwen__Qwen2.5-Coder-14B-Instruct"
    if not repo_path.exists():
        pytest.skip(f"Fixture no encontrada en: {repo_path}")

    normalizer = HuggingFaceMetadataNormalizer(repo_path)
    res = normalizer.normalize()

    assert res is not None
    assert res.repo_id == "Qwen/Qwen2.5-Coder-14B-Instruct"
    assert len(res.units) == 1

    unit = res.units[0]
    assert unit.unit_type == "monolithic"
    
    # Debe contener 1 artefacto sharded de pesos
    sharded_artifacts = [a for a in unit.direct_artifacts if a.is_sharded]
    assert len(sharded_artifacts) == 1
    assert sharded_artifacts[0].index_file == "model.safetensors.index.json"
    assert len(sharded_artifacts[0].files) == 6


def test_parse_diffusers_pipeline_components(assert_data_dir_exists):
    """
    Caso: Qwen-Image
    Verifica la detección de un pipeline Diffusers clásico con sus componentes principales.
    """
    repo_path = DATA_DIR / "image/Qwen__Qwen-Image"
    if not repo_path.exists():
        pytest.skip(f"Fixture no encontrada en: {repo_path}")

    normalizer = HuggingFaceMetadataNormalizer(repo_path)
    res = normalizer.normalize()

    assert res is not None
    assert len(res.units) == 1

    unit = res.units[0]
    assert unit.unit_type == "pipeline"

    component_names = {c.name for c in unit.components}
    expected_components = {"scheduler", "text_encoder", "tokenizer", "transformer", "vae"}
    assert expected_components.issubset(component_names)


def test_identify_multi_checkpoint_repository(assert_data_dir_exists):
    """
    Caso: Lightricks/LTX-Video
    Verifica que se detecten tanto el pipeline Diffusers como los múltiples
    checkpoints independientes (.safetensors) en la raíz como ModelUnits separadas.
    """
    repo_path = DATA_DIR / "video/Lightricks__LTX-Video"
    if not repo_path.exists():
        pytest.skip(f"Fixture no encontrada en: {repo_path}")

    normalizer = HuggingFaceMetadataNormalizer(repo_path)
    res = normalizer.normalize()

    assert res is not None
    # Debe haber un pipeline y múltiples single_checkpoints
    assert len(res.units) > 1

    unit_types = [u.unit_type for u in res.units]
    assert "pipeline" in unit_types
    assert "single_checkpoint" in unit_types

    # Verificar que las variantes con fp8 en el nombre identifiquen la precisión
    fp8_units = [u for u in res.units if "fp8" in u.name]
    for unit in fp8_units:
        assert unit.precision == "fp8"


def test_extract_auxiliary_scripts_and_templates(assert_data_dir_exists):
    """
    Caso: Qwen3-Coder-30B-A3B-Instruct
    Verifica que los archivos .jinja y .py se clasifiquen como runtime_code y no como weights.
    """
    repo_path = DATA_DIR / "code/Qwen__Qwen3-Coder-30B-A3B-Instruct"
    if not repo_path.exists():
        pytest.skip(f"Fixture no encontrada en: {repo_path}")

    normalizer = HuggingFaceMetadataNormalizer(repo_path)
    res = normalizer.normalize()

    assert res is not None
    unit = res.units[0]

    runtime_artifacts = [
        a for a in unit.direct_artifacts if a.artifact_type == "runtime_code"
    ]
    
    # Confirmar que archivos .jinja o .py fueron categorizados como runtime_code
    runtime_files = [f for a in runtime_artifacts for f in a.files]
    assert any(f.endswith(".jinja") or f.endswith(".py") for f in runtime_files)


def test_flux_multiple_root_checkpoints(assert_data_dir_exists):
    """
    Caso: black-forest-labs/FLUX.1-dev
    Verifica que los checkpoints sueltos en la raíz (ae.safetensors y flux1-dev.safetensors)
    NO se colapsen en un solo modelo, sino en 2 ModelUnits independientes.
    """
    repo_path = DATA_DIR / "image/black-forest-labs__FLUX.1-dev"
    if not repo_path.exists():
        pytest.skip(f"Fixture no encontrada en: {repo_path}")

    normalizer = HuggingFaceMetadataNormalizer(repo_path)
    res = normalizer.normalize()

    assert res is not None
    assert len(res.units) == 2

    unit_names = {u.name for u in res.units}
    assert "ae" in unit_names
    assert "flux1-dev" in unit_names


def test_wan22_pipeline_metadata(assert_data_dir_exists):
    """
    Caso: Wan2.2-T2V-A14B-Diffusers
    REGLA DE NEGOCIO CRÍTICA: 'boundary_ratio' es una propiedad/configuración de pipeline,
    NO un componente. No debe aparecer dentro de unit.components.
    """
    repo_path = DATA_DIR / "video/Wan-AI__Wan2.2-T2V-A14B-Diffusers"
    if not repo_path.exists():
        pytest.skip(f"Fixture no encontrada en: {repo_path}")

    normalizer = HuggingFaceMetadataNormalizer(repo_path)
    res = normalizer.normalize()

    assert res is not None
    unit = res.units[0]

    component_names = [c.name for c in unit.components]
    
    # Esta aserción fallará con el código actual hasta refactorizar la lectura de model_index.json
    assert "boundary_ratio" not in component_names, (
        "'boundary_ratio' fue erróneamente clasificado como un Component. "
        "Debe pertenecer a los metadatos de configuración del pipeline."
    )


# ==============================================================================
# TESTS UNITARIOS DE CLASIFICACIÓN (FileClassifier)
# ==============================================================================

@pytest.mark.parametrize(
    "filename, expected_type",
    [
        # Documentación
        ("README.md", ArtifactType.DOCUMENTATION),
        ("LICENSE", ArtifactType.DOCUMENTATION),
        ("notice.txt", ArtifactType.DOCUMENTATION),
        ("changelog.rst", ArtifactType.DOCUMENTATION),
        # Metadata
        (".gitattributes", ArtifactType.METADATA),
        (".gitignore", ArtifactType.METADATA),
        ("model_info.json", ArtifactType.METADATA),
        # Runtime Code / Scripts
        ("chat_template.jinja", ArtifactType.RUNTIME_CODE),
        ("configuration_qwen2.py", ArtifactType.RUNTIME_CODE),
        ("entrypoint.sh", ArtifactType.RUNTIME_CODE),
        # Tokenizer / Processor
        ("tokenizer.json", ArtifactType.TOKENIZER),
        ("vocab.json", ArtifactType.TOKENIZER),
        ("preprocessor_config.json", ArtifactType.PROCESSOR),
        # Weights & Configs
        ("model.safetensors", ArtifactType.WEIGHTS),
        ("config.json", ArtifactType.CONFIG),
    ],
)
def test_file_classifier_direct(filename: str, expected_type: ArtifactType):
    """Verifica que FileClassifier asigne la taxonomía correcta según el patrón del archivo."""
    assert FileClassifier.classify_filename(filename) == expected_type


# ==============================================================================
# TEST DE INTEGRACIÓN: ARTEFACTOS EN REPOSITORIO REAL / FIXTURE
# ==============================================================================

def test_qwen_runtime_code_and_metadata_classification(assert_data_dir_exists):
    """
    Verifica que en un repositorio monolítico tipo Qwen los archivos auxiliares
    como .py y .jinja se clasifiquen como RUNTIME_CODE.
    """
    repo_path = DATA_DIR / "llm/Qwen__Qwen2.5-7B-Instruct"
    if not repo_path.exists():
        pytest.skip(f"Fixture no encontrada en: {repo_path}")

    normalizer = HuggingFaceMetadataNormalizer(repo_path)
    res = normalizer.normalize()

    assert res is not None
    assert len(res.units) == 1

    unit = res.units[0]
    artifact_types = {art.name: art.artifact_type for art in unit.direct_artifacts}

    # Verificar que los archivos jinja o py tengan runtime_code
    for art_name, art_type in artifact_types.items():
        if art_name.endswith((".py", ".jinja")):
            assert art_type == ArtifactType.RUNTIME_CODE, (
                f"El archivo {art_name} debería ser RUNTIME_CODE, pero fue {art_type}"
            )


def test_repository_normalized_json_serialization(assert_data_dir_exists):
    """
    Verifica que la salida de RepositoryNormalized se pueda serializar correctamente
    a JSON sin errores de tipos Enum o Dataclass.
    """
    repo_path = DATA_DIR / "video/Wan-AI__Wan2.2-T2V-A14B-Diffusers"
    if not repo_path.exists():
        pytest.skip(f"Fixture no encontrada en: {repo_path}")

    normalizer = HuggingFaceMetadataNormalizer(repo_path)
    res = normalizer.normalize()

    assert res is not None
    dict_data = res.to_dict()

    # Debe ser serializable a string JSON sin lanzar TypeError
    json_str = json.dumps(dict_data, indent=2)
    parsed = json.loads(json_str)

    assert parsed["repo_id"] == res.repo_id
    assert "units" in parsed


# ==============================================================================
# TESTS DE GGUF Y LORA ADAPTERS
# ==============================================================================

@pytest.mark.parametrize(
    "filename, expected_quant",
    [
        ("Llama-3-8B-Instruct.Q4_K_M.gguf", "Q4_K_M"),
        ("model.Q8_0.gguf", "Q8_0"),
        ("mistral-7b-v0.1.IQ3_XS.gguf", "IQ3_XS"),
        ("model.fp16.gguf", "FP16"),
        ("unnamed_model.gguf", None),
    ],
)
def test_extract_gguf_quantization(filename: str, expected_quant: Optional[str]):
    assert FileClassifier.extract_gguf_quantization(filename) == expected_quant


@pytest.mark.parametrize(
    "filename, expected_type",
    [
        ("adapter_model.safetensors", ArtifactType.LORA_ADAPTER),
        ("adapter_config.json", ArtifactType.CONFIG),
        ("my_lora_weights.safetensors", ArtifactType.LORA_ADAPTER),
        ("model.Q4_K_M.gguf", ArtifactType.GGUF),
    ],
)
def test_lora_and_gguf_classification(filename: str, expected_type: ArtifactType):
    assert FileClassifier.classify_filename(filename) == expected_type
