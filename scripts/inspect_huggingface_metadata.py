#!/usr/bin/env python3

import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/app/data/huggingface")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"ERROR reading {path}: {exc}")
        return None


def format_size(size):
    if size is None:
        return "unknown"

    size = float(size)

    units = [
        "B",
        "KiB",
        "MiB",
        "GiB",
        "TiB",
    ]

    for unit in units:
        if size < 1024:
            return f"{size:.2f} {unit}"

        size /= 1024

    return f"{size:.2f} PiB"


def get_filename(sibling):
    if isinstance(sibling, dict):
        return sibling.get("rfilename")

    if isinstance(sibling, str):
        return sibling

    return None


def get_size(sibling):
    if not isinstance(sibling, dict):
        return None

    return sibling.get("size")


# ---------------------------------------------------------------------------
# Variant detection
# ---------------------------------------------------------------------------

VARIANT_PATTERNS = {
    "fp8": r"(?i)(^|[-_.])fp8($|[-_.])",
    "fp16": r"(?i)(^|[-_.])fp16($|[-_.])",
    "bf16": r"(?i)(^|[-_.])bf16($|[-_.])",
    "fp32": r"(?i)(^|[-_.])fp32($|[-_.])",
    "int8": r"(?i)(^|[-_.])int8($|[-_.])",
    "int4": r"(?i)(^|[-_.])int4($|[-_.])",
    "gguf": r"(?i)\.gguf$",
    "gptq": r"(?i)(^|[-_.])gptq($|[-_.])",
    "awq": r"(?i)(^|[-_.])awq($|[-_.])",
    "distilled": r"(?i)(^|[-_.])distilled($|[-_.])",
    "dev": r"(?i)(^|[-_.])dev($|[-_.])",
    "base": r"(?i)(^|[-_.])base($|[-_.])",
    "instruct": r"(?i)(^|[-_.])instruct($|[-_.])",
    "chat": r"(?i)(^|[-_.])chat($|[-_.])",
    "lora": r"(?i)(^|[-_.])lora($|[-_.])",
    "onnx": r"(?i)\.onnx($|[-_.])",
    "openvino": r"(?i)(^|[-_.])openvino($|[-_.])",
}


def detect_variants(filenames):
    detected = set()

    for filename in filenames:
        for name, pattern in VARIANT_PATTERNS.items():
            if re.search(pattern, filename):
                detected.add(name)

    return sorted(detected)


# ---------------------------------------------------------------------------
# Component detection
# ---------------------------------------------------------------------------

KNOWN_COMPONENTS = {
    "transformer",
    "transformer_2",
    "unet",
    "vae",
    "vae_1_0",
    "vae_encoder",
    "vae_decoder",
    "text_encoder",
    "text_encoder_2",
    "tokenizer",
    "tokenizer_2",
    "scheduler",
    "processor",
    "image_encoder",
    "feature_extractor",
}


def detect_components(filenames):
    components = defaultdict(list)

    for filename in filenames:
        parts = Path(filename).parts

        if len(parts) < 2:
            continue

        component = parts[0]

        if component in KNOWN_COMPONENTS:
            components[component].append(filename)

    return dict(components)


# ---------------------------------------------------------------------------
# Safetensors information
# ---------------------------------------------------------------------------

def inspect_safetensors(filenames, siblings_by_name):
    safetensors = []

    for filename in filenames:
        if not filename.endswith(".safetensors"):
            continue

        sibling = siblings_by_name.get(filename, {})

        safetensors.append(
            {
                "filename": filename,
                "size": sibling.get("size"),
                "size_human": format_size(sibling.get("size")),
            }
        )

    return safetensors


# ---------------------------------------------------------------------------
# Index files
# ---------------------------------------------------------------------------

def inspect_index_files(repo_dir, filenames):
    indexes = []

    for filename in filenames:
        if not filename.endswith(".safetensors.index.json"):
            continue

        path = repo_dir / filename
        data = load_json(path)

        entry = {
            "filename": filename,
            "loaded": data is not None,
        }

        if isinstance(data, dict):

            metadata = data.get("metadata")

            if isinstance(metadata, dict):
                entry["metadata"] = metadata

            weight_map = data.get("weight_map")

            if isinstance(weight_map, dict):

                entry["weight_map_entries"] = len(weight_map)

                unique_files = sorted(set(weight_map.values()))

                entry["weight_files"] = unique_files

        indexes.append(entry)

    return indexes


# ---------------------------------------------------------------------------
# Config inspection
# ---------------------------------------------------------------------------

def inspect_configs(repo_dir, filenames):
    configs = []

    for filename in filenames:

        if not filename.endswith("config.json"):
            continue

        path = repo_dir / filename

        data = load_json(path)

        if not isinstance(data, dict):
            continue

        interesting = {}

        for key in [
            "architectures",
            "model_type",
            "torch_dtype",
            "transformers_version",
            "diffusers_version",
            "_class_name",
            "library_name",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "intermediate_size",
            "vocab_size",
            "cross_attention_dim",
            "in_channels",
            "out_channels",
            "sample_size",
        ]:
            if key in data:
                interesting[key] = data[key]

        configs.append(
            {
                "filename": filename,
                "fields": interesting,
            }
        )

    return configs


# ---------------------------------------------------------------------------
# model_index inspection
# ---------------------------------------------------------------------------

def inspect_model_index(repo_dir, filenames):

    if "model_index.json" not in filenames:
        return None

    path = repo_dir / "model_index.json"

    data = load_json(path)

    if data is None:
        return None

    return data


# ---------------------------------------------------------------------------
# Main model inspection
# ---------------------------------------------------------------------------

def inspect_model(model_info_path: Path):

    repo_dir = model_info_path.parent

    model_info = load_json(model_info_path)

    if not isinstance(model_info, dict):
        return

    model_id = model_info.get("id")

    siblings = model_info.get("siblings") or []

    filenames = []

    siblings_by_name = {}

    for sibling in siblings:

        filename = get_filename(sibling)

        if not filename:
            continue

        filenames.append(filename)

        if isinstance(sibling, dict):
            siblings_by_name[filename] = sibling

    filenames = sorted(filenames)

    # -----------------------------------------------------------------------
    # Basic information
    # -----------------------------------------------------------------------

    print()
    print("=" * 110)
    print(f"MODEL: {model_id}")
    print(f"PATH:  {repo_dir.relative_to(ROOT)}")
    print("=" * 110)

    print("\nBASIC INFORMATION")

    fields = [
        "id",
        "sha",
        "pipeline_tag",
        "library_name",
        "private",
        "gated",
        "disabled",
        "downloads",
        "likes",
        "created_at",
        "last_modified",
    ]

    for field in fields:

        value = model_info.get(field)

        if value is not None:
            print(f"  {field}: {value}")

    # -----------------------------------------------------------------------
    # Tags
    # -----------------------------------------------------------------------

    tags = model_info.get("tags") or []

    print("\nTAGS")

    for tag in tags:
        print(f"  {tag}")

    # -----------------------------------------------------------------------
    # File statistics
    # -----------------------------------------------------------------------

    print("\nFILE STATISTICS")

    extensions = Counter()
    total_size = 0
    sized_files = 0

    for filename in filenames:

        suffix = Path(filename).suffix.lower()

        if not suffix:
            suffix = "<none>"

        extensions[suffix] += 1

        sibling = siblings_by_name.get(filename)

        if isinstance(sibling, dict):

            size = sibling.get("size")

            if isinstance(size, int):
                total_size += size
                sized_files += 1

    print(f"  total files: {len(filenames)}")
    print(f"  files with size metadata: {sized_files}")
    print(f"  total size: {format_size(total_size)}")

    print("\nFILE EXTENSIONS")

    for extension, count in sorted(
        extensions.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        print(f"  {extension}: {count}")

    # -----------------------------------------------------------------------
    # Variants
    # -----------------------------------------------------------------------

    variants = detect_variants(filenames)

    print("\nPOSSIBLE VARIANTS")

    if variants:
        for variant in variants:
            print(f"  {variant}")
    else:
        print("  none detected")

    # -----------------------------------------------------------------------
    # Components
    # -----------------------------------------------------------------------

    components = detect_components(filenames)

    print("\nCOMPONENTS")

    if components:

        for component in sorted(components):

            files = components[component]

            print(
                f"  {component}: "
                f"{len(files)} files"
            )

    else:
        print("  none detected")

    # -----------------------------------------------------------------------
    # Safetensors
    # -----------------------------------------------------------------------

    safetensors = inspect_safetensors(
        filenames,
        siblings_by_name,
    )

    print("\nSAFETENSORS")

    if safetensors:

        for item in safetensors:

            print(
                f"  {item['filename']} "
                f"({item['size_human']})"
            )

    else:
        print("  none")

    # -----------------------------------------------------------------------
    # Safetensors indexes
    # -----------------------------------------------------------------------

    indexes = inspect_index_files(
        repo_dir,
        filenames,
    )

    print("\nSAFETENSORS INDEX FILES")

    if indexes:

        for index in indexes:

            print(f"\n  {index['filename']}")

            if "metadata" in index:
                print(
                    f"    metadata: "
                    f"{index['metadata']}"
                )

            if "weight_map_entries" in index:
                print(
                    f"    weight map entries: "
                    f"{index['weight_map_entries']}"
                )

            for weight_file in index.get(
                "weight_files",
                [],
            ):
                print(
                    f"      {weight_file}"
                )

    else:
        print("  none")

    # -----------------------------------------------------------------------
    # Configs
    # -----------------------------------------------------------------------

    configs = inspect_configs(
        repo_dir,
        filenames,
    )

    print("\nCONFIG FILES")

    if configs:

        for config in configs:

            print(f"\n  {config['filename']}")

            for key, value in config["fields"].items():

                print(
                    f"    {key}: {value}"
                )

    else:
        print("  none")

    # -----------------------------------------------------------------------
    # model_index.json
    # -----------------------------------------------------------------------

    model_index = inspect_model_index(
        repo_dir,
        filenames,
    )

    print("\nMODEL INDEX")

    if model_index is None:

        print("  none")

    else:

        print(
            json.dumps(
                model_index,
                indent=2,
                ensure_ascii=False,
            )
        )

    # -----------------------------------------------------------------------
    # Directory tree
    # -----------------------------------------------------------------------

    print("\nFILE TREE")

    root_files = []
    directories = defaultdict(list)

    for filename in filenames:

        parts = Path(filename).parts

        if len(parts) == 1:
            root_files.append(filename)

        else:
            directories[parts[0]].append(filename)

    for filename in root_files:
        print(f"  {filename}")

    for directory in sorted(directories):

        print(f"\n  [{directory}]")

        for filename in sorted(directories[directory]):
            print(f"    {filename}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():

    if not ROOT.exists():

        raise SystemExit(
            f"Metadata directory does not exist: {ROOT}"
        )

    model_infos = sorted(
        ROOT.rglob("model_info.json")
    )

    if not model_infos:

        raise SystemExit(
            f"No model_info.json files found under {ROOT}"
        )

    print("=" * 110)
    print("HUGGING FACE METADATA INSPECTOR")
    print("=" * 110)
    print(f"Root: {ROOT}")
    print(f"Models: {len(model_infos)}")

    for model_info_path in model_infos:
        inspect_model(model_info_path)

    print()
    print("=" * 110)
    print("INSPECTION FINISHED")
    print("=" * 110)


if __name__ == "__main__":
    main()