import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "huggingface"


MODELS = {
    "image": [
        "stabilityai/stable-diffusion-xl-base-1.0",
        "black-forest-labs/FLUX.1-dev",
        "Qwen/Qwen-Image",
        "Qwen/Qwen-Image-Edit",
    ],
    "video": [
        "stabilityai/stable-video-diffusion-img2vid-xt",
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "Lightricks/LTX-Video",
    ],
    "code": [
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "Qwen/Qwen2.5-Coder-14B-Instruct",
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    ],
}


def safe_name(repo_id: str) -> str:
    """
    Convert a Hugging Face repo ID into a filesystem-safe name.
    """
    return repo_id.replace("/", "__")


def serialize(value):
    """
    Convert Hugging Face objects into JSON-serializable structures.
    """

    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {
            str(key): serialize(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            serialize(item)
            for item in value
        ]

    if hasattr(value, "__dict__"):
        return {
            str(key): serialize(item)
            for key, item in vars(value).items()
        }

    return str(value)


def collect_model(
    api: HfApi,
    category: str,
    repo_id: str,
):
    print()
    print("=" * 80)
    print(f"Collecting: {repo_id}")
    print(f"Category:  {category}")
    print("=" * 80)

    try:
        info = api.model_info(
            repo_id=repo_id,
            files_metadata=True,
        )
    except Exception as exc:
        print(f"ERROR: could not retrieve model info: {exc}")
        return False

    output_dir = (
        DATA_DIR
        / category
        / safe_name(repo_id)
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # ModelInfo
    # ------------------------------------------------------------------

    model_info_path = output_dir / "model_info.json"

    model_info = serialize(info)

    with model_info_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            model_info,
            file,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )

    print(f"Saved: {model_info_path}")

    # ------------------------------------------------------------------
    # model_index.json
    #
    # This is intentionally downloaded independently from model_info.
    # hf_hub_download downloads only this metadata file, not model
    # weights.
    # ------------------------------------------------------------------

    model_index_path = output_dir / "model_index.json"

    revision = getattr(info, "sha", None)

    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename="model_index.json",
            revision=revision,
        )

        with open(
            downloaded_path,
            "r",
            encoding="utf-8",
        ) as source:
            model_index = json.load(source)

        with model_index_path.open(
            "w",
            encoding="utf-8",
        ) as destination:
            json.dump(
                model_index,
                destination,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )

        print(f"Saved: {model_index_path}")

    except Exception as exc:
        print(
            "No model_index.json "
            f"or could not retrieve it: {exc}"
        )

    # ------------------------------------------------------------------
    # Collection metadata
    # ------------------------------------------------------------------

    collection_path = output_dir / "collection.json"

    collection = {
        "repo_id": repo_id,
        "category": category,
        "revision": revision,
        "collected_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    with collection_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            collection,
            file,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )

    print(f"Saved: {collection_path}")

    return True


def main():
    token = None

    api = HfApi(token=token)

    total = 0
    successful = 0

    for category, repositories in MODELS.items():

        for repo_id in repositories:
            total += 1

            if collect_model(
                api=api,
                category=category,
                repo_id=repo_id,
            ):
                successful += 1

    print()
    print("=" * 80)
    print("Collection finished")
    print("=" * 80)
    print(f"Successful: {successful}")
    print(f"Failed:     {total - successful}")
    print(f"Total:      {total}")
    print(f"Output:     {DATA_DIR}")


if __name__ == "__main__":
    main()