#!/usr/bin/env python3
"""
Discover a list of Hugging Face repositories via the model-manager API and
print a size-annotated tree (units -> components -> artifacts) plus
per-repo and grand totals.

This performs discovery only (POST /repositories) - it never triggers a
download. Use it to review what a full download would cost before running
any download commands.

Usage:
    python3 review_repositories.py [--base-url http://localhost:8000] [--json out.json]
"""

import argparse
import json
import sys
import urllib.request
import urllib.error


REPOS = [
    "google/bert_uncased_L-2_H-128_A-2",
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "Qwen/Qwen-Image",
    "stabilityai/stable-video-diffusion-img2vid-xt",
    "black-forest-labs/FLUX.1-dev",
    "Lightricks/LTX-Video",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "Qwen/Qwen2.5-Coder-14B-Instruct",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct",
]


def human_size(num_bytes):
    if num_bytes is None:
        return "unknown"
    size = float(num_bytes)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PiB"


def post_repository(base_url, repo_id, timeout=600):
    url = f"{base_url}/repositories"
    payload = json.dumps({"repo_id": repo_id}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8")), None
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return None, f"HTTP {exc.code}: {body}"
    except urllib.error.URLError as exc:
        return None, f"connection error: {exc}"


def artifact_size(artifact):
    return artifact.get("size_bytes") or 0


def sum_artifacts(artifacts):
    return sum(artifact_size(a) for a in artifacts)


def sum_unit(unit):
    total = sum_artifacts(unit.get("artifacts", []))
    for comp in unit.get("components", []):
        total += sum_artifacts(comp.get("artifacts", []))
    return total


def print_repo_tree(repo):
    repo_total = 0
    print(f"\n{'=' * 78}")
    print(f"REPOSITORY: {repo['id']}")
    print(f"{'=' * 78}")

    for unit in repo.get("units", []):
        unit_total = sum_unit(unit)
        repo_total += unit_total
        print(f"  [{unit['unit_type']}] {unit['name']}  ({human_size(unit_total)})")

        for art in unit.get("artifacts", []):
            print(f"      - {art['name']}  ({human_size(artifact_size(art))})"
                  f"{'  [sharded]' if art.get('is_sharded') else ''}")

        for comp in unit.get("components", []):
            comp_total = sum_artifacts(comp.get("artifacts", []))
            print(f"      [component] {comp['name']} ({comp['component_type']})  "
                  f"({human_size(comp_total)})")
            for art in comp.get("artifacts", []):
                print(f"          - {art['name']}  ({human_size(artifact_size(art))})"
                      f"{'  [sharded]' if art.get('is_sharded') else ''}")

    print(f"  {'-' * 74}")
    print(f"  REPO TOTAL: {human_size(repo_total)}")
    return repo_total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--json", default=None, help="optional path to dump raw results as JSON")
    parser.add_argument("--repos", nargs="*", default=None, help="override the default repo list")
    args = parser.parse_args()

    repo_ids = args.repos or REPOS
    grand_total = 0
    failures = []
    raw_results = []

    for repo_id in repo_ids:
        repo, error = post_repository(args.base_url, repo_id)
        if error:
            print(f"\n{'=' * 78}")
            print(f"REPOSITORY: {repo_id}")
            print(f"{'=' * 78}")
            print(f"  FAILED: {error}")
            failures.append((repo_id, error))
            continue

        raw_results.append(repo)
        grand_total += print_repo_tree(repo)

    print(f"\n{'#' * 78}")
    print(f"GRAND TOTAL across {len(repo_ids) - len(failures)} successful "
          f"repositories: {human_size(grand_total)}")
    if failures:
        print(f"\n{len(failures)} repositories FAILED to import:")
        for repo_id, error in failures:
            print(f"  - {repo_id}: {error}")
    print(f"{'#' * 78}\n")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(raw_results, f, indent=2)
        print(f"Raw discovery results written to {args.json}")

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()