#!/usr/bin/env python3
"""
Audits the classification/grouping output of the model-manager normalizer.

Reads the JSON produced by `review_repositories.py --json <file>` (a list of
repository response objects, same shape as POST /repositories) and checks:

  1. No file path is assigned to more than one artifact within the same
     repository (a file should belong to exactly one artifact).
  2. Distribution of artifact_type values across all discovered artifacts.
  3. Any artifact classified as "unknown" is flagged explicitly, since that
     means the classifier didn't recognize the file type.
  4. Sharded artifacts have both files and an index_file set (sanity check
     that shard grouping didn't partially fail).

This does not call the API - it only inspects already-discovered JSON, so
it's safe to re-run as many times as you like.

Usage:
    python3 classification_audit.py /tmp/discovery_results.json
"""

import argparse
import json
import sys
from collections import Counter, defaultdict


def file_path(entry):
    return entry["path"] if isinstance(entry, dict) else entry


def iter_artifacts(repo):
    """Yields (location_label, artifact) for every artifact in a repo response."""
    for unit in repo.get("units", []):
        for art in unit.get("artifacts", []):
            yield f"unit:{unit['name']}", art
        for comp in unit.get("components", []):
            for art in comp.get("artifacts", []):
                yield f"unit:{unit['name']}/component:{comp['name']}", art


def audit_repo(repo):
    repo_id = repo["id"]
    problems = []
    type_counter = Counter()

    # file path -> list of (location, artifact_name) that claimed it
    file_owners = defaultdict(list)

    for location, art in iter_artifacts(repo):
        art_type = art.get("artifact_type", "unknown")
        type_counter[art_type] += 1

        if art_type == "unknown":
            problems.append(
                f"  UNKNOWN TYPE: {location} artifact '{art['name']}' "
                f"(id={art['id']}) files={art.get('files')}"
            )

        if art.get("is_sharded") and not art.get("index_file"):
            problems.append(
                f"  SHARDED W/O INDEX: {location} artifact '{art['name']}' "
                f"(id={art['id']}) is_sharded=true but index_file is empty"
            )

        for f in art.get("files", []):
            path = file_path(f)
            file_owners[path].append(f"{location}/{art['name']} (id={art['id']})")

    duplicated = {path: owners for path, owners in file_owners.items() if len(owners) > 1}
    for path, owners in duplicated.items():
        problems.append(f"  DUPLICATE FILE: '{path}' claimed by {len(owners)} artifacts:")
        for owner in owners:
            problems.append(f"      - {owner}")

    return repo_id, type_counter, problems, len(file_owners)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_path", help="path written by review_repositories.py --json")
    args = parser.parse_args()

    with open(args.json_path, "r", encoding="utf-8") as f:
        repos = json.load(f)

    if not isinstance(repos, list):
        print("Expected a JSON list of repository objects (as produced by "
              "review_repositories.py --json).", file=sys.stderr)
        sys.exit(2)

    grand_type_counter = Counter()
    total_problems = 0
    total_files_seen = 0

    for repo in repos:
        repo_id, type_counter, problems, files_seen = audit_repo(repo)
        total_files_seen += files_seen
        grand_type_counter.update(type_counter)

        print(f"\n{'=' * 78}")
        print(f"REPOSITORY: {repo_id}")
        print(f"{'=' * 78}")
        print("  artifact_type distribution:")
        for art_type, count in sorted(type_counter.items(), key=lambda x: -x[1]):
            marker = "  <-- REVIEW" if art_type == "unknown" else ""
            print(f"    {art_type}: {count}{marker}")

        if problems:
            print(f"\n  ISSUES FOUND ({len(problems)}):")
            for p in problems:
                print(p)
            total_problems += len(problems)
        else:
            print("\n  No issues found.")

    print(f"\n{'#' * 78}")
    print("GRAND TOTAL artifact_type distribution across all repositories:")
    for art_type, count in sorted(grand_type_counter.items(), key=lambda x: -x[1]):
        marker = "  <-- REVIEW" if art_type == "unknown" else ""
        print(f"  {art_type}: {count}{marker}")
    print(f"\nTotal unique files tracked across all artifacts: {total_files_seen}")
    print(f"Total issues found: {total_problems}")
    print(f"{'#' * 78}\n")

    if total_problems > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()