#!/usr/bin/env python3
"""
Upload SpaceOmicsBench v2.1 to Hugging Face Hub as a dataset.

Usage:
    python upload_to_huggingface.py --token hf_xxxxx
    python upload_to_huggingface.py --card-only
    python upload_to_huggingface.py --dry-run

By default, this uses HF_TOKEN from the environment or a cached Hugging Face
login. Use --card-only when only README/assets need to be refreshed.
"""

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent
REPO_ID = "jang1563/SpaceOmicsBench"
CARD_PATH = PROJECT_ROOT / "docs" / "hf_dataset_card.md"
CARD_ASSETS = [
    (PROJECT_ROOT / "docs" / "assets" / "spaceomicsbench_summary.png", "assets/spaceomicsbench_summary.png"),
]


def get_card_files():
    """Return list of (local_path, repo_path) tuples for the HF card surface."""
    files = [(CARD_PATH, "README.md")]
    files.extend(CARD_ASSETS)
    return files


def get_upload_files():
    """Return list of (local_path, repo_path) tuples to upload."""
    files = []

    # data/processed/*.csv
    processed_dir = PROJECT_ROOT / "data" / "processed"
    for csv_file in sorted(processed_dir.glob("*.csv")):
        files.append((csv_file, f"data/processed/{csv_file.name}"))

    # tasks/*.json (top-level only, skip legacy/)
    tasks_dir = PROJECT_ROOT / "tasks"
    for task_file in sorted(tasks_dir.glob("*.json")):
        files.append((task_file, f"tasks/{task_file.name}"))

    # splits/*.json (top-level only, skip legacy/)
    splits_dir = PROJECT_ROOT / "splits"
    for split_file in sorted(splits_dir.glob("*.json")):
        files.append((split_file, f"splits/{split_file.name}"))

    # evaluation/llm/question_bank.json
    qb_path = PROJECT_ROOT / "evaluation" / "llm" / "question_bank.json"
    if qb_path.exists():
        files.append((qb_path, "evaluation/llm/question_bank.json"))

    # results/v2.1/*.json
    results_dir = PROJECT_ROOT / "results" / "v2.1"
    for result_file in sorted(results_dir.glob("*.json")):
        files.append((result_file, f"results/v2.1/{result_file.name}"))

    # baselines/baseline_results.json
    baseline_path = PROJECT_ROOT / "baselines" / "baseline_results.json"
    if baseline_path.exists():
        files.append((baseline_path, "baselines/baseline_results.json"))

    return files


def validate_files(files):
    missing = [str(local_path) for local_path, _ in files if not local_path.exists()]
    if missing:
        print("ERROR: missing files:")
        for path in missing:
            print(f"  {path}")
        sys.exit(1)


def print_file_list(label, files):
    size_mb = sum(local_path.stat().st_size for local_path, _ in files) / 1024 / 1024
    print(f"{label}: {len(files)} files, {size_mb:.1f} MB")
    for local_path, repo_path in files:
        size_kb = local_path.stat().st_size / 1024
        print(f"  {repo_path:64s} {size_kb:8.1f} KB")


def upload_commit(api, repo_id, files, commit_message):
    from huggingface_hub import CommitOperationAdd

    validate_files(files)
    operations = [
        CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=str(local_path))
        for local_path, repo_path in files
    ]
    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message=commit_message,
    )


def main():
    parser = argparse.ArgumentParser(description="Upload SpaceOmicsBench to Hugging Face Hub")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face API token (or set HF_TOKEN env var)")
    parser.add_argument("--repo-id", type=str, default=REPO_ID, help=f"HF repo ID (default: {REPO_ID})")
    parser.add_argument("--private", action="store_true", help="Make repository private")
    parser.add_argument("--dry-run", action="store_true", help="List files without uploading")
    parser.add_argument("--card-only", action="store_true", help="Upload only README.md and card assets")
    parser.add_argument("--skip-card", action="store_true", help="Upload dataset files without refreshing README/assets")
    args = parser.parse_args()

    if args.card_only and args.skip_card:
        parser.error("--card-only cannot be combined with --skip-card")

    card_files = [] if args.skip_card else get_card_files()
    data_files = [] if args.card_only else get_upload_files()
    all_files = card_files + data_files
    validate_files(all_files)

    print(f"SpaceOmicsBench -> {args.repo_id}")
    if card_files:
        print_file_list("Card surface", card_files)
    if data_files:
        print_file_list("Dataset payload", data_files)
    print()

    if args.dry_run:
        print("DRY RUN: no files uploaded.")
        return

    token = args.token or os.environ.get("HF_TOKEN")

    try:
        from huggingface_hub import HfApi, login
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    if token:
        login(token=token)
    else:
        print("No token provided. Trying cached login...")

    api = HfApi()

    try:
        user = api.whoami()
        print(f"Logged in as: {user['name']}")
    except Exception:
        print("ERROR: Not authenticated. Run with --token hf_xxx or set HF_TOKEN env var.")
        print("       Get token at: https://huggingface.co/settings/tokens")
        sys.exit(1)

    try:
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=args.private,
            exist_ok=True,
        )
        print(f"Repository ready: https://huggingface.co/datasets/{args.repo_id}")
    except Exception as exc:
        print(f"ERROR creating repo: {exc}")
        sys.exit(1)

    if card_files:
        print("\nUploading dataset card and assets...")
        upload_commit(
            api,
            args.repo_id,
            card_files,
            "Polish SpaceOmicsBench dataset card",
        )

    if not data_files:
        print("\nDone. Card surface uploaded to:")
        print(f"  https://huggingface.co/datasets/{args.repo_id}")
        return

    print(f"\nUploading {len(data_files)} dataset files...")
    batch_size = 20
    total_batches = (len(data_files) + batch_size - 1) // batch_size
    for i in range(0, len(data_files), batch_size):
        batch = data_files[i : i + batch_size]
        batch_num = i // batch_size + 1
        print(f"  Batch {batch_num}/{total_batches}: {batch[0][1]} ... {batch[-1][1]}")
        upload_commit(
            api,
            args.repo_id,
            batch,
            f"Upload SpaceOmicsBench v2.1 files (batch {batch_num}/{total_batches})",
        )

    print("\nDone. Dataset uploaded to:")
    print(f"  https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
