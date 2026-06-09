from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

FILES_TO_DELETE = [
    DATA_DIR / "benchmark_events.jsonl",
    DATA_DIR / "dlq.jsonl",
    DATA_DIR / "live_traffic.json",
    DATA_DIR / "dlq_intercept.json",
    DATA_DIR / "benchmark_summary.json",
]


def main():
    print("\nResetting experiment runtime files...")
    print("=" * 80)

    deleted_count = 0
    skipped_count = 0

    for file_path in FILES_TO_DELETE:
        if file_path.exists():
            file_path.unlink()
            deleted_count += 1
            print(f"Deleted: {file_path}")
        else:
            skipped_count += 1
            print(f"Skipped, not found: {file_path}")

    print("=" * 80)
    print(f"Deleted files: {deleted_count}")
    print(f"Skipped files: {skipped_count}")
    print("Experiment reset completed.\n")


if __name__ == "__main__":
    main()
