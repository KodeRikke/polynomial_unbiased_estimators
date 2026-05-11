from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REPORTS_DIR = PROJECT_ROOT / "reports"

REPORT_EXTENSIONS = {".aux", ".log", ".pdf", ".tex"}


def delete_matching_files(folder: Path, extensions: set[str]) -> None:
    """Delete files matching given extensions only in the direct folder, not subfolders."""
    if not folder.exists():
        print(f"Skipping missing folder: {folder}")
        return

    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in extensions:
            path.unlink()
            print(f"Deleted: {path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    delete_matching_files(REPORTS_DIR, REPORT_EXTENSIONS)
    print("Cleanup complete.")


if __name__ == "__main__":
    main()