from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PLOTS_DIR = PROJECT_ROOT / "plots"

PLOT_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf", ".svg"}


def delete_matching_files(folder: Path, extensions: set[str]) -> None:
    if not folder.exists():
        print(f"Skipping missing folder: {folder}")
        return

    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            path.unlink()
            print(f"Deleted: {path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    delete_matching_files(PLOTS_DIR, PLOT_EXTENSIONS)
    print("Cleanup complete.")


if __name__ == "__main__":
    main()