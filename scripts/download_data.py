"""Download and extract the SDNET2018 dataset from Mendeley Data.

Usage:
    python scripts/download_data.py --output ./data
"""

import argparse
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path


SDNET_URL = (
    "https://data.mendeley.com/public-files/datasets/z6n8jg7bky/files/"
    "1b36e246-6fce-4042-9b3f-44bb85278e6b/file_downloaded"
)
# Alternative: use the Mendeley API or manual download
GDRIVE_ID = "1xSL5mON2PbhVTsLEA5b9F9OjMH9Y7RyG"  # community mirror (may change)


def download_with_progress(url: str, dest: Path) -> None:
    print(f"Downloading {url}")

    def _progress(count, block_size, total_size):
        if total_size > 0:
            pct = count * block_size * 100 / total_size
            bar = "#" * int(pct / 2)
            sys.stdout.write(f"\r[{bar:<50}] {pct:5.1f}%")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()


def download_gdrive(file_id: str, dest: Path) -> None:
    try:
        import gdown
    except ImportError:
        print("gdown not installed. Run: pip install gdown")
        sys.exit(1)
    gdown.download(id=file_id, output=str(dest), quiet=False)


def extract(archive: Path, out_dir: Path) -> None:
    print(f"Extracting {archive} → {out_dir}")
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SDNET2018 dataset")
    parser.add_argument("--output", default="./data", help="Output directory")
    parser.add_argument("--source", choices=["mendeley", "gdrive"], default="gdrive")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "SDNET2018.zip"
    dataset_dir = out_dir / "SDNET2018"

    if dataset_dir.exists():
        print(f"Dataset already exists at {dataset_dir}")
        return

    if not zip_path.exists():
        if args.source == "gdrive":
            download_gdrive(GDRIVE_ID, zip_path)
        else:
            download_with_progress(SDNET_URL, zip_path)
    else:
        print(f"Archive already downloaded: {zip_path}")

    extract(zip_path, out_dir)

    # Clean up archive to save space
    os.remove(zip_path)
    print(f"\nDataset ready at {dataset_dir}")

    # Quick sanity check
    expected = ["D", "P", "W"]
    missing = [d for d in expected if not (dataset_dir / d).exists()]
    if missing:
        print(f"WARNING: Missing directories: {missing}")
        print("Please check the extraction or download manually from:")
        print("  https://data.mendeley.com/datasets/z6n8jg7bky/2")
    else:
        total = sum(1 for _ in dataset_dir.rglob("*.jpg"))
        print(f"Total images found: {total}")


if __name__ == "__main__":
    main()
