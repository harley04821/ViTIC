#!/usr/bin/env python3
"""Download ViTIC dataset from Google Drive.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --data_path /custom/path/data/General
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

FOLDER_ID = "1Q4hFk8uWErfUaJ6xD308k9QPRa5wg5ru"
DATASETS  = {"clothing": "amazon_clothing", "beauty": "amazon_beauty"}


def ensure_gdown():
    try:
        import gdown
        return gdown
    except ImportError:
        print("[+] gdown not found — installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gdown>=4.7"])
        import gdown
        return gdown


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./data/General",
                        help="directory where amazon_clothing/ and amazon_beauty/ will be placed")
    args = parser.parse_args()

    data_dir = Path(args.data_path)
    tmp_dir  = data_dir / ".tmp_vitIC_download"

    already = [name for name in DATASETS.values() if (data_dir / name).exists()]
    if len(already) == len(DATASETS):
        print("[✓] Dataset already exists — nothing to do.")
        return
    if already:
        print(f"[i] Partially present: {already}. Downloading missing datasets.")

    gdown = ensure_gdown()

    tmp_dir.mkdir(parents=True, exist_ok=True)
    print(f"[+] Downloading from Google Drive …  (folder id: {FOLDER_ID})")

    gdown.download_folder(
        id=FOLDER_ID,
        output=str(tmp_dir),
        quiet=False,
        use_cookies=False,
    )

    # gdown places contents in a sub-directory named after the Drive folder
    subdirs = [p for p in tmp_dir.iterdir() if p.is_dir()]
    src_root = subdirs[0] if len(subdirs) == 1 else tmp_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in DATASETS.items():
        src = src_root / src_name
        dst = data_dir / dst_name
        if not src.exists():
            print(f"[!] '{src_name}' not found in downloaded folder — skipping.")
            continue
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(src), str(dst))
        print(f"[✓] {dst}")

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("\n[✓] Done. Dataset layout:")
    for dst_name in DATASETS.values():
        p = data_dir / dst_name
        if p.exists():
            npy = list((p / "item_info").glob("*.npy")) if (p / "item_info").exists() else []
            print(f"    {p}/  ({len(npy)} embedding files)")


if __name__ == "__main__":
    main()
