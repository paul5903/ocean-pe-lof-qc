# -*- coding: utf-8 -*-
"""
Public Oceanographic Data Downloader:
Downloads open-access profiles from official GDAC Argo (Ifremer),
WOA23 (NOAA NCEI), and WOCE (UCSD CCHDO) repositories.
"""
import os
import urllib.request
from pathlib import Path
from typing import List
from . import config

PUBLIC_ARGO_URLS = [
    "https://data-argo.ifremer.fr/dac/csio/2901503/profiles/D2901503_001.nc",
    "https://data-argo.ifremer.fr/dac/csio/2901503/profiles/D2901503_002.nc",
    "https://data-argo.ifremer.fr/dac/csio/2901503/profiles/D2901503_003.nc"
]


def download_file(url: str, output_path: Path) -> bool:
    """Downloads a remote file with progress tracking."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"[CACHE] File already exists: {output_path.name}")
        return True

    print(f"[DOWNLOAD] Fetching {url} -> {output_path.name}...")
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Ocean-PE-LOF-QC/2.0 (Oceanographic Research Benchmark)"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp, open(output_path, "wb") as f:
            f.write(resp.read())
        print(f"[OK] Downloaded {output_path.name} ({output_path.stat().st_size / 1024:.1f} KB)")
        return True
    except Exception as e:
        print(f"[WARN] Failed to download {url}: {e}")
        return False


def download_public_sample_suite(target_dir: Path = None) -> List[Path]:
    """Downloads sample suite of public NetCDF datasets."""
    dest = target_dir or (config.DATA_DIR / "public_samples")
    dest.mkdir(parents=True, exist_ok=True)

    downloaded = []
    for url in PUBLIC_ARGO_URLS:
        fname = url.split("/")[-1]
        out_file = dest / fname
        if download_file(url, out_file):
            downloaded.append(out_file)
    return downloaded


if __name__ == "__main__":
    download_public_sample_suite()
