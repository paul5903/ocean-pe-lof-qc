import os
import ssl
import json
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from .. import config

SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

HEADERS = {'User-Agent': 'Mozilla/5.0'}

def download_file(url: str, output_path: Path) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        return True
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30, context=SSL_CTX) as resp, open(output_path, "wb") as f:
            f.write(resp.read())
        return True
    except Exception as e:
        print(f"Download error {url}: {e}")
        return False

def download_argo_profiles(target_dir: Path = None) -> list:
    dest = target_dir or (config.DATA_DIR / "argo_profiles")
    dest.mkdir(parents=True, exist_ok=True)
    sample_urls = [
        "https://data-argo.ifremer.fr/dac/csio/2901503/profiles/D2901503_001.nc",
        "https://data-argo.ifremer.fr/dac/csio/2901503/profiles/D2901503_002.nc",
        "https://data-argo.ifremer.fr/dac/csio/2901503/profiles/D2901503_003.nc"
    ]
    downloaded = []
    for u in sample_urls:
        p = dest / u.split("/")[-1]
        if download_file(u, p):
            downloaded.append(p)
    return downloaded

if __name__ == "__main__":
    download_argo_profiles()
