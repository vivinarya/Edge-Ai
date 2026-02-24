import sys
import requests
import zipfile
import io
import os
import pandas as pd
import numpy as np
from pathlib import Path

ZENODO_URL = "https://zenodo.org/records/10963339/files/Battery%20Dataset.zip?download=1"
ROOT = Path("c:/gru")
DATA_DIR = ROOT / "data" / "xjtu_real"

def download_and_extract():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Downloading 2.4GB XJTU Dataset from Zenodo... this will take a few minutes!")
    # Stream the download
    with requests.get(ZENODO_URL, stream=True) as r:
        r.raise_for_status()
        zip_path = DATA_DIR / "xjtu_full.zip"
        with open(zip_path, 'wb') as f:
            downloaded = 0
            for chunk in r.iter_content(chunk_size=8192*100):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if downloaded % (1024*1024*100) < 8192*100:  # print every 100MB
                        print(f"  Downloaded {downloaded / 1024 / 1024:.1f} MB...")
                        
    print("Download complete. Extracting CSV files...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        # Find just a few CSV files to parse (e.g., Battery 1)
        csv_files = [m for m in z.namelist() if m.endswith('.csv') and not m.startswith('__MACOSX')]
        print(f"Found {len(csv_files)} total CSV files in archive.")
        
        # Let's just extract the first 3 batteries (e.g. 1_1, 1_2, 2_1) to save time, or all of them
        target_files = csv_files[:5] 
        for name in target_files:
            print(f"Extracting {name}...")
            z.extract(name, DATA_DIR)
            
    print("Done extracting target files. Running the parser...")

if __name__ == "__main__":
    download_and_extract()
