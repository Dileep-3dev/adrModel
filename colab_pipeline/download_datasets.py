#!/usr/bin/env python3
"""
download_datasets.py
Downloads all public ADR datasets directly in Colab to avoid large Drive uploads.
Run this once before training to populate /content/datasets/.
"""
import os
import subprocess
import urllib.request
import zipfile
import tarfile
from pathlib import Path


def run_cmd(cmd, cwd=None):
    """Execute shell command."""
    print(f"→ {cmd}")
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


def download_file(url, dest):
    """Download file with progress."""
    print(f"Downloading {url} → {dest}")
    urllib.request.urlretrieve(url, dest)


def extract_archive(archive_path, dest_dir):
    """Extract zip or tar archive."""
    print(f"Extracting {archive_path} → {dest_dir}")
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as z:
            z.extractall(dest_dir)
    elif archive_path.endswith(('.tar.gz', '.tgz')):
        with tarfile.open(archive_path, 'r:gz') as t:
            t.extractall(dest_dir)
    os.remove(archive_path)


def main():
    # Root for all datasets
    data_root = Path("/content/datasets")
    data_root.mkdir(parents=True, exist_ok=True)
    os.chdir(data_root)

    print("=" * 60)
    print("DOWNLOADING PUBLIC ADR DATASETS")
    print("=" * 60)

    # 1. ADRtarget (GitHub)
    print("\n[1/11] ADRtarget")
    run_cmd("git clone https://github.com/wrab12/ADRtarget.git ADRtarget-master")

    # 2. CT-ADE (GitHub)
    print("\n[2/11] CT-ADE")
    run_cmd("git clone https://github.com/tatonetti-lab/CT-ADE.git CT-ADE-main-dataset")

    # 3. SIDER4 (GitHub)
    print("\n[3/11] SIDER4")
    run_cmd("git clone https://github.com/cosylabiiit/SIDER4.git SIDER4-master-dataset")

    # 4. DrugMeNot (GitHub)
    print("\n[4/11] DrugMeNot")
    run_cmd("git clone https://github.com/Sandman-Ren/DrugMeNot.git DrugMeNot-main")

    # 5. RecSys23-ADRnet (GitHub)
    print("\n[5/11] RecSys23-ADRnet")
    run_cmd("git clone https://github.com/Applied-Machine-Learning-Lab/RecSys23-ADRnet.git RecSys23-ADRnet-main")

    # 6. Hybrid-ADR-Predictor (GitHub)
    print("\n[6/11] Hybrid-ADR-Predictor")
    run_cmd("git clone https://github.com/Huzaifg/Hybrid-Adverse-Drug-Reactions-Predictor.git Hybrid-Adverse-Drug-Reactions-Predictor-main")

    # 7. SIDER raw data (public FTP - German side effect resource)
    print("\n[7/11] SIDER raw data")
    sider_dir = data_root / "siderData"
    sider_dir.mkdir(exist_ok=True)
    sider_files = [
        "http://sideeffects.embl.de/media/download/meddra_all_se.tsv.gz",
        "http://sideeffects.embl.de/media/download/meddra_freq.tsv.gz",
        "http://sideeffects.embl.de/media/download/meddra_all_indications.tsv.gz",
        "http://sideeffects.embl.de/media/download/drug_names.tsv",
    ]
    for url in sider_files:
        fname = url.split('/')[-1]
        download_file(url, sider_dir / fname)
        if fname.endswith('.gz'):
            run_cmd(f"gzip -d {fname}", cwd=sider_dir)

    # 8. ChEMBL fingerprints (mock - replace with actual download if available)
    print("\n[8/11] ChEMBL data")
    chembl_dir = data_root / "chemBlData"
    chembl_dir.mkdir(exist_ok=True)
    # Placeholder: user must provide chembl_35.fps or download from ChEMBL FTP
    print("⚠️  ChEMBL files (chembl_35.fps, chembl_35_chemreps.txt) must be manually uploaded or downloaded from:")
    print("    ftp://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/")
    (chembl_dir / "README.txt").write_text(
        "Place chembl_35.fps and chembl_35_chemreps.txt here.\n"
        "Download from: ftp://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/"
    )

    # 9. FAERS (mock - actual FAERS downloads are multi-part quarterly ZIPs)
    print("\n[9/11] FAERS data")
    faers_dir = data_root / "faersData"
    faers_dir.mkdir(exist_ok=True)
    print("⚠️  FAERS data is large and requires manual download from:")
    print("    https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html")
    (faers_dir / "README.txt").write_text(
        "Download quarterly XML/ASCII files from FDA FAERS.\n"
        "https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html"
    )

    # 10. PubChem bulk data (mock)
    print("\n[10/11] PubChem data")
    pubchem_dir = data_root / "pubChemData"
    pubchem_dir.mkdir(exist_ok=True)
    print("⚠️  PubChem bulk download optional; scripts in CT-ADE fetch CIDs on demand.")
    (pubchem_dir / "README.txt").write_text(
        "PubChem CID details are fetched by CT-ADE scripts (b0_download_pubchem_cids.py).\n"
        "Bulk files can be downloaded from: ftp://ftp.ncbi.nlm.nih.gov/pubchem/"
    )

    # 11. Drugs.com reviews (already included in your workspace CSV)
    print("\n[11/11] Drugs.com reviews (included in workspace CSVs)")
    # drugs_side_effects_drugs_com.csv should be in colab_pipeline bundle

    print("\n" + "=" * 60)
    print("✅ PUBLIC DATASETS DOWNLOADED")
    print("=" * 60)
    print(f"\nDatasets root: {data_root}")
    print("\nManual uploads needed:")
    print("  • ChEMBL fingerprints → chemBlData/")
    print("  • FAERS quarterly files → faersData/ (if desired)")
    print("  • PubChem bulk (optional)")
    print("\nNext step: Run train_all_data_model.py with DATA_ROOT=/content/datasets")


if __name__ == "__main__":
    main()
