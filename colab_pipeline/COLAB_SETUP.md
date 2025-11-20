# Google Colab Setup – Complete Steps

Follow these steps **in order** to run the ADR prediction pipeline on Google Colab without uploading 13GB of datasets.

---

## Prerequisites
- Google account with Colab access
- GitHub repo (optional, but recommended for version control)

---

## Step 1: Upload Code to GitHub (Recommended)

1. Open PowerShell in `d:\adrReports`
2. Initialize git and push only the essential files:

```powershell
cd d:\adrReports
git init
git add colab_pipeline/ requirements_colab.txt *.csv
git commit -m "Initial commit: pipeline + CSVs"
git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPO>.git
git push -u origin main
```

**Files to include:**
- `colab_pipeline/` folder (train_all_data_model.py, download_datasets.py, README.md)
- `requirements_colab.txt`
- Small CSVs: `drug_embeddings_data.csv`, `drugs_side_effects_drugs_com.csv`, `formatted_training_data.csv`, `train.csv`, `unified_adr_dataset.csv`

**Exclude large folders** (add to `.gitignore`):
```
ADRtarget-master/
CT-ADE-main-dataset/
chemBlData/
faersData/
siderData/
pubChemData/
DrugMeNot-main/
Hybrid-Adverse-Drug-Reactions-Predictor-main/
ML Clinical Trials*/
RecSys23-ADRnet-main/
SIDER4-master-dataset/
```

---

## Step 2: Open Google Colab

Go to [https://colab.research.google.com](https://colab.research.google.com) and create a new notebook.

---

## Step 3: Clone Your Repository

```python
!git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPO>.git
%cd <YOUR_REPO>
```

---

## Step 4: Install Dependencies

```python
!pip install -q -r requirements_colab.txt
```

---

## Step 5: Download Public Datasets

Run the download script to fetch all public datasets (skips the 13GB upload):

```python
!python colab_pipeline/download_datasets.py
```

**Expected output:**
- Clones 6 GitHub repos (ADRtarget, CT-ADE, SIDER4, DrugMeNot, RecSys23-ADRnet, Hybrid-ADR-Predictor)
- Downloads SIDER raw TSV files
- Creates placeholder folders for ChEMBL/FAERS/PubChem with instructions

**Time:** ~5-10 minutes depending on network speed

---

## Step 6: (Optional) Upload Manual Datasets

If you need **ChEMBL fingerprints** or **FAERS data**:

1. In Colab, click folder icon → upload files to `/content/datasets/chemBlData/` or `/content/datasets/faersData/`
2. Or mount Google Drive if you've uploaded them there:

```python
from google.colab import drive
drive.mount('/content/drive')
# Then copy: !cp /content/drive/MyDrive/chembl_35.fps /content/datasets/chemBlData/
```

---

## Step 7: Run the Training Pipeline

```python
!python colab_pipeline/train_all_data_model.py \
    --data_root /content/datasets \
    --epochs 20 \
    --batch_size 64 \
    --use_rdkit \
    --use_gpu
```

**Arguments:**
- `--data_root /content/datasets` → points to downloaded datasets
- `--epochs 20` → number of training epochs (default: 10)
- `--batch_size 64` → batch size (adjust if OOM errors occur)
- `--use_rdkit` → enable RDKit molecular descriptors
- `--use_gpu` → use Colab GPU (automatic if available)

**Training time:** 30-60 minutes on Colab T4 GPU

---

## Step 8: Download Results

After training completes, download the model and metrics:

```python
from google.colab import files
files.download('adr_model.pth')
files.download('adr_scaler.pkl')
files.download('adr_label_binarizer.pkl')
files.download('training_metrics.json')
```

---

## Troubleshooting

### Out of Memory (OOM)
Reduce batch size:
```python
!python colab_pipeline/train_all_data_model.py --batch_size 32
```

### Missing Datasets
Check `/content/datasets/` structure:
```python
!ls -lh /content/datasets/
```

### GPU Not Detected
Enable GPU: Runtime → Change runtime type → Hardware accelerator → GPU (T4)

### Git Clone Fails (Private Repo)
Generate GitHub personal access token and use:
```python
!git clone https://<TOKEN>@github.com/<USER>/<REPO>.git
```

---

## Complete Command Sequence (Copy-Paste)

```python
# 1. Clone repo
!git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPO>.git
%cd <YOUR_REPO>

# 2. Install dependencies
!pip install -q -r requirements_colab.txt

# 3. Download datasets
!python colab_pipeline/download_datasets.py

# 4. Train model
!python colab_pipeline/train_all_data_model.py \
    --data_root /content/datasets \
    --epochs 20 \
    --batch_size 64 \
    --use_rdkit \
    --use_gpu

# 5. Download artifacts
from google.colab import files
files.download('adr_model.pth')
files.download('adr_scaler.pkl')
files.download('adr_label_binarizer.pkl')
files.download('training_metrics.json')
```

---

## Summary

✅ **No 13GB upload needed** – datasets download automatically  
✅ **GitHub-based workflow** – version control + easy Colab integration  
✅ **Complete pipeline** – from download to trained model in ~40 minutes  
✅ **GPU-accelerated** – uses Colab T4 for fast training  

**Total Colab runtime:** ~40-60 minutes (download: 5-10 min, training: 30-50 min)
