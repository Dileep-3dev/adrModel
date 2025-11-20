# Colab Training Pipeline

This folder contains `train_all_data_model.py`, a self-contained script that ingests **all 16
data assets** in the repository (five root CSVs + eleven supporting directories) and trains a
multi-label ADR predictor. The pipeline is designed for Google Colab so that you can
leverage GPUs/TPUs even if your local machine cannot handle the full corpus.

## Quick start (Colab)

1. Mount the Drive (or upload the repo zip) so that `/content/adrReports` mirrors the
   workspace structure shown in this repo.
2. Install dependencies and launch the trainer:

   ```python
   !pip install -r /content/adrReports/requirements_colab.txt
   !python /content/adrReports/colab_pipeline/train_all_data_model.py \
       --data-root /content/adrReports \
       --output-dir /content/adr_model_artifacts \
       --epochs 25 --batch-size 256 --max-auto-files 40
   ```

   Adjust paths if you keep the datasets elsewhere. The script automatically detects
   whether CUDA is available and will fall back to CPU for small test runs.

3. Artifacts:
   - `adr_multilabel_model.pt` – PyTorch weights for inference.
   - `adr_preprocessor.pkl` – Scaler, label list, text encoder name, auto-ingestion
     metadata.
   - `metrics.json` – Micro/macro F1, AUROC, AUPR, and best validation loss.
   - `validation_outputs.npz` – Validation logits/predictions (handy for threshold
     tuning or downstream ensembling).

## Data coverage

| Source | How it is used |
| --- | --- |
| `formatted_training_data.csv` | Base multi-label matrix (80 ADR labels) and seed numeric features (`drug_length`, `drug_words`). |
| `unified_adr_dataset.csv` | Aggregated severity, confidence, and source richness per normalized drug. |
| `drug_embeddings_data.csv` | Provides `num_known_adrs`, data-quality signals, and ADR list text for embedding. |
| `drugs_side_effects_drugs_com.csv` | Ratings, review counts, Rx/OTC flag, pregnancy class, and rich textual descriptions. |
| `train.csv` | Harvested with the same heuristic column matcher as auto-ingest to capture any extra `(drug, ADR)` pairs or annotations from the Kaggle-style training sheet; summarized into counts + text blobs per drug. |
| `siderData` | ATC + CID counts from `drug_atc.tsv` / `drug_names.tsv`. |
| `chemBlData` / `chemBIData` | Optional RDKit descriptors from `chembl_35_chemreps.txt` (skip with `--no-chemistry` if RDKit is unavailable). Both spellings are supported automatically. |
| Remaining research folders (`ADRtarget-master`, `CT-ADE-main-dataset`, `CT-ADE-main-dataset/CT-ADE-main-dataset`, `DrugMeNot-main`, `faersData`, `Hybrid-Adverse-Drug-Reactions-Predictor-main`, `ML Clinical Trials - Galeano and Paccanaro-dataset`, `pubChemData`, `RecSys23-ADRnet-main`, `SIDER4-master-dataset`, `siderData`, `chemBlData`, `chemBIData`) | Every CSV/TSV inside these directories is auto-scanned. Heuristic column matching (`drug`, `compound`, `side_effect`, `reaction`, etc.) extracts additional `(drug, ADR)` mentions which are concatenated into a text blob per drug. This ensures no dataset is left unused even if schemas differ. |

You can override or extend the auto-ingested directories via `--auto-only-folders` if you
add new datasets later.

## Feature engineering & model

* Numeric features are scaled with `StandardScaler`.
* Text blobs (drug names + Drugs.com descriptions + embeddings + auto-ingested ADRs)
  are encoded with a configurable SentenceTransformer (defaults to
  `sentence-transformers/all-MiniLM-L6-v2`).
* Features are concatenated and fed into a three-layer fully-connected neural net with
  dropout, batch norm, and `BCEWithLogitsLoss` using label-wise positive weights.
* Metrics: micro/macro F1, micro AUROC, micro AUPR, plus per-label F1 diagnostics.

## Notable flags

- `--dry-run`: Build the dataset/preprocessor without training (useful for debugging).
- `--max-auto-files`: Cap the number of CSV/TSV files scanned per folder to balance
  runtime vs. coverage.
- `--max-auto-file-mb`: Skip overly large files when the environment cannot handle them.
- `--no-chemistry`: Disable RDKit-derived descriptors if RDKit is not installed.
- `--device`: Force `cpu`/`cuda`/`mps`.

## Next steps & tips

- Tune thresholds per ADR label using `validation_outputs.npz` for application-specific
  precision/recall trade-offs.
- To serve the model, load `adr_preprocessor.pkl`, recreate the SentenceTransformer by
  name, and apply the same feature-building logic (there is a `text_blob` column in the
  training frame to mimic at inference time).
- Extend `auto_ingest_from_directories` with dataset-specific parsers if you want more
  precise joins (e.g., mapping specific FAERS tables to DrugBank IDs).
