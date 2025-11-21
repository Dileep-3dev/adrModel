#!/usr/bin/env python3
"""Unified ADR training pipeline for Google Colab.

This script ingests every dataset that ships with the adrReports workspace (all five
root CSV/Excel tables plus eleven supporting folders), learns cross-dataset features, and trains a
multi-label neural network that predicts adverse drug reactions. It is optimized for Google
Colab, but can also run locally.

High-level flow:
     1. Load core label matrix from ``formatted_training_data.csv``.
     2. Merge rich numeric/textual features from ``unified_adr_dataset.csv``,
         ``drug_embeddings_data.csv``, ``drugs_side_effects_drugs_com.csv``, ``train.csv``,
         ``siderData`` lookups, and optional chemistry fingerprints from ``chemBlData`` /
         ``chemBIData`` (if RDKit is installed).
     3. Auto-ingest every CSV/TSV that lives inside the remaining 10+ research folders
         (ADRtarget, CT-ADE, FAERS, PubChem, RecSys23-ADRnet, SIDER4, chemBlData, etc.) to
         harvest extra (drug, ADR) mentions. These are folded into a text blob per drug so
         that the model benefits from all corpora without hand-maintaining every schema.
     4. Encode text with a SentenceTransformer, scale numeric signals, train a
       multi-label neural network with BCE loss + label-specific class weights, and
       report micro/macro F1 + AUROC.
    5. Persist model weights and preprocessing assets for downstream inference.

Example usage on Colab (after mounting Drive with the repository contents)::

    !pip install -r requirements_colab.txt
    !python colab_pipeline/train_all_data_model.py \
        --data-root /content/drive/MyDrive/adrReports \
        --output-dir /content/adr_model_artifacts \
        --epochs 25 --batch-size 256 --max-auto-files 30

The script is defensive: when a dataset is missing or too large for the current machine,
that source is skipped with a warning but processing continues so that Colab sessions
remain smooth.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import math
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score, f1_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

try:  # sentence-transformers is heavy, import lazily.
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover - handled in CLI flow
    SentenceTransformer = None  # type: ignore
    logging.warning("sentence-transformers missing: install requirements_colab.txt")

try:  # Optional, used for chemBlData fingerprints.
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
except ImportError:  # pragma: no cover - RDKit optional
    Chem = None
    Descriptors = None
    rdMolDescriptors = None

# Keywords for auto-ingestion heuristics.
DRUG_COLUMN_KEYWORDS = [
    "drug", "compound", "product", "medicine", "therapy", "treatment", "name",
    "molecule", "agent", "substance"
]
ADR_COLUMN_KEYWORDS = [
    "adr", "side", "reaction", "event", "effect", "issue", "symptom",
    "tox", "injury", "complaint", "ae"
]

SEVERITY_MAP = {
    "unknown": 0.5,
    "mild": 0.33,
    "moderate": 0.66,
    "severe": 1.0,
    "serious": 0.9,
    "life-threatening": 1.0,
}

RANDOM_SEED = 1337
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def normalize_name(name: str) -> str:
    """Lowercase alphanumeric normalization to align drugs across datasets."""
    if not isinstance(name, str):
        return ""
    cleaned = ''.join(ch.lower() if ch.isalnum() else ' ' for ch in name)
    return ' '.join(cleaned.split())


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def safe_read_table(path: Path, *, nrows: Optional[int] = None,
                    sep: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Read CSV/TSV defensively, returning None on parser failures."""
    if not path.exists():
        logging.warning("Missing file: %s", path)
        return None
    if sep is None:
        if path.suffix.lower() == '.tsv':
            sep = '\t'
        elif path.suffix.lower() == '.csv':
            sep = ','
        else:
            sep = ','
    try:
        return pd.read_csv(
            path,
            sep=sep,
            nrows=nrows,
            encoding='utf-8',
            on_bad_lines='skip',
            low_memory=False,
        )
    except Exception as exc:  # pragma: no cover - resiliency path
        logging.warning("Failed to read %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Core dataset loaders
# ---------------------------------------------------------------------------

def load_formatted_training_data(path: Path) -> Tuple[pd.DataFrame, List[str]]:
    logging.info("Loading primary label matrix from %s", path)
    df = safe_read_table(path)
    if df is None or df.empty:
        raise ValueError(
            "formatted_training_data.csv is required and may not be empty."
        )
    label_cols = [c for c in df.columns if c.startswith('label_')]
    if not label_cols:
        raise ValueError("No label_ columns found in formatted_training_data.csv")
    df['drug_name_clean'] = df['drug_name'].map(normalize_name)
    logging.info("Loaded %d labeled drugs with %d ADR targets",
                 len(df), len(label_cols))
    return df, label_cols


def load_unified_stats(path: Path) -> pd.DataFrame:
    logging.info("Aggregating signals from unified_adr_dataset.csv")
    df = safe_read_table(path)
    if df is None:
        return pd.DataFrame()
    df['drug_name_clean'] = df['drug_name'].map(normalize_name)
    df['severity_score'] = df['severity'].fillna('unknown').map(SEVERITY_MAP).fillna(0.5)
    df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')
    agg = df.groupby('drug_name_clean').agg({
        'severity_score': ['mean', 'std'],
        'confidence': ['mean', 'std'],
        'adverse_reaction': 'nunique',
        'source': 'nunique',
        'dataset': 'nunique',
        'severity': 'count',
    })
    agg.columns = ['unified_severity_mean', 'unified_severity_std',
                   'unified_confidence_mean', 'unified_confidence_std',
                   'unified_unique_adrs', 'unified_source_count',
                   'unified_dataset_count', 'unified_event_count']
    agg = agg.reset_index()
    logging.info("Unified stats available for %d drugs", len(agg))
    return agg


def _parse_str_list(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    text = text.strip()
    if not text:
        return []
    try:
        data = ast.literal_eval(text)
        if isinstance(data, (list, tuple)):
            return [str(x) for x in data]
    except (ValueError, SyntaxError):
        pass
    return [text]


def load_drug_embeddings(path: Path) -> pd.DataFrame:
    logging.info("Loading drug_embeddings_data.csv features")
    df = safe_read_table(path)
    if df is None:
        return pd.DataFrame()
    df['drug_name_clean'] = df['drug_name_clean'].map(normalize_name)
    df['num_known_adrs'] = pd.to_numeric(df['num_known_adrs'], errors='coerce')
    df['avg_severity_embedding'] = df['severity'].apply(lambda xs: np.mean([
        SEVERITY_MAP.get(str(x).lower(), 0.5) for x in _parse_str_list(xs)
    ]) if isinstance(xs, str) else 0.5)
    df['embedding_adr_text'] = df['adr_clean'].apply(
        lambda x: ' '.join(_parse_str_list(x))
    )
    cols = ['drug_name_clean', 'num_known_adrs', 'avg_severity_embedding',
            'confidence', 'data_quality_score', 'embedding_adr_text']
    logging.info("Embedding metadata for %d drugs", df['drug_name_clean'].nunique())
    return df[cols].drop_duplicates('drug_name_clean')


def load_drugscom_features(path: Path) -> pd.DataFrame:
    logging.info("Loading drugs.com side effect metadata")
    df = safe_read_table(path)
    if df is None:
        return pd.DataFrame()
    df['drug_name_clean'] = df['drug_name'].map(normalize_name)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df['no_of_reviews'] = pd.to_numeric(df['no_of_reviews'], errors='coerce')
    df['activity_pct'] = pd.to_numeric(
        df['activity'].str.replace('%', '', regex=False), errors='coerce'
    )
    df['side_effects_text'] = (
        df['side_effects'].fillna('') + ' ' +
        df['medical_condition_description'].fillna('')
    )
    # Aggregate per drug to avoid duplicating label rows during feature merge.
    agg = df.groupby('drug_name_clean').agg({
        'rating': 'mean',
        'no_of_reviews': 'sum',
        'activity_pct': 'mean',
        'rx_otc': lambda x: next((v for v in x if isinstance(v, str) and v), ''),
        'pregnancy_category': lambda x: next((v for v in x if isinstance(v, str) and v), ''),
        'drug_classes': lambda x: ' '.join(sorted({str(v) for v in x if isinstance(v, str)})),
        'side_effects_text': lambda x: ' '.join(sorted({str(v) for v in x if isinstance(v, str)})),
        'medical_condition': lambda x: ' '.join(sorted({str(v) for v in x if isinstance(v, str)})),
        'medical_condition_description': lambda x: ' '.join(
            sorted({str(v) for v in x if isinstance(v, str)})
        ),
    }).reset_index()
    return agg


def load_sider_features(root: Path) -> pd.DataFrame:
    atc_path = root / 'drug_atc.tsv'
    names_path = root / 'drug_names.tsv'
    if not atc_path.exists() or not names_path.exists():
        logging.warning("siderData TSVs missing, skipping")
        return pd.DataFrame()
    atc = safe_read_table(atc_path, sep='\t')
    names = safe_read_table(names_path, sep='\t')
    if atc is None or names is None:
        return pd.DataFrame()
    names.columns = ['cid', 'drug_name']
    atc.columns = ['cid', 'atc_code']
    merged = names.merge(atc, on='cid', how='left')
    merged['drug_name_clean'] = merged['drug_name'].map(normalize_name)
    agg = merged.groupby('drug_name_clean').agg({
        'atc_code': 'nunique',
        'cid': 'nunique'
    }).rename(columns={
        'atc_code': 'sider_atc_count',
        'cid': 'sider_cid_count'
    }).reset_index()
    logging.info("SIDER features for %d drugs", len(agg))
    return agg


def load_chembl_features(root: Path, max_rows: int = 20000) -> pd.DataFrame:
    """Optionally derive simple descriptors from chembl SMILES."""
    if Chem is None:
        logging.warning("RDKit not installed; skipping chemBlData fingerprints")
        return pd.DataFrame()
    smiles_path = root / 'chembl_35_chemreps.txt'
    if not smiles_path.exists():
        logging.warning("chembl_35_chemreps.txt missing, skipping chemistry features")
        return pd.DataFrame()
    logging.info("Parsing chemistry descriptors (may take a while)...")
    # File structure: CHEMBLID\tSMILES\t...; only use first two columns.
    df = safe_read_table(smiles_path, nrows=max_rows, sep='\t')
    if df is None or df.empty:
        return pd.DataFrame()
    df.columns = [c.lower() for c in df.columns]
    chembl_name_col = None
    for col in df.columns:
        if 'name' in col or 'pref_name' in col:
            chembl_name_col = col
            break
    if chembl_name_col is None:
        chembl_name_col = df.columns[0]
    smiles_col = None
    for col in df.columns:
        if 'smile' in col:
            smiles_col = col
            break
    if smiles_col is None:
        logging.warning("Could not find SMILES column in chembl reps; skipping")
        return pd.DataFrame()
    records = []
    for _, row in df.iterrows():
        smiles = row[smiles_col]
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        if mol is None:
            continue
        num_rings = rdMolDescriptors.CalcNumRings(mol) if rdMolDescriptors else Chem.GetSSSR(mol)
        desc = {
            'chembl_num_atoms': float(mol.GetNumAtoms()),
            'chembl_num_rings': float(num_rings),
            'chembl_mol_wt': float(Descriptors.MolWt(mol)),
            'chembl_logp': float(Descriptors.MolLogP(mol)),
            'chembl_tpsa': float(Descriptors.TPSA(mol)),
        }
        name = row.get(chembl_name_col)
        desc['drug_name_clean'] = normalize_name(str(name))
        records.append(desc)
    if not records:
        logging.warning("No valid RDKit descriptors parsed; skipping chemistry features")
        return pd.DataFrame()
    logging.info("Computed RDKit descriptors for %d molecules", len(records))
    df_desc = pd.DataFrame(records)
    numeric_cols = [
        'chembl_num_atoms',
        'chembl_num_rings',
        'chembl_mol_wt',
        'chembl_logp',
        'chembl_tpsa',
    ]
    agg = df_desc.groupby('drug_name_clean')[numeric_cols].mean().reset_index()
    return agg


# ---------------------------------------------------------------------------
# Shared mention-harvesting helpers
# ---------------------------------------------------------------------------

def _identify_drug_adr_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    lower_cols = {col: col.lower() for col in df.columns}
    drug_cols = [col for col, low in lower_cols.items()
                 if any(key in low for key in DRUG_COLUMN_KEYWORDS)]
    adr_cols = [col for col, low in lower_cols.items()
                if any(key in low for key in ADR_COLUMN_KEYWORDS)]
    return drug_cols, adr_cols


def _harvest_mentions(df: pd.DataFrame, drug_cols: Sequence[str],
                      adr_cols: Sequence[str]) -> Dict[str, List[str]]:
    aggregated: Dict[str, List[str]] = defaultdict(list)
    subset = df[drug_cols + adr_cols].dropna()
    for _, row in subset.iterrows():
        for d_col in drug_cols:
            drug_token = normalize_name(row[d_col])
            if not drug_token:
                continue
            for a_col in adr_cols:
                adr_token = normalize_name(str(row[a_col]))
                if adr_token:
                    aggregated[drug_token].append(adr_token)
    return aggregated


# ---------------------------------------------------------------------------
# Auto-ingestion of remaining research datasets
# ---------------------------------------------------------------------------

def auto_ingest_from_directories(
    directories: Sequence[Path],
    *,
    skip_paths: Sequence[Path],
    max_files_per_dir: int = 25,
    sample_rows: int = 5000,
    max_file_mb: int = 200,
) -> Tuple[pd.DataFrame, List[Dict[str, int]]]:
    """Harvest (drug, ADR) mentions from arbitrary CSV/TSV files."""
    skip_set = {p.resolve() for p in skip_paths}
    aggregated: Dict[str, List[str]] = defaultdict(list)
    usage_stats: List[Dict[str, int]] = []
    for directory in directories:
        if not directory.exists():
            logging.warning("Auto-ingest directory missing: %s", directory)
            continue
        processed = 0
        files = list(directory.rglob('*'))
        for file_path in files:
            if processed >= max_files_per_dir:
                break
            if file_path.is_dir():
                continue
            if file_path.suffix.lower() not in {'.csv', '.tsv'}:
                continue
            if file_path.resolve() in skip_set:
                continue
            try:
                size_mb = file_path.stat().st_size / (1024 * 1024)
            except OSError:
                size_mb = 0
            if max_file_mb and size_mb > max_file_mb:
                logging.debug("Skipping %s (%.1f MB > limit)", file_path, size_mb)
                continue
            df = safe_read_table(file_path, nrows=sample_rows)
            if df is None or df.empty:
                continue
            drug_cols, adr_cols = _identify_drug_adr_columns(df)
            if not drug_cols or not adr_cols:
                continue
            processed += 1
            mentions = _harvest_mentions(df, drug_cols, adr_cols)
            for drug, adrs in mentions.items():
                aggregated[drug].extend(adrs)
        usage_stats.append({'dataset': directory.name, 'files_used': processed})
    if not aggregated:
        return pd.DataFrame(), usage_stats
    records = []
    for drug, adrs in aggregated.items():
        uniq = sorted(set(adrs))
        records.append({
            'drug_name_clean': drug,
            'auto_adr_text': ' '.join(uniq),
            'auto_adr_count': len(uniq),
        })
    logging.info("Auto-ingested ADR snippets for %d drugs", len(records))
    return pd.DataFrame(records), usage_stats


def load_train_mentions(path: Path) -> pd.DataFrame:
    """Harvest drug/ADR snippets from the root-level train.csv (if provided)."""
    if not path.exists():
        logging.warning("train.csv missing at %s; skipping supplemental mentions", path)
        return pd.DataFrame()
    logging.info("Loading supplemental ADR mentions from train.csv")
    df = safe_read_table(path)
    if df is None or df.empty:
        logging.warning("train.csv empty or unreadable; skipping")
        return pd.DataFrame()
    drug_cols, adr_cols = _identify_drug_adr_columns(df)
    if not drug_cols or not adr_cols:
        logging.warning("train.csv does not expose recognizable drug/ADR columns; skipping")
        return pd.DataFrame()
    mentions = _harvest_mentions(df, drug_cols, adr_cols)
    if not mentions:
        logging.warning("No valid drug/ADR pairs discovered in train.csv")
        return pd.DataFrame()
    records = []
    for drug, adrs in mentions.items():
        uniq = sorted(set(adrs))
        records.append({
            'drug_name_clean': drug,
            'train_text_blob': ' '.join(uniq),
            'train_adr_count': len(uniq),
        })
    logging.info("train.csv contributed ADR snippets for %d drugs", len(records))
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Feature assembly and model training
# ---------------------------------------------------------------------------

def build_feature_table(
    labels_df: pd.DataFrame,
    feature_frames: Sequence[pd.DataFrame],
) -> Tuple[pd.DataFrame, List[str]]:
    base = labels_df[['drug_name', 'drug_name_clean', 'drug_length', 'drug_words']].copy()
    feature_df = base
    for frame in feature_frames:
        if frame is None or frame.empty:
            continue
        overlap_cols = [c for c in frame.columns if c != 'drug_name_clean']
        feature_df = feature_df.merge(frame, on='drug_name_clean', how='left')
    text_cols = [
        'side_effects_text', 'medical_condition_description', 'embedding_adr_text',
        'auto_adr_text', 'drug_classes', 'medical_condition', 'train_text_blob'
    ]
    for col in text_cols:
        if col not in feature_df.columns:
            feature_df[col] = ''
    feature_df['text_blob'] = feature_df.apply(
        lambda row: ' '.join(filter(None, [
            row['drug_name'],
            str(row.get('side_effects_text', '')),
            str(row.get('medical_condition_description', '')),
            str(row.get('embedding_adr_text', '')),
            str(row.get('auto_adr_text', '')),
            str(row.get('drug_classes', '')),
            str(row.get('medical_condition', '')),
            str(row.get('train_text_blob', '')),
        ])), axis=1
    )
    numeric_cols = []
    for col in feature_df.columns:
        if col in {'drug_name', 'drug_name_clean', 'text_blob'}:
            continue
        if pd.api.types.is_numeric_dtype(feature_df[col]):
            numeric_cols.append(col)
    feature_df[numeric_cols] = feature_df[numeric_cols].fillna(0.0)
    logging.info("Numeric feature columns: %d", len(numeric_cols))
    return feature_df, numeric_cols


class NumpyDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class MultiLabelNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def compute_pos_weights(y: np.ndarray) -> torch.Tensor:
    positives = y.sum(axis=0)
    negatives = y.shape[0] - positives
    pos_weights = (negatives + 1e-3) / (positives + 1e-3)
    return torch.from_numpy(pos_weights).float()


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    label_cols: Sequence[str],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    val_split: float,
    device: torch.device,
) -> Tuple[MultiLabelNet, Dict[str, float], Dict[str, np.ndarray]]:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=RANDOM_SEED
    )
    train_ds = NumpyDataset(X_train, y_train)
    val_ds = NumpyDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = MultiLabelNet(X.shape[1], y.shape[1]).to(device)
    pos_weights = compute_pos_weights(y).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_val_loss = math.inf
    best_state = None
    history: List[Dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})
        logging.info("Epoch %02d | train %.4f | val %.4f", epoch, train_loss, val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
    if best_state is None:
        best_state = model.state_dict()
    model.load_state_dict(best_state)

    # Final evaluation on validation set with best weights.
    model.eval()
    with torch.no_grad():
        val_logits = []
        for xb, _ in val_loader:
            xb = xb.to(device)
            val_logits.append(model(xb).cpu())
    val_logits = torch.cat(val_logits, dim=0)
    val_probs = torch.sigmoid(val_logits).numpy()
    metrics = compute_metrics(y_val, val_probs, label_cols)
    metrics['best_val_loss'] = best_val_loss
    return model, metrics, {'y_val': y_val, 'val_probs': val_probs, 'history': history}


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                    label_cols: Sequence[str]) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'aupr_micro': average_precision_score(y_true, y_prob, average='micro'),
    }
    try:
        metrics['auroc_micro'] = roc_auc_score(y_true, y_prob, average='micro')
    except ValueError:
        metrics['auroc_micro'] = float('nan')
    per_label_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    metrics['per_label_f1'] = dict(zip(label_cols, per_label_f1))
    return metrics


# ---------------------------------------------------------------------------
# CLI orchestration
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ADR model over all data sources")
    parser.add_argument('--data-root', type=Path, default=Path('.'),
                        help='Workspace root containing the 16 datasets')
    parser.add_argument('--output-dir', type=Path, default=Path('adr_model_artifacts'))
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--val-split', type=float, default=0.2)
    parser.add_argument('--embedding-model', type=str,
                        default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--max-auto-files', type=int, default=25,
                        help='Max CSV/TSV files per directory for auto-ingest stage')
    parser.add_argument('--sample-rows', type=int, default=5000,
                        help='Rows per auxiliary file when auto-ingesting')
    parser.add_argument('--max-auto-file-mb', type=int, default=200)
    parser.add_argument('--auto-only-folders', nargs='*', default=None,
                        help='Override default research folders for auto-ingest')
    parser.add_argument('--no-chemistry', action='store_true',
                        help='Skip RDKit descriptor extraction even if RDKit is present')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--dry-run', action='store_true',
                        help='Load data and build features without training (debug)')
    return parser.parse_args(argv)


def ensure_sentence_transformer(model_name: str) -> SentenceTransformer:
    if SentenceTransformer is None:
        raise ImportError(
            "sentence-transformers not installed. pip install -r requirements_colab.txt"
        )
    logging.info("Loading text encoder: %s", model_name)
    return SentenceTransformer(model_name)


def encode_text(texts: Sequence[str], encoder: SentenceTransformer) -> np.ndarray:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    emb = encoder.encode(list(texts), batch_size=64, show_progress_bar=True,
                         convert_to_numpy=True, device=device)
    return emb


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    setup_logging(args.verbose)
    data_root = args.data_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    formatted_path = data_root / 'formatted_training_data.csv'
    labels_df, label_cols = load_formatted_training_data(formatted_path)

    chembl_candidates = [
        data_root / 'chemBlData',
        data_root / 'chemBIData',
    ]
    chembl_root = next((path for path in chembl_candidates if path.exists()), None)

    core_frames = [
        load_unified_stats(data_root / 'unified_adr_dataset.csv'),
        load_drug_embeddings(data_root / 'drug_embeddings_data.csv'),
        load_drugscom_features(data_root / 'drugs_side_effects_drugs_com.csv'),
        load_sider_features(data_root / 'siderData'),
        load_train_mentions(data_root / 'train.csv'),
    ]
    if not args.no_chemistry and chembl_root is not None:
        core_frames.append(load_chembl_features(chembl_root))
    elif not args.no_chemistry:
        logging.warning("chemBlData / chemBIData folder missing; skipping chemistry features")

    skip_files = [
        data_root / 'formatted_training_data.csv',
        data_root / 'unified_adr_dataset.csv',
        data_root / 'drug_embeddings_data.csv',
        data_root / 'drugs_side_effects_drugs_com.csv',
        data_root / 'train.csv',
    ]
    default_auto_dirs = [
        'ADRtarget-master', 'CT-ADE-main-dataset', 'CT-ADE-main-dataset/CT-ADE-main-dataset',
        'DrugMeNot-main', 'faersData', 'Hybrid-Adverse-Drug-Reactions-Predictor-main',
        'ML Clinical Trials - Galeano and Paccanaro-dataset', 'pubChemData',
        'RecSys23-ADRnet-main', 'SIDER4-master-dataset', 'siderData',
        'chemBlData', 'chemBIData'
    ]
    auto_dirs = args.auto_only_folders if args.auto_only_folders else default_auto_dirs
    auto_dirs = [data_root / Path(d) for d in auto_dirs]
    auto_frame, usage = auto_ingest_from_directories(
        auto_dirs,
        skip_paths=skip_files,
        max_files_per_dir=args.max_auto_files,
        sample_rows=args.sample_rows,
        max_file_mb=args.max_auto_file_mb,
    )
    for stat in usage:
        logging.info("Auto-ingest %s: %d files", stat['dataset'], stat['files_used'])
    core_frames.append(auto_frame)

    feature_df, numeric_cols = build_feature_table(labels_df, core_frames)
    numeric_data = feature_df[numeric_cols].values.astype(np.float32)
    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(numeric_data)

    encoder = ensure_sentence_transformer(args.embedding_model)
    text_embeddings = encode_text(feature_df['text_blob'].tolist(), encoder)
    X = np.hstack([numeric_scaled, text_embeddings])
    y = labels_df[label_cols].values.astype(np.float32)
    logging.info("Final feature matrix shape: %s", X.shape)

    artifacts = {
        'label_columns': label_cols,
        'numeric_columns': numeric_cols,
        'scaler': scaler,
        'text_model_name': args.embedding_model,
        'usage_stats': usage,
    }

    if args.dry_run:
        joblib.dump(artifacts, args.output_dir / 'preprocessor_only.pkl')
        logging.info("Dry run complete; artifacts saved. Skipping training per flag.")
        return

    device = torch.device(args.device)
    logging.info("Training on device: %s", device)
    model, metrics, extras = train_model(
        X, y, label_cols,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_split=args.val_split,
        device=device,
    )
    artifacts['metrics'] = metrics
    artifacts['training_history'] = extras['history']
    torch.save(model.state_dict(), args.output_dir / 'adr_multilabel_model.pt')
    joblib.dump(artifacts, args.output_dir / 'adr_preprocessor.pkl')
    np.savez_compressed(args.output_dir / 'validation_outputs.npz', **extras)
    with open(args.output_dir / 'metrics.json', 'w', encoding='utf-8') as fp:
        json.dump(metrics, fp, indent=2)
    logging.info("Training complete. Metrics: %s", metrics)


if __name__ == '__main__':  # pragma: no cover
    main()
