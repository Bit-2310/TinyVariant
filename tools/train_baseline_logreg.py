#!/usr/bin/env python
"""
Train a simple logistic regression baseline on the ClinVar balanced dataset.

Feature encoding:
  - Gene, chromosome, ref/alt nucleotides, amino-acid change: one-hot categorical
  - Protein position: integer (standardized)

Outputs accuracy and ROC AUC on the held-out test split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/clinvar/processed/clinvar_missense_balanced.tsv"),
        help="Balanced ClinVar TSV (must match TRM preprocessing).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of samples to use for training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stratified split.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to dump metrics as JSON.",
    )
    return parser.parse_args()


AA_PROPERTY = {
    'A': 'nonpolar',
    'R': 'positive',
    'N': 'polar',
    'D': 'negative',
    'C': 'polar',
    'Q': 'polar',
    'E': 'negative',
    'G': 'nonpolar',
    'H': 'positive',
    'I': 'nonpolar',
    'L': 'nonpolar',
    'K': 'positive',
    'M': 'nonpolar',
    'F': 'nonpolar',
    'P': 'nonpolar',
    'S': 'polar',
    'T': 'polar',
    'W': 'nonpolar',
    'Y': 'polar',
    'V': 'nonpolar',
    '*': 'stop'
}


def split_dataframe(df: pd.DataFrame, train_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train_idx = []
    test_idx = []

    for label, group in df.groupby("Label"):
        indices = group.index.to_numpy()
        rng.shuffle(indices)
        cutoff = int(len(indices) * train_ratio)
        train_idx.extend(indices[:cutoff])
        test_idx.extend(indices[cutoff:])

    train_df = df.loc[sorted(train_idx)].reset_index(drop=True)
    test_df = df.loc[sorted(test_idx)].reset_index(drop=True)
    return train_df, test_df


def _augment_df(df: pd.DataFrame) -> pd.DataFrame:
    counts = df['GeneSymbol'].value_counts()
    common_genes = counts[counts >= 5].index
    df = df.copy()
    df['GeneBucket'] = df['GeneSymbol'].where(df['GeneSymbol'].isin(common_genes), '<RARE>')
    df['Consequence'] = df['Name'].str.extract('\(([^)]+)\)')
    df['Consequence'] = df['Consequence'].where(df['Consequence'].notna(), '<unknown>')
    df['AAPropFrom'] = df['ProteinFrom'].map(AA_PROPERTY).fillna('<unknown>')
    df['AAPropTo'] = df['ProteinTo'].map(AA_PROPERTY).fillna('<unknown>')
    df['AAChangeClass'] = df.apply(lambda r: 'same' if r['AAPropFrom'] == r['AAPropTo'] else r['AAPropFrom'] + '->' + r['AAPropTo'], axis=1)
    return df


def prepare_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, OneHotEncoder, StandardScaler]:
    df = _augment_df(df)
    cat_features = df[[
        'GeneBucket',
        'Chromosome',
        'ReviewStatus',
        'Consequence',
        'AAPropFrom',
        'AAPropTo',
        'AAChangeClass',
        'RefAllele',
        'AltAllele',
        'ProteinFrom',
        'ProteinTo',
    ]]
    pos_feature = df[['ProteinPos']].astype(np.float32)
    labels = df['Label'].to_numpy(dtype=np.int64)

    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    X_cat = ohe.fit_transform(cat_features)

    scaler = StandardScaler()
    X_pos = scaler.fit_transform(pos_feature)

    X = np.hstack([X_cat.toarray(), X_pos])
    return X, labels, ohe, scaler


def prepare_features_with_encoders(
    df: pd.DataFrame, ohe: OneHotEncoder, scaler: StandardScaler
) -> Tuple[np.ndarray, np.ndarray]:
    df = _augment_df(df)
    cat_features = df[[
        'GeneBucket',
        'Chromosome',
        'ReviewStatus',
        'Consequence',
        'AAPropFrom',
        'AAPropTo',
        'AAChangeClass',
        'RefAllele',
        'AltAllele',
        'ProteinFrom',
        'ProteinTo',
    ]]
    pos_feature = df[['ProteinPos']].astype(np.float32)
    labels = df['Label'].to_numpy(dtype=np.int64)

    X_cat = ohe.transform(cat_features)
    X_pos = scaler.transform(pos_feature)
    X = np.hstack([X_cat.toarray(), X_pos])
    return X, labels


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input, sep="\t")
    train_df, test_df = split_dataframe(df, args.train_ratio, args.seed)

    X_train, y_train, ohe, scaler = prepare_features(train_df)
    X_test, y_test = prepare_features_with_encoders(test_df, ohe, scaler)

    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    metrics = {
        "accuracy": float(accuracy),
        "roc_auc": float(auc),
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
    }

    print(json.dumps(metrics, indent=2))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
