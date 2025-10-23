#!/usr/bin/env python
"""
Convert the balanced ClinVar TSV into the TinyRecursiveModels puzzle dataset format.

Each variant becomes a single-example "puzzle" where the input sequence encodes
gene, chromosome, nucleotide change, amino-acid change, and protein position.
The model is trained to predict a binary label token at the final position.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

SEQ_LEN = 13  # CLS + 6 feature slots + 5 position digits + label slot
PAD_TOKEN = "PAD"
CLS_TOKEN = "CLS"
LABEL_SLOT_TOKEN = "LABEL_SLOT"
LABEL_TOKENS = {
    "Benign": "LABEL_BENIGN",
    "Pathogenic": "LABEL_PATHOGENIC",
}

SPECIAL_TOKENS = [PAD_TOKEN, CLS_TOKEN, LABEL_SLOT_TOKEN] + list(LABEL_TOKENS.values())

RANDOM_SEED = 42


def get_repo_version() -> str:
    version_file = Path(__file__).resolve().parents[1] / "VERSION"
    try:
        return version_file.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return "unknown"


def get_git_commit() -> str:
    import subprocess

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return commit.decode("utf-8").strip()
    except Exception:
        return "unknown"


@dataclass
class EncodedDataset:
    inputs: np.ndarray
    labels: np.ndarray
    puzzle_identifiers: np.ndarray
    puzzle_indices: np.ndarray
    group_indices: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/clinvar/processed/clinvar_missense_balanced.tsv"),
        help="Path to balanced ClinVar TSV (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/clinvar/processed/clinvar_trm"),
        help="Destination directory for TRM-ready dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of samples for training split (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for stratified split (default: %(default)s)",
    )
    return parser.parse_args()


def load_balanced_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    return df


def build_vocab(df: pd.DataFrame) -> Dict[str, int]:
    token_to_id: Dict[str, int] = {}
    for token in SPECIAL_TOKENS:
        token_to_id[token] = len(token_to_id)

    def add_token(prefix: str, value: str) -> None:
        token = f"{prefix}:{value}"
        if token not in token_to_id:
            token_to_id[token] = len(token_to_id)

    for gene in sorted(df["GeneSymbol"].unique()):
        add_token("GENE", gene)
    for chrom in sorted(df["Chromosome"].unique()):
        add_token("CHR", chrom)
    for nuc in ["A", "C", "G", "T"]:
        add_token("NUC", nuc)
    for aa in sorted(set(df["ProteinFrom"]) | set(df["ProteinTo"])):
        add_token("AA", aa)
    for digit in "0123456789":
        add_token("DIGIT", digit)

    return token_to_id


def token_id(token_to_id: Dict[str, int], token: str) -> int:
    return token_to_id[token]


def encode_variant(
    row: pd.Series, token_to_id: Dict[str, int]
) -> Tuple[List[int], List[int]]:
    seq: List[int] = []
    lbls: List[int] = []

    seq.append(token_id(token_to_id, CLS_TOKEN))

    seq.append(token_id(token_to_id, f"GENE:{row['GeneSymbol']}"))
    seq.append(token_id(token_to_id, f"CHR:{row['Chromosome']}"))
    seq.append(token_id(token_to_id, f"NUC:{row['RefAllele']}"))
    seq.append(token_id(token_to_id, f"NUC:{row['AltAllele']}"))
    seq.append(token_id(token_to_id, f"AA:{row['ProteinFrom']}"))
    seq.append(token_id(token_to_id, f"AA:{row['ProteinTo']}"))

    pos_str = f"{int(row['ProteinPos']):05d}"
    for digit in pos_str:
        seq.append(token_id(token_to_id, f"DIGIT:{digit}"))

    seq.append(token_id(token_to_id, LABEL_SLOT_TOKEN))

    assert len(seq) == SEQ_LEN, f"Unexpected sequence length {len(seq)}"

    lbls = [0] * SEQ_LEN
    label_name = row["LabelName"]
    label_token = LABEL_TOKENS[label_name]
    lbls[-1] = token_id(token_to_id, label_token)

    return seq, lbls


def encode_dataframe(
    df: pd.DataFrame, token_to_id: Dict[str, int]
) -> EncodedDataset:
    inputs = np.zeros((len(df), SEQ_LEN), dtype=np.int32)
    labels = np.zeros((len(df), SEQ_LEN), dtype=np.int32)
    puzzle_identifiers = np.zeros(len(df), dtype=np.int32)

    for idx, row in df.reset_index(drop=True).iterrows():
        seq, lbls = encode_variant(row, token_to_id)
        inputs[idx] = np.array(seq, dtype=np.int32)
        labels[idx] = np.array(lbls, dtype=np.int32)
        puzzle_identifiers[idx] = int(row["VariantIdentifier"])

    puzzle_indices = np.arange(0, len(df) + 1, dtype=np.int32)
    group_indices = puzzle_indices.copy()

    return EncodedDataset(
        inputs=inputs,
        labels=labels,
        puzzle_identifiers=puzzle_identifiers,
        puzzle_indices=puzzle_indices,
        group_indices=group_indices,
    )


def stratified_split(
    df: pd.DataFrame, train_ratio: float, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train_indices: List[int] = []
    test_indices: List[int] = []

    for label, group in df.groupby("LabelName"):
        indices = group.index.to_numpy()
        rng.shuffle(indices)
        cutoff = int(len(indices) * train_ratio)
        train_indices.extend(indices[:cutoff])
        test_indices.extend(indices[cutoff:])

    train_df = df.loc[sorted(train_indices)].reset_index(drop=True)
    test_df = df.loc[sorted(test_indices)].reset_index(drop=True)
    return train_df, test_df


def save_split(
    split_dir: Path,
    dataset: EncodedDataset,
    vocab_size: int,
    pad_id: int,
    num_identifiers: int,
    code_version: str,
    git_commit: str,
) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)

    np.save(split_dir / "all__inputs.npy", dataset.inputs)
    np.save(split_dir / "all__labels.npy", dataset.labels)
    np.save(split_dir / "all__puzzle_identifiers.npy", dataset.puzzle_identifiers)
    np.save(split_dir / "all__puzzle_indices.npy", dataset.puzzle_indices)
    np.save(split_dir / "all__group_indices.npy", dataset.group_indices)

    metadata = {
        "seq_len": SEQ_LEN,
        "vocab_size": vocab_size,
        "pad_id": pad_id,
        "ignore_label_id": pad_id,
        "blank_identifier_id": 0,
        "num_puzzle_identifiers": num_identifiers,
        "total_groups": int(dataset.group_indices[-1]),
        "mean_puzzle_examples": 1.0,
        "total_puzzles": int(dataset.group_indices[-1]),
        "sets": ["all"],
        "code_version": code_version,
        "git_commit": git_commit,
    }

    with open(split_dir / "dataset.json", "w") as f:
        json.dump(metadata, f, indent=2)


def save_identifiers(output_dir: Path, identifiers: List[str]) -> None:
    mapping = ["<blank>"] + identifiers
    with open(output_dir / "identifiers.json", "w") as f:
        json.dump(mapping, f, indent=2)


def save_vocab(output_dir: Path, token_to_id: Dict[str, int]) -> None:
    vocab_items = sorted(token_to_id.items(), key=lambda kv: kv[1])
    with open(output_dir / "vocab.json", "w") as f:
        json.dump(vocab_items, f, indent=2)


if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_balanced_table(args.input)
    df = df.reset_index(drop=True)
    df["VariantIdentifier"] = np.arange(1, len(df) + 1, dtype=np.int32)

    token_to_id = build_vocab(df)
    vocab_size = len(token_to_id)
    pad_id = token_to_id[PAD_TOKEN]
    num_identifiers = int(df["VariantIdentifier"].max()) + 1

    train_df, test_df = stratified_split(df, args.train_ratio, args.seed)

    train_dataset = encode_dataframe(train_df, token_to_id)
    test_dataset = encode_dataframe(test_df, token_to_id)

    code_version = get_repo_version()
    git_commit = get_git_commit()

    save_split(
        args.output_dir / "train",
        train_dataset,
        vocab_size,
        pad_id,
        num_identifiers,
        code_version,
        git_commit,
    )
    save_split(
        args.output_dir / "test",
        test_dataset,
        vocab_size,
        pad_id,
        num_identifiers,
        code_version,
        git_commit,
    )

    identifier_strings = (
        df.sort_values("VariantIdentifier")
        .apply(lambda row: f"{row['VariationID']}|{row['Name']}", axis=1)
        .tolist()
    )
    save_identifiers(args.output_dir, identifier_strings)
    save_vocab(args.output_dir, token_to_id)

    print(f"Wrote ClinVar TRM dataset to {args.output_dir}")
def get_repo_version() -> str:
    version_file = Path(__file__).resolve().parents[1] / "VERSION"
    try:
        return version_file.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return "unknown"


def get_git_commit() -> str:
    import subprocess

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return commit.decode("utf-8").strip()
    except Exception:
        return "unknown"
