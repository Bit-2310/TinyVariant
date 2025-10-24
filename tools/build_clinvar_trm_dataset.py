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
from typing import Any, Dict, List, Tuple, Set
from collections import Counter

import numpy as np
import pandas as pd

SEQ_LEN = 25  # CLS + core features + phenotype/provenance tokens + phenotype tertiary + 5 position digits + label slot
PAD_TOKEN = "PAD"
CLS_TOKEN = "CLS"
LABEL_SLOT_TOKEN = "LABEL_SLOT"
LABEL_TOKENS = {
    "Benign": "LABEL_BENIGN",
    "Pathogenic": "LABEL_PATHOGENIC",
}

SPECIAL_TOKENS = [PAD_TOKEN, CLS_TOKEN, LABEL_SLOT_TOKEN] + list(LABEL_TOKENS.values())

RANDOM_SEED = 42

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

AA_CHANGE_DEFAULT = '<unknown_change>'

PHENOTYPE_OTHER = "<other>"
PHENOTYPE_NONE = "<none>"
PHENOTYPE_TOP_K = 50
PHENOTYPE_SOURCE_TOP_K = 8
PHENOTYPE_SLOT_COLUMN_NAMES = [
    "PhenotypePrimaryToken",
    "PhenotypeSecondaryToken",
    "PhenotypeTertiaryToken",
]
PHENOTYPE_TOKEN_PREFIXES = [
    "PHENO_PRIMARY",
    "PHENO_SECONDARY",
    "PHENO_TERTIARY",
]

PHENOTYPE_COUNT_BUCKETS = (
    (0, "none"),
    (1, "one"),
    (2, "two"),
    (3, "three_plus"),
)

SUBMITTER_BUCKETS = (
    (0, "0"),
    (1, "1"),
    (3, "2_3"),
    (5, "4_5"),
    (float("inf"), "6_plus"),
)

EVAL_YEAR_BUCKETS = (
    (2010, "pre2010"),
    (2015, "2010_2014"),
    (2020, "2015_2019"),
    (float("inf"), "2020_plus"),
)


@dataclass(frozen=True)
class FeatureBuckets:
    phenotype_terms: Set[str]
    phenotype_sources: Set[str]


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


def _split_pipe_field(raw: Any) -> List[str]:
    if not isinstance(raw, str):
        return []
    return [part for part in raw.split("|") if part]


def _select_top_values(sequences: List[List[str]], top_k: int) -> Set[str]:
    counter: Counter[str] = Counter()
    for seq in sequences:
        counter.update(seq)
    return {value for value, _ in counter.most_common(top_k)}


def _bucket_count(value: int) -> str:
    for threshold, name in PHENOTYPE_COUNT_BUCKETS:
        if value <= threshold:
            return name
    return PHENOTYPE_COUNT_BUCKETS[-1][1]


def _bucket_submitters(value: int) -> str:
    for threshold, name in SUBMITTER_BUCKETS:
        if value <= threshold:
            return name
    return SUBMITTER_BUCKETS[-1][1]


def _bucket_eval_year(iso_date: str) -> str:
    if not isinstance(iso_date, str) or len(iso_date) < 4:
        return "unknown"
    try:
        year = int(iso_date[:4])
    except ValueError:
        return "unknown"
    for threshold, name in EVAL_YEAR_BUCKETS:
        if year < threshold:
            return name
    return EVAL_YEAR_BUCKETS[-1][1]


def build_feature_buckets(df: pd.DataFrame) -> FeatureBuckets:
    phenotypes = df["PhenotypeTermsList"].tolist()
    sources = df["PhenotypeSourcesList"].tolist()
    top_terms = _select_top_values(phenotypes, PHENOTYPE_TOP_K)
    top_sources = _select_top_values(sources, PHENOTYPE_SOURCE_TOP_K)
    return FeatureBuckets(phenotype_terms=top_terms, phenotype_sources=top_sources)


def apply_feature_buckets(df: pd.DataFrame, buckets: FeatureBuckets) -> pd.DataFrame:
    df = df.copy()

    def _map_term(term: str) -> str:
        return term if term in buckets.phenotype_terms else PHENOTYPE_OTHER

    def _phenotype_slots(terms: List[str]) -> List[str]:
        slots: List[str] = []
        seen = set()
        for term in terms:
            if term in seen:
                continue
            seen.add(term)
            slots.append(_map_term(term))
            if len(slots) == len(PHENOTYPE_SLOT_COLUMN_NAMES):
                break
        while len(slots) < len(PHENOTYPE_SLOT_COLUMN_NAMES):
            slots.append(PHENOTYPE_NONE)
        return slots

    def _choose_source(sources: List[str]) -> str:
        if not sources:
            return PHENOTYPE_NONE
        source = sources[0]
        return source if source in buckets.phenotype_sources else PHENOTYPE_OTHER

    phenotype_slot_values = df["PhenotypeTermsList"].apply(_phenotype_slots)
    for idx, column_name in enumerate(PHENOTYPE_SLOT_COLUMN_NAMES):
        df[column_name] = phenotype_slot_values.apply(lambda values, i=idx: values[i])

    df["PhenotypeCountBucket"] = df["PhenotypeTermCount"].apply(_bucket_count)
    df["PhenotypeSourceToken"] = df["PhenotypeSourcesList"].apply(_choose_source)
    df["SubmitterBucket"] = df["SubmitterCount"].apply(_bucket_submitters)
    df["EvalRecencyBucket"] = df["LastEvaluatedISO"].apply(_bucket_eval_year)

    return df


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
    parser.add_argument(
        "--phenotype-ablation",
        action="store_true",
        help="Replace phenotype tokens with <none> to evaluate impact of phenotype features.",
    )
    parser.add_argument(
        "--provenance-ablation",
        action="store_true",
        help="Zero out provenance/submitter/evaluation buckets to ablate provenance features.",
    )
    return parser.parse_args()


def load_balanced_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    # Ensure expected enrichment columns exist; fallback for older datasets
    if "PhenotypeTerms" not in df.columns:
        df["PhenotypeTerms"] = ""
    if "PhenotypeSources" not in df.columns:
        df["PhenotypeSources"] = ""
    if "SubmitterCount" not in df.columns:
        number_submitters = df.get("NumberSubmitters")
        if number_submitters is None:
            submitter_series = pd.Series([0] * len(df))
        else:
            submitter_series = pd.to_numeric(number_submitters, errors="coerce").fillna(0)
        df["SubmitterCount"] = submitter_series.astype(int)
    else:
        df["SubmitterCount"] = pd.to_numeric(df["SubmitterCount"], errors="coerce").fillna(0).astype(int)
    if "LastEvaluatedISO" not in df.columns:
        last_eval = df.get("LastEvaluated")
        if last_eval is None:
            df["LastEvaluatedISO"] = [""] * len(df)
        else:
            iso_series = pd.to_datetime(last_eval, errors="coerce")
            df["LastEvaluatedISO"] = iso_series.dt.strftime("%Y-%m-%d").fillna("")
    else:
        df["LastEvaluatedISO"] = df["LastEvaluatedISO"].fillna("")

    df["PhenotypeTerms"] = df["PhenotypeTerms"].fillna("")
    df["PhenotypeSources"] = df["PhenotypeSources"].fillna("")

    df["PhenotypeTermsList"] = df["PhenotypeTerms"].apply(_split_pipe_field)
    df["PhenotypeSourcesList"] = df["PhenotypeSources"].apply(_split_pipe_field)
    df["PhenotypeTermCount"] = df["PhenotypeTermsList"].apply(len).astype(int)
    phenotype_ids_series = df.get("PhenotypeIDs")
    if phenotype_ids_series is None:
        phenotype_ids_series = pd.Series([""] * len(df))
    else:
        phenotype_ids_series = phenotype_ids_series.fillna("")
    df["PhenotypeIDCount"] = phenotype_ids_series.apply(_split_pipe_field).apply(len).astype(int)

    counts = df['GeneSymbol'].value_counts()
    common_genes = counts[counts >= 5].index
    df['GeneBucket'] = df['GeneSymbol'].where(df['GeneSymbol'].isin(common_genes), '<RARE>')

    df['Consequence'] = df['Name'].str.extract('\(([^)]+)\)')
    df['Consequence'] = df['Consequence'].where(df['Consequence'].notna(), '<unknown>')

    df['AAPropFrom'] = df['ProteinFrom'].map(AA_PROPERTY).fillna('<unknown>')
    df['AAPropTo'] = df['ProteinTo'].map(AA_PROPERTY).fillna('<unknown>')
    df['AAChangeClass'] = df.apply(lambda r: 'same' if r['AAPropFrom'] == r['AAPropTo'] else r['AAPropFrom'] + '->' + r['AAPropTo'], axis=1)
    return df


def build_vocab(df: pd.DataFrame) -> Dict[str, int]:
    token_to_id: Dict[str, int] = {}
    for token in SPECIAL_TOKENS:
        token_to_id[token] = len(token_to_id)

    def add_token(prefix: str, value: str) -> None:
        token = f"{prefix}:{value}"
        if token not in token_to_id:
            token_to_id[token] = len(token_to_id)

    for gene_bucket in sorted(df["GeneBucket"].unique()):
        add_token("GENE", gene_bucket)
    for chrom in sorted(df["Chromosome"].unique()):
        add_token("CHR", chrom)
    for nuc in ["A", "C", "G", "T"]:
        add_token("NUC", nuc)
    for aa in sorted(set(df["ProteinFrom"]) | set(df["ProteinTo"])):
        add_token("AA", aa)
    for prop in sorted(df["AAPropFrom"].unique()):
        add_token("AAPROP", prop)
    for prop in sorted(df["AAPropTo"].unique()):
        add_token("AAPROP", prop)
    for change in sorted(df["AAChangeClass"].unique()):
        add_token("AACHANGE", change)
    for status in sorted(df["ReviewStatus"].unique()):
        add_token("REV", status)
    for consequence in sorted(df["Consequence"].unique()):
        add_token("CONSEQ", consequence)
    for column_name, prefix in zip(PHENOTYPE_SLOT_COLUMN_NAMES, PHENOTYPE_TOKEN_PREFIXES):
        for value in sorted(df[column_name].unique()):
            add_token(prefix, value)
    for source in sorted(df["PhenotypeSourceToken"].unique()):
        add_token("PHENO_SOURCE", source)
    for bucket in sorted(df["PhenotypeCountBucket"].unique()):
        add_token("PHENO_COUNT", bucket)
    for bucket in sorted(df["SubmitterBucket"].unique()):
        add_token("SUBMITTERS", bucket)
    for bucket in sorted(df["EvalRecencyBucket"].unique()):
        add_token("EVAL", bucket)
    for digit in "0123456789":
        add_token("DIGIT", digit)

    return token_to_id


def apply_ablation_flags(
    df: pd.DataFrame,
    phenotype_ablation: bool,
    provenance_ablation: bool,
) -> pd.DataFrame:
    df = df.copy()

    if phenotype_ablation:
        df["PhenotypeTerms"] = ""
        df["PhenotypeSources"] = ""
        df["PhenotypeTermsList"] = [[] for _ in range(len(df))]
        df["PhenotypeSourcesList"] = [[] for _ in range(len(df))]
        df["PhenotypeTermCount"] = 0
        df["PhenotypeIDCount"] = 0

        for column_name in PHENOTYPE_SLOT_COLUMN_NAMES:
            df[column_name] = PHENOTYPE_NONE

        df["PhenotypeSourceToken"] = PHENOTYPE_NONE
        df["PhenotypeCountBucket"] = PHENOTYPE_COUNT_BUCKETS[0][1]

    if provenance_ablation:
        df["SubmitterBucket"] = SUBMITTER_BUCKETS[0][1]
        df["EvalRecencyBucket"] = EVAL_YEAR_BUCKETS[0][1]
        if not phenotype_ablation:
            df["PhenotypeSourceToken"] = PHENOTYPE_NONE
            df["PhenotypeCountBucket"] = PHENOTYPE_COUNT_BUCKETS[0][1]

    return df


def token_id(token_to_id: Dict[str, int], token: str) -> int:
    return token_to_id[token]


def encode_variant(
    row: pd.Series, token_to_id: Dict[str, int]
) -> Tuple[List[int], List[int]]:
    seq: List[int] = []
    lbls: List[int] = []

    seq.append(token_id(token_to_id, CLS_TOKEN))

    seq.append(token_id(token_to_id, f"GENE:{row['GeneBucket']}"))
    seq.append(token_id(token_to_id, f"CHR:{row['Chromosome']}"))
    seq.append(token_id(token_to_id, f"REV:{row['ReviewStatus']}"))
    seq.append(token_id(token_to_id, f"NUC:{row['RefAllele']}"))
    seq.append(token_id(token_to_id, f"NUC:{row['AltAllele']}"))
    seq.append(token_id(token_to_id, f"AA:{row['ProteinFrom']}"))
    seq.append(token_id(token_to_id, f"AA:{row['ProteinTo']}"))
    seq.append(token_id(token_to_id, f"AAPROP:{row['AAPropFrom']}"))
    seq.append(token_id(token_to_id, f"AAPROP:{row['AAPropTo']}"))
    seq.append(token_id(token_to_id, f"AACHANGE:{row['AAChangeClass']}"))
    seq.append(token_id(token_to_id, f"CONSEQ:{row['Consequence']}"))
    for column_name, prefix in zip(PHENOTYPE_SLOT_COLUMN_NAMES, PHENOTYPE_TOKEN_PREFIXES):
        seq.append(token_id(token_to_id, f"{prefix}:{row[column_name]}"))
    seq.append(token_id(token_to_id, f"PHENO_SOURCE:{row['PhenotypeSourceToken']}"))
    seq.append(token_id(token_to_id, f"PHENO_COUNT:{row['PhenotypeCountBucket']}"))
    seq.append(token_id(token_to_id, f"SUBMITTERS:{row['SubmitterBucket']}"))
    seq.append(token_id(token_to_id, f"EVAL:{row['EvalRecencyBucket']}"))

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
    feature_metadata: Dict[str, Any],
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
        "feature_metadata": feature_metadata,
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

    feature_buckets = build_feature_buckets(df)
    df = apply_feature_buckets(df, feature_buckets)
    df = apply_ablation_flags(
        df,
        phenotype_ablation=args.phenotype_ablation,
        provenance_ablation=args.provenance_ablation,
    )

    token_to_id = build_vocab(df)
    vocab_size = len(token_to_id)
    pad_id = token_to_id[PAD_TOKEN]
    num_identifiers = int(df["VariantIdentifier"].max()) + 1

    feature_metadata = {
        "phenotype_terms_top": sorted(feature_buckets.phenotype_terms),
        "phenotype_sources_top": sorted(feature_buckets.phenotype_sources),
        "phenotype_top_k": PHENOTYPE_TOP_K,
        "phenotype_source_top_k": PHENOTYPE_SOURCE_TOP_K,
        "phenotype_slot_columns": PHENOTYPE_SLOT_COLUMN_NAMES,
        "phenotype_count_buckets": [name for _, name in PHENOTYPE_COUNT_BUCKETS],
        "submitter_buckets": [name for _, name in SUBMITTER_BUCKETS],
        "eval_year_buckets": [name for _, name in EVAL_YEAR_BUCKETS],
        "phenotype_ablation": bool(args.phenotype_ablation),
        "provenance_ablation": bool(args.provenance_ablation),
    }

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
        feature_metadata,
    )
    save_split(
        args.output_dir / "test",
        test_dataset,
        vocab_size,
        pad_id,
        num_identifiers,
        code_version,
        git_commit,
        feature_metadata,
    )

    identifier_strings = (
        df.sort_values("VariantIdentifier")
        .apply(lambda row: f"{row['VariationID']}|{row['Name']}", axis=1)
        .tolist()
    )
    save_identifiers(args.output_dir, identifier_strings)
    save_vocab(args.output_dir, token_to_id)

    print(f"Wrote ClinVar TRM dataset to {args.output_dir}")
