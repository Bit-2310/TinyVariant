#!/usr/bin/env python
"""
Filter ClinVar's variant summary down to high-confidence missense SNVs and
emit a balanced pathogenic vs benign table for downstream TinyVariant work.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


ALLOWED_REVIEW_STATUSES = {
    "criteria provided, single submitter",
    "criteria provided, multiple submitters, no conflicts",
    "reviewed by expert panel",
    "practice guideline",
}

AA_THREE_TO_ONE = {
    "Ala": "A",
    "Arg": "R",
    "Asn": "N",
    "Asp": "D",
    "Cys": "C",
    "Gln": "Q",
    "Glu": "E",
    "Gly": "G",
    "His": "H",
    "Ile": "I",
    "Leu": "L",
    "Lys": "K",
    "Met": "M",
    "Phe": "F",
    "Pro": "P",
    "Ser": "S",
    "Thr": "T",
    "Trp": "W",
    "Tyr": "Y",
    "Val": "V",
}

PROTEIN_PATTERN = re.compile(
    r"p\.\(?(?P<from>[A-Za-z]{3})(?P<pos>\d+)(?P<to>[A-Za-z]{3}|=|Ter|\*)\)?"
)

PHENOTYPE_PLACEHOLDERS = {
    "not provided",
    "not specified",
    "not applicable",
    "not surveyed",
    "see cases",
    "unknown",
    "",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/clinvar/raw/variant_summary.txt.gz"),
        help="Path to ClinVar variant_summary.txt.gz (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/clinvar/processed"),
        help="Directory to store processed outputs (default: %(default)s)",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=2500,
        help="Cap the number of examples per class (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed for sampling (default: %(default)s)",
    )
    return parser.parse_args()


def get_repo_version() -> str:
    version_file = Path(__file__).resolve().parents[1] / "VERSION"
    try:
        return version_file.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return "unknown"


def get_git_commit() -> str:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return commit.decode("utf-8").strip()
    except Exception:
        return "unknown"


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def _dedupe_preserve(seq: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in seq:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            ordered.append(item)
    return ordered


def parse_phenotype_terms(raw: str) -> List[str]:
    if not isinstance(raw, str):
        return []

    tokens = []
    for part in raw.replace("||", "|").split("|"):
        cleaned = normalize_whitespace(part)
        if not cleaned:
            continue
        if cleaned.lower() in PHENOTYPE_PLACEHOLDERS:
            continue
        tokens.append(cleaned)
    return _dedupe_preserve(tokens)


def parse_phenotype_ids(raw: str) -> List[str]:
    if not isinstance(raw, str):
        return []

    tokens: List[str] = []
    for group in raw.split("||"):
        for part in group.split(","):
            cleaned = normalize_whitespace(part)
            if not cleaned:
                continue
            tokens.append(cleaned)
    return _dedupe_preserve(tokens)


def extract_phenotype_sources(phenotype_ids: List[str]) -> List[str]:
    sources: List[str] = []
    seen = set()
    for identifier in phenotype_ids:
        source = identifier.split(":", 1)[0] if ":" in identifier else "UNKNOWN"
        if source not in seen:
            seen.add(source)
            sources.append(source)
    return sources


def load_variant_summary(path: Path) -> pd.DataFrame:
    usecols = [
        "#AlleleID",
        "Type",
        "Name",
        "GeneID",
        "GeneSymbol",
        "ClinicalSignificance",
        "ClinSigSimple",
        "LastEvaluated",
        "ReviewStatus",
        "OriginSimple",
        "Assembly",
        "Chromosome",
        "Start",
        "Stop",
        "RCVaccession",
        "PhenotypeList",
        "PhenotypeIDS",
        "NumberSubmitters",
        "SubmitterCategories",
        "Guidelines",
        "ReferenceAlleleVCF",
        "AlternateAlleleVCF",
        "VariationID",
    ]
    df = pd.read_csv(
        path,
        sep="\t",
        compression="gzip",
        usecols=usecols,
        dtype=str,
        low_memory=False,
    )
    return df


def parse_protein_change(name: str) -> Tuple[str, int, str] | None:
    if not isinstance(name, str):
        return None
    match = PROTEIN_PATTERN.search(name)
    if not match:
        return None
    from_aa, pos, to_aa = match.group("from"), match.group("pos"), match.group("to")
    if to_aa in {"=", "Ter", "*"}:
        return None
    if from_aa == to_aa:
        return None
    if from_aa not in AA_THREE_TO_ONE or to_aa not in AA_THREE_TO_ONE:
        return None
    return AA_THREE_TO_ONE[from_aa], int(pos), AA_THREE_TO_ONE[to_aa]


def filter_high_confidence(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ClinSigSimple"] = pd.to_numeric(df["ClinSigSimple"], errors="coerce")
    df = df[df["ClinSigSimple"].isin([0, 1])]
    df = df[df["Assembly"] == "GRCh38"]
    df = df[df["Type"].str.lower() == "single nucleotide variant"]
    df = df[df["OriginSimple"].str.contains("germline", case=False, na=False)]
    df = df[
        df["ReviewStatus"]
        .str.lower()
        .isin({status.lower() for status in ALLOWED_REVIEW_STATUSES})
    ]

    df = df[
        df["ReferenceAlleleVCF"].str.len().eq(1)
        & df["ReferenceAlleleVCF"].str.upper().isin(list("ACGT"))
        & df["AlternateAlleleVCF"].str.len().eq(1)
        & df["AlternateAlleleVCF"].str.upper().isin(list("ACGT"))
    ]

    protein = df["Name"].apply(parse_protein_change)
    mask = protein.notna()
    df = df.loc[mask].copy()
    protein = protein[mask]
    df["ProteinFrom"] = [p[0] for p in protein]
    df["ProteinPos"] = [p[1] for p in protein]
    df["ProteinTo"] = [p[2] for p in protein]
    df["RefAllele"] = df["ReferenceAlleleVCF"].str.upper()
    df["AltAllele"] = df["AlternateAlleleVCF"].str.upper()

    df = df.drop_duplicates(subset=["VariationID"])
    return df


def enrich_annotations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    submitter_count = pd.to_numeric(df["NumberSubmitters"], errors="coerce").fillna(0)
    df["SubmitterCount"] = submitter_count.astype(int)

    submitter_cats = pd.to_numeric(df["SubmitterCategories"], errors="coerce").fillna(0)
    df["SubmitterCategories"] = submitter_cats.astype(int)

    df["LastEvaluatedISO"] = (
        pd.to_datetime(df["LastEvaluated"], errors="coerce")
        .dt.strftime("%Y-%m-%d")
        .fillna("")
    )

    phenotype_terms = df["PhenotypeList"].apply(parse_phenotype_terms)
    phenotype_ids = df["PhenotypeIDS"].apply(parse_phenotype_ids)

    df["PhenotypeTerms"] = phenotype_terms.apply(lambda terms: "|".join(terms))
    df["PhenotypeIDs"] = phenotype_ids.apply(lambda ids: "|".join(ids))
    df["PhenotypeTermCount"] = phenotype_terms.apply(len).astype(int)
    df["PhenotypeIDCount"] = phenotype_ids.apply(len).astype(int)
    df["PrimaryPhenotype"] = phenotype_terms.apply(
        lambda terms: terms[0] if terms else "<none>"
    )
    df["PhenotypeSources"] = phenotype_ids.apply(
        lambda ids: "|".join(extract_phenotype_sources(ids))
    )

    return df


def balance_classes(
    df: pd.DataFrame, max_per_class: int, seed: int
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    class_map = {1: "Pathogenic", 0: "Benign"}
    splits = []
    counts = {}
    for label, group in df.groupby("ClinSigSimple"):
        label_name = class_map.get(label, str(label))
        take = min(len(group), max_per_class)
        sampled = group.sample(n=take, random_state=seed)
        splits.append(sampled)
        counts[label_name] = take
    if len(splits) < 2:
        raise ValueError(
            "Expected at least two ClinSigSimple classes after filtering; "
            f"found {sorted(counts.keys())}"
        )
    result = pd.concat(splits).sample(frac=1, random_state=seed).reset_index(drop=True)
    result = result.assign(
        Label=result["ClinSigSimple"].map({1: 1, 0: 0}).astype(int),
        LabelName=result["ClinSigSimple"].map(class_map),
    )
    return result, counts


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    code_version = get_repo_version()
    git_commit = get_git_commit()
    dataset_version = f"clinvar_missense_balanced_v{code_version}"

    df_raw = load_variant_summary(args.input)
    df_filtered = filter_high_confidence(df_raw)
    df_enriched = enrich_annotations(df_filtered)
    df_balanced, counts = balance_classes(
        df_enriched, max_per_class=args.max_per_class, seed=args.seed
    )

    output_tsv = args.output_dir / "clinvar_missense_balanced.tsv"
    df_balanced.to_csv(output_tsv, sep="\t", index=False)

    filtered_with_pheno = int(df_enriched["PhenotypeTermCount"].gt(0).sum())
    balanced_with_pheno = int(df_balanced["PhenotypeTermCount"].gt(0).sum())

    stats = {
        "input_rows": len(df_raw),
        "filtered_rows": len(df_filtered),
        "balanced_rows": len(df_balanced),
        "per_class_counts": counts,
        "code_version": code_version,
        "git_commit": git_commit,
        "dataset_version": dataset_version,
        "filters": {
            "assembly": "GRCh38",
            "type": "single nucleotide variant",
            "origin": "germline",
            "review_status": sorted(ALLOWED_REVIEW_STATUSES),
            "clin_sig_simple": {1: "Pathogenic", 0: "Benign"},
        },
        "phenotype_coverage": {
            "filtered_with_annotations": filtered_with_pheno,
            "balanced_with_annotations": balanced_with_pheno,
        },
    }
    output_stats = args.output_dir / "clinvar_missense_balanced_stats.json"
    with output_stats.open("w") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved balanced dataset to {output_tsv}")
    print(f"Saved stats to {output_stats}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
