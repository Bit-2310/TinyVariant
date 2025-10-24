import json
from pathlib import Path

import pandas as pd
import pytest

from tools.build_clinvar_trm_dataset import (
    SEQ_LEN,
    AA_PROPERTY,
    PHENOTYPE_COUNT_BUCKETS,
    PHENOTYPE_TOP_K,
    PHENOTYPE_SOURCE_TOP_K,
    PHENOTYPE_SLOT_COLUMN_NAMES,
    PHENOTYPE_TOKEN_PREFIXES,
    SUBMITTER_BUCKETS,
    EVAL_YEAR_BUCKETS,
    build_feature_buckets,
    build_vocab,
    encode_variant,
    load_balanced_table,
    apply_feature_buckets,
)

DATA_PATH = Path("data/clinvar/processed/clinvar_missense_balanced.tsv")
TRAIN_METADATA_PATH = Path("data/clinvar/processed/clinvar_trm/train/dataset.json")


@pytest.mark.skipif(not DATA_PATH.exists(), reason="ClinVar balanced TSV not found; run dataset builder first.")
def test_no_clinsig_leakage_and_sequence_length():
    processed = load_balanced_table(DATA_PATH)
    feature_buckets = build_feature_buckets(processed)
    processed = apply_feature_buckets(processed, feature_buckets)
    sample = processed.sample(n=min(64, len(processed)), random_state=42)

    vocab = build_vocab(sample)
    assert not any(key.startswith("CLIN:") for key in vocab.keys()), (
        "ClinicalSignificance-derived tokens should be excluded from vocabulary"
    )

    seq, labels = encode_variant(sample.iloc[0], vocab)
    assert len(seq) == SEQ_LEN
    assert len(labels) == SEQ_LEN


@pytest.mark.skipif(not DATA_PATH.exists(), reason="ClinVar balanced TSV not found; run dataset builder first.")
def test_case_study_variant_properties():
    processed = load_balanced_table(DATA_PATH)
    feature_buckets = build_feature_buckets(processed)
    processed = apply_feature_buckets(processed, feature_buckets)
    row = processed.iloc[0]

    expected_from = AA_PROPERTY.get(row["ProteinFrom"], "<unknown>")
    expected_to = AA_PROPERTY.get(row["ProteinTo"], "<unknown>")
    assert row["AAPropFrom"] == expected_from
    assert row["AAPropTo"] == expected_to

    counts = processed["GeneSymbol"].value_counts()
    gene = row["GeneSymbol"]
    if counts.get(gene, 0) >= 5:
        assert row["GeneBucket"] == gene
    else:
        assert row["GeneBucket"] == "<RARE>"


@pytest.mark.skipif(not DATA_PATH.exists(), reason="ClinVar balanced TSV not found; run dataset builder first.")
def test_phenotype_tokens_present():
    processed = load_balanced_table(DATA_PATH)
    feature_buckets = build_feature_buckets(processed)
    processed = apply_feature_buckets(processed, feature_buckets)

    subset = processed[processed["PhenotypeTermCount"] > 0]
    if subset.empty:
        pytest.skip("No phenotype annotations present in processed dataset.")

    row = subset.iloc[0]
    vocab = build_vocab(processed)
    seq, _ = encode_variant(row, vocab)
    id_to_token = {idx: token for token, idx in vocab.items()}
    tokens = [id_to_token[idx] for idx in seq]

    for column_name, prefix in zip(PHENOTYPE_SLOT_COLUMN_NAMES, PHENOTYPE_TOKEN_PREFIXES):
        if row[column_name] != "<none>":
            assert any(tok.startswith(f"{prefix}:") and tok.endswith(row[column_name]) for tok in tokens)

    assert any(tok.startswith("SUBMITTERS:") for tok in tokens)
    assert any(tok.startswith("EVAL:") for tok in tokens)


@pytest.mark.skipif(not DATA_PATH.exists(), reason="ClinVar balanced TSV not found; run dataset builder first.")
def test_bucketed_fields_within_expected_ranges():
    processed = load_balanced_table(DATA_PATH)
    feature_buckets = build_feature_buckets(processed)
    processed = apply_feature_buckets(processed, feature_buckets)

    primary_expected = set(feature_buckets.phenotype_terms) | {"<none>", "<other>"}
    secondary_expected = primary_expected
    source_expected = set(feature_buckets.phenotype_sources) | {"<none>", "<other>"}
    count_expected = {name for _, name in PHENOTYPE_COUNT_BUCKETS}
    submitter_expected = {name for _, name in SUBMITTER_BUCKETS}
    eval_expected = {name for _, name in EVAL_YEAR_BUCKETS} | {"unknown"}

    assert set(processed["PhenotypePrimaryToken"].unique()).issubset(primary_expected)
    assert set(processed["PhenotypeSecondaryToken"].unique()).issubset(secondary_expected)
    assert set(processed["PhenotypeTertiaryToken"].unique()).issubset(primary_expected)
    assert set(processed["PhenotypeSourceToken"].unique()).issubset(source_expected)
    assert set(processed["PhenotypeCountBucket"].unique()).issubset(count_expected)
    assert set(processed["SubmitterBucket"].unique()).issubset(submitter_expected)
    assert set(processed["EvalRecencyBucket"].unique()).issubset(eval_expected)

    phenotype_counts = processed["PhenotypeTermCount"]
    assert (phenotype_counts >= 0).all()
    assert phenotype_counts.max() >= PHENOTYPE_TOP_K // 10  # sanity: some records should have multiple terms


@pytest.mark.skipif(not TRAIN_METADATA_PATH.exists(), reason="TRM train metadata not found; run dataset builder first.")
def test_dataset_metadata_records_feature_info():
    metadata = json.loads(TRAIN_METADATA_PATH.read_text())
    feature_meta = metadata.get("feature_metadata")
    assert feature_meta is not None, "feature_metadata should be stored in dataset.json"

    for key in (
        "phenotype_terms_top",
        "phenotype_sources_top",
        "phenotype_top_k",
        "phenotype_source_top_k",
        "phenotype_slot_columns",
        "phenotype_count_buckets",
        "submitter_buckets",
        "eval_year_buckets",
    ):
        assert key in feature_meta, f"Missing {key} in feature metadata"

    assert feature_meta["phenotype_top_k"] == PHENOTYPE_TOP_K
    assert feature_meta["phenotype_source_top_k"] == PHENOTYPE_SOURCE_TOP_K
    assert feature_meta["phenotype_slot_columns"] == PHENOTYPE_SLOT_COLUMN_NAMES
    assert len(feature_meta["phenotype_terms_top"]) <= PHENOTYPE_TOP_K
    assert len(feature_meta["phenotype_sources_top"]) <= PHENOTYPE_SOURCE_TOP_K
