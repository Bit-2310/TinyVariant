import pandas as pd
import pytest
from pathlib import Path

from tools.build_clinvar_trm_dataset import (
    SEQ_LEN,
    build_vocab,
    encode_variant,
    load_balanced_table,
)

DATA_PATH = Path("data/clinvar/processed/clinvar_missense_balanced.tsv")


@pytest.mark.skipif(not DATA_PATH.exists(), reason="ClinVar balanced TSV not found; run dataset builder first.")
def test_no_clinsig_leakage_and_sequence_length():
    processed = load_balanced_table(DATA_PATH)
    sample = processed.sample(n=min(64, len(processed)), random_state=42)

    vocab = build_vocab(sample)
    assert not any(key.startswith("CLIN:") for key in vocab.keys()), (
        "ClinicalSignificance-derived tokens should be excluded from vocabulary"
    )

    seq, labels = encode_variant(sample.iloc[0], vocab)
    assert len(seq) == SEQ_LEN
    assert len(labels) == SEQ_LEN
