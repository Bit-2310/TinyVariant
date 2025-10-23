import pandas as pd
from pathlib import Path

from tools.build_clinvar_trm_dataset import (
    SEQ_LEN,
    build_vocab,
    encode_variant,
    load_balanced_table,
)


def test_clinsig_not_in_vocab(tmp_path):
    data = pd.DataFrame(
        [
            {
                "#AlleleID": "1",
                "Type": "single nucleotide variant",
                "Name": "NM_000000.1(GENE1):c.1A>T (p.Met1Leu)",
                "GeneSymbol": "GENE1",
                "ClinicalSignificance": "Pathogenic",
                "ClinSigSimple": 1,
                "OriginSimple": "germline",
                "Assembly": "GRCh38",
                "Chromosome": "1",
                "Start": 100,
                "Stop": 100,
                "ReviewStatus": "criteria provided, single submitter",
                "VariationID": "VAR1",
                "ReferenceAlleleVCF": "A",
                "AlternateAlleleVCF": "T",
                "ProteinFrom": "M",
                "ProteinPos": 1,
                "ProteinTo": "L",
                "RefAllele": "A",
                "AltAllele": "T",
                "Label": 1,
                "LabelName": "Pathogenic",
            },
            {
                "#AlleleID": "2",
                "Type": "single nucleotide variant",
                "Name": "NM_000000.1(GENE2):c.2G>A (p.Gly2Asp)",
                "GeneSymbol": "GENE2",
                "ClinicalSignificance": "Benign",
                "ClinSigSimple": 0,
                "OriginSimple": "germline",
                "Assembly": "GRCh38",
                "Chromosome": "2",
                "Start": 200,
                "Stop": 200,
                "ReviewStatus": "reviewed by expert panel",
                "VariationID": "VAR2",
                "ReferenceAlleleVCF": "G",
                "AlternateAlleleVCF": "A",
                "ProteinFrom": "G",
                "ProteinPos": 2,
                "ProteinTo": "D",
                "RefAllele": "G",
                "AltAllele": "A",
                "Label": 0,
                "LabelName": "Benign",
            },
        ]
    )

    tsv_path = Path(tmp_path) / "sample.tsv"
    data.to_csv(tsv_path, sep="\t", index=False)

    processed = load_balanced_table(tsv_path)
    vocab = build_vocab(processed)

    assert not any(key.startswith("CLIN:") for key in vocab.keys()), "Clinical significance should not leak into vocab"

    seq, labels = encode_variant(processed.iloc[0], vocab)
    assert len(seq) == SEQ_LEN
    assert len(labels) == SEQ_LEN
