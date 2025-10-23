# TinyVariant data workspace

This directory keeps public datasets used by the TinyVariant proof of concept.

## Layout

- `clinvar/raw/` – untouched downloads (e.g., NCBI’s `variant_summary.txt.gz` dump).
- `clinvar/processed/` – cleaned NumPy/metadata artifacts that plug into the TRM pipeline.
- `download_data.sh` – helper script that pulls the upstream files.

## ClinVar variant summary

We start from the ClinVar tab-delimited export because it bundles per-variant
labels, review statuses, genomic coordinates, and consequence strings in one
place. Expect roughly 350 MB on disk for the compressed file (≈1.5 GB when
unpacked).

Download with:

```bash
bash data/download_data.sh
```

Pass `--force` to re-download files even if they already exist. The script uses
`aria2c` when available for multi-connection downloads, and falls back to `wget`
otherwise.

Once the raw dump is in place we will generate a balanced, missense-only dataset
under `clinvar/processed/` ready for the TinyVariant training loop.

To build the default 5 k balanced slice:
```bash
python tools/prepare_clinvar_dataset.py
python tools/build_clinvar_trm_dataset.py
```

To expand to 10 k (5 k per class):
```bash
python tools/prepare_clinvar_dataset.py --max-per-class 5000
python tools/build_clinvar_trm_dataset.py
```
