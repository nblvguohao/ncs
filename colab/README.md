# AlphaFold-Multimer Colab Notebook

## Quick Start

1. Open Google Colab: https://colab.research.google.com/
2. Set Runtime -> GPU (A100 recommended)
3. Upload `run_af_multimer_colab.py`
4. Run all cells

## What it does

- Downloads all required data from GitHub (no manual upload needed)
- Runs AlphaFold-Multimer on 10 GPCR-Gα complexes
- Analyzes predicted contacts vs BW sites
- Performs statistical tests (Fisher's exact test)
- Generates publication-quality figures
- Packages all results for download

## Expected Runtime

- **A100 GPU**: 2-4 hours
- **V100 GPU**: 4-8 hours  
- **T4 GPU**: 8-16 hours

## Output

- `af_validation_results.csv` - Contact analysis per receptor
- `af_validation_summary.json` - Statistical summary
- `fig_af_validation.png` - 4-panel figure
- `pdb_structures/*.pdb` - Predicted structures
- `af_multimer_results.zip` - All results packaged

## Key Analysis

Tests whether FDR-significant BW sites show higher contact frequency with Gα in predicted structures, validating our coupling determinants.
