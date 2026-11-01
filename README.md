# BioSciReview-Bench

A benchmark for detecting, locating and classifying scientific methodology errors in bioinformatics code that can run without obvious failures.

[Paper](https://aclanthology.org/2026.realm-1.65/) · [Dataset guide](DATASET.md) · [Licenses](LICENSING.md)

## Dataset and evaluation

The dataset contains **200 synthetic code snippets in 100 correct–buggy pairs**: 50 Python pairs and 50 R pairs, covering five bioinformatics domains and six error categories.

We evaluate six coding-agent configurations: Codex with GPT-5.4 and GPT-5.5; Claude Code with Sonnet 4.6, Opus 4.8, DeepSeek-V4-Flash and DeepSeek-V4-Pro. Each configuration reviews each snippet once, using the same prompt and output format. Each review receives one snippet without labels or its paired example.

Metrics measure detection, line localization and full diagnosis (detection + line + error category). Localization is evaluated at the exact line and within ±3 lines. Confidence intervals resample complete pairs.

## Download and reproduce

[Download the dataset and results](https://doi.org/10.5281/zenodo.23862936). Evaluation code is hosted in this repository.

Save the downloaded ZIP as `dataset.zip`. With Python 3.11+, extract the data and results into this repository:

```bash
unzip dataset.zip 'data/*' 'results/*'
python3 -m pip install -r requirements.txt
python3 scripts/validate_dataset_counts.py
python3 scripts/quality_check.py
python3 -m unittest discover -s tests -v
```

Reproduce the analysis from the included predictions, without new model calls:

```bash
python3 eval/analysis.py
```

The script uses the single extracted results directory; use `--run NAME` to select one when several exist. It overwrites result tables, so keep the original ZIP for comparison. To run new evaluations, follow [EVALUATION.md](EVALUATION.md).

## License and citation

Code and code samples: [MIT](LICENSE). Annotations, documentation and results: [CC BY 4.0](LICENSES/CC-BY-4.0.txt). See [LICENSING.md](LICENSING.md) for scope.

Please cite the [REALM 2026 paper](https://doi.org/10.18653/2026.realm-1.65) and the [dataset](https://doi.org/10.5281/zenodo.23862936). [CITATION.cff](CITATION.cff) provides the published paper as the preferred citation.
