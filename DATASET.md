# BioSciReview-Bench dataset

Junhao Qiu, Independent Researcher · REALM at EMNLP 2026

DOI: [10.5281/zenodo.23862936](https://doi.org/10.5281/zenodo.23862936)

200 synthetic bioinformatics code snippets form 100 correct–buggy pairs: 50 Python pairs and 50 R pairs, across five domains and six methodology-error categories. The archive also includes predictions from six coding-agent configurations and their evaluation results.

## Contents

| Location | Contents |
| --- | --- |
| `data/snippets/` | Python and R code samples |
| `data/annotations.jsonl` | Labels, error locations, categories and explanations |
| `data/taxonomy.yaml` | Domain and error definitions |
| `results/<run>/rendered/` | Evaluation inputs, prompts, labels and line mappings |
| `results/<run>/parsed/` | 1,200 predictions: 200 per configuration |
| `results/<run>/scores/` | Evaluation metrics |
| `results/<run>/analysis/` | Tables, confidence intervals and error analyses |

`<run>` is the results directory supplied in the archive. `pair_id` links matched samples. `bug_present` gives the label; `bug_category` gives the error type. `bug_line` refers to the original source. Scoring uses `rendered_bug_line`, which refers to the numbered evaluation input. Keep labels and pair information hidden from evaluated agents.

## Use and citation

Follow the [code repository](https://github.com/qchiujunhao/BioSciReview-Bench) README to extract the data and results. `DATASET_SHA256SUMS.txt` in the archive lists file checksums; the separately downloaded `SHA256SUMS.txt` verifies the ZIP.

Code samples use MIT. Annotations, documentation and results use CC BY 4.0; see `LICENSING.md`. Cite the [REALM 2026 paper](https://aclanthology.org/2026.realm-1.65/) and this dataset using `DATASET_CITATION.cff`.
