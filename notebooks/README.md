# Notebooks (planned)

Notebooks are only for exploration, analysis and the demo. All training and
evaluation runs through the scripts in `src/` so results are reproducible.

| Notebook | Owner | What it does |
| --- | --- | --- |
| `01_eda.ipynb` | Justin | Class counts, animals per image, box-size histograms, sample grid (works with `data/annotations/*.json`; no GPU) |
| `02_error_analysis.ipynb` | Fanyou | Reads `results/predictions/*.csv` and `results/results.csv`; confusion pairs, mistake galleries, robustness line chart |
| `demo.ipynb` | Jiacheng | Runs `src.pipeline.detect_and_classify` on new photos |
| `colab_runner.ipynb` | Jiacheng | Clone repo, install, download data to `/content`, save runs to Google Drive |
