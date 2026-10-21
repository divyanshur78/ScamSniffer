# ScamSniffer

> **⚠️ Disclaimer:** ScamSniffer is a research and educational tool. Predictions are
> probabilistic and **are not guaranteed to be correct**. Do not rely solely on this
> tool to make security decisions. Always apply human judgement and consult qualified
> cybersecurity professionals.

AI-powered phishing detection for suspicious URLs, emails, SMS, and chat messages
using machine learning.

---

## Features

- 🔗 **URL analysis** — entropy, TLD reputation, keyword heuristics, IP-hostname detection
- 📧 **Email / SMS / chat classification** — TF-IDF + n-gram text pipeline
- 🌐 **Streamlit web interface** — simple, browser-based UI with confidence scores
- 📊 **Model evaluation** — precision, recall, F1-score, confusion matrix reports
- 🔒 **Safe input handling** — length limits, type validation, no URL fetching

## Project Structure

```
ScamSniffer/
├── app/                    # Streamlit web interface
├── src/                    # Core library
│   ├── features/           # URL & text feature extractors
│   └── models/             # sklearn pipeline wrapper
├── scripts/                # Training CLI
├── data/samples/           # Synthetic sample data for CI
├── models/                 # Persisted model artefacts (git-ignored)
├── reports/                # Evaluation plots & metrics (git-ignored)
├── tests/                  # pytest unit tests
└── docs/                   # Architecture & usage documentation
```

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements-dev.txt
```

### 2. Train the models

```bash
python scripts/train.py --data-dir data/samples --output-dir models/
```

### 3. Run the web app

```bash
streamlit run app/main.py
```

### 4. Run the test suite

```bash
pytest --cov=src --cov-report=term-missing
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| ML framework | scikit-learn |
| Data | pandas, numpy |
| Model persistence | joblib |
| Web UI | Streamlit |
| URL parsing | tldextract |
| Input validation | validators |
| Testing | pytest, pytest-cov |
| Linting | Ruff |

## Data

See [`data/README.md`](data/README.md) for dataset provenance, labelling conventions,
and recommended public phishing datasets for production training.

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — system design & data flow
- [`docs/usage.md`](docs/usage.md) — detailed usage guide
- [`docs/data_sources.md`](docs/data_sources.md) — data sourcing guidance

## Licence

MIT — see [LICENSE](LICENSE).
