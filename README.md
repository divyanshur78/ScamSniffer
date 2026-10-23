# ScamSniffer 🛡️

### AI-Powered Phishing Detection for URLs, Emails, SMS, and Chat Messages

ScamSniffer is a machine-learning-based cybersecurity project designed to identify potentially suspicious URLs and phishing-related messages. It combines URL feature extraction, text classification, input validation, and a Streamlit web interface to help users assess potential phishing threats.

> ⚠️ **Disclaimer:** ScamSniffer is a research and educational project. Its predictions are probabilistic and are not guaranteed to be correct. Do not rely solely on this tool to make security decisions. Always apply human judgment and consult qualified cybersecurity professionals when necessary.

---

## Features

- 🔗 **URL Phishing Detection:** Analyzes URL characteristics, including entropy, suspicious keywords, top-level domains, and IP-based hostnames.
- 📧 **Message Classification:** Uses a TF-IDF and n-gram text pipeline to classify email, SMS, and chat messages.
- 🌐 **Streamlit Web Interface:** Provides a browser-based interface for submitting URLs and messages for analysis.
- 📊 **Model Evaluation:** Supports classification metrics, model comparison, and evaluation reports.
- 🔒 **Input Validation:** Applies input validation and length limits to help handle potentially unsafe inputs.
- 🧪 **Automated Testing:** Includes tests for detection logic, feature extraction, validation, model pipelines, and adversarial inputs.

---

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Machine Learning | scikit-learn |
| Data Processing | pandas, NumPy |
| Model Persistence | joblib |
| Web Interface | Streamlit |
| URL Analysis | tldextract, validators |
| Testing | pytest, pytest-cov |
| Code Quality | Ruff |

---

## Project Structure

```text
ScamSniffer/
├── app/
│   └── main.py
├── data/
│   ├── README.md
│   ├── raw/                 # Local training datasets; excluded from Git
│   └── samples/
│       ├── urls.csv
│       └── messages.csv
├── models/
│   ├── url_model.joblib            # Active URL model
│   ├── text_model.joblib           # Active text model
│   └── url_model_phiusiil.joblib   # Experimental model
├── reports/                 # Generated evaluation reports
├── scripts/
│   ├── train.py
│   └── train_phiusiil.py
├── src/
│   ├── features/
│   │   ├── text_features.py
│   │   └── url_features.py
│   ├── models/
│   │   └── pipeline.py
│   ├── data_utils.py
│   ├── detector.py
│   ├── evaluation.py
│   ├── model_comparison.py
│   └── validator.py
├── tests/
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
└── README.md
```

**Note:** The active model files (`models/url_model.joblib` and `models/text_model.joblib`) are included in the repository. The experimental model (`models/url_model_phiusiil.joblib`), raw training datasets, and generated reports are excluded from Git. A fresh repository clone will include the active models but may not include the experimental model, raw datasets, or generated reports.
---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/divyanshur78/ScamSniffer.git
cd ScamSniffer
```

Replace `<YOUR_GITHUB_REPOSITORY_URL>` with the actual URL of your GitHub repository.

### 2. Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell prevents virtual environment activation, you can run the environment's Python executable directly.

### 3. Install dependencies

Install the application dependencies:

```powershell
python -m pip install -r requirements.txt
```

For development and testing, install the additional dependencies:

```powershell
python -m pip install -r requirements-dev.txt
```

---

## Model Setup and Training

ScamSniffer currently uses two active model files:

- `models/url_model.joblib`
- `models/text_model.joblib`

The active model files are included in the Git repository:

- `models/url_model.joblib`
- `models/text_model.joblib`

No separate model download is required when cloning the repository normally. Ensure both files are present in the `models/` directory before running the application.

The experimental model, `models/url_model_phiusiil.joblib`, is excluded from Git and is not required to run the application.

Before running the application from a fresh clone, ensure the required model files are available in the `models/` directory.

### Train with sample datasets

The project provides a training script:

```powershell
python scripts/train.py --data-dir data/samples --output-dir models/
```

The bundled sample datasets are small synthetic examples intended for testing and continuous integration. They are not representative of real-world phishing data and should not be used as the sole basis for production training.

### Experimental PhiUSIIL model

The project also includes `scripts/train_phiusiil.py` for experimentation with the PhiUSIIL phishing URL dataset.

The experimental model is saved separately as `models/url_model_phiusiil.joblib`. It does not automatically replace the active URL model.

Evaluate experimental models carefully before considering them for application use.

---

## Run the Application

After installing dependencies and ensuring the active model files are available, start the Streamlit application:

```powershell
python -m streamlit run app/main.py
```

Streamlit will display a local address in the terminal. Open that address in your browser to use ScamSniffer.

---

## Run Tests

Run the complete test suite:

```powershell
python -m pytest -q
```

To generate coverage information:

```powershell
python -m pytest --cov=src --cov-report=term-missing
```

**Development test status:** The project passed 676 automated tests in the development environment at the time of documentation. Results may change if the code, dependencies, or environment changes.

---

## Data and Privacy

The `data/samples/` directory contains small synthetic datasets used for testing and continuous integration.

Refer to [`data/README.md`](data/README.md) for information about:

- Dataset structure and label conventions
- Data provenance and suggested public datasets
- Training usage
- Privacy and legal considerations

Do not add personal, confidential, or sensitive information to project datasets or test fixtures without appropriate authorization.

---

## Limitations

- Predictions do not guarantee that a URL or message is malicious or legitimate.
- Detection quality depends on the training data, feature extraction, model design, and evaluation methodology.
- Synthetic sample datasets are not sufficient to establish real-world detection performance.
- Legitimate URLs or messages may be flagged incorrectly.
- Phishing attempts may go undetected.
- Experimental models require separate evaluation before use.
- ScamSniffer is not a replacement for browser security protections, email security systems, or professional security analysis.

---

## Future Improvements

Potential future improvements include:

- Evaluation using larger, representative, independently labelled datasets
- Improved detection of previously unseen phishing techniques
- More comprehensive false-positive and false-negative analysis
- Improved reproducibility of model training and evaluation
- Clearer documentation of model performance and limitations

---

## Project Status

ScamSniffer is a research and educational cybersecurity project. Further evaluation is required before relying on it in real-world security environments.
