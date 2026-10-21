# Data Directory

This directory holds labelled datasets used to train and evaluate ScamSniffer's
detection models. **No real user data is ever stored here.**

## Structure

```
data/
└── samples/
    ├── urls.csv       # Labelled URL dataset  (columns: url, label)
    └── messages.csv   # Labelled text dataset (columns: text, type, label)
```

## Label Convention

| Value | Meaning |
|-------|---------|
| `0`   | Legitimate / benign |
| `1`   | Phishing / scam |

The `type` column in `messages.csv` may be one of: `email`, `sms`, `chat`.

## Data Sources & Provenance

The sample files bundled in this repository are **small synthetic examples**
created solely for unit tests and CI. They are not representative of real-world
phishing distributions and must not be used as the sole basis for production
model training.

Recommended public datasets for training a production model:

| Dataset | URL | Notes |
|---------|-----|-------|
| PhishTank | <https://www.phishtank.com/developer_info.php> | Verified phishing URLs |
| OpenPhish | <https://openphish.com/> | Phishing feeds |
| UCI SMS Spam Collection | <https://archive.ics.uci.edu/dataset/228/sms+spam+collection> | SMS spam/ham |
| Enron Email Dataset | <https://www.cs.cmu.edu/~enron/> | Email corpus |
| TREC 2007 Spam Corpus | <https://plg.uwaterloo.ca/~gvcormac/treccorpus07/> | Email spam |

## Usage

Scripts read from this directory automatically:

```bash
python scripts/train.py --data-dir data/samples --output-dir models/
```

## Privacy & Legal

Ensure any data you add complies with the licence of the source dataset and
does not contain personally identifiable information (PII).
