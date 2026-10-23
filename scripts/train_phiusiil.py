
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ruff: noqa: E402
from src.features.url_features import extract_url_features
from src.models.pipeline import build_url_pipeline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="data/raw/PhiUSIIL_Phishing_URL_Dataset.csv",
    )
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output", default="models/url_model_phiusiil.joblib")
    parser.add_argument("--report", default="reports/phiusiil_evaluation.json")
    args = parser.parse_args()

    dataset_path = ROOT / args.dataset
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print("Loading PhiUSIIL dataset...")
    df = pd.read_csv(dataset_path, usecols=["URL", "label"])
    df = df.dropna(subset=["URL", "label"]).copy()
    df["URL"] = df["URL"].astype(str).str.strip()
    df = df[df["URL"] != ""]
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("Expected dataset labels 0 and 1.")

    df = df.drop_duplicates(subset=["URL"], keep="first").reset_index(drop=True)

    y = (df["label"] == 0).astype(int).tolist()
    urls = df["URL"].tolist()

    print(f"Unique URLs: {len(urls):,}")
    print(f"Phishing: {sum(y):,} | Legitimate: {len(y) - sum(y):,}")
    print("Extracting the existing 26 URL-only features...")

    features = []
    valid_urls = []
    valid_labels = []

    for i, (url, label) in enumerate(zip(urls, y), start=1):
        try:
            row = extract_url_features(url)
            features.append(row)
            valid_urls.append(url)
            valid_labels.append(label)
        except (ValueError, TypeError, AttributeError):
            continue

        if i % 10000 == 0:
            print(f"  Processed {i:,}/{len(urls):,} URLs")

    if len(features) < 10 or len(set(valid_labels)) != 2:
        raise ValueError("Not enough valid URLs or one class is missing.")

    print(f"Valid URLs: {len(features):,}")

    try:
        import tldextract

        extractor = tldextract.TLDExtract(suffix_list_urls=())
        groups = []
        for url in valid_urls:
            ext = extractor(url)
            domain = ".".join(part for part in (ext.domain, ext.suffix) if part)
            groups.append(domain or url)
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        train_idx, test_idx = next(
            splitter.split(features, valid_labels, groups=groups)
        )
        if len(set(valid_labels[i] for i in train_idx)) < 2 or len(
            set(valid_labels[i] for i in test_idx)
        ) < 2:
            raise ValueError("Domain split did not include both classes.")
        print("Split strategy: registered-domain grouped holdout")
    except (ValueError, ImportError):
        indices = list(range(len(features)))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=valid_labels,
        )
        print("Split strategy: stratified holdout (domain grouping unavailable)")

    X_train = [features[i] for i in train_idx]
    y_train = [valid_labels[i] for i in train_idx]
    X_test = [features[i] for i in test_idx]
    y_test = [valid_labels[i] for i in test_idx]

    print(f"Training: {len(X_train):,} | Testing: {len(X_test):,}")
    print("Training Random Forest...")
    model = build_url_pipeline(random_state=args.random_state)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test).tolist()
    probabilities = model.predict_proba(X_test)[:, 1].tolist()

    report = {
        "dataset": "PhiUSIIL Phishing URL Dataset",
        "dataset_source": "UCI Machine Learning Repository",
        "features": "Existing ScamSniffer URL-only features",
        "dataset_label_mapping": {"0": "phishing", "1": "legitimate"},
        "model_label_mapping": {"0": "legitimate", "1": "phishing"},
        "total_valid_unique_urls": len(features),
        "train_samples": len(train_idx),
        "test_samples": len(test_idx),
        "split_strategy": "registered-domain grouped holdout",
        "accuracy": accuracy_score(y_test, predictions),
        "precision_phishing": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall_phishing": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1_phishing": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "confusion_matrix_labels": ["legitimate", "phishing"],
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["legitimate", "phishing"],
            output_dict=True,
            zero_division=0,
        ),
        "limitations": [
            "Offline URL features only; webpage-only dataset columns were not used.",
            "Evaluation results do not guarantee real-world performance.",
            "Domain grouping can fall back to a stratified split if required.",
        ],
    }

    report_path = ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)

    print("\nEvaluation results")
    print(f"Accuracy:          {report['accuracy']:.4f}")
    print(f"Precision phishing:{report['precision_phishing']:.4f}")
    print(f"Recall phishing:   {report['recall_phishing']:.4f}")
    print(f"F1 phishing:       {report['f1_phishing']:.4f}")
    print(f"ROC-AUC:           {report['roc_auc']:.4f}")
    print("Confusion matrix [legitimate, phishing]:")
    print(report["confusion_matrix"])
    print(f"\nCandidate model saved to: {output_path}")
    print(f"Evaluation report saved to: {report_path}")
    print("The existing url_model.joblib has NOT been replaced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
