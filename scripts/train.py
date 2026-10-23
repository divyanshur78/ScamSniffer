
from __future__ import annotations

# ruff: noqa: E402
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# Ruff: project-root setup must run before importing local modules.

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_utils import (
    DeduplicationReport,
    DomainSplitReport,
    dataset_summary,
    deduplicate_urls,
    domain_grouped_split,
)
from src.evaluation import evaluate
from src.features.url_features import extract_url_features
from src.model_comparison import (
    compare_text_models,
    compare_url_models,
    print_comparison_table,
)
from src.models.pipeline import build_text_pipeline, build_url_pipeline

URL_CSV = "urls.csv"
MSG_CSV = "messages.csv"
MIN_CLASS_COUNT = 2
DEFAULT_CV_FOLDS = 5


def load_url_data(
    data_dir: Path,
    deduplicate: bool = True,
) -> tuple[list[dict], list[int], list[str], DeduplicationReport | None]:


    csv_path = data_dir / URL_CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"URL dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    _validate_columns(df, required={"url", "label"}, source=str(csv_path))
    _validate_labels_column(df, source=str(csv_path))
    _validate_no_nulls(df[["url", "label"]], source=str(csv_path))
    _validate_min_class_counts(df["label"], source=str(csv_path))

    raw_urls   = df["url"].tolist()
    labels_raw = [int(label) for label in df["label"].tolist()]

    dedup_report: DeduplicationReport | None = None
    if deduplicate:
        raw_urls, labels_raw, dedup_report = deduplicate_urls(raw_urls, labels_raw)
        if dedup_report.n_removed:
            print(
                f"  ⚠  Deduplication: removed {dedup_report.n_removed} row(s) "
                f"({len(dedup_report.exact_duplicate_indices)} exact duplicate(s), "
                f"{len(dedup_report.conflicting_label_indices)} conflicting label(s))."
            )
        if dedup_report.near_duplicate_pairs:
            print(
                f"  ℹ  {len(dedup_report.near_duplicate_pairs)} near-duplicate pair(s) "
                "detected (reported only, not removed)."
            )

    feature_dicts: list[dict] = []
    skipped_indices: list[int] = []
    for i, raw_url in enumerate(raw_urls):
        try:
            feature_dicts.append(extract_url_features(raw_url))
        except ValueError:
            skipped_indices.append(i)
            feature_dicts.append(None)

    if skipped_indices:
        print(f"  ⚠  {len(skipped_indices)} URL(s) failed feature extraction and will be dropped.")


    valid_pairs = [
        (features, label, url)
        for features, label, url in zip(feature_dicts, labels_raw, raw_urls)
        if features is not None
    ]

    if not valid_pairs:
        raise ValueError("No valid URL features could be extracted from the dataset.")

    features, labels, valid_urls = zip(*valid_pairs)
    return (
        list(features),
        [int(label) for label in labels],
        list(valid_urls),
        dedup_report,
    )



def load_message_data(data_dir: Path) -> tuple[list[str], list[int]]:

    csv_path = data_dir / MSG_CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"Messages dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    _validate_columns(df, required={"text", "label"}, source=str(csv_path))
    _validate_labels_column(df, source=str(csv_path))
    _validate_no_nulls(df[["text", "label"]], source=str(csv_path))
    _validate_min_class_counts(df["label"], source=str(csv_path))

    texts = df["text"].tolist()
    labels = [int(label) for label in df["label"].tolist()]
    return texts, labels



def _validate_columns(
    df: pd.DataFrame, required: set[str], source: str
) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{source}: missing required columns: {sorted(missing)}. "
            f"Found: {sorted(df.columns.tolist())}."
        )


def _validate_labels_column(df: pd.DataFrame, source: str) -> None:
    bad = set(df["label"].dropna().unique().tolist()) - {0, 1}
    if bad:
        raise ValueError(
            f"{source}: 'label' column contains values outside {{0, 1}}: "
            f"{sorted(bad)}."
        )


def _validate_no_nulls(df: pd.DataFrame, source: str) -> None:
    null_counts = df.isnull().sum()
    bad_cols = null_counts[null_counts > 0]
    if not bad_cols.empty:
        details = ", ".join(f"{col}={n}" for col, n in bad_cols.items())
        raise ValueError(f"{source}: null values found — {details}.")


def _validate_min_class_counts(
    labels: pd.Series, source: str, minimum: int = MIN_CLASS_COUNT
) -> None:
    counts = labels.value_counts()
    for cls in [0, 1]:
        n = int(counts.get(cls, 0))
        if n < minimum:
            raise ValueError(
                f"{source}: class {cls} has only {n} sample(s); "
                f"minimum required is {minimum}."
            )


def _split_stratified(
    n: int,
    labels: list[int],
    test_size: float,
    random_state: int,
) -> tuple[list[int], list[int]]:

    arr = np.array(labels)
    classes, counts = np.unique(arr, return_counts=True)
    min_count = counts.min()

    n_test = max(1, int(round(n * test_size)))
    n_train = n - n_test
    min_test_per_class = max(1, int(round(min_count * test_size)))

    can_stratify = (
        min_test_per_class >= 1
        and (min_count - min_test_per_class) >= 1
        and n_train >= len(classes)
        and n_test >= len(classes)
    )

    if can_stratify:
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        train_idx, test_idx = next(sss.split(np.zeros(n), arr))
        return train_idx.tolist(), test_idx.tolist()

    rng = np.random.default_rng(random_state)
    train_idx, test_idx = [], []
    for cls in classes:
        cls_indices = np.where(arr == cls)[0]
        rng.shuffle(cls_indices)
        n_cls_test = max(1, int(round(len(cls_indices) * test_size)))
        test_idx.extend(cls_indices[:n_cls_test].tolist())
        train_idx.extend(cls_indices[n_cls_test:].tolist())
    return train_idx, test_idx


def _domain_split_or_fallback(
    urls: list[str],
    labels: list[int],
    test_size: float,
    random_state: int,
) -> tuple[list[int], list[int], DomainSplitReport]:

    train_idx, test_idx, report = domain_grouped_split(
        urls, labels, test_size=test_size, random_state=random_state
    )
    return train_idx, test_idx, report


def train_url_model(
    features: list[dict],
    labels: list[int],
    test_size: float,
    random_state: int,
    urls: list[str] | None = None,
) -> tuple[object, object]:

    if urls is not None:
        train_idx, test_idx, split_report = _domain_split_or_fallback(
            urls, labels, test_size, random_state
        )
        print(f"  Split strategy : {split_report.strategy}")
        if split_report.cv_warning if hasattr(split_report, 'cv_warning') else False:
            print(f"  ⚠  {split_report.reason}")
        else:
            print(f"  Split reason   : {split_report.reason}")
    else:
        train_idx, test_idx = _split_stratified(len(features), labels, test_size, random_state)

    X_train = [features[i] for i in train_idx]
    y_train = [labels[i] for i in train_idx]
    X_test  = [features[i] for i in test_idx]
    y_test  = [labels[i] for i in test_idx]

    pipeline = build_url_pipeline(random_state=random_state)
    pipeline.fit(X_train, y_train)

    y_pred  = pipeline.predict(X_test).tolist()
    y_proba = pipeline.predict_proba(X_test)[:, 1].tolist()

    result = evaluate(y_test, y_pred, y_proba)
    return pipeline, result


def train_text_model(
    texts: list[str],
    labels: list[int],
    test_size: float,
    random_state: int,
) -> tuple[object, object]:
    train_idx, test_idx = _split_stratified(
        len(texts), labels, test_size, random_state
    )
    X_train = [texts[i] for i in train_idx]
    y_train = [labels[i] for i in train_idx]
    X_test  = [texts[i] for i in test_idx]
    y_test  = [labels[i] for i in test_idx]

    pipeline = build_text_pipeline(random_state=random_state)
    pipeline.fit(X_train, y_train)

    y_pred  = pipeline.predict(X_test).tolist()
    y_proba = pipeline.predict_proba(X_test)[:, 1].tolist()

    result = evaluate(y_test, y_pred, y_proba)
    return pipeline, result


def save_model(pipeline: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"  [saved] Model -> {path}")


def save_report(result: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.as_dict(), indent=2), encoding="utf-8")
    print(f"  [saved] Report -> {path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train ScamSniffer URL and text detection models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/samples"),
        help="Directory containing urls.csv and messages.csv.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("models"),
        help="Directory for saved model artefacts.",
    )
    parser.add_argument(
        "--report-dir", type=Path, default=Path("reports"),
        help="Directory for evaluation JSON reports.",
    )
    parser.add_argument(
        "--random-state", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.25,
        help="Fraction of data held out for evaluation.",
    )
    parser.add_argument(
        "--compare", action="store_true", default=False,
        help="Run multi-model cross-validation comparison before training.",
    )
    parser.add_argument(
        "--no-dedup", action="store_true", default=False,
        help="Skip deduplication of the URL dataset.",
    )
    return parser


def _wrap_comparison(results: list) -> object:
    class _ComparisonReport:
        def as_dict(self):
            return {"model_comparison": [r.as_dict() for r in results]}
    return _ComparisonReport()


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    print("\n==========================================")
    print("  ScamSniffer - Model Training")
    print("==========================================")
    print(f"  data-dir    : {args.data_dir}")
    print(f"  output-dir  : {args.output_dir}")
    print(f"  report-dir  : {args.report_dir}")
    print(f"  random-state: {args.random_state}")
    print(f"  test-size   : {args.test_size}")
    print()
    print("  WARNING: The bundled data/samples/ datasets are SYNTHETIC.")
    print("  Metrics produced here do NOT reflect real-world accuracy.")
    print()

    print("-- URL model -----------------------------")
    try:
        url_features, url_labels, url_raw, url_dedup = load_url_data(args.data_dir)
        print(f"  Loaded {len(url_features)} URL samples (after deduplication).")
        print(dataset_summary(url_raw, url_labels))

        if args.compare:
            print()
            print("  -- URL model comparison (cross-validation) --")
            url_comparison = compare_url_models(
                url_features, url_labels,
                cv=DEFAULT_CV_FOLDS, random_state=args.random_state,
            )
            print_comparison_table(url_comparison)
            save_report(
                _wrap_comparison(url_comparison),
                args.report_dir / "url_model_comparison.json",
            )
            print()

        url_pipeline, url_result = train_url_model(
            url_features, url_labels, args.test_size, args.random_state,
            urls=url_raw,
        )
        print(f"  Class counts : {url_result.class_counts}")
        print(f"  Precision    : {url_result.precision:.4f}")
        print(f"  Recall       : {url_result.recall:.4f}")
        print(f"  F1-score     : {url_result.f1:.4f}")
        if url_result.roc_auc is not None:
            print(f"  ROC-AUC      : {url_result.roc_auc:.4f}")
        print(f"  Confusion matrix:\n{url_result.confusion_matrix}")
        save_model(url_pipeline, args.output_dir / "url_model.joblib")
        save_report(url_result, args.report_dir / "url_evaluation.json")
    except Exception as exc:
        print(f"  FAILED - URL model: {exc}", file=sys.stderr)
        return 1

    print()
    print("-- Text model ----------------------------")
    try:
        texts, msg_labels = load_message_data(args.data_dir)
        print(f"  Loaded {len(texts)} message samples.")

        if args.compare:
            print()
            print("  -- Text model comparison (cross-validation) --")
            text_comparison = compare_text_models(
                texts, msg_labels,
                cv=DEFAULT_CV_FOLDS, random_state=args.random_state,
            )
            print_comparison_table(text_comparison)
            save_report(
                _wrap_comparison(text_comparison),
                args.report_dir / "text_model_comparison.json",
            )
            print()

        text_pipeline, text_result = train_text_model(
            texts, msg_labels, args.test_size, args.random_state
        )
        print(f"  Class counts : {text_result.class_counts}")
        print(f"  Precision    : {text_result.precision:.4f}")
        print(f"  Recall       : {text_result.recall:.4f}")
        print(f"  F1-score     : {text_result.f1:.4f}")
        if text_result.roc_auc is not None:
            print(f"  ROC-AUC      : {text_result.roc_auc:.4f}")
        print(f"  Confusion matrix:\n{text_result.confusion_matrix}")
        save_model(text_pipeline, args.output_dir / "text_model.joblib")
        save_report(text_result, args.report_dir / "text_evaluation.json")
    except Exception as exc:
        print(f"  FAILED - Text model: {exc}", file=sys.stderr)
        return 1

    print()
    print("==========================================")
    print("  Training complete.")
    print("==========================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
