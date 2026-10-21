"""
src/data_utils.py
=================
Dataset quality utilities for ScamSniffer training data.

Provides deterministic, offline-only helpers for:
  - Exact and near-duplicate URL detection
  - Conflicting-label detection (same URL, different labels)
  - Registered-domain extraction for grouping
  - Domain-grouped train/test splitting (with documented fallback)
  - Dataset provenance summary reporting

⚠️  No network requests are made at any point.  tldextract uses its bundled
    snapshot; no live Public Suffix List update is performed.

Public API
----------
DeduplicationReport
    Dataclass summarising exact duplicates, near-duplicates, and conflicting
    labels found in a URL list.

deduplicate_urls(urls, labels)
    Remove exact duplicates and conflicting-label rows; return cleaned
    (urls, labels) plus a DeduplicationReport.

domain_grouped_split(urls, labels, test_size, random_state, min_group_size)
    Split a URL dataset by registered domain to prevent data leakage between
    train and test sets.  Falls back to stratified split when the dataset is
    too small for a clean domain split.  Returns (train_idx, test_idx, report).

DomainSplitReport
    Dataclass describing which strategy was used and summary statistics.

dataset_summary(urls, labels)
    Return a plain-text summary of dataset statistics for logging/documentation.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

import tldextract

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _registered_domain(url: str) -> str:
    """
    Return the registered domain (domain + suffix) from *url*, lower-cased.

    Returns an empty string for bare IPs, localhost, or unparseable inputs.
    """
    try:
        ext = tldextract.extract(url)
        if ext.domain and ext.suffix:
            return f"{ext.domain}.{ext.suffix}".lower()
    except Exception:
        pass
    return ""


def _normalise_url(url: str) -> str:
    """
    Return a case-folded, scheme-stripped form of *url* for near-duplicate
    detection.  Strips trailing slashes and lowercases the entire URL.
    """
    normalised = url.strip().lower()
    # Strip scheme
    normalised = re.sub(r"^https?://", "", normalised)
    # Strip trailing slash
    normalised = normalised.rstrip("/")
    return normalised


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


@dataclass
class DeduplicationReport:
    """
    Summary of duplicates and conflicts found in a URL dataset.

    Attributes
    ----------
    exact_duplicate_indices   : list[int]
        Row indices that are exact string duplicates of an earlier row.
    near_duplicate_pairs      : list[tuple[int, int]]
        Pairs of row indices whose normalised forms (scheme-stripped,
        case-folded, slash-trimmed) are identical.
    conflicting_label_indices : list[int]
        Row indices whose URL appeared earlier with a *different* label.
    n_removed                 : int
        Total number of rows removed (exact duplicates + conflicting labels).
    """

    exact_duplicate_indices:   list[int] = field(default_factory=list)
    near_duplicate_pairs:      list[tuple[int, int]] = field(default_factory=list)
    conflicting_label_indices: list[int] = field(default_factory=list)
    n_removed:                 int = 0

    def as_dict(self) -> dict:
        return {
            "exact_duplicate_indices":   self.exact_duplicate_indices,
            "near_duplicate_pairs":      [(a, b) for a, b in self.near_duplicate_pairs],
            "conflicting_label_indices": self.conflicting_label_indices,
            "n_removed":                 self.n_removed,
        }


def deduplicate_urls(
    urls: Sequence[str],
    labels: Sequence[int],
) -> tuple[list[str], list[int], DeduplicationReport]:
    """
    Detect and remove exact duplicates and conflicting-label rows.

    Strategy
    --------
    1. Exact duplicates: identical URL string.  Keep first occurrence.
    2. Conflicting labels: same URL string appears with both label 0 and 1.
       Both the earlier and later occurrences are removed.
    3. Near-duplicates (scheme/case/slash differences) are *reported* but
       NOT automatically removed, because the model may legitimately want
       the same path under http and https as two separate training samples.

    Parameters
    ----------
    urls   : sequence of URL strings (already validated or raw).
    labels : corresponding integer labels (0 or 1).

    Returns
    -------
    (cleaned_urls, cleaned_labels, report)
    """
    report = DeduplicationReport()

    # --- Pass 1: detect conflicting labels ---
    # Map url -> first seen label
    first_seen_label: dict[str, int] = {}
    conflict_urls: set[str] = set()
    for url, label in zip(urls, labels):
        if url in first_seen_label:
            if first_seen_label[url] != label:
                conflict_urls.add(url)
        else:
            first_seen_label[url] = label

    # --- Pass 2: detect exact duplicates and build keep set ---
    seen_urls: set[str] = set()
    to_remove: set[int] = set()

    for i, (url, label) in enumerate(zip(urls, labels)):
        if url in conflict_urls:
            report.conflicting_label_indices.append(i)
            to_remove.add(i)
            continue
        if url in seen_urls:
            report.exact_duplicate_indices.append(i)
            to_remove.add(i)
        else:
            seen_urls.add(url)

    report.n_removed = len(to_remove)

    # --- Pass 3: near-duplicate detection (report only) ---
    norm_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, url in enumerate(urls):
        if i not in to_remove:
            norm_to_indices[_normalise_url(url)].append(i)
    for norm, indices in norm_to_indices.items():
        if len(indices) >= 2:
            for j in range(len(indices)):
                for k in range(j + 1, len(indices)):
                    report.near_duplicate_pairs.append((indices[j], indices[k]))

    # Build cleaned lists
    cleaned_urls   = [u for i, u in enumerate(urls)   if i not in to_remove]
    cleaned_labels = [l for i, l in enumerate(labels) if i not in to_remove]

    return cleaned_urls, cleaned_labels, report


# ---------------------------------------------------------------------------
# Domain-grouped split
# ---------------------------------------------------------------------------


@dataclass
class DomainSplitReport:
    """
    Metadata about which splitting strategy was applied.

    Attributes
    ----------
    strategy              : str
        ``"domain_grouped"`` or ``"stratified_fallback"``.
    reason                : str
        Human-readable explanation of why this strategy was chosen.
    n_total               : int   Total samples after deduplication.
    n_train               : int   Training set size.
    n_test                : int   Test set size.
    n_domains             : int   Number of distinct registered domains.
    n_single_sample_domains: int  Domains with only one sample.
    train_class_counts    : dict  {label: count} for train set.
    test_class_counts     : dict  {label: count} for test set.
    """

    strategy:               str = ""
    reason:                 str = ""
    n_total:                int = 0
    n_train:                int = 0
    n_test:                 int = 0
    n_domains:              int = 0
    n_single_sample_domains: int = 0
    train_class_counts:     dict = field(default_factory=dict)
    test_class_counts:      dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "strategy":               self.strategy,
            "reason":                 self.reason,
            "n_total":                self.n_total,
            "n_train":                self.n_train,
            "n_test":                 self.n_test,
            "n_domains":              self.n_domains,
            "n_single_sample_domains": self.n_single_sample_domains,
            "train_class_counts":     self.train_class_counts,
            "test_class_counts":      self.test_class_counts,
        }


# Minimum number of independent domain groups needed per class for a clean
# domain-grouped split (must leave at least one domain per class in test).
_MIN_DOMAIN_GROUPS_PER_CLASS = 2


def domain_grouped_split(
    urls: Sequence[str],
    labels: Sequence[int],
    test_size: float = 0.25,
    random_state: int = 42,
    min_group_size: int = 1,
) -> tuple[list[int], list[int], DomainSplitReport]:
    """
    Split URL indices by registered domain to prevent data leakage.

    Each registered domain is placed entirely in either train or test,
    so the model is evaluated on domains it has never seen.  This is the
    preferred strategy when enough distinct domains per class are available.

    Fallback
    --------
    When there are fewer than ``_MIN_DOMAIN_GROUPS_PER_CLASS`` distinct
    domain groups per class, a stratified index split is used instead and
    the report documents the reason clearly.

    Parameters
    ----------
    urls          : sequence of URL strings.
    labels        : corresponding labels (0 = legitimate, 1 = phishing).
    test_size     : fraction of *domains* to place in test (not samples).
    random_state  : seed for deterministic shuffling.
    min_group_size: minimum samples a domain group must have to participate
                    in the split; singletons are documented in the report.

    Returns
    -------
    (train_indices, test_indices, report)
        All indices are integers into the original *urls* / *labels* sequences.
    """
    import numpy as np

    urls_list   = list(urls)
    labels_list = list(labels)
    n           = len(urls_list)
    report      = DomainSplitReport(n_total=n)

    # --- Group indices by registered domain ---
    domain_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, url in enumerate(urls_list):
        rd = _registered_domain(url) or f"__no_domain_{i}__"
        domain_to_indices[rd].append(i)

    report.n_domains = len(domain_to_indices)
    report.n_single_sample_domains = sum(
        1 for idxs in domain_to_indices.values() if len(idxs) == 1
    )

    # --- Determine dominant label per domain group ---
    # A domain group gets the label of its majority sample.
    # Ties go to the phishing class (1) as the conservative choice.
    domain_labels: dict[str, int] = {}
    for domain, idxs in domain_to_indices.items():
        cls_counts = defaultdict(int)
        for i in idxs:
            cls_counts[labels_list[i]] += 1
        domain_labels[domain] = max(cls_counts, key=lambda c: (cls_counts[c], c))

    # Count domain groups per class
    cls_domains: dict[int, list[str]] = defaultdict(list)
    for domain, cls in domain_labels.items():
        cls_domains[cls].append(domain)

    can_domain_split = all(
        len(cls_domains.get(cls, [])) >= _MIN_DOMAIN_GROUPS_PER_CLASS
        for cls in [0, 1]
    )

    if not can_domain_split:
        reason = (
            f"Insufficient distinct domain groups for a clean domain-grouped split "
            f"(need >={_MIN_DOMAIN_GROUPS_PER_CLASS} per class; "
            f"found: {dict((c, len(ds)) for c, ds in cls_domains.items())}). "
            "Falling back to stratified index split."
        )
        report.strategy = "stratified_fallback"
        report.reason   = reason
        train_idx, test_idx = _stratified_fallback(labels_list, test_size, random_state)
    else:
        # --- Domain-grouped split ---
        rng = np.random.default_rng(random_state)
        test_domains:  set[str] = set()
        train_domains: set[str] = set()

        for cls in [0, 1]:
            d_list = cls_domains[cls][:]
            rng.shuffle(d_list)
            n_test_domains = max(1, int(round(len(d_list) * test_size)))
            test_domains.update(d_list[:n_test_domains])
            train_domains.update(d_list[n_test_domains:])

        train_idx = [
            i for domain, idxs in domain_to_indices.items()
            if domain in train_domains
            for i in idxs
        ]
        test_idx = [
            i for domain, idxs in domain_to_indices.items()
            if domain in test_domains
            for i in idxs
        ]
        # Domains that ended up in neither (due to rounding) go to train
        neither = [
            i for domain, idxs in domain_to_indices.items()
            if domain not in train_domains and domain not in test_domains
            for i in idxs
        ]
        train_idx.extend(neither)

        report.strategy = "domain_grouped"
        report.reason   = (
            f"Domain-grouped split: {len(test_domains)} domain(s) → test, "
            f"{len(train_domains)} domain(s) → train."
        )

    report.n_train = len(train_idx)
    report.n_test  = len(test_idx)

    # Compute per-class counts for report
    train_counts: dict[int, int] = defaultdict(int)
    test_counts:  dict[int, int] = defaultdict(int)
    for i in train_idx:
        train_counts[labels_list[i]] += 1
    for i in test_idx:
        test_counts[labels_list[i]] += 1
    report.train_class_counts = dict(train_counts)
    report.test_class_counts  = dict(test_counts)

    return train_idx, test_idx, report


def _stratified_fallback(
    labels: list[int],
    test_size: float,
    random_state: int,
) -> tuple[list[int], list[int]]:
    """Balanced class-stratified index split (no sklearn dependency)."""
    import numpy as np

    arr     = np.array(labels)
    classes = np.unique(arr)
    rng     = np.random.default_rng(random_state)
    train_idx: list[int] = []
    test_idx:  list[int] = []

    for cls in classes:
        cls_indices = np.where(arr == cls)[0]
        rng.shuffle(cls_indices)
        n_cls_test = max(1, int(round(len(cls_indices) * test_size)))
        test_idx.extend(cls_indices[:n_cls_test].tolist())
        train_idx.extend(cls_indices[n_cls_test:].tolist())

    return train_idx, test_idx


# ---------------------------------------------------------------------------
# Dataset summary
# ---------------------------------------------------------------------------


def dataset_summary(urls: Sequence[str], labels: Sequence[int]) -> str:
    """
    Return a plain-text summary of a URL dataset for logging and provenance.

    Parameters
    ----------
    urls   : sequence of URL strings.
    labels : corresponding integer labels.

    Returns
    -------
    Multi-line string describing class balance, domain diversity, and TLD
    distribution.
    """
    from collections import Counter

    n = len(urls)
    label_counts = Counter(labels)
    tld_counts: Counter = Counter()
    domain_set: set[str] = set()

    for url in urls:
        ext = tldextract.extract(url)
        if ext.suffix:
            tld_counts[ext.suffix.lower()] += 1
        rd = _registered_domain(url)
        if rd:
            domain_set.add(rd)

    top_tlds = tld_counts.most_common(10)

    lines = [
        f"Dataset summary: {n} samples",
        f"  Legitimate (0) : {label_counts.get(0, 0)}",
        f"  Phishing   (1) : {label_counts.get(1, 0)}",
        f"  Distinct domains: {len(domain_set)}",
        f"  Top TLDs: "
        + ", ".join(f"{tld}({c})" for tld, c in top_tlds),
    ]
    return "\n".join(lines)
