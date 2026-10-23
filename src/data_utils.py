
from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

import tldextract


def _registered_domain(url: str) -> str:

    try:
        ext = tldextract.extract(url)
        if ext.domain and ext.suffix:
            return f"{ext.domain}.{ext.suffix}".lower()
    except Exception:
        pass
    return ""


def _normalise_url(url: str) -> str:

    normalised = url.strip().lower()
    normalised = re.sub(r"^https?://", "", normalised)
    normalised = normalised.rstrip("/")
    return normalised

@dataclass
class DeduplicationReport:

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

    report = DeduplicationReport()

    first_seen_label: dict[str, int] = {}
    conflict_urls: set[str] = set()
    for url, label in zip(urls, labels):
        if url in first_seen_label:
            if first_seen_label[url] != label:
                conflict_urls.add(url)
        else:
            first_seen_label[url] = label

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

    norm_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, url in enumerate(urls):
        if i not in to_remove:
            norm_to_indices[_normalise_url(url)].append(i)
    for norm, indices in norm_to_indices.items():
        if len(indices) >= 2:
            for j in range(len(indices)):
                for k in range(j + 1, len(indices)):
                    report.near_duplicate_pairs.append((indices[j], indices[k]))

    cleaned_urls   = [u for i, u in enumerate(urls)   if i not in to_remove]
    cleaned_labels = [
    label for i, label in enumerate(labels) if i not in to_remove
]

    return cleaned_urls, cleaned_labels, report

@dataclass
class DomainSplitReport:

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


_MIN_DOMAIN_GROUPS_PER_CLASS = 2


def domain_grouped_split(
    urls: Sequence[str],
    labels: Sequence[int],
    test_size: float = 0.25,
    random_state: int = 42,
    min_group_size: int = 1,
) -> tuple[list[int], list[int], DomainSplitReport]:

    import numpy as np

    urls_list   = list(urls)
    labels_list = list(labels)
    n           = len(urls_list)
    report      = DomainSplitReport(n_total=n)

    domain_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, url in enumerate(urls_list):
        rd = _registered_domain(url) or f"__no_domain_{i}__"
        domain_to_indices[rd].append(i)

    report.n_domains = len(domain_to_indices)
    report.n_single_sample_domains = sum(
        1 for idxs in domain_to_indices.values() if len(idxs) == 1
    )

    domain_labels: dict[str, int] = {}
    for domain, idxs in domain_to_indices.items():
        cls_counts = defaultdict(int)
        for i in idxs:
            cls_counts[labels_list[i]] += 1
        domain_labels[domain] = max(cls_counts, key=lambda c: (cls_counts[c], c))

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


def dataset_summary(urls: Sequence[str], labels: Sequence[int]) -> str:

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
        "  Top TLDs: "
        + ", ".join(f"{tld}({c})" for tld, c in top_tlds),
    ]
    return "\n".join(lines)
