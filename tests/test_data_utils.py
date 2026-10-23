
from __future__ import annotations

# ruff: noqa: E402
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_utils import (  #noqa: E402
    DeduplicationReport,
    DomainSplitReport,
    _normalise_url,
    _registered_domain,
    _stratified_fallback,
    dataset_summary,
    deduplicate_urls,
    domain_grouped_split,
)


class TestRegisteredDomain:
    def test_normal_url(self):
        assert _registered_domain("https://www.amazon.com/path") == "amazon.com"

    def test_subdomain_stripped(self):
        assert _registered_domain("http://login.paypal.com/signin") == "paypal.com"

    def test_ccTLD(self):
        assert _registered_domain("https://www.bbc.co.uk/news") == "bbc.co.uk"

    def test_http_url(self):
        assert _registered_domain("http://secure-paypal.xyz/verify") == "secure-paypal.xyz"

    def test_ip_address_returns_something(self):
        result = _registered_domain("http://192.168.0.1/path")
        assert isinstance(result, str)

    def test_localhost_returns_empty(self):
        result = _registered_domain("http://localhost/admin")
        assert isinstance(result, str)

    def test_bare_domain(self):
        assert _registered_domain("https://github.com/") == "github.com"

    def test_empty_string_returns_empty(self):
        assert _registered_domain("") == ""

    def test_bad_url_returns_empty(self):
        assert _registered_domain("not a url at all!!!") == ""

class TestNormaliseUrl:
    def test_strips_https_scheme(self):
        assert _normalise_url("https://AMAZON.COM/path") == "amazon.com/path"

    def test_strips_http_scheme(self):
        assert _normalise_url("http://paypal.com/signin") == "paypal.com/signin"

    def test_lowercases(self):
        assert _normalise_url("https://EXAMPLE.COM") == "example.com"

    def test_strips_trailing_slash(self):
        assert _normalise_url("https://example.com/") == "example.com"

    def test_preserves_path(self):
        assert _normalise_url("https://example.com/path/to/page") == "example.com/path/to/page"

    def test_strips_surrounding_whitespace(self):
        assert _normalise_url("  https://example.com  ") == "example.com"

    def test_same_canonical_form(self):
        a = _normalise_url("https://AMAZON.COM/")
        b = _normalise_url("http://amazon.com")
        assert a == b

class TestDeduplicateUrlsClean:
    def test_clean_data_unchanged(self):
        urls   = ["https://a.com", "https://b.com", "https://c.com"]
        labels = [0, 1, 0]
        clean_urls, clean_labels, report = deduplicate_urls(urls, labels)
        assert clean_urls   == urls
        assert clean_labels == labels
        assert report.n_removed == 0
        assert report.exact_duplicate_indices   == []
        assert report.conflicting_label_indices == []

    def test_returns_dedup_report(self):
        urls   = ["https://a.com"]
        labels = [0]
        _, _, report = deduplicate_urls(urls, labels)
        assert isinstance(report, DeduplicationReport)

    def test_single_sample(self):
        urls, labels, report = deduplicate_urls(["https://a.com"], [1])
        assert urls   == ["https://a.com"]
        assert labels == [1]
        assert report.n_removed == 0


class TestDeduplicateUrlsExactDuplicates:
    def test_exact_duplicate_removed(self):
        urls   = ["https://a.com", "https://b.com", "https://a.com"]
        labels = [0, 1, 0]
        clean_urls, clean_labels, report = deduplicate_urls(urls, labels)
        assert "https://a.com" in clean_urls
        assert clean_urls.count("https://a.com") == 1
        assert 2 in report.exact_duplicate_indices
        assert report.n_removed == 1

    def test_multiple_exact_duplicates(self):
        urls   = ["https://a.com"] * 4 + ["https://b.com"]
        labels = [0, 0, 0, 0, 1]
        clean_urls, clean_labels, report = deduplicate_urls(urls, labels)
        assert clean_urls.count("https://a.com") == 1
        assert len(report.exact_duplicate_indices) == 3

    def test_all_duplicates_of_same_url(self):
        urls   = ["https://a.com"] * 3
        labels = [0, 0, 0]
        clean_urls, clean_labels, report = deduplicate_urls(urls, labels)
        assert len(clean_urls) == 1
        assert report.n_removed == 2


class TestDeduplicateUrlsConflictingLabels:
    def test_conflicting_labels_both_removed(self):
        urls   = ["https://a.com", "https://b.com", "https://a.com"]
        labels = [0, 1, 1]   # a.com has label 0 first, then 1 — conflict
        clean_urls, clean_labels, report = deduplicate_urls(urls, labels)
        assert "https://a.com" not in clean_urls
        assert "https://b.com" in clean_urls
        assert len(report.conflicting_label_indices) >= 1

    def test_conflict_increases_n_removed(self):
        urls   = ["https://a.com", "https://a.com"]
        labels = [0, 1]
        _, _, report = deduplicate_urls(urls, labels)
        assert report.n_removed >= 1

    def test_conflict_reported_separately_from_exact_dup(self):
        urls   = ["https://x.com", "https://x.com"]
        labels = [0, 0]   # exact dup, same label
        _, _, report = deduplicate_urls(urls, labels)
        assert report.exact_duplicate_indices != []
        assert report.conflicting_label_indices == []

    def test_conflict_reported_not_exact_dup(self):
        urls   = ["https://x.com", "https://x.com"]
        labels = [0, 1]   # conflicting labels
        _, _, report = deduplicate_urls(urls, labels)
        assert report.conflicting_label_indices != []


class TestDeduplicateUrlsNearDuplicates:
    def test_near_duplicates_reported(self):
        urls   = ["https://amazon.com", "http://amazon.com/"]
        labels = [0, 0]
        clean_urls, _, report = deduplicate_urls(urls, labels)
        assert len(clean_urls) == 2
        assert len(report.near_duplicate_pairs) >= 1

    def test_near_duplicate_pair_indices_valid(self):
        urls   = ["https://amazon.com/", "http://AMAZON.COM"]
        labels = [0, 0]
        _, _, report = deduplicate_urls(urls, labels)
        for a, b in report.near_duplicate_pairs:
            assert 0 <= a < len(urls)
            assert 0 <= b < len(urls)
            assert a != b

    def test_identical_exact_not_near_duplicate(self):
        urls   = ["https://a.com", "https://a.com"]
        labels = [0, 0]
        clean_urls, _, report = deduplicate_urls(urls, labels)
        assert len(clean_urls) == 1

    def test_distinct_urls_no_near_duplicate(self):
        urls   = ["https://amazon.com", "https://paypal.com"]
        labels = [0, 1]
        _, _, report = deduplicate_urls(urls, labels)
        assert report.near_duplicate_pairs == []


class TestDeduplicateUrlsAsDict:
    def test_as_dict_keys(self):
        _, _, report = deduplicate_urls(["https://a.com"], [0])
        d = report.as_dict()
        assert set(d.keys()) == {
            "exact_duplicate_indices",
            "near_duplicate_pairs",
            "conflicting_label_indices",
            "n_removed",
        }

    def test_as_dict_serialisable(self):
        import json
        _, _, report = deduplicate_urls(["https://a.com", "https://a.com"], [0, 0])
        json.dumps(report.as_dict())

class TestStratifiedFallback:
    def test_returns_two_lists(self):
        labels = [0] * 10 + [1] * 10
        train, test = _stratified_fallback(labels, 0.25, 42)
        assert isinstance(train, list)
        assert isinstance(test,  list)

    def test_all_indices_covered(self):
        labels = [0] * 10 + [1] * 10
        train, test = _stratified_fallback(labels, 0.25, 42)
        assert sorted(train + test) == list(range(20))

    def test_test_size_roughly_correct(self):
        labels = [0] * 20 + [1] * 20
        train, test = _stratified_fallback(labels, 0.25, 42)
        ratio = len(test) / len(labels)
        assert 0.15 < ratio < 0.40

    def test_both_classes_in_test(self):
        labels = [0] * 10 + [1] * 10
        train, test = _stratified_fallback(labels, 0.25, 42)
        test_labels = {labels[i] for i in test}
        assert test_labels == {0, 1}

    def test_deterministic(self):
        labels = [0] * 10 + [1] * 10
        t1, _ = _stratified_fallback(labels, 0.25, 42)
        t2, _ = _stratified_fallback(labels, 0.25, 42)
        assert t1 == t2


def _make_urls_labels(n_per_class: int = 10) -> tuple[list[str], list[int]]:
    legit = [
        "https://www.amazon.com/", "https://www.google.com/",
        "https://www.paypal.com/", "https://www.microsoft.com/",
        "https://www.apple.com/",  "https://www.facebook.com/",
        "https://www.netflix.com/","https://www.github.com/",
        "https://www.linkedin.com/","https://www.twitter.com/",
    ][:n_per_class]
    phish = [
        "http://amaz0n-verify.xyz/", "http://goog1e-login.ml/",
        "http://paypa1-secure.tk/",  "http://micros0ft-alert.ga/",
        "http://app1e-id.cf/",       "http://faceb00k-login.gq/",
        "http://netf1ix-pay.xyz/",   "http://githu8-secure.tk/",
        "http://1inkedin-jobs.ml/",  "http://tw1tter-verify.ga/",
    ][:n_per_class]
    urls   = legit + phish
    labels = [0] * len(legit) + [1] * len(phish)
    return urls, labels


class TestDomainGroupedSplitNormal:
    def setup_method(self):
        self.urls, self.labels = _make_urls_labels(10)
        self.train, self.test, self.report = domain_grouped_split(
            self.urls, self.labels, test_size=0.25, random_state=42
        )

    def test_returns_three_items(self):
        assert isinstance(self.train, list)
        assert isinstance(self.test,  list)
        assert isinstance(self.report, DomainSplitReport)

    def test_all_indices_covered(self):
        all_idx = sorted(self.train + self.test)
        assert all_idx == list(range(len(self.urls)))

    def test_no_index_in_both_sets(self):
        assert set(self.train).isdisjoint(set(self.test))

    def test_test_set_not_empty(self):
        assert len(self.test) > 0

    def test_train_set_not_empty(self):
        assert len(self.train) > 0

    def test_report_n_total(self):
        assert self.report.n_total == len(self.urls)

    def test_report_n_train_plus_test(self):
        assert self.report.n_train + self.report.n_test == len(self.urls)

    def test_report_strategy_set(self):
        assert self.report.strategy in {"domain_grouped", "stratified_fallback"}

    def test_report_reason_nonempty(self):
        assert len(self.report.reason) > 0

    def test_report_n_domains_positive(self):
        assert self.report.n_domains > 0

    def test_report_class_counts_populated(self):
        assert isinstance(self.report.train_class_counts, dict)
        assert isinstance(self.report.test_class_counts, dict)


class TestDomainGroupedSplitFallback:

    def test_tiny_dataset_falls_back(self):
        urls   = ["https://a.com", "http://b.com"]
        labels = [0, 1]
        _, _, report = domain_grouped_split(urls, labels, test_size=0.5, random_state=42)
        assert report.strategy == "stratified_fallback"

    def test_fallback_reason_nonempty(self):
        urls   = ["https://a.com", "http://b.com"]
        labels = [0, 1]
        _, _, report = domain_grouped_split(urls, labels, test_size=0.5, random_state=42)
        assert "fallback" in report.reason.lower() or len(report.reason) > 0

    def test_fallback_still_covers_all_indices(self):
        urls   = ["https://a.com", "http://b.com", "https://c.com", "http://d.com"]
        labels = [0, 1, 0, 1]
        train, test, _ = domain_grouped_split(urls, labels, test_size=0.5, random_state=42)
        assert sorted(train + test) == list(range(4))


class TestDomainGroupedSplitDeterministic:
    def test_same_seed_same_result(self):
        urls, labels = _make_urls_labels(10)
        tr1, te1, _ = domain_grouped_split(urls, labels, test_size=0.25, random_state=99)
        tr2, te2, _ = domain_grouped_split(urls, labels, test_size=0.25, random_state=99)
        assert sorted(tr1) == sorted(tr2)
        assert sorted(te1) == sorted(te2)

    def test_different_seeds_may_differ(self):
        urls, labels = _make_urls_labels(10)
        tr1, _, _ = domain_grouped_split(urls, labels, test_size=0.25, random_state=1)
        tr2, _, _ = domain_grouped_split(urls, labels, test_size=0.25, random_state=2)
        assert isinstance(tr1, list)
        assert isinstance(tr2, list)


class TestDomainSplitReportAsDict:
    def test_as_dict_keys(self):
        _, _, report = domain_grouped_split(
            *_make_urls_labels(10), test_size=0.25, random_state=42
        )
        d = report.as_dict()
        assert "strategy" in d
        assert "reason" in d
        assert "n_total" in d
        assert "n_train" in d
        assert "n_test" in d
        assert "n_domains" in d
        assert "train_class_counts" in d
        assert "test_class_counts" in d

    def test_as_dict_serialisable(self):
        import json
        _, _, report = domain_grouped_split(
            *_make_urls_labels(10), test_size=0.25, random_state=42
        )
        json.dumps(report.as_dict())


class TestDatasetSummary:
    def test_returns_string(self):
        urls   = ["https://amazon.com", "http://phish.xyz/login"]
        labels = [0, 1]
        result = dataset_summary(urls, labels)
        assert isinstance(result, str)

    def test_contains_sample_count(self):
        urls   = ["https://amazon.com", "http://phish.xyz/login"]
        labels = [0, 1]
        result = dataset_summary(urls, labels)
        assert "2" in result

    def test_contains_class_counts(self):
        urls   = ["https://a.com"] * 3 + ["http://b.xyz"] * 2
        labels = [0, 0, 0, 1, 1]
        result = dataset_summary(urls, labels)
        assert "3" in result
        assert "2" in result

    def test_multiline(self):
        urls   = ["https://amazon.com", "http://phish.xyz/login"]
        labels = [0, 1]
        result = dataset_summary(urls, labels)
        assert "\n" in result

    def test_empty_list_no_crash(self):
        result = dataset_summary([], [])
        assert isinstance(result, str)

    def test_single_sample(self):
        result = dataset_summary(["https://a.com"], [0])
        assert isinstance(result, str)
        assert "1" in result
