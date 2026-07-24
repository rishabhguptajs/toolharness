"""Agreement metrics for benchmark validation (M6).

Pure functions over label lists — no dependency on the detectors or any dataset,
so they are trivially unit-testable and reused for both the deterministic
per-detector P/R/F1 and the LLM-judge Cohen's kappa.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass
class BinaryMetrics:
    """Precision/recall/F1 for a binary classifier (positive = "should fire")."""

    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 1.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.n if self.n else 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "accuracy": round(self.accuracy, 4),
            "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
        }


def binary_metrics(y_true: Sequence[bool], y_pred: Sequence[bool]) -> BinaryMetrics:
    """Build a confusion matrix from truth/prediction label sequences."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")
    tp = fp = fn = tn = 0
    for t, p in zip(y_true, y_pred, strict=True):
        if t and p:
            tp += 1
        elif not t and p:
            fp += 1
        elif t and not p:
            fn += 1
        else:
            tn += 1
    return BinaryMetrics(tp=tp, fp=fp, fn=fn, tn=tn)


@dataclass
class KappaResult:
    kappa: float
    observed_agreement: float
    expected_agreement: float
    n: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "kappa": round(self.kappa, 4),
            "observed_agreement": round(self.observed_agreement, 4),
            "expected_agreement": round(self.expected_agreement, 4),
            "n": self.n,
        }


def cohen_kappa(rater_a: Sequence[Any], rater_b: Sequence[Any]) -> KappaResult:
    """Cohen's kappa between two raters over categorical labels.

    kappa = (po - pe) / (1 - pe), where ``po`` is observed agreement and ``pe`` is
    the agreement expected by chance from each rater's marginal distribution.
    Returns kappa=1.0 for the degenerate case of perfect agreement with a single
    category (pe == 1), matching the common convention.
    """
    if len(rater_a) != len(rater_b):
        raise ValueError("rater_a and rater_b must be the same length")
    n = len(rater_a)
    if n == 0:
        return KappaResult(kappa=1.0, observed_agreement=1.0, expected_agreement=1.0, n=0)

    po = sum(1 for a, b in zip(rater_a, rater_b, strict=True) if a == b) / n
    count_a = Counter(rater_a)
    count_b = Counter(rater_b)
    categories = set(count_a) | set(count_b)
    pe = sum((count_a[c] / n) * (count_b[c] / n) for c in categories)

    if pe >= 1.0:
        kappa = 1.0 if po >= 1.0 else 0.0
    else:
        kappa = (po - pe) / (1 - pe)
    return KappaResult(kappa=kappa, observed_agreement=po, expected_agreement=pe, n=n)
