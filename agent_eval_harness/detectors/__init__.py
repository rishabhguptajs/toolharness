"""Failure-mode detectors.

Two groups:
  * ``DETERMINISTIC_DETECTORS`` — the five deterministic (or deterministic-anchored)
    modes from M1: wrong-args, hallucinated, redundant, missing-verification,
    unsafe-call.
  * ``HYBRID_DETECTORS`` — the three judgment-heavy modes added in M3: wrong-tool
    (M1), ignored-output (M4), premature-stop (M7). Each has deterministic anchors
    and escalates ambiguous cases to the LLM judge when one is wired into
    ``DetectorContext.judge`` (otherwise it runs heuristic-only).

``ALL_DETECTORS`` is the full eight, ordered M1..M8.
"""

from agent_eval_harness.detectors.base import Detector, DetectorContext, DetectorResult
from agent_eval_harness.detectors.m1_wrong_tool import WrongToolDetector
from agent_eval_harness.detectors.m2_wrong_args import WrongArgsDetector
from agent_eval_harness.detectors.m3_hallucinated import HallucinatedCallDetector
from agent_eval_harness.detectors.m4_ignored_output import IgnoredOutputDetector
from agent_eval_harness.detectors.m5_redundant import RedundantCallDetector
from agent_eval_harness.detectors.m6_missing_verification import MissingVerificationDetector
from agent_eval_harness.detectors.m7_premature_stop import PrematureStopDetector
from agent_eval_harness.detectors.m8_unsafe_call import UnsafeCallDetector

DETERMINISTIC_DETECTORS: list[Detector] = [
    WrongArgsDetector(),
    HallucinatedCallDetector(),
    RedundantCallDetector(),
    MissingVerificationDetector(),
    UnsafeCallDetector(),
]

HYBRID_DETECTORS: list[Detector] = [
    WrongToolDetector(),
    IgnoredOutputDetector(),
    PrematureStopDetector(),
]

# Full eight, canonical M1..M8 order.
ALL_DETECTORS: list[Detector] = [
    WrongToolDetector(),
    WrongArgsDetector(),
    HallucinatedCallDetector(),
    IgnoredOutputDetector(),
    RedundantCallDetector(),
    MissingVerificationDetector(),
    PrematureStopDetector(),
    UnsafeCallDetector(),
]

__all__ = [
    "Detector",
    "DetectorContext",
    "DetectorResult",
    "WrongToolDetector",
    "WrongArgsDetector",
    "HallucinatedCallDetector",
    "IgnoredOutputDetector",
    "RedundantCallDetector",
    "MissingVerificationDetector",
    "PrematureStopDetector",
    "UnsafeCallDetector",
    "DETERMINISTIC_DETECTORS",
    "HYBRID_DETECTORS",
    "ALL_DETECTORS",
]
