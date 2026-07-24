"""Failure-mode detectors.

M1 ships the five deterministic (or deterministic-anchored) detectors. The three
judgment-heavy modes — wrong_tool (M1), ignored_output (M4), premature_stop (M7)
— arrive with the LLM-judge layer in milestone M3.
"""

from agent_eval_harness.detectors.base import Detector, DetectorContext, DetectorResult
from agent_eval_harness.detectors.m2_wrong_args import WrongArgsDetector
from agent_eval_harness.detectors.m3_hallucinated import HallucinatedCallDetector
from agent_eval_harness.detectors.m5_redundant import RedundantCallDetector
from agent_eval_harness.detectors.m6_missing_verification import MissingVerificationDetector
from agent_eval_harness.detectors.m8_unsafe_call import UnsafeCallDetector

DETERMINISTIC_DETECTORS: list[Detector] = [
    WrongArgsDetector(),
    HallucinatedCallDetector(),
    RedundantCallDetector(),
    MissingVerificationDetector(),
    UnsafeCallDetector(),
]

__all__ = [
    "Detector",
    "DetectorContext",
    "DetectorResult",
    "WrongArgsDetector",
    "HallucinatedCallDetector",
    "RedundantCallDetector",
    "MissingVerificationDetector",
    "UnsafeCallDetector",
    "DETERMINISTIC_DETECTORS",
]
