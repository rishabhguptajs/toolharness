"""Benchmark converters: public tool-calling datasets -> NormalizedSession.

These turn a benchmark *case* (a task + advertised functions + gold answer) into
scoring-ready sessions with a known ground-truth label, so detector precision/
recall can be measured against curated data (M6).
"""
