#!/usr/bin/env python3
"""
RLHF Rubric Calibration Kit — Deterministic Inter-Annotator Agreement & Bilingual Drift Auditor
Author: MD Rafiul Islam Shubho (ORCID: 0009-0009-9956-4511)
Organization: Vision Flow Group — Applied AI & RLHF Evaluation Division

Zero-dependency standard-library CLI for auditing RLHF preference annotations across
Bangla-English bilingual tasks:
  1. Pairwise Cohen's Kappa & Fleiss' Kappa across N human/LLM raters
  2. Rubric dimension variance & tie-rate drift detection
  3. Bangla-English code-switch ratio & Unicode normalization sanity checks
"""

from __future__ import annotations
import argparse
import json
import math
import statistics
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def cohens_kappa(rater_a: Sequence[str], rater_b: Sequence[str]) -> float:
    if len(rater_a) != len(rater_b) or not rater_a:
        raise ValueError("Rater sequences must be non-empty and equal length.")
    n = len(rater_a)
    labels = sorted(set(rater_a) | set(rater_b))
    po = sum(1 for a, b in zip(rater_a, rater_b) if a == b) / n
    ca, cb = Counter(rater_a), Counter(rater_b)
    pe = sum((ca[k] / n) * (cb[k] / n) for k in labels)
    if math.isclose(1.0 - pe, 0.0):
        return 1.0 if math.isclose(po, 1.0) else 0.0
    return round((po - pe) / (1.0 - pe), 4)


def fleiss_kappa(matrix: Sequence[Sequence[int]]) -> float:
    """Compute Fleiss' kappa for N items x K categories rating count matrix."""
    n_items = len(matrix)
    if n_items == 0:
        return 0.0
    n_raters = sum(matrix[0])
    if n_raters <= 1:
        return 1.0
    k = len(matrix[0])
    p_j = [sum(row[j] for row in matrix) / (n_items * n_raters) for j in range(k)]
    p_i = [
        (sum(c * c for c in row) - n_raters) / (n_raters * (n_raters - 1))
        for row in matrix
    ]
    p_bar = sum(p_i) / n_items
    p_e = sum(p * p for p in p_j)
    if math.isclose(1.0 - p_e, 0.0):
        return 1.0
    return round((p_bar - p_e) / (1.0 - p_e), 4)


def script_mix_profile(text: str) -> Dict[str, float]:
    """Measure Bengali vs Latin character balance and NFC normalization consistency."""
    normalized = unicodedata.normalize("NFC", text)
    bn = sum(1 for ch in normalized if "\u0980" <= ch <= "\u09ff")
    lat = sum(1 for ch in normalized if ("A" <= ch <= "Z") or ("a" <= ch <= "z"))
    total = max(bn + lat, 1)
    return {
        "bengali_ratio": round(bn / total, 4),
        "latin_ratio": round(lat / total, 4),
        "nfc_clean": normalized == text,
    }


def audit_jsonl(records: List[Dict]) -> Dict:
    """Run full rubric calibration audit over JSONL evaluation records."""
    rater_a, rater_b = [], []
    score_deltas: List[float] = []
    drift_flags: List[Dict] = []

    for idx, rec in enumerate(records):
        la, lb = str(rec.get("rater_a", "")), str(rec.get("rater_b", ""))
        if la and lb:
            rater_a.append(la)
            rater_b.append(lb)
        sa, sb = rec.get("score_a"), rec.get("score_b")
        if isinstance(sa, (int, float)) and isinstance(sb, (int, float)):
            delta = abs(float(sa) - float(sb))
            score_deltas.append(delta)
            if delta >= 2.0:
                drift_flags.append({
                    "id": rec.get("id", f"row-{idx}"),
                    "score_delta": delta,
                    "script_profile": script_mix_profile(str(rec.get("response", ""))),
                })

    kappa = cohens_kappa(rater_a, rater_b) if rater_a else None
    mean_delta = round(statistics.mean(score_deltas), 4) if score_deltas else 0.0
    return {
        "total_items": len(records),
        "cohens_kappa": kappa,
        "mean_score_delta": mean_delta,
        "high_drift_count": len(drift_flags),
        "calibration_status": "PASS" if (kappa is None or kappa >= 0.72) and mean_delta <= 0.75 else "RECALIBRATE",
        "flagged_items": drift_flags[:25],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic RLHF Rubric Calibration Auditor")
    parser.add_argument("--input", type=Path, help="Path to JSONL annotation file")
    parser.add_argument("--demo", action="store_true", help="Run built-in bilingual RLHF self-test")
    args = parser.parse_args()

    if args.demo or not args.input:
        sample = [
            {"id": "bn-001", "rater_a": "A_better", "rater_b": "A_better", "score_a": 4.5, "score_b": 4.5, "response": "বাংলা AI evaluation rubric calibration"},
            {"id": "bn-002", "rater_a": "B_better", "rater_b": "B_better", "score_a": 4.0, "score_b": 3.5, "response": "Deterministic prosody scoring in Bangla S2S"},
            {"id": "bn-003", "rater_a": "Tie", "rater_b": "A_better", "score_a": 3.0, "score_b": 5.0, "response": "Code-switch fluency test বাংলা এবং English"},
            {"id": "bn-004", "rater_a": "A_better", "rater_b": "A_better", "score_a": 5.0, "score_b": 4.5, "response": "Accurate factual grounding with zero hallucination"},
        ]
        print(json.dumps(audit_jsonl(sample), ensure_ascii=False, indent=2))
        return

    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(json.dumps(audit_jsonl(rows), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
