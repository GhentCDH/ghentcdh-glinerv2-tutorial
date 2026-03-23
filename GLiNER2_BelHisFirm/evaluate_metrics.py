#!/usr/bin/env python3
"""Compute entity-level NER metrics for BelHisFirm validation data."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def normalize_entity_text(text: str) -> str:
    """Normalize entity text for matching while preserving strict semantics."""
    return re.sub(r"\s+", " ", text).strip().casefold()


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def extract_truth_entities(record: dict[str, Any]) -> dict[str, list[str]]:
    entities = record.get("entities", {})
    if not isinstance(entities, dict):
        raise ValueError("Ground-truth record has invalid 'entities' format")

    result: dict[str, list[str]] = {}
    for label, values in entities.items():
        if not isinstance(values, list):
            raise ValueError(f"Ground-truth label '{label}' is not a list")
        result[label] = [v for v in values if isinstance(v, str) and v.strip()]
    return result


def extract_pred_entities(record: dict[str, Any]) -> dict[str, list[str]]:
    entities = record.get("entities", [])
    if not isinstance(entities, list) or not entities:
        raise ValueError("Prediction record has invalid 'entities' format")

    # Current evaluator output stores one dict wrapped in a list.
    label_map = entities[0]
    if not isinstance(label_map, dict):
        raise ValueError("Prediction entities[0] is not a dictionary")

    result: dict[str, list[str]] = {}
    for label, values in label_map.items():
        if not isinstance(values, list):
            raise ValueError(f"Prediction label '{label}' is not a list")
        extracted: list[str] = []
        for item in values:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    extracted.append(text)
            elif isinstance(item, str) and item.strip():
                extracted.append(item)
        result[label] = extracted
    return result


def evaluate(ground_truth: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if len(ground_truth) != len(predictions):
        raise ValueError(
            f"Length mismatch: ground-truth has {len(ground_truth)} records, "
            f"predictions has {len(predictions)}"
        )

    label_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for truth_record, pred_record in zip(ground_truth, predictions):
        truth_entities = extract_truth_entities(truth_record)
        pred_entities = extract_pred_entities(pred_record)

        labels = set(truth_entities) | set(pred_entities)
        for label in labels:
            truth_counter = Counter(normalize_entity_text(v) for v in truth_entities.get(label, []))
            pred_counter = Counter(normalize_entity_text(v) for v in pred_entities.get(label, []))

            tp = sum((truth_counter & pred_counter).values())
            fp = sum((pred_counter - truth_counter).values())
            fn = sum((truth_counter - pred_counter).values())

            label_counts[label]["tp"] += tp
            label_counts[label]["fp"] += fp
            label_counts[label]["fn"] += fn

    # Stable ordering in output
    sorted_labels = sorted(label_counts.keys())

    per_label: dict[str, dict[str, float | int]] = {}
    tp_total = fp_total = fn_total = 0

    for label in sorted_labels:
        tp = int(label_counts[label]["tp"])
        fp = int(label_counts[label]["fp"])
        fn = int(label_counts[label]["fn"])

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)

        per_label[label] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": tp + fn,
            "predicted": tp + fp,
            "precision": precision,
            "recall": recall,
            "f1": f1(precision, recall),
        }

        tp_total += tp
        fp_total += fp
        fn_total += fn

    precision_micro = safe_div(tp_total, tp_total + fp_total)
    recall_micro = safe_div(tp_total, tp_total + fn_total)

    macro_precision = safe_div(sum(item["precision"] for item in per_label.values()), len(per_label))
    macro_recall = safe_div(sum(item["recall"] for item in per_label.values()), len(per_label))
    macro_f1 = safe_div(sum(item["f1"] for item in per_label.values()), len(per_label))

    return {
        "records_evaluated": len(ground_truth),
        "matching": "exact normalized text match (casefold + whitespace collapse)",
        "micro": {
            "tp": tp_total,
            "fp": fp_total,
            "fn": fn_total,
            "precision": precision_micro,
            "recall": recall_micro,
            "f1": f1(precision_micro, recall_micro),
        },
        "macro": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1,
        },
        "per_label": per_label,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NER metrics from validation and prediction JSON files")
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("data/val.json"),
        help="Path to validation ground-truth JSON",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("gliner2_evaluation_BelHisFirm_base.json"),
        help="Path to model predictions JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("metrics_BelHisFirm_base.json"),
        help="Path for metrics JSON output",
    )
    args = parser.parse_args()

    ground_truth = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))

    results = evaluate(ground_truth, predictions)

    args.output.write_text(json.dumps(results, ensure_ascii=True, indent=2), encoding="utf-8")

    micro = results["micro"]
    print(f"Records: {results['records_evaluated']}")
    print(
        f"Micro P/R/F1: {micro['precision']:.4f} / {micro['recall']:.4f} / {micro['f1']:.4f} "
        f"(TP={micro['tp']}, FP={micro['fp']}, FN={micro['fn']})"
    )
    print(f"Saved detailed metrics to: {args.output}")


if __name__ == "__main__":
    main()

