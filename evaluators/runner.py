"""
Evaluation Runner
=================

Main orchestration for running evaluations and generating reports.

This module ties together evaluators, metrics, and data loading
into a complete evaluation pipeline.

Usage:
    from evaluators.runner import run_evaluation

    results = run_evaluation(
        data=samples,
        model="gpt-5-mini-2025-08-07",
    )

Or from command line:
    uv run verify_evaluator.py --sample 50
"""

import os
import csv
import json
import random
from datetime import datetime
from collections import defaultdict
from typing import Optional

from .openai_evaluator import OpenAIEvaluator
from .cerebras_evaluator import CerebrasEvaluator
from .gemini_evaluator import GeminiEvaluator
from .metrics import calculate_metrics, interpret_kappa


def load_dataset(filepath: str) -> list[dict]:
    """
    Load labeled dataset from CSV.

    Expected CSV format:
        id,question,response,label,category
        1,"What is 2+2?","The answer is 4.","pass","math"
        ...

    Args:
        filepath: Path to the CSV file

    Returns:
        List of dicts with keys: id, question, response, label, category
    """
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        # Only return rows with labels (skip unlabeled data)
        return [row for row in reader if row.get("label")]


def get_evaluator(model: str):
    """
    Get the appropriate evaluator for a model.

    Args:
        model: Model ID (e.g., "gpt-5-mini-2025-08-07", "llama3.1-8b", "gemini-2.0-flash-lite")

    Returns:
        Evaluator instance (OpenAIEvaluator, CerebrasEvaluator, or GeminiEvaluator)
    """
    if model in CerebrasEvaluator.SUPPORTED_MODELS:
        return CerebrasEvaluator(model=model)
    elif model in GeminiEvaluator.SUPPORTED_MODELS:
        return GeminiEvaluator(model=model)
    else:
        return OpenAIEvaluator(model=model)


def run_evaluation(
    data: list[dict],
    model: str,
    progress_callback: Optional[callable] = None,
) -> list[dict]:
    """
    Run evaluation on a dataset.

    Args:
        data: List of samples (each with "question", "response", "label" keys)
        model: Model ID to use for evaluation
        progress_callback: Optional callback(current, total, accuracy_so_far)

    Returns:
        List of result dicts with keys:
            - id: Sample ID
            - category: Question category
            - label: Ground truth label
            - pred: Predicted label
            - match: Whether prediction matches label
            - reasoning: Evaluator's reasoning
    """
    evaluator = get_evaluator(model)
    results = []

    for i, row in enumerate(data, 1):
        pred, reasoning = evaluator.evaluate(row["question"], row["response"])

        match = pred == row["label"]
        results.append({
            "id": row.get("id", str(i)),
            "category": row.get("category", ""),
            "label": row["label"],
            "pred": pred,
            "match": match,
            "reasoning": reasoning,
        })

        if progress_callback and (i % 10 == 0 or i == len(data)):
            correct = sum(1 for r in results if r["match"])
            progress_callback(i, len(data), correct / i)

    return results


def print_results(results: list[dict], model: str, data_file: str):
    """
    Print formatted evaluation results.

    Args:
        results: List of result dicts from run_evaluation
        model: Model ID used
        data_file: Path to the data file
    """
    metrics = calculate_metrics(results)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # Confusion matrix
    print("\n### Confusion Matrix ###")
    print("                  Predicted")
    print("                  Pass    Fail")
    print(f"  Actual Pass     {metrics['tp']:4d}    {metrics['fn']:4d}")
    print(f"  Actual Fail     {metrics['fp']:4d}    {metrics['tn']:4d}")

    # Overall metrics
    print("\n### Overall Metrics ###")
    ci = metrics["accuracy_ci"]
    print(f"  Accuracy:      {metrics['accuracy']*100:.1f}% (95% CI: {ci[0]*100:.1f}%-{ci[1]*100:.1f}%)")
    print(f"  Cohen's Kappa: {metrics['kappa']:.3f} - {interpret_kappa(metrics['kappa'])}")

    # Pass detection
    print("\n### Pass Detection ###")
    print(f"  Precision: {metrics['precision_pass']*100:.1f}%")
    print(f"  Recall:    {metrics['recall_pass']*100:.1f}%")
    print(f"  F1 Score:  {metrics['f1_pass']*100:.1f}%")

    # Fail detection
    print("\n### Fail Detection (Critical per Eugene Yan) ###")
    print(f"  Precision: {metrics['precision_fail']*100:.1f}%")
    print(f"  Recall:    {metrics['recall_fail']*100:.1f}%  <-- Key metric: catching failures")
    print(f"  F1 Score:  {metrics['f1_fail']*100:.1f}%")

    # Per-category breakdown
    categories = defaultdict(list)
    for r in results:
        if r["category"]:
            categories[r["category"]].append(r)

    if categories:
        print("\n### Per-Category Accuracy ###")
        for cat, cat_results in sorted(categories.items()):
            cat_metrics = calculate_metrics(cat_results)
            fail_rate = sum(1 for r in cat_results if r["label"] == "fail") / len(cat_results) * 100
            print(
                f"  {cat:25s}: {cat_metrics['accuracy']*100:5.1f}% acc, "
                f"{cat_metrics['kappa']:.2f} kappa, {fail_rate:.0f}% fail rate "
                f"({len(cat_results)} samples)"
            )

    # Disagreements
    disagreements = [r for r in results if not r["match"]]
    if disagreements:
        print(f"\n### Sample Disagreements (showing up to 10) ###")
        for d in disagreements[:10]:
            print(f"\n  ID {d['id']}: Ground truth={d['label']}, Predicted={d['pred']}")
            if d["reasoning"]:
                reasoning = d["reasoning"][:100] + "..." if len(d["reasoning"]) > 100 else d["reasoning"]
                print(f"  Reasoning: {reasoning}")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION (Eugene Yan's Guidelines)")
    print("=" * 70)

    if metrics["kappa"] >= 0.6:
        print("  Kappa >= 0.6: Excellent agreement - evaluator is reliable")
    elif metrics["kappa"] >= 0.4:
        print("  ~ Kappa 0.4-0.6: Substantial agreement - acceptable for most use cases")
    else:
        print("  Kappa < 0.4: Fair/slight agreement - consider improving prompts")

    if metrics["recall_fail"] >= 0.8:
        print("  Fail Recall >= 80%: Good at catching failures")
    elif metrics["recall_fail"] >= 0.6:
        print("  ~ Fail Recall 60-80%: Moderate failure detection")
    else:
        print("  Fail Recall < 60%: Poor failure detection - many defects will be missed")

    print(f"\n  Note: Human inter-rater reliability is often only kappa 0.2-0.3")
    print(f"  Your evaluator: kappa {metrics['kappa']:.3f}")


def save_run(
    results: list[dict],
    model: str,
    data_file: str,
    output_file: str = "results/runs.jsonl",
    category_filter: Optional[str] = None,
):
    """
    Save run results to JSONL file for historical tracking.

    Args:
        results: List of result dicts
        model: Model ID used
        data_file: Path to the data file
        output_file: Path to output JSONL file
        category_filter: Category filter used (if any)
    """
    metrics = calculate_metrics(results)

    # Per-category breakdown
    categories = defaultdict(list)
    for r in results:
        if r["category"]:
            categories[r["category"]].append(r)

    run_record = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "dataset": data_file,
        "sample_size": len(results),
        "category_filter": category_filter,
        "metrics": {
            "accuracy": round(metrics["accuracy"], 4),
            "accuracy_ci": [round(metrics["accuracy_ci"][0], 4), round(metrics["accuracy_ci"][1], 4)],
            "kappa": round(metrics["kappa"], 4),
            "precision_pass": round(metrics["precision_pass"], 4),
            "recall_pass": round(metrics["recall_pass"], 4),
            "f1_pass": round(metrics["f1_pass"], 4),
            "precision_fail": round(metrics["precision_fail"], 4),
            "recall_fail": round(metrics["recall_fail"], 4),
            "f1_fail": round(metrics["f1_fail"], 4),
        },
        "confusion_matrix": {
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "tn": metrics["tn"],
        },
        "per_category": {
            cat: {
                "accuracy": round(calculate_metrics(cat_results)["accuracy"], 4),
                "kappa": round(calculate_metrics(cat_results)["kappa"], 4),
                "samples": len(cat_results),
            }
            for cat, cat_results in categories.items()
        }
        if categories
        else {},
    }

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    # Append to JSONL file
    with open(output_file, "a") as f:
        f.write(json.dumps(run_record) + "\n")

    print(f"\n  Run saved to: {output_file}")
