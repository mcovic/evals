"""
Evaluators Package
==================

Modular LLM evaluation framework implementing Eugene Yan's methodology.

Usage:
    from evaluators import OpenAIEvaluator, CerebrasEvaluator, calculate_metrics

    # Create an evaluator
    evaluator = OpenAIEvaluator(model="gpt-5-mini-2025-08-07")

    # Evaluate a response
    verdict, reasoning = evaluator.evaluate(question, response)

    # Calculate metrics
    metrics = calculate_metrics(results)

See:
    - docs/glossary.md for terminology
    - docs/tutorial/ for learning progression
    - examples/ for simple examples
"""

from .base import BaseEvaluator
from .openai_evaluator import OpenAIEvaluator
from .cerebras_evaluator import CerebrasEvaluator
from .gemini_evaluator import GeminiEvaluator
from .metrics import calculate_metrics, interpret_kappa, confidence_interval

__all__ = [
    "BaseEvaluator",
    "OpenAIEvaluator",
    "CerebrasEvaluator",
    "GeminiEvaluator",
    "calculate_metrics",
    "interpret_kappa",
    "confidence_interval",
]
