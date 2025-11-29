"""
Gemini Evaluator
================

Evaluator using Google's Gemini API.

PROS:
- Free tier (200 RPD for Flash Lite)
- Simple API
- Fast inference

CONS:
- No strict schema enforcement in free tier
- Text parsing is fragile ("pass" vs "Pass" vs "I think it passes")
- Daily quota limits hit quickly

WARNING: This evaluator uses text parsing, which is less reliable than
function calling. See docs/tutorial/03_function_calling.md for why this matters.

Example:
    from evaluators import GeminiEvaluator

    evaluator = GeminiEvaluator(model="gemini-2.0-flash-lite")
    verdict, reasoning = evaluator.evaluate(question, response)
"""

import os
import time
import random
from dotenv import load_dotenv

from .base import BaseEvaluator

load_dotenv()


class GeminiEvaluator(BaseEvaluator):
    """
    Evaluator using Google's Gemini API.

    WARNING: Uses text parsing instead of function calling.
    This is less reliable but works with Gemini's free tier.

    Attributes:
        model: The Gemini model to use (e.g., "gemini-2.0-flash-lite")
        retries: Number of retries on transient failures
        client: Gemini GenerativeModel instance

    Supported models:
        - gemini-2.0-flash-lite (free tier, 200 RPD)
        - gemini-2.0-flash
        - gemini-1.5-flash
    """

    SUPPORTED_MODELS = {
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    }

    def __init__(self, model: str = "gemini-2.0-flash-lite", retries: int = 3):
        """
        Initialize the Gemini evaluator.

        Args:
            model: Gemini model ID (default: gemini-2.0-flash-lite)
            retries: Number of retries on failure (default: 3)

        Raises:
            ValueError: If GOOGLE_API_KEY is not set
            ImportError: If google-generativeai is not installed
        """
        super().__init__(model=model, retries=retries)

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")

        try:
            import google.generativeai as genai

            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(model)
        except ImportError:
            raise ImportError(
                "google-generativeai package not installed. "
                "Run: uv add google-generativeai"
            )

    def get_prompt(self, question: str, response: str) -> str:
        """
        Generate the evaluation prompt.

        Note: We ask for a single word to minimize parsing issues,
        but the model may still return variations like "Pass." or "I pass this."
        """
        return f"""You are evaluating whether an AI response correctly and helpfully answers a question.

Question: {question}

Response: {response}

Evaluate the response on these criteria:
1. Is the response factually correct?
2. Does it actually answer the question asked?
3. Is it complete and not misleading?

Reply with exactly one word: 'pass' if correct and helpful, 'fail' if wrong, incomplete, or misleading."""

    def evaluate(self, question: str, response: str) -> tuple[str, str]:
        """
        Evaluate a question/response pair.

        WARNING: Uses text parsing which is fragile. The model might return:
        - "Pass" or "PASS" (case variations)
        - "pass." (with punctuation)
        - "I think it passes" (embedded in sentence)

        We handle common cases but this is inherently less reliable than
        function calling with strict schema.

        Args:
            question: The question that was asked
            response: The AI-generated response to evaluate

        Returns:
            Tuple of (verdict, reasoning):
            - verdict: "pass" or "fail" (parsed from text)
            - reasoning: Raw model output for debugging
        """
        prompt = self.get_prompt(question, response)

        for attempt in range(self.retries):
            try:
                result = self.client.generate_content(prompt)
                raw_text = result.text.strip()

                # Fragile parsing - see docs/common_mistakes.md#1
                # We check the first 20 chars to avoid false positives
                # from words like "bypass" or "compassionate"
                text_lower = raw_text.lower()[:20]

                if "pass" in text_lower:
                    verdict = "pass"
                elif "fail" in text_lower:
                    verdict = "fail"
                else:
                    # Default to fail if unclear
                    verdict = "fail"

                return verdict, f"Raw: {raw_text[:100]}"

            except Exception as e:
                error_str = str(e)

                # Handle rate limiting (Gemini returns 429)
                if "429" in error_str or "quota" in error_str.lower():
                    if attempt < self.retries - 1:
                        # Longer backoff for rate limits
                        delay = (3**attempt) + random.uniform(1, 3)
                        time.sleep(delay)
                        continue

                # Standard backoff for other errors
                if attempt < self.retries - 1:
                    delay = (2**attempt) + random.uniform(0, 1)
                    time.sleep(delay)
                else:
                    return "fail", f"Error after {self.retries} attempts: {error_str[:100]}"

        return "fail", "Max retries exceeded"
