# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Moutasem Akkad <moutasemalakkad@gmail.com>
"""LLM backends for the judge — each is a Callable[[str], str].

The judge depends only on "give a prompt, get text back", so a backend is just
a function. That dependency injection keeps the judge testable with no network:
echo_backend returns a fixed string for tests; gemini_backend calls Gemini and
lives behind the optional [gemini] extra.
"""
from __future__ import annotations

from typing import Callable


def echo_backend(response: str) -> Callable[[str], str]:
    """Return a backend that ignores the prompt and always returns `response`.

    The seam that keeps the judge testable without a network: a test injects a
    canned VERDICT/REASON block and asserts how the judge parses it. Also useful
    as a stub when wiring the harness before a real model is configured.
    """

    def _call(prompt: str) -> str:
        return response

    return _call


def gemini_backend(
    model: str = "gemini-2.5-pro",
    project: str | None = None,
    location: str = "us-central1",
    thinking_budget: int = 512,
    max_output_tokens: int = 1024,
    temperature: float = 0.0,
) -> Callable[[str], str]:
    """Return a backend that sends the prompt to Gemini and returns its text.

    Behind the [gemini] extra; google-genai is imported lazily so importing
    this module never requires the extra and the core install stays stdlib-only.

    thinking_budget is kept non-zero on purpose: judging SQL equivalence is
    open-ended reasoning, and Gemini 2.5 spends reasoning tokens *before* the
    answer — a zero budget plus the short output cap would truncate the two-line
    verdict. temperature=0 makes the verdict reproducible. A model at least as
    strong as the one under test reduces self-preference bias (a judge favoring
    output shaped like its own).
    """
    from google import genai  # lazy: keeps core install stdlib-only
    from google.genai import types

    client = genai.Client(vertexai=True, project=project, location=location)

    def _call(prompt: str) -> str:
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                thinking_config=types.ThinkingConfig(
                    thinking_budget=thinking_budget
                ),
            ),
        )
        return resp.text or ""

    return _call
