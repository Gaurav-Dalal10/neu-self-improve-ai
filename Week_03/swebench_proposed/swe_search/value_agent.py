from __future__ import annotations

import os
from typing import Tuple

import openai
from dotenv import load_dotenv

from swe_search.reward import compute_reward
from swe_search.state import SearchState


class ValueAgent:
    def __init__(self, model: str = "gpt-4o-mini", use_llm: bool = True):
        load_dotenv(override=True)
        self.model = model
        self.use_llm = use_llm
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=api_key) if (use_llm and api_key) else None

    def score(self, state: SearchState) -> Tuple[float, str]:
        heuristic = compute_reward(state)
        patch = (state.terminal_patch or "").strip()
        if not self.client or not patch:
            reason = "No patch generated" if not patch else "Heuristic-only score"
            return heuristic, reason

        prompt = (
            "Score this candidate software patch from 0 to 1 for likelihood of fixing issue. "
            "Return JSON with keys score and explanation.\n"
            f"Patch preview:\n{patch[:4000]}"
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=300,
            )
            text = (resp.choices[0].message.content or "").strip()
            # Accept lightweight parsing without strict JSON dependency.
            lower = text.lower()
            score = heuristic
            if "score" in lower:
                import re

                m = re.search(r"score[^0-9]*([01](?:\.\d+)?)", lower)
                if m:
                    score = max(0.0, min(1.0, float(m.group(1))))
            explanation = text[:1200] if text else "LLM returned empty explanation"
            # Blend with heuristic to reduce instability.
            blended = round(0.7 * score + 0.3 * heuristic, 4)
            return blended, explanation
        except Exception as exc:
            return heuristic, f"Value model fallback due to error: {exc}"
