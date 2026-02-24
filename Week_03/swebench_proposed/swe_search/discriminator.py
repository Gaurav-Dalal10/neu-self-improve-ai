from __future__ import annotations

import os
from typing import List

import openai
from dotenv import load_dotenv

from swe_search.candidate_pool import ScoredCandidate
from swe_search.state import SearchState


class Discriminator:
    def __init__(self, model: str = "gpt-4o", use_llm: bool = True):
        load_dotenv(override=True)
        self.model = model
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=api_key) if (use_llm and api_key) else None

    def choose(self, candidates: List[ScoredCandidate]) -> ScoredCandidate:
        if not candidates:
            raise ValueError("No candidates provided")
        if len(candidates) == 1 or not self.client:
            return max(candidates, key=lambda c: c.score)

        lines = []
        for idx, cand in enumerate(candidates, 1):
            patch_preview = (cand.state.terminal_patch or "")[:2000]
            lines.append(
                f"Candidate {idx}\n"
                f"score={cand.score}\n"
                f"value_explanation={cand.explanation[:600]}\n"
                f"patch=\n{patch_preview}\n"
            )
        prompt = (
            "You are a discriminator selecting the best candidate patch. "
            "Return only the best candidate number as an integer.\n\n"
            + "\n\n".join(lines)
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=20,
            )
            text = (resp.choices[0].message.content or "").strip()
            import re

            m = re.search(r"(\d+)", text)
            if m:
                idx = int(m.group(1)) - 1
                if 0 <= idx < len(candidates):
                    return candidates[idx]
        except Exception:
            pass
        return max(candidates, key=lambda c: c.score)
