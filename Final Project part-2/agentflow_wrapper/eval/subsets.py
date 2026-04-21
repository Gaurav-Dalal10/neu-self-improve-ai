import random
from typing import Dict, List


def sample_examples(rows: List[Dict], n_examples: int, seed: int) -> List[Dict]:
    if n_examples <= 0 or n_examples >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    chosen = sorted(idx[:n_examples])
    return [rows[i] for i in chosen]
