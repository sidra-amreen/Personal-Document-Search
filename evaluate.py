import json
from pathlib import Path

import numpy as np

from engine import SearchEngine

QUERIES = json.loads(Path("eval_queries.json").read_text())


def evaluate(engine, alpha, k=3):
    rr, hit1, hitk = [], 0, 0
    for q, target in QUERIES:
        ranked = [Path(h.path).name for h in engine.search(q, k=len(engine.files), alpha=alpha)]
        r = ranked.index(target) + 1 if target in ranked else len(ranked) + 1
        rr.append(1 / r); hit1 += r == 1; hitk += r <= k
    n = len(QUERIES)
    return hit1 / n, hitk / n, float(np.mean(rr))


if __name__ == "__main__":
    e = SearchEngine().build("sample_docs")
    print(f"{len(QUERIES)} queries\n{'mode':<10}{'R@1':>7}{'R@3':>7}{'MRR':>7}")
    for name, a in [("keyword", 1.0), ("semantic", 0.0), ("hybrid", 0.6)]:
        r1, r3, mrr = evaluate(e, a)
        print(f"{name:<10}{r1:>7.2f}{r3:>7.2f}{mrr:>7.2f}")
