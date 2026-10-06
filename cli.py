"""CLI:  python cli.py index ./docs   |   python cli.py search "query" -k 5   |   python cli.py chat"""
import argparse
from pathlib import Path

from engine import SearchEngine, highlight

INDEX = "index.joblib"


def show(engine, query, k, alpha):
    for i, h in enumerate(engine.search(query, k, alpha), 1):
        print(f"\n{i}. {Path(h.path).name}   (score {h.score:.2f}, kw {h.bm25:.1f}, sem {h.semantic:.2f})")
        print("   " + highlight(h.snippet, query, engine, "\033[1;33m{}\033[0m"))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("index"); a.add_argument("folder")
    s = sub.add_parser("search"); s.add_argument("query"); s.add_argument("-k", type=int, default=5)
    s.add_argument("--alpha", type=float, default=0.6)
    sub.add_parser("chat")
    args = ap.parse_args()

    if args.cmd == "index":
        e = SearchEngine().build(args.folder)
        e.save(INDEX)
        print(f"Indexed {len(e.files)} files, {len(e.chunks)} chunks -> {INDEX}")
    else:
        e = SearchEngine.load(INDEX)
        if args.cmd == "search":
            show(e, args.query, args.k, args.alpha)
        else:
            while (q := input("\nsearch> ").strip()) not in {"q", "quit", "exit"}:
                if q:
                    show(e, q, 5, 0.6)


if __name__ == "__main__":
    main()
