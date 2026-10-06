# Personal Document Search

Search your own .txt / .md / .pdf / .docx files by meaning, not just exact words. Runs fully offline.

## How it works
1. **Load + chunk**: files are split into overlapping ~120-word chunks.
2. **Keyword signal**: BM25 over a sparse term matrix (exact-match strength, rare words weigh more).
3. **Semantic signal**: TF-IDF → truncated SVD (LSA), so related words that co-occur land close together.
4. **Fusion**: `alpha * BM25 + (1 - alpha) * LSA` after min-max normalisation; best chunk per document is returned with a highlighted snippet.

## Run
```bash
pip install -r requirements.txt
python cli.py index sample_docs          # build index.joblib
python cli.py search "bake with a starter" -k 3
python cli.py chat                       # interactive
streamlit run app.py                     # web UI
python evaluate.py                       # keyword vs semantic vs hybrid (R@1, R@3, MRR)
```

## Notes / next steps
- The sample corpus is tiny (13 docs), so LSA has little to learn from; it improves with hundreds of documents. Point it at a real folder.
- Add your own queries to `eval_queries.json` as `[query, expected_filename]` to tune `alpha`.
- For true semantic matching, add sentence-transformer embeddings (e.g. `all-MiniLM-L6-v2`) as a third score in `SearchEngine.scores`.
- The index rebuilds fully when files change (`is_stale`); for very large folders, make it incremental.
- Scanned PDFs need OCR first; they contain no extractable text.
