"""Hybrid search: BM25 (exact keywords) + LSA (latent semantic similarity)."""
import re
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize

from loaders import read_file, scan_folder


def chunk_text(text: str, size: int = 120, overlap: int = 30):
    words = text.split()
    if not words:
        return []
    step = max(size - overlap, 1)
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words) - overlap, 1), step)]


def _minmax(x):
    rng = x.max() - x.min()
    return (x - x.min()) / rng if rng > 1e-12 else np.zeros_like(x)


@dataclass
class Hit:
    path: str
    score: float
    snippet: str
    bm25: float
    semantic: float


class SearchEngine:
    def __init__(self, k1=1.5, b=0.75, n_topics=100):
        self.k1, self.b, self.n_topics = k1, b, n_topics
        self.chunks, self.meta, self.files = [], [], {}   # meta[i] = path of chunk i

    # ---------- indexing ----------
    def build(self, folder: str, chunk_size=120, overlap=30):
        self.chunks, self.meta, self.files = [], [], {}
        for path, mtime in scan_folder(folder):
            self.files[str(path)] = mtime
            for c in chunk_text(read_file(path), chunk_size, overlap):
                self.chunks.append(c)
                self.meta.append(str(path))
        if not self.chunks:
            raise ValueError(f"No readable documents found in {folder}")

        self.cv = CountVectorizer(stop_words="english", lowercase=True,
                                  token_pattern=r"(?u)\b[a-zA-Z0-9][a-zA-Z0-9+#.-]*\b")
        tf = self.cv.fit_transform(self.chunks).tocsr().astype(np.float32)

        # BM25 weight matrix
        N = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        self.idf = np.log(1 + (N - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.asarray(tf.sum(axis=1)).ravel()
        norm = self.k1 * (1 - self.b + self.b * dl / max(dl.mean(), 1e-9))
        coo = tf.tocoo()
        w = coo.data * (self.k1 + 1) / (coo.data + norm[coo.row]) * self.idf[coo.col]
        self.bm25_w = sp.csc_matrix((w, (coo.row, coo.col)), shape=tf.shape)

        # LSA: TF-IDF -> truncated SVD
        self.tfidf = TfidfTransformer(sublinear_tf=True).fit(tf)
        X = self.tfidf.transform(tf)
        k = max(2, min(self.n_topics, X.shape[0] - 1, X.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=0).fit(X)
        self.Z = normalize(self.svd.transform(X))
        return self

    # ---------- querying ----------
    def scores(self, query: str):
        vocab = self.cv.vocabulary_
        idx = [vocab[t] for t in self.cv.build_analyzer()(query) if t in vocab]
        bm25 = np.asarray(self.bm25_w[:, idx].sum(axis=1)).ravel() if idx else np.zeros(len(self.chunks))
        q = self.tfidf.transform(self.cv.transform([query]))
        sem = self.Z @ normalize(self.svd.transform(q)).ravel()
        return bm25, sem

    def search(self, query: str, k=5, alpha=0.6) -> list:
        """alpha = weight on BM25 (1.0 = pure keyword, 0.0 = pure semantic)."""
        bm25, sem = self.scores(query)
        fused = alpha * _minmax(bm25) + (1 - alpha) * _minmax(sem)
        hits, seen = [], set()
        for i in np.argsort(-fused):
            path = self.meta[i]
            if path in seen:
                continue                    # one best chunk per document
            seen.add(path)
            hits.append(Hit(path, float(fused[i]), self._snippet(self.chunks[i], query),
                            float(bm25[i]), float(sem[i])))
            if len(hits) == k:
                break
        return hits

    def _snippet(self, text, query, width=260):
        terms = [t for t in self.cv.build_analyzer()(query)]
        pos = [m.start() for t in terms for m in [re.search(re.escape(t), text, re.I)] if m]
        start = max(min(pos) - 60, 0) if pos else 0
        s = text[start:start + width]
        return ("…" if start else "") + s + ("…" if start + width < len(text) else "")

    # ---------- persistence ----------
    def is_stale(self, folder: str) -> bool:
        return {str(p): m for p, m in scan_folder(folder)} != self.files

    def save(self, path="index.joblib"):
        joblib.dump(self, path)

    @staticmethod
    def load(path="index.joblib"):
        return joblib.load(path)


def highlight(text: str, query: str, engine: SearchEngine, fmt="**{}**") -> str:
    terms = sorted({t for t in engine.cv.build_analyzer()(query)}, key=len, reverse=True)
    if not terms:
        return text
    pat = re.compile("(" + "|".join(re.escape(t) for t in terms) + ")", re.I)
    return pat.sub(lambda m: fmt.format(m.group(0)), text)
