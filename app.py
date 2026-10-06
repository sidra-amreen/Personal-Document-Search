from pathlib import Path

import streamlit as st

from engine import SearchEngine, highlight

st.set_page_config(page_title="Personal Document Search", page_icon="🔎", layout="wide")
INDEX = "index.joblib"

st.title("🔎 Personal Document Search")
folder = st.sidebar.text_input("Folder to index", "sample_docs")
alpha = st.sidebar.slider("Keyword ↔ Semantic", 0.0, 1.0, 0.6, 0.1,
                          help="1.0 = exact keyword (BM25), 0.0 = latent-semantic (LSA)")
k = st.sidebar.slider("Results", 1, 15, 5)

if st.sidebar.button("(Re)build index") or "engine" not in st.session_state:
    try:
        if Path(INDEX).exists() and "engine" not in st.session_state:
            eng = SearchEngine.load(INDEX)
            if eng.is_stale(folder):
                eng = SearchEngine().build(folder)
        else:
            eng = SearchEngine().build(folder)
        eng.save(INDEX)
        st.session_state.engine = eng
    except Exception as e:
        st.sidebar.error(str(e)); st.stop()

eng = st.session_state.engine
st.sidebar.caption(f"{len(eng.files)} files · {len(eng.chunks)} chunks")

query = st.text_input("Search your documents", placeholder="e.g. how to cook with sourdough starter")
if query:
    for h in eng.search(query, k, alpha):
        with st.container(border=True):
            st.markdown(f"**📄 {Path(h.path).name}** &nbsp; `{h.score:.2f}`")
            st.markdown(highlight(h.snippet, query, eng, "<mark>{}</mark>"), unsafe_allow_html=True)
            st.caption(h.path)
