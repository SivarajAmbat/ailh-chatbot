# training_chatbot.py
import os
import glob
import re
import pickle
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

import streamlit as st

# -------------------------
# Configuration
# -------------------------
DATA_DIR = "data"       # folder with excel files
INDEX_DIR = "index_data"  # folder to persist index + metadata
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, good quality

if not os.path.exists(INDEX_DIR):
    os.makedirs(INDEX_DIR)



import os
os.environ["CURL_CA_BUNDLE"] = ""

# -------------------------
# Utilities: load & normalize
# -------------------------
def load_all_excels(folder: str) -> pd.DataFrame:
    """Load all excels from folder into one DataFrame and normalize columns."""
    files = glob.glob(os.path.join(folder, "*.xlsx")) + glob.glob(os.path.join(folder, "*.xls"))
    dfs = []
    for f in files:
        try:
            df = pd.read_excel(f, engine="openpyxl")
        except Exception:
            df = pd.read_excel(f)
        df["__source_file"] = os.path.basename(f)
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()  # empty
    combined = pd.concat(dfs, ignore_index=True)
    # Normalize column names to lowercase stripped
    combined.columns = [c.strip() if isinstance(c, str) else c for c in combined.columns]
    lowermap = {c: c.lower() for c in combined.columns}
    combined = combined.rename(columns=lowermap)
    # Ensure columns exist
    required = ["date", "topic", "explanation", "category", "reference material", "session recording"]
    for req in required:
        if req not in combined.columns:
            combined[req] = None
    # Clean up types
    try:
        # combined["date"] = pd.to_datetime(combined["date"], errors="coerce").dt.date
        # combined["date"] = pd.to_datetime(combined["date"], format="%B %d, %Y", errors="coerce").dt.date

        combined["date"] = pd.to_datetime(combined["date"].astype(str).str.strip(), errors="coerce").dt.date

        invalid = combined[combined["date"].isna()]
        if not invalid.empty:
            st.write("Unparseable date values:")
            st.write(invalid["original_date_column"])  # Replace with actual column name if needed


    except Exception:
        pass
    combined["topic"] = combined["topic"].astype(str).str.strip()
    combined["explanation"] = combined["explanation"].astype(str).str.strip()
    combined["category"] = combined["category"].astype(str).str.strip()
    combined["reference material"] = combined["reference material"].astype(str).replace("nan", "")
    combined["session recording"] = combined["session recording"].astype(str).replace("nan", "")
    # Create a short text field used for semantic search
    combined["search_text"] = combined.apply(
        lambda r: " | ".join(filter(None, [str(r.get("topic","")), str(r.get("explanation","")), str(r.get("category",""))])),
        axis=1
    )
    combined = combined.reset_index().rename(columns={"index": "row_id"})
    return combined

# -------------------------
# Embedding / Index building
# -------------------------
class SearchIndex:
    def __init__(self, model_name=EMBED_MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.index = None  # faiss index
        self.embeddings = None  # numpy array (n, d)
        self.df = None  # dataframe with metadata
        self.dim = self.model.get_sentence_embedding_dimension()

    def build(self, df: pd.DataFrame, text_col: str = "search_text", persist=True):
        """Build embeddings + FAISS index from dataframe."""
        self.df = df.copy().reset_index(drop=True)
        texts = self.df[text_col].fillna("").tolist()
        # compute embeddings in batches to avoid OOM
        batch_size = 128
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            emb = self.model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
            embs.append(emb)
        self.embeddings = np.vstack(embs).astype("float32")
        # Build FAISS index (inner product on normalized embeddings = cosine)
        faiss.normalize_L2(self.embeddings)
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(self.embeddings)
        if persist:
            self.save(os.path.join(INDEX_DIR, "faiss.index"), os.path.join(INDEX_DIR, "meta.pkl"))

    def save(self, index_path: str, meta_path: str):
        faiss.write_index(self.index, index_path)
        with open(meta_path, "wb") as f:
            pickle.dump({"df": self.df, "embeddings_shape": self.embeddings.shape}, f)

    def load(self, index_path: str, meta_path: str):
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError("Index or meta not found.")
        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        self.df = meta["df"]
        # we don't re-load embeddings into memory unless needed
        self.dim = self.model.get_sentence_embedding_dimension()

    def semantic_search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        q_emb = self.model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_emb)
        scores, idxs = self.index.search(q_emb, top_k)
        idxs = idxs[0]
        scores = scores[0]
        results = []
        for i, idx in enumerate(idxs):
            if idx == -1:
                continue
            results.append((int(idx), float(scores[i])))
        return results

# -------------------------
# Intent handling: rule-based + semantic fallback
# -------------------------
def find_exact_topic_rows(df: pd.DataFrame, topic: str) -> pd.DataFrame:
    # exact or case-insensitive substring match on topic column
    mask = df["topic"].str.lower().str.contains(topic.lower(), na=False)
    return df[mask]

def list_topics(df: pd.DataFrame, limit: Optional[int] = None) -> List[str]:
    topics = df["topic"].dropna().unique().tolist()
    topics_sorted = sorted(topics, key=lambda s: s.lower())
    return topics_sorted[:limit] if limit else topics_sorted

def topics_in_category(df: pd.DataFrame, category: str) -> List[str]:
    mask = df["category"].str.lower().str.contains(category.lower(), na=False)
    return sorted(df.loc[mask, "topic"].dropna().unique().tolist())

def dates_for_keyword(df: pd.DataFrame, keyword: str) -> List[str]:
    mask = df["search_text"].str.lower().str.contains(keyword.lower(), na=False)
    dates = df.loc[mask, "date"].dropna().astype(str).unique().tolist()
    return sorted(dates)

def get_links_for_topic(df: pd.DataFrame, topic: str) -> List[Dict[str,str]]:
    rows = find_exact_topic_rows(df, topic)
    links = []
    for _, r in rows.iterrows():
        links.append({
            "topic": r["topic"],
            "date": str(r.get("date", "")),
            "ppt": r.get("reference material", ""),
            "recording": r.get("session recording", ""),
            "source_file": r.get("__source_file", "")
        })
    return links

def extract_intent_and_arg(query: str) -> Tuple[str, Optional[str]]:
    """Simple heuristics to map query to intent + argument."""
    q = query.strip().lower()
    # Patterns
    if re.match(r"^(list|show).*(topics|topic)", q):
        # maybe category requested: "list topics in <category>"
        m = re.search(r"in\s+([a-z0-9 _-]+)$", q)
        if m:
            return ("list_topics_in_category", m.group(1).strip())
        return ("list_topics", None)
    if re.match(r"^(explain|what is|explanation).*(of|for)?\s*", q):
        m = re.search(r"(?:of|for)\s+(.+)$", q)
        if m:
            return ("explain_topic", m.group(1).strip())
    if re.match(r"^(dates|when).*(topic|keyword|discussed|discuss)", q):
        m = re.search(r"dates?.* (?:for|of)?\s*(.+)$", q)
        if m:
            return ("dates_for_keyword", m.group(1).strip())
    if re.match(r"^(ppt|presentation|slides).*(for)?", q):
        m = re.search(r"(?:for|about)\s+(.+)$", q)
        if m:
            return ("ppt_for_topic", m.group(1).strip())
    if re.match(r"^(recording|video).*(for)?", q):
        m = re.search(r"(?:for|about)\s+(.+)$", q)
        if m:
            return ("recording_for_topic", m.group(1).strip())
    # direct: "explain <topic>"
    m = re.match(r"^explain\s+(.+)$", q)
    if m:
        return ("explain_topic", m.group(1).strip())
    # direct: "topics in <category>"
    m = re.match(r"^topics in\s+(.+)$", q)
    if m:
        return ("list_topics_in_category", m.group(1).strip())
    # fallback: None => semantic
    return ("semantic", query)

def handle_query(search_index: SearchIndex, query: str, top_k: int = 5) -> str:
    intent, arg = extract_intent_and_arg(query)
    df = search_index.df
    # safety guard if df missing
    if df is None or df.empty:
        return "No data loaded. Put your Excel files into the folder and rebuild the index."
    if intent == "list_topics":
        topics = list_topics(df)
        return "Topics across sessions:\n\n" + "\n".join(f"- {t}" for t in topics)
    elif intent == "list_topics_in_category" and arg:
        topics = topics_in_category(df, arg)
        if not topics:
            return f"No topics found in category '{arg}'."
        return f"Topics in category '{arg}':\n\n" + "\n".join(f"- {t}" for t in topics)
    elif intent == "explain_topic" and arg:
        rows = find_exact_topic_rows(df, arg)
        if not rows.empty:
            out = []
            for _, r in rows.iterrows():
                out.append(f"Topic: {r['topic']}\nDate: {r.get('date','')}\nCategory: {r.get('category','')}\nExplanation: {r.get('explanation','')}\nPPT: {r.get('reference material','')}\nRecording: {r.get('session recording','')}")
            return "\n\n---\n\n".join(out)
        # fallback to semantic
        sem = search_index.semantic_search(arg, top_k=top_k)
        if not sem:
            return f"No explanation found for '{arg}'."
        lines = []
        for idx, score in sem:
            r = df.iloc[idx]
            lines.append(f"Match (score {score:.3f}):\nTopic: {r['topic']}\nExplanation: {r['explanation']}\nDate: {r.get('date','')}\nCategory: {r.get('category','')}\nPPT: {r.get('reference material','')}\nRecording: {r.get('session recording','')}")
        return "\n\n---\n\n".join(lines)
    elif intent == "dates_for_keyword" and arg:
        dates = dates_for_keyword(df, arg)
        if not dates:
            return f"No dates found for '{arg}'."
        return f"Dates when '{arg}' was discussed:\n\n" + "\n".join(f"- {d}" for d in dates)
    elif intent == "ppt_for_topic" and arg:
        links = get_links_for_topic(df, arg)
        if links:
            return "\n".join([f"{l['date']} - {l['topic']}\nPPT: {l['ppt']}\nRecording: {l['recording']}\nFile: {l['source_file']}" for l in links])
        # fallback semantic
        sem = search_index.semantic_search(arg, top_k=top_k)
        if not sem:
            return f"No PPT/Recording found for '{arg}'."
        out = []
        for idx, score in sem:
            r = df.iloc[idx]
            out.append(f"Match (score {score:.3f}): {r['topic']} - PPT: {r.get('reference material','')} - Recording: {r.get('session recording','')}")
        return "\n".join(out)
    elif intent == "recording_for_topic" and arg:
        return handle_query(search_index, "ppt for " + arg, top_k=top_k)  # same treatment
    elif intent == "semantic":
        # do semantic search and show nearest matches
        sem = search_index.semantic_search(arg, top_k=top_k)
        if not sem:
            return "No relevant results found."
        out = []
        for idx, score in sem:
            r = df.iloc[idx]
            out.append(f"Score: {score:.3f}\nTopic: {r['topic']}\nExplanation (excerpt): {r['explanation'][:400]}\nDate: {r.get('date','')}\nCategory: {r.get('category','')}\nPPT: {r.get('reference material','')}\nRecording: {r.get('session recording','')}\n---")
        return "\n\n".join(out)
    else:
        return "Sorry, I couldn't understand the request. Try: 'list topics', 'explain <topic>', 'topics in <category>', 'dates for <keyword>', 'ppt for <topic>'."

# -------------------------
# Streamlit UI
# -------------------------
def main():
    st.set_page_config(page_title="Training Sessions Chatbot", layout="wide")
    st.title("💬 AI Learning Hours Assistant")
    st.write(
        """
        This is a simple chatbot that can help you with the content discussed during the sessions"
        """
    )


    st.markdown("""
    <style>
        /* Override markdown styles to use normal font size */
        .markdown-text-container h1,
        .markdown-text-container h2,
        .markdown-text-container h3,
        .markdown-text-container h4,
        .markdown-text-container h5,
        .markdown-text-container h6 {
            font-size: 1rem !important;
            font-weight: normal !important;
            margin: 0 !important;
        }
        .markdown-text-container p {
            font-size: 1rem !important;
            margin: 0 !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar: load / build / persist
    # st.sidebar.title("Data & Index")
    data_dir = "data"
    # refresh = st.sidebar.button("Load & (re)build index")
    # load_saved = st.sidebar.button("Load saved index (if exists)")

    refresh = True
    load_saved = False

    st.write(data_dir)
    st.write(refresh)
    st.write(load_saved)


    # instantiate
    idx_obj = st.session_state.get("idx_obj", None)
    if idx_obj is None:
        idx_obj = SearchIndex()
        st.session_state["idx_obj"] = idx_obj


    if refresh:
        with st.spinner("Please wait. Loading data and building index ..."):
            df = load_all_excels(data_dir)
            if df.empty:
                st.error(f"No data files found in {data_dir} directory. Please contact developer for assistance.")
                return
            st.session_state["df"] = df
            idx_obj.build(df)
            st.success(f"Loaded {len(df)} rows and built index.")
            refresh = False
            load_saved = True
    elif load_saved:
        try:
            idx_obj.load(os.path.join(INDEX_DIR, "faiss.index"), os.path.join(INDEX_DIR, "meta.pkl"))
            st.session_state["df"] = idx_obj.df
            st.success("Loaded saved index and metadata.")
        except Exception as e:
            st.error(f"Failed to load saved index: {e}")

    # If session has df, show quick stats
    df = st.session_state.get("df", None)
    if df is not None:
        st.sidebar.write(f"Rows: {len(df)}")
        st.sidebar.write(f"Unique Topics: {df['topic'].nunique()}")
        st.sidebar.write(f"Unique Categories: {df['category'].nunique()}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("Example queries:")
    st.sidebar.markdown(
        """
        - list topics\n- list topics from any session \n- list the model releases\n- list the discussions in any category\n- what's new about anything\n- PPT or recording for any session
        """
    )

    # Main: Chat interface
    st.header("Ask about sessions")
    query = st.chat_input("Ask anything", key="query_input")
    # top_k = st.slider("Number of results to return (semantic)", min_value=1, max_value=10, value=5)
    top_k = 15
    if query:
        if df is None:
            st.error("No data loaded. Click 'Load & (re)build index' on the sidebar first.")
        else:
            with st.spinner("Searching..."):
                response = handle_query(idx_obj, query, top_k=top_k)
                with st.chat_message("assistant"):
                    st.markdown(response)

    # Bonus: quick tables
    st.header("Browse data")
    if df is not None:
        with st.expander("Show raw rows"):
            st.dataframe(df[["date", "topic", "category", "reference material", "session recording", "__source_file"]].sort_values(by="date", na_position="first"))

        st.write("Quick: list topics by category")
        cats = sorted(df["category"].dropna().unique().tolist())
        cat_choice = st.selectbox("Select category", ["(all)"] + cats)
        if st.button("Show topics in category"):
            if cat_choice == "(all)":
                results = list_topics(df)
            else:
                results = topics_in_category(df, cat_choice)
            st.write(f"Topics ({len(results)}):")
            for t in results:
                st.write("- " + t)

if __name__ == "__main__":
    main()
