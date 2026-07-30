"""Streamlit UI: upload lectures/PDFs, ask grounded questions."""

from pathlib import Path

import streamlit as st

from ingest.pdf_ingest import extract_pdf
from ingest.video_ingest import transcribe_video
from processing.chunker import chunk_records, group_segments
from embeddings.embedder import embed_texts
from database.faiss_manager import FaissManager
from retrieval.retriever import retrieve
from generation.generator import generate_answer
from utils.manifest import already_ingested, mark_ingested, load_manifest
from utils.config import UPLOAD_DIR

st.set_page_config(page_title="Video RAG Assistant", layout="centered")


@st.cache_resource
def get_manager():
    return FaissManager()


if "history" not in st.session_state:
    st.session_state.history = []

manager = get_manager()

with st.sidebar:
    st.subheader("Indexed files")
    files = load_manifest()
    if files:
        for f in files:
            st.write(f"📄 {f}")
    else:
        st.caption("No files ingested yet.")

st.title("Video RAG Assistant")
st.caption(f"Index contains {manager.size} chunks")

uploaded = st.file_uploader(
    "Upload a lecture video or PDF", type=["mp4", "mov", "pdf"]
)

if uploaded and st.button("Ingest"):
    safe_name = Path(uploaded.name).name  # strip any path components from the filename

    if already_ingested(safe_name):
        st.warning(f"{safe_name} is already in the index.")
    else:
        save_path = UPLOAD_DIR / safe_name
        try:
            with open(save_path, "wb") as f:
                f.write(uploaded.getbuffer())

            with st.spinner(f"Processing {safe_name}..."):
                if safe_name.lower().endswith(".pdf"):
                    chunks = chunk_records(extract_pdf(str(save_path)))
                else:
                    chunks = group_segments(transcribe_video(str(save_path)))

                if not chunks:
                    st.warning(
                        f"No extractable text/audio found in {safe_name} — nothing was added."
                    )
                else:
                    vectors = embed_texts([c["text"] for c in chunks])
                    manager.add(vectors, chunks)
                    mark_ingested(safe_name)
                    st.success(f"Added {len(chunks)} chunks from {safe_name}")
                    st.rerun()
        except Exception as e:
            st.error(f"Failed to process {safe_name}: {e}")

st.divider()
st.subheader("Ask a question")

question = st.chat_input("Ask about your uploaded content")

if question:
    try:
        chunks = retrieve(question, manager)
        answer = generate_answer(question, chunks, st.session_state.history)
    except Exception as e:
        answer = f"Something went wrong while generating an answer: {e}"

    st.session_state.history.append({"role": "user", "text": question})
    st.session_state.history.append({"role": "assistant", "text": answer})

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.write(turn["text"])