# RAG-Based Video AI Assistant

An AI study assistant that answers questions from your own lecture videos and PDFs — and only from them. Built with Retrieval-Augmented Generation (RAG): it retrieves relevant passages from your uploaded content, then asks Gemini to answer strictly using that context. If the answer isn't in your materials, it says so instead of guessing.

## Why RAG instead of just prompting an LLM

Pasting entire lecture transcripts into a prompt doesn't scale — context windows are finite, cost scales with tokens sent, and models get less reliable with huge unfocused inputs. RAG instead: embeds your content into a searchable vector index once, then at query time retrieves only the handful of chunks relevant to the specific question. This keeps prompts small, cheap, and focused — and lets the index grow to any number of documents.

## Architecture

video-rag-assistant/
├── app.py # Streamlit UI — orchestration only, no ML logic
├── ingest/
│ ├── video_ingest.py # mp4/mov -> Whisper -> timestamped segments
│ └── pdf_ingest.py # PDF -> page-level text
├── processing/
│ └── chunker.py # chunk_records() slices PDFs; group_segments() merges video segments
├── embeddings/
│ └── embedder.py # BGE-M3, loaded once, normalized vectors
├── database/
│ └── faiss_manager.py # FAISS index + parallel metadata store, kept in sync by position
├── retrieval/
│ └── retriever.py # embed query -> search -> similarity threshold filter
├── generation/
│ └── generator.py # grounding prompt + Gemini call
├── utils/
│ ├── config.py # all tunables: chunk size, top-k, threshold, model names
│ └── manifest.py # tracks ingested filenames, prevents duplicate ingestion
├── data/ # gitignored: uploads, FAISS index, manifest
└── requirements.txt


**Data flow:** file upload → extract/transcribe (with metadata: page number or timestamp) → chunk to ~500 characters → embed with BGE-M3 → store in FAISS. At query time: embed the question with the same model → search FAISS for the top-k nearest chunks → drop any chunk scoring below the similarity threshold → build a grounding prompt from the survivors → Gemini generates a cited answer, or the app refuses if nothing passed the threshold.

## Tech stack

| Component | Choice | Why |
|---|---|---|
| Speech-to-text | OpenAI Whisper (`base`) | Runs locally, free, gives segment-level timestamps for citation |
| Embeddings | BGE-M3 (Sentence Transformers) | Strong open-source multilingual model, runs locally at zero cost per call |
| Vector search | FAISS (`IndexFlatIP`) | Exact search is fast enough at this scale; normalized vectors make inner product equal cosine similarity |
| Generation | Gemini (`gemini-flash-latest`) | Hosted, high-quality; alias name avoids breakage when Google retires specific model versions |
| UI | Streamlit | Fast to build a working chat interface; `@st.cache_resource` keeps heavy models loaded once per session |
| PDF parsing | PyMuPDF | Fast, reliable extraction with page numbers |

## Two safeguards against hallucination

1. **Similarity threshold** — every retrieved chunk has a score; anything below the configured threshold is dropped before Gemini is ever called.
2. **Grounding prompt** — Gemini is explicitly instructed to answer only from the provided context and say so if the answer isn't there.

## Setup

```bash
git clone https://github.com/siddhantsantoshnarlawar-afk/video-rag-assistant.git
cd video-rag-assistant
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Create a `.env` file in the project root:

GEMINI_API_KEY=your_key_here


Get a key from [aistudio.google.com](https://aistudio.google.com).

## Running

```bash
streamlit run app.py
```

Upload a lecture video (.mp4/.mov) or PDF, click Ingest, then ask questions in the chat box. The sidebar shows every file currently in the index.

## Known limitations (found during testing)

- **Broad/structural questions underperform.** Top-k semantic search retrieves chunks similar in *meaning* to the question, but has no concept of document structure — asking "list all topics in this PDF" retrieves semantically-matching chunks rather than the actual table-of-contents page.
- **Multi-document summarization is weak.** With a fixed top-k across the whole index, a question like "summarize both documents" tends to surface chunks from whichever document scores marginally higher, rather than balanced coverage.
- **Similarity threshold was empirically tuned**, not guessed — testing showed unrelated sentence pairs scoring up to ~0.45 under BGE-M3, so the threshold was raised from an initial 0.4 to 0.5 to avoid answering off-topic questions.

## Future improvements

- Hybrid search (keyword + semantic) to complement pure vector similarity
- Cross-encoder reranking of top-k results for better precision
- Query classification to route structural/broad questions differently from pointed factual ones
- Docker containerization
- User authentication and per-user indexes
- Cloud deployment
- Response caching for repeated questions
- OCR support for scanned/image-based PDFs
- Multi-language support (BGE-M3 is already multilingual; Whisper needs explicit language pinning for non-English audio)
- Voice-based question input