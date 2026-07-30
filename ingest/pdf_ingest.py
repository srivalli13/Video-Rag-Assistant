"""Extract text from PDF files, page by page, with metadata."""

from pathlib import Path
import fitz  # PyMuPDF's import name


def extract_pdf(pdf_path: str) -> list[dict]:
    """
    Read a PDF and return a list of page records.

    Each record: {"text": str, "page": int, "source": str}
    Empty pages (images-only, blanks) are skipped.
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)

    pages = []
    for page_number, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if not text:
            continue  # skip pages with no extractable text
        pages.append({
            "text": text,
            "page": page_number,
            "source": pdf_path.name,
        })

    doc.close()
    return pages


if __name__ == "__main__":
    # Quick manual test: python ingest/pdf_ingest.py path/to/file.pdf
    import sys

    result = extract_pdf(sys.argv[1])
    print(f"Extracted {len(result)} non-empty pages from {sys.argv[1]}")
    print("--- First page preview ---")
    print(result[0]["text"][:400])