"""
data_loader.py — Downloads, cleans, and returns raw text from both datasets.
No RAG framework used. All cleaning is manual string/pandas operations.

Student: Najart Rauf Awuni | Index: 10022300177
"""

import io
import re
import logging
import requests
import pandas as pd
import pypdf

logger = logging.getLogger(__name__)

CSV_URL = (
    "https://github.com/GodwinDansoAcity/acitydataset/raw/main/"
    "Ghana_Election_Result.csv"
)
PDF_URL = (
    "https://mofep.gov.gh/sites/default/files/budget-statements/"
    "2025-Budget-Statement-and-Economic-Policy_v4.pdf"
)


# ─────────────────────────────────────────────
#  CLEANING HELPERS
# ─────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """
    Remove non-printable characters, collapse whitespace, and strip HTML
    remnants. Keeping punctuation because it matters for sentence chunking.
    """
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)   # keep printable ASCII
    text = re.sub(r"[ \t]+", " ", text)              # collapse spaces/tabs
    text = re.sub(r"\n{3,}", "\n\n", text)           # max 2 blank lines
    return text.strip()


def _clean_csv_row(row: pd.Series) -> str:
    """Convert one CSV row to a readable sentence for embedding."""
    parts = []
    for col, val in row.items():
        if pd.notna(val) and str(val).strip():
            parts.append(f"{col}: {val}")
    return " | ".join(parts)


# ─────────────────────────────────────────────
#  LOADERS
# ─────────────────────────────────────────────

def load_csv(url: str = CSV_URL) -> list[dict]:
    """
    Download Ghana Election CSV, clean each row, return list of dicts
    with keys: 'text' (the cleaned sentence) and 'source'.
    """
    logger.info("Downloading CSV from %s", url)
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    df = pd.read_csv(io.StringIO(response.text))
    logger.info("CSV loaded — %d rows, %d columns", len(df), len(df.columns))

    # --- Data cleaning ---
    df.columns = [c.strip().replace("\n", " ") for c in df.columns]  # tidy headers
    df.drop_duplicates(inplace=True)
    df.dropna(how="all", inplace=True)

    # Convert numeric vote columns to int where possible
    for col in df.columns:
        if "vote" in col.lower() or "result" in col.lower() or "seat" in col.lower():
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    records = []
    for _, row in df.iterrows():
        text = _clean_csv_row(row)
        if len(text) > 20:                          # skip near-empty rows
            records.append({"text": text, "source": "Ghana_Election_Result.csv"})

    logger.info("CSV produced %d clean records", len(records))
    return records


def load_pdf(url: str = PDF_URL) -> list[dict]:
    """
    Download Ghana 2025 Budget PDF, extract text page-by-page via pypdf,
    clean it, return list of dicts with 'text', 'source', and 'page'.
    """
    logger.info("Downloading PDF from %s", url)
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    pdf_bytes = io.BytesIO(response.content)
    reader = pypdf.PdfReader(pdf_bytes)
    logger.info("PDF loaded — %d pages", len(reader.pages))

    records = []
    for page_num, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        cleaned = _clean_text(raw)
        if len(cleaned) > 50:                       # skip mostly-blank pages
            records.append({
                "text": cleaned,
                "source": "2025-Budget-Statement.pdf",
                "page": page_num,
            })

    logger.info("PDF produced %d non-empty pages", len(records))
    return records


def load_all_documents() -> list[dict]:
    """
    Entry point: load both datasets and merge into one document list.
    Each document: {'text': str, 'source': str, 'page'?: int}
    """
    docs = []
    try:
        docs.extend(load_csv())
    except Exception as exc:
        logger.error("CSV load failed: %s", exc)

    try:
        docs.extend(load_pdf())
    except Exception as exc:
        logger.error("PDF load failed: %s", exc)

    logger.info("Total documents loaded: %d", len(docs))
    return docs
