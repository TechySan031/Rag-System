"""
Document parser for PDF and Markdown files.
Extracts text with metadata (source, page number, file type).
"""
import fitz  # PyMuPDF
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class Document:
    """A parsed document unit with text content and metadata."""
    text: str
    metadata: dict = field(default_factory=dict)


def parse_pdf(filepath: str | Path) -> list[Document]:
    """
    Parse a PDF file into a list of Documents, one per page.
    Uses PyMuPDF (fitz) for robust text extraction.
    """
    filepath = Path(filepath)
    documents = []

    with fitz.open(str(filepath)) as pdf:
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            text = page.get_text("text").strip()

            if not text:
                continue

            documents.append(Document(
                text=text,
                metadata={
                    "source": filepath.name,
                    "page": page_num + 1,  # 1-indexed
                    "type": "pdf",
                    "total_pages": len(pdf),
                }
            ))

    return documents


def parse_markdown(filepath: str | Path) -> list[Document]:
    """
    Parse a Markdown file into a single Document.
    Preserves the full text as one unit (chunking is separate).
    """
    filepath = Path(filepath)
    text = filepath.read_text(encoding="utf-8").strip()

    if not text:
        return []

    return [Document(
        text=text,
        metadata={
            "source": filepath.name,
            "page": 1,
            "type": "markdown",
        }
    )]


def parse_text(filepath: str | Path) -> list[Document]:
    """Parse a plain text file into a single Document."""
    filepath = Path(filepath)
    text = filepath.read_text(encoding="utf-8").strip()

    if not text:
        return []

    return [Document(
        text=text,
        metadata={
            "source": filepath.name,
            "page": 1,
            "type": "text",
        }
    )]


def parse_file(filepath: str | Path) -> list[Document]:
    """
    Auto-detect file type and route to the correct parser.
    Raises ValueError for unsupported file types.
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    parsers = {
        ".pdf": parse_pdf,
        ".md": parse_markdown,
        ".markdown": parse_markdown,
        ".txt": parse_text,
    }

    parser = parsers.get(suffix)
    if parser is None:
        raise ValueError(f"Unsupported file type: {suffix}. Supported: {list(parsers.keys())}")

    return parser(filepath)
