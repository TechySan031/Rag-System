r"""
Document chunker with fixed-size + overlap and structure-aware strategies.
Produces chunks suitable for embedding and retrieval.

Two modes:
  1. Fixed-size + overlap: Simple, predictable baseline
  2. Structure-aware: Splits on markdown headers, section boundaries, and
     paragraph breaks first, then applies fixed-size within sections.
     Preserves heading context in chunk metadata.

Mode selected via STRUCTURE_AWARE_CHUNKING config flag.
"""
import re
import hashlib
from app.core.parser import Document


def _generate_chunk_id(source: str, page: int, offset: int) -> str:
    """Generate a deterministic chunk ID from source metadata."""
    raw = f"{source}:{page}:{offset}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[tuple[str, int]]:
    """
    Split text into overlapping chunks (fixed-size strategy).
    Returns list of (chunk_text, char_offset) tuples.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at a sentence boundary (. ! ? newline) near the end
        if end < len(text):
            search_start = max(start + chunk_size // 2, start)
            best_break = end
            for i in range(end, search_start, -1):
                if text[i - 1] in ".!?\n":
                    best_break = i
                    break
            end = best_break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append((chunk, start))

        # Advance with overlap
        start = end - overlap if end < len(text) else len(text)

    return chunks


def _split_by_structure(text: str) -> list[dict]:
    """
    Split text by structural boundaries (headers, section breaks).
    Returns list of dicts with 'text', 'heading', and 'offset'.
    """
    # Match markdown-style headers: # Title, ## Section, etc.
    header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

    sections = []
    last_end = 0
    current_heading = ""

    for match in header_pattern.finditer(text):
        # Capture text before this header
        before_text = text[last_end:match.start()].strip()
        if before_text:
            sections.append({
                "text": before_text,
                "heading": current_heading,
                "offset": last_end,
            })

        current_heading = match.group(2).strip()
        last_end = match.end()

    # Capture remaining text
    remaining = text[last_end:].strip()
    if remaining:
        sections.append({
            "text": remaining,
            "heading": current_heading,
            "offset": last_end,
        })

    # If no headers found, fall back to paragraph splitting
    if len(sections) <= 1:
        sections = _split_by_paragraphs(text)

    return sections


def _split_by_paragraphs(text: str) -> list[dict]:
    """
    Split text on double newlines (paragraph boundaries).
    Groups small paragraphs together.
    """
    paragraphs = re.split(r'\n\s*\n', text)
    sections = []
    offset = 0

    for para in paragraphs:
        para = para.strip()
        if para:
            sections.append({
                "text": para,
                "heading": "",
                "offset": offset,
            })
        offset += len(para) + 2  # account for the split delimiter

    return sections


def _structure_aware_chunk(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[tuple[str, int, str]]:
    """
    Structure-aware chunking: split by headers/sections first,
    then apply fixed-size chunking within each section.

    Returns list of (chunk_text, offset, heading) tuples.
    """
    sections = _split_by_structure(text)
    chunks = []

    for section in sections:
        section_text = section["text"]
        heading = section["heading"]
        base_offset = section["offset"]

        if len(section_text) <= chunk_size:
            # Section fits in one chunk
            chunks.append((section_text, base_offset, heading))
        else:
            # Apply fixed-size chunking within the section
            sub_chunks = chunk_text(section_text, chunk_size, overlap)
            for sub_text, sub_offset in sub_chunks:
                # Prepend heading as context if available
                if heading:
                    contextualized = f"[Section: {heading}]\n{sub_text}"
                else:
                    contextualized = sub_text
                chunks.append((contextualized, base_offset + sub_offset, heading))

    return chunks


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 512,
    overlap: int = 64,
    structure_aware: bool = True,
) -> list[Document]:
    """
    Chunk a list of Documents into smaller pieces.

    Args:
        documents: List of parsed Document objects
        chunk_size: Max characters per chunk
        overlap: Overlap between consecutive chunks
        structure_aware: If True, use heading/section-based splitting

    Each chunk inherits the parent document's metadata plus chunk-specific info.
    """
    chunked = []

    for doc in documents:
        if structure_aware:
            text_chunks = _structure_aware_chunk(doc.text, chunk_size, overlap)
            for chunk_text_content, offset, heading in text_chunks:
                chunk_id = _generate_chunk_id(
                    doc.metadata.get("source", "unknown"),
                    doc.metadata.get("page", 1),
                    offset,
                )
                chunk_metadata = {
                    **doc.metadata,
                    "chunk_id": chunk_id,
                    "chunk_offset": offset,
                    "chunk_size": len(chunk_text_content),
                    "heading": heading,
                }
                chunked.append(Document(
                    text=chunk_text_content,
                    metadata=chunk_metadata,
                ))
        else:
            # Original fixed-size strategy
            text_chunks = chunk_text(doc.text, chunk_size, overlap)
            for chunk_text_content, offset in text_chunks:
                chunk_id = _generate_chunk_id(
                    doc.metadata.get("source", "unknown"),
                    doc.metadata.get("page", 1),
                    offset,
                )
                chunk_metadata = {
                    **doc.metadata,
                    "chunk_id": chunk_id,
                    "chunk_offset": offset,
                    "chunk_size": len(chunk_text_content),
                }
                chunked.append(Document(
                    text=chunk_text_content,
                    metadata=chunk_metadata,
                ))

    return chunked
