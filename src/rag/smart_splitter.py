"""
Smart paper-aware document splitter.

Unlike blind fixed-size chunking, this splitter:
1. Detects academic paper section boundaries (Abstract, Introduction, etc.)
2. Splits within section boundaries, preserving semantic units
3. Adapts chunk size per section type (e.g., smaller for abstract, larger for experiments)
4. Injects rich metadata: section name, page range, heading level
"""

import re
from typing import List, Tuple, Optional, Dict
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---- Section detection patterns ----

SECTION_PATTERNS = [
    # Numbered sections: "1. Introduction", "2. Related Work", "3.1 Method"
    (re.compile(r"^\s*(?:\d+\.)+\s*(\w[\w\s]{2,60})\s*$", re.MULTILINE), "numbered"),
    # Bold / heading-style: "Introduction", "ABSTRACT", "Abstract"
    (re.compile(
        r"^\s*(?:ABSTRACT|Abstract|Introduction|Related\s+Work|Background|"
        r"Method(?:ology)?|Experiment|Evaluation|Results?|Discussion|"
        r"Conclusion|References?|Acknowledgments?|Appendix)\s*$",
        re.MULTILINE | re.IGNORECASE,
    ), "named"),
    # Markdown-style: "## Introduction", "### 3.1 Method"
    (re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE), "markdown"),
]

# Sections that benefit from smaller chunks (semantically dense)
NARROW_SECTIONS = {"abstract", "introduction", "conclusion", "discussion", "related work", "background"}
# Sections that can use larger chunks (descriptive/technical)
WIDE_SECTIONS = {"experiment", "evaluation", "results", "method", "methodology", "implementation"}


def _detect_sections(text: str) -> List[Tuple[int, int, str]]:
    """Find section boundaries in paper text.

    Returns list of (start_pos, end_pos, section_title).
    Boundaries are non-overlapping and sorted by position.
    """
    boundaries: List[Tuple[int, str]] = []

    for pattern, style in SECTION_PATTERNS:
        for m in pattern.finditer(text):
            pos = m.start()
            title = m.group(1).strip() if style != "named" else m.group(0).strip()
            boundaries.append((pos, title))

    boundaries.sort(key=lambda x: x[0])

    # Deduplicate adjacent same-name headers
    deduped = []
    for pos, title in boundaries:
        if deduped and abs(pos - deduped[-1][0]) < 10 and deduped[-1][1] == title:
            continue
        deduped.append((pos, title))

    # Build (start, end, title) pairs
    sections = []
    for i, (pos, title) in enumerate(deduped):
        end = deduped[i + 1][0] if i + 1 < len(deduped) else len(text)
        sections.append((pos, end, title))

    return sections


def _get_chunk_params(section_title: str) -> Tuple[int, int]:
    """Determine chunk size and overlap for a section type."""
    title_lower = section_title.lower().strip("# ")

    for narrow_kw in NARROW_SECTIONS:
        if narrow_kw in title_lower:
            return (400, 80)

    for wide_kw in WIDE_SECTIONS:
        if wide_kw in title_lower:
            return (800, 120)

    return (500, 100)  # default


def smart_split_documents(
    documents: List[Document],
    default_chunk_size: int = 500,
    default_chunk_overlap: int = 100,
) -> List[Document]:
    """Split documents with section awareness.

    Args:
        documents: Raw documents from loader (typically one per page).
        default_chunk_size: Fallback when no section detected.
        default_chunk_overlap: Fallback overlap.

    Returns:
        List of chunk Documents with enriched metadata.
    """
    # Merge all page texts into one for section detection
    full_text = "\n\n".join(d.page_content for d in documents)

    # Also track page boundaries for metadata
    page_map: Dict[int, int] = {}  # char_pos -> page_number
    offset = 0
    for i, doc in enumerate(documents):
        page_map[offset] = i + 1
        offset += len(doc.page_content) + 2  # +2 for the "\n\n" join

    # Detect sections
    sections = _detect_sections(full_text)
    if not sections:
        # Fallback: no sections detected, use standard recursive splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=default_chunk_size,
            chunk_overlap=default_chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return splitter.split_documents(documents)

    # Split per section with section-appropriate chunk params
    all_chunks = []
    for start, end, title in sections:
        section_text = full_text[start:end].strip()
        if not section_text:
            continue

        chunk_size, chunk_overlap = _get_chunk_params(title)

        # Find which pages this section spans
        page_nums = set()
        for pos, pn in page_map.items():
            if start <= pos < end:
                page_nums.add(pn)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # Split the section text
        section_doc = Document(
            page_content=section_text,
            metadata={
                "section": title,
                "pages": sorted(page_nums) if page_nums else [],
                "source": documents[0].metadata.get("source", "") if documents else "",
            },
        )
        sub_chunks = splitter.split_documents([section_doc])

        # Enrich each sub-chunk with section metadata
        for chunk in sub_chunks:
            chunk.metadata["section"] = title
            chunk.metadata["source"] = section_doc.metadata["source"]
            chunk.metadata["chunk_strategy"] = "section-aware"

        all_chunks.extend(sub_chunks)

    return all_chunks


def split_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    smart: bool = True,
) -> List[Document]:
    """Public API: split documents with optional smart section detection.

    Args:
        documents: Raw documents.
        chunk_size: Override default.
        chunk_overlap: Override default.
        smart: Enable section-aware splitting (default True).

    Returns:
        Chunked documents.
    """
    from ..config import Config

    chunk_size = chunk_size or Config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP

    if smart and Config.SMART_CHUNKING:
        return smart_split_documents(documents, chunk_size, chunk_overlap)

    # Standard fallback
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)
