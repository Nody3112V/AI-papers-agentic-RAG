"""Hierarchical, table-aware Markdown chunker with contextual header injection."""

from __future__ import annotations

import re
from dataclasses import dataclass

PAGE_MARKER_RE = re.compile(r"^<!--\s*PAGE:\s*(\d+)\s*-->$")
HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$")
TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
TABLE_CAPTION_RE = re.compile(r"^\s*(?:\*\*|__)?Table\s+\d+[:.\s]", re.IGNORECASE)


@dataclass
class PaperChunk:
    arxiv_id: str
    title: str
    year: int
    topic_cluster: str
    page_number: int
    section_path: str
    chunk_index: int
    has_table: bool
    text: str
    embedding_text: str


def estimate_tokens(text: str) -> int:
    """Fast deterministic token estimator (~1.3 tokens per whitespace-delimited word)."""
    words = text.split()
    if not words:
        return 0
    return max(1, int(len(words) * 1.3))


def _split_large_table(table_lines: list[str], max_tokens: int) -> list[str]:
    """Split oversized Markdown tables by row groups while repeating the table header."""
    caption_lines: list[str] = []
    idx = 0
    while idx < len(table_lines) and not TABLE_LINE_RE.match(table_lines[idx]):
        caption_lines.append(table_lines[idx])
        idx += 1

    body = table_lines[idx:]
    if len(body) <= 2:
        return ["\n".join(table_lines).strip()]

    header = body[:2]  # Header row + separator row (|---|---|)
    data_rows = body[2:]
    header_prefix = "\n".join(caption_lines + header)
    header_tokens = estimate_tokens(header_prefix)

    chunks: list[str] = []
    current_rows: list[str] = []
    current_tokens = header_tokens

    for row in data_rows:
        row_tokens = estimate_tokens(row)
        if current_rows and current_tokens + row_tokens > max_tokens:
            chunks.append("\n".join(caption_lines + header + current_rows).strip())
            current_rows = [row]
            current_tokens = header_tokens + row_tokens
        else:
            current_rows.append(row)
            current_tokens += row_tokens

    if current_rows:
        chunks.append("\n".join(caption_lines + header + current_rows).strip())
    return chunks


def _split_text_recursive(
    text: str,
    *,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
) -> list[str]:
    """Split prose text by paragraphs/words respecting token size and overlap."""
    text = text.strip()
    if not text:
        return []
    if estimate_tokens(text) <= chunk_size_tokens:
        return [text]

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = estimate_tokens(para)
        if para_tokens > chunk_size_tokens:
            # Flush current buffer first
            if current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts = []
                current_tokens = 0
            # Split long paragraph by words with overlap
            words = para.split()
            max_words = max(20, int(chunk_size_tokens / 1.3))
            overlap_words = max(0, int(chunk_overlap_tokens / 1.3))
            step = max(1, max_words - overlap_words)
            for start in range(0, len(words), step):
                segment = " ".join(words[start : start + max_words]).strip()
                if segment:
                    chunks.append(segment)
                if start + max_words >= len(words):
                    break
            continue

        if current_parts and current_tokens + para_tokens > chunk_size_tokens:
            chunks.append("\n\n".join(current_parts))
            # Build overlap tail from previous chunk words
            overlap_words_count = max(0, int(chunk_overlap_tokens / 1.3))
            prev_words = chunks[-1].split()
            tail = " ".join(prev_words[-overlap_words_count:]) if overlap_words_count else ""
            current_parts = [tail, para] if tail else [para]
            current_tokens = estimate_tokens("\n\n".join(current_parts))
        else:
            current_parts.append(para)
            current_tokens += para_tokens

    if current_parts:
        chunks.append("\n\n".join(current_parts))
    return chunks


def _format_embedding_text(
    *,
    title: str,
    year: int,
    section_path: str,
    chunk_text: str,
    inject_header: bool,
) -> str:
    if not inject_header:
        return chunk_text
    return f"Paper: {title} ({year}) | Section: {section_path}\n\n{chunk_text}"


def chunk_markdown(
    markdown: str,
    *,
    arxiv_id: str,
    title: str,
    year: int,
    topic_cluster: str,
    chunk_size_tokens: int = 512,
    chunk_overlap_tokens: int = 64,
    max_table_chunk_tokens: int = 700,
    inject_contextual_header: bool = True,
) -> list[PaperChunk]:
    """Split paper Markdown into hierarchical chunks while keeping Markdown tables intact."""
    lines = markdown.replace("\r\n", "\n").split("\n")

    current_page = 1
    heading_stack: dict[int, str] = {}
    blocks: list[tuple[int, str, bool, str]] = []  # (page, section_path, is_table, content)

    prose_buf: list[str] = []
    prose_page = 1

    def _current_section() -> str:
        if not heading_stack:
            return "Introduction"
        return " > ".join(heading_stack[lvl] for lvl in sorted(heading_stack))

    def _flush_prose() -> None:
        nonlocal prose_buf
        content = "\n".join(prose_buf).strip()
        if content:
            blocks.append((prose_page, _current_section(), False, content))
        prose_buf = []

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()

        # 1. Page marker
        page_match = PAGE_MARKER_RE.match(stripped)
        if page_match:
            _flush_prose()
            current_page = int(page_match.group(1))
            prose_page = current_page
            i += 1
            continue

        # 2. Markdown heading
        heading_match = HEADING_RE.match(stripped)
        if heading_match:
            _flush_prose()
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()
            heading_stack[level] = heading_text
            for deeper in [lvl for lvl in heading_stack if lvl > level]:
                del heading_stack[deeper]
            prose_page = current_page
            i += 1
            continue

        # 3. Markdown table block (optionally preceded by a "Table N:" caption line)
        if TABLE_LINE_RE.match(line):
            caption: list[str] = []
            if prose_buf and TABLE_CAPTION_RE.match(prose_buf[-1].strip()):
                caption.append(prose_buf.pop())
            _flush_prose()

            table_start_page = current_page
            table_lines = caption
            while i < n and TABLE_LINE_RE.match(lines[i]):
                table_lines.append(lines[i])
                i += 1
            blocks.append((table_start_page, _current_section(), True, "\n".join(table_lines)))
            prose_page = current_page
            continue

        if not prose_buf:
            prose_page = current_page
        prose_buf.append(line)
        i += 1

    _flush_prose()

    # Convert blocks into PaperChunk objects
    results: list[PaperChunk] = []
    chunk_index = 0
    for page_no, section_path, is_table, content in blocks:
        if is_table:
            if estimate_tokens(content) <= max_table_chunk_tokens:
                pieces = [content.strip()]
            else:
                pieces = _split_large_table(content.split("\n"), max_table_chunk_tokens)
        else:
            pieces = _split_text_recursive(
                content,
                chunk_size_tokens=chunk_size_tokens,
                chunk_overlap_tokens=chunk_overlap_tokens,
            )

        for piece in pieces:
            if not piece.strip():
                continue
            emb_text = _format_embedding_text(
                title=title,
                year=year,
                section_path=section_path,
                chunk_text=piece,
                inject_header=inject_contextual_header,
            )
            results.append(
                PaperChunk(
                    arxiv_id=arxiv_id,
                    title=title,
                    year=year,
                    topic_cluster=topic_cluster,
                    page_number=page_no,
                    section_path=section_path,
                    chunk_index=chunk_index,
                    has_table=is_table,
                    text=piece,
                    embedding_text=emb_text,
                )
            )
            chunk_index += 1

    return results
