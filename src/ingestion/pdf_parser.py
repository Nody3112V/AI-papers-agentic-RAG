"""Layout-aware PDF parsing (Docling GPU / PyMuPDF4LLM fallback) with academic cleanup."""

from __future__ import annotations

import re
from pathlib import Path

from src.core.logging import get_logger

logger = get_logger(__name__)

PAGE_MARKER_FMT = "<!-- PAGE: {page} -->"

# Match level-1 or level-2 Markdown headings for excluded academic tail sections.
_EXCLUDE_HEADING_TEMPLATE = r"(?im)^#{1,2}\s+(?:\d+\.?\s*)?(?:%s)\b.*$"


def clean_academic_markdown(
    markdown: str,
    exclude_sections: list[str] | None = None,
) -> str:
    """Remove trailing references/acknowledgments, fix hyphenation, and strip line numbers."""
    text = markdown.replace("\r\n", "\n")

    # 1. Cut off excluded tail sections (References, Bibliography, Acknowledgments, etc.)
    if exclude_sections:
        escaped = "|".join(re.escape(s.strip()) for s in exclude_sections if s.strip())
        if escaped:
            pattern = re.compile(_EXCLUDE_HEADING_TEMPLATE % escaped)
            match = pattern.search(text)
            if match:
                text = text[: match.start()].rstrip()

    # 2. Rejoin hyphenated line breaks: "re-\ntrieval" -> "retrieval"
    text = re.sub(r"([A-Za-z]{2,})-\n([a-z]{2,})", r"\1\2", text)

    # 3. Strip review-draft left-margin line numbers (e.g., "001 ", "142 ") at line start
    text = re.sub(r"(?m)^\s*\d{3,4}\s+(?=[A-Za-z])", "", text)

    # 4. Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def _parse_with_docling(
    pdf_path: Path,
    *,
    do_ocr: bool = False,
    do_table_structure: bool = True,
) -> tuple[str, int]:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pipeline_options = PdfPipelineOptions(
        do_ocr=do_ocr,
        do_table_structure=do_table_structure,
    )
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    result = converter.convert(str(pdf_path))
    doc = result.document
    page_count = len(doc.pages) if getattr(doc, "pages", None) else 1

    # Export page-by-page when supported so every chunk retains exact page_number.
    page_parts: list[str] = []
    if page_count > 1:
        for page_no in range(1, page_count + 1):
            try:
                page_md = doc.export_to_markdown(page_no=page_no)
            except TypeError:
                page_parts = [PAGE_MARKER_FMT.format(page=1), doc.export_to_markdown()]
                break
            if page_md and page_md.strip():
                page_parts.append(PAGE_MARKER_FMT.format(page=page_no))
                page_parts.append(page_md.strip())
    else:
        page_parts = [PAGE_MARKER_FMT.format(page=1), doc.export_to_markdown()]

    return "\n\n".join(page_parts), page_count


def _parse_with_pymupdf4llm(pdf_path: Path) -> tuple[str, int]:
    import pymupdf4llm

    pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
    parts: list[str] = []
    for idx, page_item in enumerate(pages, start=1):
        text = page_item.get("text", "") if isinstance(page_item, dict) else str(page_item)
        parts.append(PAGE_MARKER_FMT.format(page=idx))
        parts.append(text.strip())
    return "\n\n".join(parts), max(len(pages), 1)


def parse_pdf_to_markdown(
    pdf_path: str | Path,
    markdown_out_path: str | Path,
    *,
    parser: str = "docling",
    do_ocr: bool = False,
    do_table_structure: bool = True,
    exclude_sections: list[str] | None = None,
    overwrite: bool = False,
) -> tuple[str, int]:
    """Parse a PDF into cleaned Markdown with page markers; resumes from cache if present."""
    out_path = Path(markdown_out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        cached = out_path.read_text(encoding="utf-8")
        page_markers = re.findall(r"<!-- PAGE: (\d+) -->", cached)
        page_count = max((int(p) for p in page_markers), default=1)
        return cached, page_count

    pdf = Path(pdf_path)
    raw_md: str
    page_count: int

    if parser == "docling":
        try:
            raw_md, page_count = _parse_with_docling(
                pdf,
                do_ocr=do_ocr,
                do_table_structure=do_table_structure,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Docling failed on {pdf.name} ({exc}); falling back to PyMuPDF4LLM.")
            raw_md, page_count = _parse_with_pymupdf4llm(pdf)
    else:
        raw_md, page_count = _parse_with_pymupdf4llm(pdf)

    cleaned = clean_academic_markdown(raw_md, exclude_sections=exclude_sections)
    out_path.write_text(cleaned, encoding="utf-8")
    return cleaned, page_count
