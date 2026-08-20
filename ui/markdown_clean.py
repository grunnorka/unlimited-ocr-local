"""Strip Unlimited-OCR grounding tokens into clean markdown."""

from __future__ import annotations

import re

DET_RE = re.compile(
    r"<\|det\|>([^<\s]+)(?:\s*\[[^\]]*\])?\s*<\|/det\|>(.*)",
    re.DOTALL,
)
REF_OPEN_RE = re.compile(r"<\|ref\|>(.*?)(?:<\|/ref\|>|$)", re.DOTALL)
DET_ANY_RE = re.compile(r"<\|det\|>.*?(?:<\|/det\|>|$)", re.DOTALL)
ORPHAN_TOKEN_RE = re.compile(r"<\|/?[a-z_]+\|>", re.IGNORECASE)

# Categories that become markdown headings
HEADING_LEVELS = {
    "title": 1,
    "doc_title": 1,
    "header": 2,
    "sub_title": 2,
    "subtitle": 2,
    "section": 2,
    "section_header": 2,
}


def _format_block(category: str, lines: list[str]) -> str:
    content = "\n".join(lines).strip()
    if not content:
        return ""

    cat = (category or "text").lower().strip()
    if cat in ("image", "figure", "picture"):
        return ""
    if cat in ("footer", "page_footer", "page_number"):
        return f"*{content}*"
    if cat in ("list", "list_item", "bullet"):
        if content.startswith(("-", "*", "•", "1.", "2.", "3.")):
            return content
        return f"- {content}"
    if cat in ("code", "code_block"):
        return f"```\n{content}\n```"
    if cat in ("equation", "formula", "display_formula"):
        return content
    if cat in ("table",):
        return content

    level = HEADING_LEVELS.get(cat)
    if level is not None:
        # Prefer first line as heading if multi-line
        first, *rest = content.split("\n", 1)
        heading = f"{'#' * level} {first.strip()}"
        if rest and rest[0].strip():
            return f"{heading}\n\n{rest[0].strip()}"
        return heading

    return content


def to_clean_markdown(raw: str) -> str:
    """Convert raw OCR output with <|det|> / <|ref|> tokens to clean markdown."""
    if not raw:
        return ""

    # Unwrap <|ref|>…<|/ref|> (keep inner text)
    text = REF_OPEN_RE.sub(lambda m: m.group(1), raw)

    blocks: list[tuple[str, list[str]]] = []
    cur_cat: str | None = None
    cur_lines: list[str] | None = None

    for line in text.splitlines():
        line = line.rstrip()
        if not line:
            continue
        m = DET_RE.match(line)
        if m:
            category, content = m.group(1).strip(), m.group(2).strip()
            if category.lower() in ("image", "figure", "picture"):
                if cur_lines is not None:
                    blocks.append((cur_cat or "text", cur_lines))
                    cur_lines = None
                    cur_cat = None
                continue
            if cur_lines is not None:
                blocks.append((cur_cat or "text", cur_lines))
            cur_cat = category
            cur_lines = [content] if content else []
            continue
        if cur_lines is None:
            cur_cat = "text"
            cur_lines = []
        cur_lines.append(line)

    if cur_lines is not None:
        blocks.append((cur_cat or "text", cur_lines))

    if not blocks:
        # Fallback: strip any leftover tokens from plain / streaming text
        cleaned = DET_ANY_RE.sub("", text)
        cleaned = ORPHAN_TOKEN_RE.sub("", cleaned)
        return cleaned.strip()

    parts = [_format_block(cat, lines) for cat, lines in blocks]
    return "\n\n".join(p for p in parts if p).strip()
