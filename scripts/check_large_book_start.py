#!/usr/bin/env python3
"""Red/green check: large-book Start blockers in UI + analyze."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "ui/static/index.html").read_text()
APP = (ROOT / "ui/app.py").read_text()


def main() -> int:
    bugs: list[tuple[str, bool]] = []

    # Analyze must not block Start: background helper, not awaited in addFiles loop
    add_start = HTML.find("async function addFiles")
    add_end = HTML.find("async function selectFile", add_start)
    add = HTML[add_start:add_end]
    bugs.append(
        (
            "addFiles_awaits_analyze_before_select",
            "await analyzeFile(" in add or "await runAnalyzeInBackground(" in add,
        )
    )
    bugs.append(
        (
            "addFiles_missing_background_analyze",
            "void runAnalyzeInBackground(" not in add,
        )
    )

    sel_start = HTML.find("async function selectFile")
    sel_end = HTML.find("function appendOcrSettings", sel_start)
    sel = HTML[sel_start:sel_end]
    bugs.append(
        (
            "selectFile_loads_full_pdf_arrayBuffer_for_preview",
            "arrayBuffer()" in sel,
        )
    )
    m = re.search(r"Math\.min\(total,\s*(\d+)\)", sel)
    preview_cap = int(m.group(1)) if m else (1 if "getPage(1)" in sel else 999)
    bugs.append(("selectFile_preview_renders_many_pages", preview_cap > 1))

    start_s = HTML.find("async function startOCR")
    start_e = HTML.find("async function streamOcr", start_s)
    start = HTML[start_s:start_e]
    bugs.append(
        (
            "startOCR_calls_heavy_selectFile",
            "await selectFile(item.id" in start and "skipPreview: true" not in start,
        )
    )

    analyze_fn = APP.split("def _analyze_pdf")[1].split("def _analyze_image")[0]
    bugs.append(
        (
            "analyze_extracts_images_on_every_page",
            "for i, page in enumerate(doc):" in analyze_fn
            or "for page in doc:" in analyze_fn,
        )
    )

    bugs.append(
        (
            "boot_no_health_retry",
            "scheduleHealthPoll" not in HTML and "pollHealth" not in HTML,
        )
    )

    sync_s = HTML.find("function syncExportButtons")
    sync_e = HTML.find("function renderFileList", sync_s)
    sync = HTML[sync_s:sync_e]
    bugs.append(
        (
            "start_requires_analysis_complete",
            ("status === 'ready'" in sync or 'status == "ready"' in sync)
            and "analysis" in sync,
        )
    )

    bugs.append(
        (
            "getPdfDoc_still_uses_arrayBuffer",
            "arrayBuffer()" in HTML[HTML.find("async function getPdfDocForItem"):HTML.find("async function pdfPageToDataUrl")],
        )
    )

    print("LARGE_BOOK_START_LOOP")
    red = 0
    for name, is_bug in bugs:
        status = "RED" if is_bug else "GREEN"
        if is_bug:
            red += 1
        print(f"  {status} {name}")
    print(f"RED_COUNT={red}")
    return 1 if red else 0


if __name__ == "__main__":
    sys.exit(main())
