"""Serve the official Unlimited-OCR UI against the local vLLM API."""

from __future__ import annotations

import base64
import io
import json
import mimetypes
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import fitz
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from PIL import Image, UnidentifiedImageError

from markdown_clean import to_clean_markdown

API_BASE = os.environ.get("OCR_API_BASE", "http://unlimited-ocr:8000/v1")
MODEL = os.environ.get("OCR_MODEL", "baidu/Unlimited-OCR")
# Tuned for RTX 5080 16 GB + desktop display (~2 GB): keep headroom under
# vLLM --max-model-len 16384 so prompt + image tokens always fit.
MAX_MODEL_LEN = int(os.environ.get("OCR_MAX_MODEL_LEN", "16384"))
RESERVED_INPUT_TOKENS = int(os.environ.get("OCR_RESERVED_INPUT_TOKENS", "4096"))
DEFAULT_MAX_TOKENS = int(os.environ.get("OCR_MAX_TOKENS", "8192"))
# Hard ceiling for completions (never == max-model-len). Dense pages may use this.
MAX_OUTPUT_TOKENS = max(1, MAX_MODEL_LEN - RESERVED_INPUT_TOKENS)
STATIC_DIR = Path(__file__).resolve().parent / "static"


def clamp_max_tokens(requested: int | None) -> int:
    """Cap completion budget so input + output cannot exceed the model context."""
    value = int(requested) if requested else DEFAULT_MAX_TOKENS
    return max(1, min(value, MAX_OUTPUT_TOKENS))

app = FastAPI(title="Unlimited-OCR UI")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def client() -> OpenAI:
    return OpenAI(api_key="EMPTY", base_url=API_BASE, timeout=3600)


def encode_bytes(data: bytes, mime: str) -> dict:
    b64 = base64.b64encode(data).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{b64}"},
    }


def normalize_prompt(prompt: str) -> str:
    prompt = (prompt or "document parsing.").strip()
    if not prompt.startswith("<image>"):
        prompt = f"<image>{prompt}"
    return prompt


def _color_space_name(cs) -> str | None:
    if cs is None:
        return None
    name = getattr(cs, "name", None)
    if isinstance(name, str) and name:
        return name
    return str(cs)


def _recommend(
    *,
    multi_page: bool,
    pages: int,
    max_edge: int,
    file_size: int,
) -> dict:
    reasons: list[str] = []
    window_size = 1024 if multi_page else 128
    ngram_size = 35
    ngram_enabled = True
    mode = "base" if multi_page else "long"
    prompt = "Multi page parsing." if multi_page else "document parsing."

    if multi_page:
        reasons.append(f"Multi-page document ({pages} pages) → window_size {window_size}")
        reasons.append("Multi-page uses base mode (no crop) + Multi page parsing prompt")
    else:
        reasons.append(f"Single page → window_size {window_size}")
        if max_edge >= 2200:
            mode = "base"
            reasons.append(f"Large image (max edge {max_edge}px) → base mode for accuracy")
        else:
            reasons.append("Long/gundam mode for fast single-page parsing")

    # Completion budget: default 8192. Single dense pages may use up to MAX_OUTPUT_TOKENS.
    # Multi-page batches stay at 8192 — 12288 + window 1024 thrashs 16 GB VRAM over long runs.
    # Never recommend max_tokens == max-model-len (breaks image/prompt requests).
    max_tokens = DEFAULT_MAX_TOKENS
    dense = (not multi_page) and (
        pages >= 8 or max_edge >= 2200 or file_size > 8 * 1024 * 1024
    )
    if dense:
        max_tokens = min(MAX_OUTPUT_TOKENS, max(DEFAULT_MAX_TOKENS, 12288))
        reasons.append(
            f"Dense single page → max_tokens {max_tokens} "
            f"(16 GB safe cap; reserve {RESERVED_INPUT_TOKENS} for input under {MAX_MODEL_LEN})"
        )
    elif multi_page:
        reasons.append(
            f"Multi-page batch → max_tokens {max_tokens} "
            f"(sustained 16 GB; hard cap {MAX_OUTPUT_TOKENS})"
        )
    else:
        reasons.append(
            f"max_tokens {max_tokens} (RTX 5080 16 GB default; hard cap {MAX_OUTPUT_TOKENS})"
        )

    reasons.append(f"N-gram logits processor on (ngram_size={ngram_size})")

    return {
        "window_size": window_size,
        "ngram_size": ngram_size,
        "ngram_enabled": ngram_enabled,
        "max_tokens": max_tokens,
        "mode": mode,
        "prompt": prompt,
        "reasons": reasons,
    }


def _analyze_pdf(raw: bytes, filename: str) -> dict:
    doc = fitz.open(stream=raw, filetype="pdf")
    try:
        pages_meta = []
        max_edge = 0
        color_modes: set[str] = set()
        for i, page in enumerate(doc):
            rect = page.rect
            w_pt, h_pt = float(rect.width), float(rect.height)
            # PDF user space is 72 pt/inch
            w_in, h_in = w_pt / 72.0, h_pt / 72.0
            w_px_200 = int(round(w_pt * 200 / 72))
            h_px_200 = int(round(h_pt * 200 / 72))
            max_edge = max(max_edge, w_px_200, h_px_200)
            cs = None
            try:
                # Sample first image on page if any
                for img in page.get_images(full=True)[:1]:
                    xref = img[0]
                    info = doc.extract_image(xref)
                    cs = info.get("cs-name") or info.get("colorspace")
                    break
            except Exception:  # noqa: BLE001
                pass
            if cs:
                color_modes.add(str(cs))
            pages_meta.append(
                {
                    "page": i + 1,
                    "width_pt": round(w_pt, 2),
                    "height_pt": round(h_pt, 2),
                    "width_in": round(w_in, 3),
                    "height_in": round(h_in, 3),
                    "width_px_at_200dpi": w_px_200,
                    "height_px_at_200dpi": h_px_200,
                    "dpi_native": 72,
                }
            )

        pages = len(pages_meta)
        multi = pages > 1
        first = pages_meta[0] if pages_meta else {}
        return {
            "ok": True,
            "format": "pdf",
            "filename": filename,
            "file_size_bytes": len(raw),
            "pages": pages,
            "multi_page": multi,
            "width": first.get("width_px_at_200dpi"),
            "height": first.get("height_px_at_200dpi"),
            "width_pt": first.get("width_pt"),
            "height_pt": first.get("height_pt"),
            "dpi": 72,
            "dpi_note": "PDF native is 72 pt/inch; UI renders pages at ~200 DPI for OCR",
            "color_mode": ", ".join(sorted(color_modes)) if color_modes else "mixed/unknown",
            "page_details": pages_meta[:20],
            "recommendations": _recommend(
                multi_page=multi,
                pages=pages,
                max_edge=max_edge,
                file_size=len(raw),
            ),
        }
    finally:
        doc.close()


def _analyze_image(raw: bytes, filename: str, mime: str | None) -> dict:
    try:
        img = Image.open(io.BytesIO(raw))
    except UnidentifiedImageError as exc:
        raise ValueError(f"Unrecognized image: {exc}") from exc

    img.load()
    width, height = img.size
    dpi = None
    if img.info.get("dpi"):
        d = img.info["dpi"]
        if isinstance(d, (tuple, list)) and d:
            dpi = float(d[0])
        elif isinstance(d, (int, float)):
            dpi = float(d)
    fmt = (img.format or Path(filename).suffix.lstrip(".").upper() or "IMAGE").lower()
    mode = img.mode
    frames = getattr(img, "n_frames", 1) or 1
    multi = frames > 1
    pages = frames

    return {
        "ok": True,
        "format": fmt,
        "mime": mime,
        "filename": filename,
        "file_size_bytes": len(raw),
        "pages": pages,
        "multi_page": multi,
        "width": width,
        "height": height,
        "dpi": dpi,
        "color_mode": mode,
        "recommendations": _recommend(
            multi_page=multi,
            pages=pages,
            max_edge=max(width, height),
            file_size=len(raw),
        ),
    }


@app.get("/")
async def homepage():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health():
    try:
        models = client().models.list()
        names = [m.id for m in models.data]
        return {"ok": True, "models": names}
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(
            {"ok": False, "error": str(exc)},
            status_code=503,
        )


def _parse_smi_float(value: str) -> float | None:
    value = (value or "").strip()
    if not value or value.upper() in {"N/A", "[N/A]", "NAN"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


@app.get("/api/gpu")
async def gpu_status():
    """Host GPU metrics via nvidia-smi (requires CDI/GPU access on the UI container)."""
    if not shutil.which("nvidia-smi"):
        return {"ok": False, "error": "nvidia-smi not available"}

    query = (
        "name,utilization.gpu,memory.used,memory.total,"
        "temperature.gpu,power.draw"
    )
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc)}

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "nvidia-smi failed").strip()
        return {"ok": False, "error": err}

    gpus: list[dict] = []
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        name, util_s, mem_used_s, mem_total_s = parts[:4]
        temp_s = parts[4] if len(parts) > 4 else ""
        power_s = parts[5] if len(parts) > 5 else ""
        util = _parse_smi_float(util_s)
        mem_used = _parse_smi_float(mem_used_s)
        mem_total = _parse_smi_float(mem_total_s)
        temp = _parse_smi_float(temp_s)
        power = _parse_smi_float(power_s)
        gpus.append(
            {
                "name": name,
                "utilization": int(util) if util is not None else None,
                "memory_used_mib": int(mem_used) if mem_used is not None else None,
                "memory_total_mib": int(mem_total) if mem_total is not None else None,
                "temperature_c": int(temp) if temp is not None else None,
                "power_w": round(power, 1) if power is not None else None,
            }
        )

    if not gpus:
        return {"ok": False, "error": "no GPU data"}

    primary = gpus[0]
    return {"ok": True, "gpus": gpus, **primary}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    """Inspect uploaded PDF/image and recommend OCR settings."""
    raw = await file.read()
    filename = file.filename or "upload"
    mime = file.content_type or mimetypes.guess_type(filename)[0]
    is_pdf = (mime == "application/pdf") or filename.lower().endswith(".pdf")

    try:
        if is_pdf:
            result = _analyze_pdf(raw, filename)
        else:
            result = _analyze_image(raw, filename, mime)
        return result
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


@app.post("/api/to_markdown")
async def api_to_markdown(raw: str = Form(...)):
    """Convert raw OCR text (with grounding tokens) to clean markdown."""
    return {"markdown": to_clean_markdown(raw)}


@app.post("/api/explode_pdf")
async def explode_pdf(pdf_file: UploadFile = File(...)):
    raw = await pdf_file.read()
    tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        tmp_pdf.write(raw)
        tmp_pdf.close()
        doc = fitz.open(tmp_pdf.name)
        mat = fitz.Matrix(200 / 72, 200 / 72)
        pages = []
        try:
            for i, page in enumerate(doc):
                pix = page.get_pixmap(matrix=mat)
                png = pix.tobytes("png")
                pages.append(
                    {
                        "orig_name": f"page_{i + 1:04d}.png",
                        "data_url": "data:image/png;base64,"
                        + base64.b64encode(png).decode("utf-8"),
                    }
                )
        finally:
            doc.close()
        return {"pages": pages}
    finally:
        try:
            os.unlink(tmp_pdf.name)
        except OSError:
            pass


@app.post("/api/ocr")
async def run_ocr(
    prompt: str = Form("document parsing."),
    mode: str = Form("gundam"),
    window_size: int | None = Form(None),
    ngram_size: int = Form(35),
    ngram_enabled: bool = Form(True),
    max_tokens: int | None = Form(None),
    multi: bool = Form(False),
    image: UploadFile | None = File(None),
    image_data_url: str | None = Form(None),
):
    """Stream OCR tokens as SSE: data: {"text": "...", "done": bool}."""
    parts: list[dict] = []

    if image_data_url:
        # data:image/png;base64,...
        if "," in image_data_url:
            header, b64 = image_data_url.split(",", 1)
            mime = "image/png"
            if header.startswith("data:") and ";base64" in header:
                mime = header[5:].split(";")[0] or mime
            parts.append(encode_bytes(base64.b64decode(b64), mime))
        else:
            return JSONResponse({"error": "Invalid image_data_url"}, status_code=400)
    elif image is not None:
        data = await image.read()
        mime = image.content_type or mimetypes.guess_type(image.filename or "")[0] or "image/png"
        parts.append(encode_bytes(data, mime))
    else:
        return JSONResponse({"error": "No image provided"}, status_code=400)

    prompt_text = normalize_prompt(prompt)
    win = int(window_size) if window_size else (1024 if multi else 128)
    tokens = clamp_max_tokens(max_tokens)
    ngram = int(ngram_size) if ngram_enabled else 0

    def event_stream():
        # Mid-stream: raw text only (cleaning every token is O(n²) and bloats SSE).
        # Clean markdown once at the end; the UI also cleans/throttles live preview.
        accumulated = ""
        try:
            extra: dict = {"skip_special_tokens": False}
            if ngram_enabled and ngram > 0:
                extra["vllm_xargs"] = {"ngram_size": ngram, "window_size": win}

            stream = client().chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt_text}, *parts],
                    }
                ],
                max_tokens=tokens,
                temperature=0.0,
                stream=True,
                extra_body=extra,
            )
            for chunk in stream:
                delta = ""
                if chunk.choices:
                    delta = chunk.choices[0].delta.content or ""
                if delta:
                    accumulated += delta
                    payload = json.dumps(
                        {"text": accumulated, "done": False}
                    )
                    yield f"data: {payload}\n\n"
            md = to_clean_markdown(accumulated)
            payload = json.dumps(
                {"text": accumulated, "markdown": md, "done": True}
            )
            yield f"data: {payload}\n\n"
        except Exception as exc:  # noqa: BLE001
            payload = json.dumps(
                {"text": "", "markdown": "", "done": True, "error": str(exc)}
            )
            yield f"data: {payload}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "7860")),
        log_level="info",
    )
