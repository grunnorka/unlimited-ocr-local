"""Serve the official Unlimited-OCR UI against the local vLLM API."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import tempfile
from pathlib import Path

import fitz
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI

API_BASE = os.environ.get("OCR_API_BASE", "http://unlimited-ocr:8000/v1")
MODEL = os.environ.get("OCR_MODEL", "baidu/Unlimited-OCR")
MAX_TOKENS = int(os.environ.get("OCR_MAX_TOKENS", "8192"))
STATIC_DIR = Path(__file__).resolve().parent / "static"

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


def window_for(mode: str, multi: bool) -> int:
    if multi:
        return 1024
    # long/gundam and base both use 128 for single-page in the official demo
    return 128


def normalize_prompt(prompt: str) -> str:
    prompt = (prompt or "document parsing.").strip()
    if not prompt.startswith("<image>"):
        prompt = f"<image>{prompt}"
    return prompt


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
    image: UploadFile | None = File(None),
    image_data_url: str | None = Form(None),
):
    """Stream OCR tokens as SSE: data: {"text": "...", "done": bool}."""
    parts: list[dict] = []
    multi = False

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
    window_size = window_for(mode, multi)

    def event_stream():
        accumulated = ""
        try:
            stream = client().chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt_text}, *parts],
                    }
                ],
                max_tokens=MAX_TOKENS,
                temperature=0.0,
                stream=True,
                extra_body={
                    "skip_special_tokens": False,
                    "vllm_xargs": {"ngram_size": 35, "window_size": window_size},
                },
            )
            for chunk in stream:
                delta = ""
                if chunk.choices:
                    delta = chunk.choices[0].delta.content or ""
                if delta:
                    accumulated += delta
                    payload = json.dumps({"text": accumulated, "done": False})
                    yield f"data: {payload}\n\n"
            payload = json.dumps({"text": accumulated, "done": True})
            yield f"data: {payload}\n\n"
        except Exception as exc:  # noqa: BLE001
            payload = json.dumps({"text": "", "done": True, "error": str(exc)})
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
