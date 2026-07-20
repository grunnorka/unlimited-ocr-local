"""Minimal OpenAI-compatible client for Unlimited-OCR (vLLM)."""

from __future__ import annotations

import argparse
import base64
import mimetypes
import sys
import time
from pathlib import Path

from openai import OpenAI

# Avoid Windows cp1252 crashes on CJK / special tokens in OCR output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def encode_image(path: Path) -> dict:
    mime, _ = mimetypes.guess_type(path.name)
    mime = mime or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{data}"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Call Unlimited-OCR via vLLM")
    parser.add_argument("image", type=Path, help="Path to an image file")
    parser.add_argument(
        "--prompt",
        default="<image>document parsing.",
        help="Prompt must start with literal <image>",
    )
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--window-size", type=int, default=128)
    args = parser.parse_args()

    if not args.image.is_file():
        raise SystemExit(f"Image not found: {args.image}")

    client = OpenAI(api_key="EMPTY", base_url=args.base_url, timeout=3600)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": args.prompt},
                encode_image(args.image),
            ],
        }
    ]

    start = time.time()
    response = client.chat.completions.create(
        model="baidu/Unlimited-OCR",
        messages=messages,
        max_tokens=args.max_tokens,
        temperature=0.0,
        extra_body={
            "skip_special_tokens": False,
            "vllm_xargs": {"ngram_size": 35, "window_size": args.window_size},
        },
    )
    print(f"Response costs: {time.time() - start:.2f}s")
    print(response.choices[0].message.content or "")


if __name__ == "__main__":
    main()
