# Unlimited-OCR (local Docker)

Local Docker stack for [baidu/Unlimited-OCR](https://github.com/baidu/Unlimited-OCR): vLLM OpenAI-compatible API + web UI, tuned for Windows + NVIDIA GPU.

Upstream model/API: [baidu/Unlimited-OCR](https://github.com/baidu/Unlimited-OCR) · Official interactive demo style: [Hugging Face Spaces](https://huggingface.co/spaces/baidu/Unlimited-OCR)

## What’s included

| Service | Port | Role |
|---------|------|------|
| `unlimited-ocr` | `8000` | vLLM server (`vllm/vllm-openai:unlimited-ocr`) |
| `ui` | `7860` | Web UI (Spaces-style frontend → local API) |

## Requirements

- Docker Desktop with NVIDIA GPU support
- ~8 GB+ VRAM (tested on RTX 5080 16 GB)
- Disk space for the model + image (several GB on first run)

## Start

```powershell
docker compose up -d --build
docker compose logs -f
```

First boot downloads `baidu/Unlimited-OCR` from Hugging Face into a Docker volume. Wait until the OCR service is healthy, then open the UI.

Optional: copy `.env.example` to `.env` and set `HF_TOKEN` if Hugging Face rate-limits you.

## UI

Open **http://localhost:7860**

Current UI matches the official Spaces-style frontend (document upload, Long/Base, NGRAM, streaming raw OCR text).

### UI roadmap

Baidu’s README visualization ([`long-horizon-ocr.gif`](https://github.com/baidu/Unlimited-OCR/blob/main/assets/long-horizon-ocr.gif)) shows a richer three-panel grounding demo that is **not** shipped as a runnable app. We want to bring that experience to this local stack — tracked in GitHub Issues:

- Three-panel layout: input · raw output · bounding-box overlay
- Clean markdown copy (strip `<|det|>` / `<|ref|>` grounding tokens)
- Export helpers (`.md` / download)

See **Issues** on this repo for details.

## Call the API

```powershell
pip install openai
python ocr_client.py .\inputs\your-image.png
```

Or any OpenAI client at `http://localhost:8000/v1`.

Important request options (empty output if missing):

- Prompt must start with literal `<image>` (e.g. `<image>document parsing.`)
- `skip_special_tokens: false`
- `vllm_xargs`: `{ "ngram_size": 35, "window_size": 128 }` (use `1024` for multi-page)

## Stop

```powershell
docker compose down
```

## Notes

- Image: `vllm/vllm-openai:unlimited-ocr` (CUDA 13.0; fits RTX 50-series)
- Hopper-only alternative from upstream: `vllm/vllm-openai:unlimited-ocr-cu129`
- Compose uses `--gpu-memory-utilization 0.85` and `--max-model-len 16384` so Windows desktop VRAM use does not block startup on 16 GB GPUs
- Put files in `inputs/`; optional results in `outputs/`

## License

This repo’s Docker/UI glue code is MIT. The Unlimited-OCR model and upstream project remain under their own licenses (upstream is MIT).
