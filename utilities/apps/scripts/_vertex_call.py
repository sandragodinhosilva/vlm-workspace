#!/usr/bin/env python3
"""Vertex/gemini call helper for vibe_test.py.

The vibe-test app runs in vlm-post-training-home-venv (was video-sft-vlm-home-venv,
merged+deleted 2026-06-30), which lacks the Google Cloud
SDK that litellm needs for Vertex. This helper runs in the EVAL venv
(vlm-post-training-home-venv, which has google-genai + a working gemini path) and
is invoked via subprocess. Reuses query_server's gemini routing so the behaviour
matches the trace-gen pipeline exactly.

stdin  : JSON {model, prompt, system, video_path?, image_paths?, max_tokens, temperature}
stdout : JSON {content, thinking, status}  (thinking = gemini's reasoning summary if any)

Run with: /home/sgsilva/vlm-post-training-home-venv/bin/python _vertex_call.py
(GOOGLE_APPLICATION_CREDENTIALS must be set; .bashrc exports it.)
"""
import sys, json
from pathlib import Path

REPO = Path("/home/sgsilva/vlm-post-training")
sys.path.insert(0, str(REPO / "inference"))


def main():
    req = json.load(sys.stdin)
    model = req["model"]
    if not model.startswith("vertex_ai/"):
        model = f"vertex_ai/{model}"
    prompt = req["prompt"]
    system = req.get("system") or None
    video_path = req.get("video_path")
    image_paths = req.get("image_paths") or None
    max_tokens = int(req.get("max_tokens", 4096))
    temperature = float(req.get("temperature", 0.3))

    import query_server as Q

    try:
        if video_path:
            # gemini genai SDK path — single video file, region=global (its default)
            gemini_model = model.split("/", 1)[1]
            out = Q.query_gemini_with_genai_sdk(
                model_name=gemini_model, prompt=prompt, video_path=video_path,
                max_tokens=max_tokens, temperature=temperature, system_prompt=system)
        else:
            # text / image path via litellm (base_url already pinned to /global/)
            out = Q.query_with_litellm(
                model_name=model, prompt=prompt, image_paths=image_paths,
                video_path=None, max_tokens=max_tokens, temperature=temperature,
                system_prompt=system)
        # query_* helpers may return str or (content, usage)
        content = out[0] if isinstance(out, tuple) else out
        content = content or ""
        print(json.dumps({"content": content, "thinking": "",
                          "status": f"OK — vertex {model} ({len(content)} chars)"}))
    except Exception as e:
        print(json.dumps({"content": "", "thinking": "",
                          "status": f"ERROR (vertex {model}): {type(e).__name__}: {e}"}))


if __name__ == "__main__":
    main()
