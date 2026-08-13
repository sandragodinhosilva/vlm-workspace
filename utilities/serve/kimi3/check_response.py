#!/usr/bin/env python
"""Send a single "hi" to the running Kimi-K3 server via litellm and print the reply.

The base URL is discovered the same way the shell helpers do it: from run/endpoint.env
(written by serve.sh once the server is healthy). Override with the KIMI_BASE_URL env var.

Run with the project venv:
    .venv/bin/python check_response.py
"""

import os
import sys
from pathlib import Path

from litellm import completion

KIMI_DIR = Path(__file__).resolve().parent
ENDPOINT_FILE = KIMI_DIR / "run" / "endpoint.env"


def resolve_base_url() -> str:
    """Prefer KIMI_BASE_URL, else read BASE_URL from run/endpoint.env."""
    override = os.environ.get("KIMI_BASE_URL")
    if override:
        return override
    if ENDPOINT_FILE.exists():
        for line in ENDPOINT_FILE.read_text().splitlines():
            if line.startswith("BASE_URL="):
                return line.split("=", 1)[1].strip()
    sys.exit(
        f"Could not resolve the endpoint (no KIMI_BASE_URL and no BASE_URL in {ENDPOINT_FILE}).\n"
        "Start the server first with ./serve.sh, or set KIMI_BASE_URL."
    )


def main() -> None:
    base_url = resolve_base_url()
    print(f"-> hosted_vllm/kimi-k3 @ {base_url}", file=sys.stderr)

    resp = completion(
        model="hosted_vllm/kimi-k3",
        messages=[{"role": "user", "content": "hi"}],
        api_base=base_url,
        api_key="dummy",  # vLLM does not check the key; litellm requires a non-empty value
    )
    print(resp.choices[0].message.content)


if __name__ == "__main__":
    main()
