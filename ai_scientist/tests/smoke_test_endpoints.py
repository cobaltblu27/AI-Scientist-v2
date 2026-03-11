#!/usr/bin/env python3
"""Low-cost smoke tests for wired LLM/VLM endpoints."""

from __future__ import annotations

import argparse
import base64
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DASHSCOPE_BASE_URL = "https://coding-intl.dashscope.aliyuncs.com/v1/"
DASHSCOPE_API_VARS = ("GLM_API_KEY", "ALIBABA_API_KEY", "DASHSCOPE_API_KEY")
DEFAULT_VLM_IMAGE = Path(__file__).with_name("vlm_test_image.png")


@dataclass
class TestResult:
    name: str
    ok: bool
    seconds: float
    detail: str
    tb: str | None = None


def _find_dashscope_key_var() -> str | None:
    import os

    for var in DASHSCOPE_API_VARS:
        if os.environ.get(var):
            return var
    return None


def _shorten(text: str, limit: int = 120) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _run_case(name: str, fn) -> TestResult:
    started = time.perf_counter()
    try:
        detail = fn()
        return TestResult(
            name=name,
            ok=True,
            seconds=time.perf_counter() - started,
            detail=detail,
        )
    except Exception as exc:  # noqa: BLE001
        return TestResult(
            name=name,
            ok=False,
            seconds=time.perf_counter() - started,
            detail=f"{exc.__class__.__name__}: {exc}",
            tb=traceback.format_exc(),
        )


def _encode_image_as_data_uri(image_path: Path) -> str:
    if not image_path.exists():
        raise FileNotFoundError(f"VLM test image not found: {image_path}")
    suffix = image_path.suffix.lower()
    if suffix == ".png":
        mime = "image/png"
    elif suffix in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    else:
        raise ValueError(
            f"Unsupported test image format: {suffix}. Use .png, .jpg, or .jpeg."
        )
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def _smoke_llm(model: str, timeout_s: float, max_tokens: int) -> str:
    from ai_scientist.llm import create_client

    client, client_model = create_client(model)
    if not str(client.base_url).startswith(DASHSCOPE_BASE_URL):
        raise ValueError(
            f"Unexpected base_url for {model}: {client.base_url} (expected DashScope)"
        )

    response = client.chat.completions.create(
        model=client_model,
        messages=[
            {"role": "system", "content": "Reply very briefly."},
            {"role": "user", "content": "Reply with OK."},
        ],
        temperature=0,
        max_tokens=max_tokens,
        n=1,
        timeout=timeout_s,
    )
    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise ValueError("Empty response content")
    return f"response={_shorten(text)}"


def _smoke_tree_backend(model: str, timeout_s: float, max_tokens: int) -> str:
    from ai_scientist.treesearch.backend.backend_openai import get_ai_client

    client = get_ai_client(model, max_retries=0)
    if not str(client.base_url).startswith(DASHSCOPE_BASE_URL):
        raise ValueError(
            f"Unexpected base_url for {model}: {client.base_url} (expected DashScope)"
        )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Reply very briefly."},
            {"role": "user", "content": "Reply with OK."},
        ],
        temperature=0,
        max_tokens=max_tokens,
        n=1,
        timeout=timeout_s,
    )
    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise ValueError("Empty response content")
    return f"response={_shorten(text)}"


def _smoke_vlm(model: str, timeout_s: float, max_tokens: int, image_path: Path) -> str:
    from ai_scientist.vlm import create_client

    client, client_model = create_client(model)
    if not str(client.base_url).startswith(DASHSCOPE_BASE_URL):
        raise ValueError(
            f"Unexpected base_url for {model}: {client.base_url} (expected DashScope)"
        )
    data_uri = _encode_image_as_data_uri(image_path)

    response = client.chat.completions.create(
        model=client_model,
        messages=[
            {"role": "system", "content": "Reply very briefly."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What do you see? Reply with one short phrase."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_uri,
                        },
                    },
                ],
            },
        ],
        temperature=0,
        max_tokens=max_tokens,
        n=1,
        timeout=timeout_s,
    )
    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise ValueError("Empty response content")
    return f"response={_shorten(text)}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test low-token calls for wired DashScope endpoints."
    )
    parser.add_argument(
        "--llm-models",
        nargs="+",
        default=["glm-5", "qwen3.5-plus"],
        help="Text models tested through ai_scientist.llm.create_client().",
    )
    parser.add_argument(
        "--tree-models",
        nargs="+",
        default=["glm-5", "qwen3.5-plus"],
        help="Models tested through treesearch backend_openai.get_ai_client().",
    )
    parser.add_argument(
        "--vlm-model",
        default="qwen3.5-plus",
        help="VLM model tested through ai_scientist.vlm.create_client().",
    )
    parser.add_argument(
        "--vlm-image",
        default=str(DEFAULT_VLM_IMAGE),
        help="Path to local test image for VLM probe (.png/.jpg/.jpeg).",
    )
    parser.add_argument(
        "--skip-vlm",
        action="store_true",
        help="Skip VLM smoke test.",
    )
    parser.add_argument(
        "--skip-tree",
        action="store_true",
        help="Skip tree backend smoke tests.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=12,
        help="Max completion tokens per request (keep low to reduce cost).",
    )
    parser.add_argument(
        "--show-traceback",
        action="store_true",
        help="Print traceback details for failures.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    vlm_image = Path(args.vlm_image).expanduser().resolve()

    key_var = _find_dashscope_key_var()
    if not key_var:
        print(
            "ERROR: Missing DashScope key. Set one of: "
            + ", ".join(DASHSCOPE_API_VARS)
        )
        return 2

    print(f"Using API key from environment variable: {key_var}")
    print(
        f"Request budget: max_tokens={args.max_tokens}, timeout={args.timeout}s per call"
    )
    if not args.skip_vlm:
        print(f"VLM test image: {vlm_image}")

    results: list[TestResult] = []

    for model in args.llm_models:
        results.append(
            _run_case(
                f"llm.create_client:{model}",
                lambda m=model: _smoke_llm(m, args.timeout, args.max_tokens),
            )
        )

    if not args.skip_tree:
        for model in args.tree_models:
            results.append(
                _run_case(
                    f"treesearch.backend_openai:{model}",
                    lambda m=model: _smoke_tree_backend(
                        m, args.timeout, args.max_tokens
                    ),
                )
            )

    if not args.skip_vlm:
        results.append(
            _run_case(
                f"vlm.create_client:{args.vlm_model}",
                lambda: _smoke_vlm(
                    args.vlm_model, args.timeout, args.max_tokens, vlm_image
                ),
            )
        )

    failures = []
    print("\nSmoke test results:")
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        print(f"- [{status}] {result.name} ({result.seconds:.2f}s) -> {result.detail}")
        if not result.ok:
            failures.append(result)

    if failures and args.show_traceback:
        print("\nTracebacks for failed tests:")
        for failure in failures:
            print(f"--- {failure.name} ---")
            print(failure.tb or "No traceback available.")

    print(
        f"\nSummary: {len(results) - len(failures)}/{len(results)} passed, {len(failures)} failed."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
