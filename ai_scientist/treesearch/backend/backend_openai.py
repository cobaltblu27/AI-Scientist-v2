import json
import logging
import os
import re
import time

from .utils import FunctionSpec, OutputType, opt_messages_to_list, backoff_create
from funcy import notnone, once, select_values
import openai
from rich import print

logger = logging.getLogger("ai-scientist")

COPILOT_MODEL_PREFIX = "copilot-"
COPILOT_BASE_URL = "https://api.githubcopilot.com/"
COPILOT_INTEGRATION_ID = "vscode-chat"
GLM_MODEL_PREFIX = "glm-"
QWEN_MODEL_PREFIX = "qwen"
GLM_BASE_URL = "https://coding-intl.dashscope.aliyuncs.com/v1/"

OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


def _get_glm_api_key() -> str:
    api_key = (
        os.environ.get("GLM_API_KEY")
        or os.environ.get("ALIBABA_API_KEY")
        or os.environ.get("DASHSCOPE_API_KEY")
    )
    if not api_key:
        raise ValueError(
            "GLM_API_KEY, ALIBABA_API_KEY, or DASHSCOPE_API_KEY environment variable not set"
        )
    return api_key


def get_ai_client(model: str, max_retries=2) -> openai.OpenAI:
    if model.startswith("ollama/"):
        client = openai.OpenAI(
            base_url="http://localhost:11434/v1", 
            max_retries=max_retries
        )
    elif model.startswith(COPILOT_MODEL_PREFIX):
        if "GITHUB_COPILOT_API_KEY" not in os.environ:
            raise ValueError("GITHUB_COPILOT_API_KEY environment variable not set")

        client = openai.OpenAI(
            api_key=os.environ["GITHUB_COPILOT_API_KEY"],
            base_url=COPILOT_BASE_URL,
            default_headers={
                "Accept": "application/json",
                "Copilot-Integration-Id": COPILOT_INTEGRATION_ID,
            },
            max_retries=max_retries,
        )
    elif model.startswith(GLM_MODEL_PREFIX) or model.startswith(QWEN_MODEL_PREFIX):
        client = openai.OpenAI(
            api_key=_get_glm_api_key(),
            base_url=GLM_BASE_URL,
            max_retries=max_retries,
        )
    else:
        client = openai.OpenAI(max_retries=max_retries)
    return client


def _extract_json_from_text(raw_text: str) -> dict:
    text = raw_text.strip()
    if not text:
        raise AssertionError("No JSON object found in message content")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match is None:
            match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match is None:
            raise AssertionError("No JSON object found in message content")
        return json.loads(match.group(1))


def _parse_function_output(choice, func_spec: FunctionSpec) -> dict:
    # Try new-style tool_calls first (OpenAI API v1+)
    if choice.message.tool_calls:
        assert (
            choice.message.tool_calls[0].function.name == func_spec.name
        ), "Function name mismatch"
        try:
            print(f"[cyan]Raw func call response: {choice}[/cyan]")
            return json.loads(choice.message.tool_calls[0].function.arguments)
        except json.JSONDecodeError as e:
            logger.error(
                f"Error decoding the function arguments: {choice.message.tool_calls[0].function.arguments}"
            )
            raise e

    # Fallback to old-style function_call (deprecated but still used by some models)
    if choice.message.function_call:
        assert (
            choice.message.function_call.name == func_spec.name
        ), "Function name mismatch (function_call)"
        try:
            print(f"[cyan]Raw func call response (function_call): {choice}[/cyan]")
            return json.loads(choice.message.function_call.arguments)
        except json.JSONDecodeError as e:
            logger.error(
                f"Error decoding the function_call arguments: {choice.message.function_call.arguments}"
            )
            raise e

    # Some DashScope responses omit tool calls entirely and only leave text or reasoning.
    raw_content = choice.message.content or ""
    if raw_content.strip():
        return _extract_json_from_text(raw_content)

    reasoning_content = getattr(choice.message, "reasoning_content", "") or ""
    if reasoning_content.strip():
        return _extract_json_from_text(reasoning_content)

    raise AssertionError("function_call is empty and no JSON object found in message content")


def _build_attempt_kwargs(
    base_kwargs: dict, *, func_spec: FunctionSpec | None, is_glm_model: bool, attempt: int
) -> dict:
    attempt_kwargs = dict(base_kwargs)
    if func_spec is not None and is_glm_model and attempt > 1:
        # First attempt keeps the provider default thinking mode for quality.
        # Only fall back to non-thinking mode if structured output parsing failed.
        extra_body = dict(attempt_kwargs.get("extra_body") or {})
        extra_body["enable_thinking"] = False
        attempt_kwargs["extra_body"] = extra_body
    return attempt_kwargs


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    client = get_ai_client(model_kwargs.get("model"), max_retries=0)
    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore

    messages = opt_messages_to_list(system_message, user_message)
    model_name = str(filtered_kwargs.get("model", ""))
    is_glm_model = model_name.startswith(GLM_MODEL_PREFIX)
    is_dashscope_model = model_name.startswith(
        (GLM_MODEL_PREFIX, QWEN_MODEL_PREFIX)
    )

    if func_spec is not None:
        filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
        # DashScope thinking mode rejects forced function/tool selection.
        # Keep tools, but do not force tool_choice for GLM/Qwen models.
        if not is_dashscope_model:
            filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict

    if filtered_kwargs.get("model", "").startswith("ollama/"):
       filtered_kwargs["model"] = filtered_kwargs["model"].replace("ollama/", "")
    elif filtered_kwargs.get("model", "").startswith(COPILOT_MODEL_PREFIX):
        filtered_kwargs["model"] = filtered_kwargs["model"].removeprefix(
            COPILOT_MODEL_PREFIX
        )

    attempts = 3 if func_spec is not None and is_dashscope_model else 1
    req_time = 0.0
    in_tokens = 0
    out_tokens = 0

    for attempt in range(1, attempts + 1):
        attempt_kwargs = _build_attempt_kwargs(
            filtered_kwargs,
            func_spec=func_spec,
            is_glm_model=is_glm_model,
            attempt=attempt,
        )
        t0 = time.time()
        completion = backoff_create(
            client.chat.completions.create,
            OPENAI_TIMEOUT_EXCEPTIONS,
            messages=messages,
            **attempt_kwargs,
        )
        req_time += time.time() - t0

        choice = completion.choices[0]
        in_tokens += completion.usage.prompt_tokens
        out_tokens += completion.usage.completion_tokens

        try:
            if func_spec is None:
                output = choice.message.content
            else:
                output = _parse_function_output(choice, func_spec)
            break
        except (AssertionError, json.JSONDecodeError) as e:
            if attempt == attempts:
                raise e

            logger.warning(
                "Structured output parse failed for model=%s function=%s attempt=%d/%d "
                "(finish_reason=%s, content_len=%d, reasoning_len=%d, non_thinking=%s). Retrying.",
                model_name,
                func_spec.name,
                attempt,
                attempts,
                choice.finish_reason,
                len(choice.message.content or ""),
                len(getattr(choice.message, "reasoning_content", "") or ""),
                bool(attempt_kwargs.get("extra_body", {}).get("enable_thinking") is False),
            )
            print(
                f"[yellow]Structured output parse failed for {model_name} on attempt "
                f"{attempt}/{attempts}. Retrying...[/yellow]"
            )

    info = {
        "system_fingerprint": completion.system_fingerprint,
        "model": completion.model,
        "created": completion.created,
    }

    return output, req_time, in_tokens, out_tokens, info
