#!/usr/bin/env python3
"""
Unified script to query vision models using LiteLLM
Supports vLLM servers, Gemini, and other providers
Supports both image sequence and video modes
"""

import argparse
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import cv2
import base64
import re
import litellm
from litellm import completion
from openai import OpenAI
import logging
import os
import time
import random

# Import Google Gen AI SDK for direct Vertex AI access
from google import genai
from google.genai import types
from google.genai.types import HttpOptions

def retry_with_exponential_backoff(
    func,
    max_retries=5,
    initial_wait=1.0,
    max_wait=60.0,
    exponential_base=2.0,
    jitter=True
):
    """
    Retry a function with exponential backoff for rate limit errors.

    Args:
        func: Function to retry (should be a callable that takes no arguments)
        max_retries: Maximum number of retry attempts
        initial_wait: Initial wait time in seconds
        max_wait: Maximum wait time in seconds
        exponential_base: Base for exponential backoff
        jitter: Whether to add random jitter to wait time

    Returns:
        Result of the function call

    Raises:
        The last exception if all retries fail
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e

            # Check if this is a rate limit error
            error_str = str(e).lower()
            is_rate_limit = (
                'ratelimiterror' in error_str or
                'rate limit' in error_str or
                '429' in error_str or
                'resource exhausted' in error_str or
                'resource_exhausted' in error_str
            )

            # If not a rate limit error, raise immediately
            if not is_rate_limit:
                raise

            # If this was the last attempt, raise
            if attempt >= max_retries:
                print(f"\n  ⚠ Max retries ({max_retries}) reached. Giving up.")
                raise

            # Calculate wait time with exponential backoff
            wait_time = min(initial_wait * (exponential_base ** attempt), max_wait)

            # Add jitter to prevent thundering herd
            if jitter:
                wait_time = wait_time * (0.5 + random.random())

            print(f"\n  ⚠ Rate limit error (attempt {attempt + 1}/{max_retries + 1})")
            print(f"  ⏳ Waiting {wait_time:.1f}s before retry...")
            time.sleep(wait_time)

    # Should never reach here, but just in case
    raise last_exception


# Model configurations
# For vLLM servers, use the actual model path (what the server was started with)
# For Gemini, use format: "vertex_ai/<model-name>"
MODELS = {
    # vLLM models (requires server running on localhost or custom base_url)
    "qwen3-vl-30b-a3b-instruct": {
        "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "provider": "vllm"
    },
    "qwen3-vl-30b-a3b-thinking": {
        "model": "Qwen/Qwen3-VL-30B-A3B-Thinking",
        "provider": "vllm"
    },
    "qwen3-vl-32b-instruct": {
        "model": "Qwen/Qwen3-VL-32B-Instruct",
        "provider": "vllm"
    },
    "qwen3-vl-32b-thinking": {
        "model": "Qwen/Qwen3-VL-32B-Thinking",
        "provider": "vllm"
    },
    "qwen3-vl-235b-a22b-thinking": {
        "model": "Qwen/Qwen3-VL-235B-A22B-Thinking",
        "provider": "vllm"
    },
    "qwen3-vl-235b-a22b-instruct": {
        "model": "Qwen/Qwen3-VL-235B-A22B-Instruct",
        "provider": "vllm"
    },
    "qwen3-vl-4b-instruct": {
        "model": "Qwen/Qwen3-VL-4B-Instruct",
        "provider": "vllm"
    },
    "qwen3-vl-4b-thinking": {
        "model": "Qwen/Qwen3-VL-4B-Thinking",
        "provider": "vllm"
    },
    "qwen3-vl-8b-instruct": {
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "provider": "vllm"
    },
    "qwen3-vl-8b-thinking": {
        "model": "Qwen/Qwen3-VL-8B-Thinking",
        "provider": "vllm"
    },
    "glm-4.5v": {
        "model": "zai-org/GLM-4.5V",
        "provider": "vllm"
    },
    "glm-4.6v": {
        "model": "/mnt/data/shared/models/GLM-4.6V",
        "provider": "vllm"
    },
    "internvl3.5-241b": {
        "model": "OpenGVLab/InternVL3_5-241B-A28B-HF",
        "provider": "vllm"
    },
    "kimi-k2.5": {
        "model": "/mnt/data/shared/models/Kimi-K2.5/",
        "provider": "vllm",
        "image_mode_only": True  # Video support only available via Moonshot's official API
    },
    "qwen3.5-397b-a17b": {
        "model": "/mnt/data/shared/models/Qwen3.5-397B-A17B/",
        "provider": "vllm"
    },
    "qwen3.5-122b-a10b": {
        "model": "/mnt/data/shared/models/Qwen3.5-122B-A10B",
        "provider": "vllm"
    },
    "qwen3.5-27b": {
        "model": "/mnt/data/shared/models/Qwen3.5-27B",
        "provider": "vllm"
    },
    "sft_qwen35_27b_human_reps_only": {
        "model": "/mnt/data/sgsilva/models/qwen35-27b-human-reps-only-step339",
        "provider": "vllm"
    },
    "sft_qwen35_27b_human_reps_plus_llm_fms": {
        "model": "/mnt/data/sgsilva/models/qwen35-27b-human-reps-plus-llm-fms-step1785",
        "provider": "vllm"
    },
    "sft_qwen35_27b_llm_fms_1epoch": {
        # pmartins' LLM-FMS-only SFT (1 epoch). Re-evaluated on the 1105 test set
        # for an apples-to-apples comparison against our variants.
        "model": "/mnt/data/pmartins/vlm_ckpts/sft_qwen_35_27b_llm_fms_1epoch/hf",
        "provider": "vllm"
    },
    "sft_qwen35_27b_oracle_obs": {
        # Redesigned variant: SFT target = 397B oracle-mode [VISUAL OBSERVATIONS].
        # Final checkpoint (step_339, 3 epochs).
        "model": "/mnt/data/sgsilva/models/qwen35-27b-oracle-obs-step339",
        "provider": "vllm"
    },
    "sft_qwen35_27b_oracle_obs_ep1": {
        # Oracle-obs variant, 1-epoch checkpoint (step_113). Evaluated alongside
        # step_339 to check for overfitting (training loss fell to ~0.05).
        "model": "/mnt/data/sgsilva/models/qwen35-27b-oracle-obs-step113",
        "provider": "vllm"
    },
    "sft_qwen35_27b_oracle_obs_ep2": {
        # Oracle-obs variant, 2-epoch checkpoint (step_226).
        "model": "/mnt/data/sgsilva/models/qwen35-27b-oracle-obs-step226",
        "provider": "vllm"
    },
    "qwen3.5-35b-a3b": {
        "model": "Qwen/Qwen3.5-35B-A3B",
        "provider": "vllm"
    },
    "qwen3.5-4b": {
        "model": "Qwen/Qwen3.5-4B",
        "provider": "vllm"
    },
    "qwen3.5-9b": {
        "model": "Qwen/Qwen3.5-9B",
        "provider": "vllm"
    },
    # Gemini models (uses Vertex AI)
    "gemini-3-pro-preview": {
        "model": "vertex_ai/gemini-3-pro-preview",
        "provider": "vertex_ai"
    },
    "gemini-3-flash-preview": {
        "model": "vertex_ai/gemini-3-flash-preview",
        "provider": "vertex_ai"
    },
}

# Default sampling parameters
DEFAULT_MAX_TOKENS = 32768
DEFAULT_TEMPERATURE = 0
# Anti-runaway stop sequences: thinkON reasoners occasionally collapse into a single repeated
# punctuation token and generate to max_tokens (verified 2026-06-23). 8 consecutive identical
# punctuation chars never occur in a legitimate answer, so stopping on them only truncates the
# degenerate tail (clean responses are byte-identical with/without). Override via env if needed.
RUNAWAY_STOP = os.environ.get("RUNAWAY_STOP", "!!!!!!!!,????????,........,--------").split(",")


def encode_images_to_video(image_paths: List[Path], fps: float = 1.0, output_path: str = None, mirror: bool = False) -> str:
    """
    Encode a sequence of images to an MP4 video file using OpenCV.

    Args:
        image_paths: List of image file paths (sorted in order)
        fps: Frames per second for the output video
        output_path: Optional output path for the video. If None, creates a temp file.
        mirror: If True, horizontally flip all frames before encoding

    Returns:
        Path to the created video file
    """
    if not image_paths:
        raise ValueError("No images provided for video encoding")

    # Create output path if not provided
    if output_path is None:
        temp_dir = tempfile.mkdtemp()
        output_path = str(Path(temp_dir) / "output.mp4")

    # Read first image to get dimensions
    first_frame = cv2.imread(str(image_paths[0]))
    if first_frame is None:
        raise ValueError(f"Failed to read first image: {image_paths[0]}")

    height, width, _ = first_frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not video_writer.isOpened():
        raise RuntimeError("Failed to create video writer")

    # Write each image to the video
    for img_path in image_paths:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"Warning: Failed to read image {img_path}, skipping...")
            continue

        # Apply horizontal flip if requested
        if mirror:
            frame = cv2.flip(frame, 1)

        # Ensure frame has the same dimensions
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))

        video_writer.write(frame)

    # Release the video writer
    video_writer.release()

    print(f"Created video: {output_path} ({len(image_paths)} frames at {fps} FPS)")

    return output_path


def query_gemini_with_genai_sdk(
    model_name: str,
    prompt: str,
    video_path: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    sampling_fps: float = None,
    project_id: str = "swordhealth-ai-research",
    location: str = "global",
    system_prompt: str = None
):
    """
    Query Gemini models using Google Gen AI SDK directly through Vertex AI.

    Args:
        model_name: Model name (e.g., "gemini-3-pro-preview", "gemini-2.5-flash")
        prompt: The user prompt
        video_path: Path to video file
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        sampling_fps: FPS rate for Gemini to sample frames at (default: 1.0)
        project_id: GCP project ID
        location: GCP location (use 'global' for Gemini 3 models)
        system_prompt: Optional system prompt (will be prepended to the user prompt for Gemini)

    Returns:
        Tuple of (response_text, usage_dict)
    """
    # Set environment variables for Vertex AI mode
    os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = 'True'
    os.environ['GOOGLE_CLOUD_PROJECT'] = project_id
    os.environ['GOOGLE_CLOUD_LOCATION'] = location

    # Initialize client with proper timeout configuration
    # Note: We don't pass timeout to HttpOptions as it doesn't properly configure
    # the underlying httpx read timeout. The SDK uses sensible defaults.
    client = genai.Client(
        http_options=HttpOptions(api_version="v1")
    )

    # Read video file
    with open(video_path, "rb") as f:
        video_bytes = f.read()

    # Set default sampling_fps if not specified
    if sampling_fps is None:
        sampling_fps = 1.0

    # Generate response with inline video data
    config_params = {
        'max_output_tokens': max_tokens,
        'temperature': temperature,
        'thinking_config': types.ThinkingConfig(
            include_thoughts=True
        )
    }

    # Add system instruction if provided
    if system_prompt:
        config_params['system_instruction'] = system_prompt

    response = client.models.generate_content(
        model=model_name,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        inline_data=types.Blob(
                            data=video_bytes,
                            mime_type="video/mp4"
                        ),
                        video_metadata=types.VideoMetadata(
                            fps=sampling_fps
                        )
                    ),
                    types.Part(text=prompt)
                ]
            )
        ],
        config=types.GenerateContentConfig(**config_params)
    )

    # Extract usage information
    usage = None
    if hasattr(response, 'usage_metadata'):
        usage = {
            'prompt_tokens': getattr(response.usage_metadata, 'prompt_token_count', 0),
            'completion_tokens': getattr(response.usage_metadata, 'candidates_token_count', 0),
            'total_tokens': getattr(response.usage_metadata, 'total_token_count', 0)
        }

        # Add token details if available (includes reasoning tokens for thinking models)
        if hasattr(response.usage_metadata, 'cached_content_token_count'):
            usage['cached_tokens'] = response.usage_metadata.cached_content_token_count

    # Extract thinking trace from thought field if available (for Gemini thinking models)
    reasoning_content = None
    if hasattr(response, 'candidates') and response.candidates:
        candidate = response.candidates[0]
        if hasattr(candidate, 'content') and candidate.content.parts:
            # Collect all thought parts
            thoughts = []
            for part in candidate.content.parts:
                if hasattr(part, 'thought') and part.thought:
                    # The thought field is a boolean flag - when True, part.text contains the thinking
                    if isinstance(part.thought, bool) and part.text:
                        thoughts.append(part.text)
                    elif hasattr(part.thought, 'text'):
                        thoughts.append(part.thought.text)
                    else:
                        thoughts.append(str(part.thought))

            if thoughts:
                reasoning_content = '\n'.join(thoughts)
                print(f"Captured thinking trace: {len(reasoning_content)} characters")

    # Add reasoning_content to usage dict if available
    if reasoning_content and usage:
        usage['reasoning_content'] = reasoning_content

    return response.text, usage


def query_with_vllm_direct(
    model_name: str,
    prompt: str,
    image_paths: List[Path] = None,
    video_path: str = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    api_base: str = None,
    num_frames: int = None,
    sampling_fps: float = None,
    system_prompt: str = None,
    timeout: float = 600.0
):
    """
    Query a vLLM server directly using OpenAI client (no LiteLLM).

    Args:
        model_name: Model name on the vLLM server
        prompt: The user prompt
        image_paths: List of image file paths (for image sequence mode)
        video_path: Path to video file (for video mode)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        api_base: Base URL for vLLM server
        num_frames: Number of frames to sample from video (for vLLM video models)
        sampling_fps: FPS rate for VLM to sample frames at (for vLLM video models)
        system_prompt: Optional system prompt to prepend to messages
        timeout: Request timeout in seconds (default: 600.0)

    Returns:
        Tuple of (response_text, usage_dict)
    """
    # Build message content
    content = [{"type": "text", "text": prompt}]

    if video_path:
        # Video mode
        with open(video_path, "rb") as f:
            video_data = base64.b64encode(f.read()).decode("utf-8")

        content.append({
            "type": "video_url",
            "video_url": {
                "url": f"data:video/mp4;base64,{video_data}"
            }
        })
    elif image_paths:
        # Image sequence mode
        for img_path in image_paths:
            with open(img_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            # Determine mime type
            ext = img_path.suffix.lower()
            mime_type = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.webp': 'image/webp',
            }.get(ext, 'image/jpeg')

            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{image_data}"}
            })

    # Build messages array with optional system prompt
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})

    # Ensure api_base ends with /v1
    if not api_base.endswith('/v1'):
        api_base = f"{api_base}/v1"

    # Create OpenAI client for vLLM
    client = OpenAI(
        api_key="EMPTY",
        base_url=api_base,
        timeout=timeout,
    )

    # Build extra_body for mm_processor_kwargs
    extra_body = {}
    if video_path:
        extra_body = {
                "mm_processor_kwargs": {
                    "fps": list(sampling_fps),
    }
}

    # Wrap the completion call with retry logic for rate limits
    def _make_request():
        kwargs = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            # Anti-runaway stop: the thinkON reasoner occasionally collapses into a single-token
            # wall (e.g. "!!!!!!!!...") and burns to max_tokens (verified 2026-06-23, ~50x wasted
            # compute per runaway). 8 consecutive identical punctuation chars never occur in a real
            # answer, so this only truncates the degenerate tail — clean responses are untouched.
            "stop": RUNAWAY_STOP,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body

        return client.chat.completions.create(**kwargs)

    response = retry_with_exponential_backoff(_make_request)

    # Return both content and usage information
    content = response.choices[0].message.content
    usage = {
        "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
        "completion_tokens": response.usage.completion_tokens if response.usage else None,
        "total_tokens": response.usage.total_tokens if response.usage else None,
    } if hasattr(response, 'usage') else None

    return content, usage


def query_with_litellm(
    model_name: str,
    prompt: str,
    image_paths: List[Path] = None,
    video_path: str = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    api_base: str = None,
    num_frames: int = None,
    sampling_fps: float = None,
    system_prompt: str = None,
    top_p: float = None,
    top_k: int = None,
    timeout: float = 600.0,
    disable_thinking: bool = False
):
    """
    Query a model using LiteLLM with images or video.

    Args:
        model_name: LiteLLM model identifier (e.g., "openai/model" or "vertex_ai/gemini")
        prompt: The user prompt
        image_paths: List of image file paths (for image sequence mode)
        video_path: Path to video file (for video mode)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        api_base: Base URL for OpenAI-compatible servers (vLLM)
        num_frames: Number of frames to sample from video (for vLLM video models) - DEPRECATED, use sampling_fps
        sampling_fps: FPS rate for VLM to sample frames at (for vLLM video models)
        system_prompt: Optional system prompt to prepend to messages
        timeout: Request timeout in seconds (default: 600.0)

    Returns:
        Model response text
    """
    # For Gemini models (vertex_ai/), use the dedicated Gemini SDK for video inputs
    # For text-only, continue to LiteLLM path below
    if not api_base and model_name.startswith("vertex_ai/") and (video_path or image_paths):
        if not video_path:
            raise ValueError("query_gemini_with_genai_sdk only supports video_path, not image_paths")

        # Extract model name (e.g., "vertex_ai/gemini-2.5-flash" -> "gemini-2.5-flash")
        gemini_model = model_name.split("/", 1)[1]

        return query_gemini_with_genai_sdk(
            model_name=gemini_model,
            prompt=prompt,
            video_path=video_path,
            max_tokens=max_tokens,
            temperature=temperature,
            sampling_fps=sampling_fps
        )

    # For vLLM and other providers, use LiteLLM
    # Build message content
    content = [{"type": "text", "text": prompt}]

    if video_path:
        # Video mode
        with open(video_path, "rb") as f:
            video_data = base64.b64encode(f.read()).decode("utf-8")

        # Check if this is a Vertex AI model (Gemini)
        is_vertex_ai = model_name.startswith("vertex_ai/")

        if is_vertex_ai:
            # Gemini/Vertex AI uses "file" type with file_data
            content.append({
                "type": "file",
                "file": {
                    "file_data": f"data:video/mp4;base64,{video_data}"
                }
            })
        else:
            # vLLM servers use video_url format
            content.append({
                "type": "video_url",
                "video_url": {
                    "url": f"data:video/mp4;base64,{video_data}"
                }
            })
    elif image_paths:
        # Image sequence mode
        for img_path in image_paths:
            with open(img_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            # Determine mime type
            ext = img_path.suffix.lower()
            mime_type = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.webp': 'image/webp',
            }.get(ext, 'image/jpeg')

            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{image_data}"}
            })

    # Build messages array with optional system prompt
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})

    # Call LiteLLM
    # For vLLM servers, we need to use the "openai/" prefix with custom base URL
    if api_base:
        # Use openai/ prefix to tell LiteLLM to use OpenAI client
        # Ensure api_base ends with /v1
        if not api_base.endswith('/v1'):
            api_base = f"{api_base}/v1"

        litellm_model = f"openai/{model_name}"
        kwargs = {
            "model": litellm_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "api_base": api_base,
            "api_key": "EMPTY",
            "timeout": timeout,
            # Anti-runaway stop (see _make_request above): cut the thinkON single-token "!!!!!!!!"
            # collapse so a runaway dies at ~hundreds of tokens, not max_tokens. Harmless to real text.
            "stop": RUNAWAY_STOP,
        }

        # Add optional sampling parameters if specified
        if top_p is not None:
            kwargs["top_p"] = top_p
        if top_k is not None:
            kwargs["top_k"] = top_k

        # Build extra_body dict for vLLM
        extra_body_dict = {}

        # Add FPS parameter for vLLM video models to control frame sampling
        if sampling_fps is not None and video_path:
            extra_body_dict["mm_processor_kwargs"] = {
                "fps": sampling_fps,  # Sample at this FPS rate
            }

        # Add chat_template_kwargs if thinking is disabled
        if disable_thinking:
            extra_body_dict["chat_template_kwargs"] = {
                "enable_thinking": False
            }

        if extra_body_dict:
            kwargs["extra_body"] = extra_body_dict
    else:
        # For other providers (like Gemini), construct base_url for Vertex AI
        if model_name.startswith("vertex_ai/"):
            # Extract the model name from vertex_ai/model-name
            gemini_model = model_name.split("/")[-1]
            base_url = f"https://aiplatform.googleapis.com/v1/projects/swordhealth-ai-research/locations/global/publishers/google/models/{gemini_model}"
            kwargs = {
                "model": model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "base_url": base_url,
                "timeout": timeout,
                # Enable thinking mode for Gemini to get reasoning summary
                "thinking_config": {
                    "include_thoughts": True
                }
            }

            # Add optional sampling parameters if specified
            if top_p is not None:
                kwargs["top_p"] = top_p
            if top_k is not None:
                kwargs["top_k"] = top_k
        else:
            # For other providers, use the model name as-is
            kwargs = {
                "model": model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "timeout": timeout,
            }

            # Add optional sampling parameters if specified
            if top_p is not None:
                kwargs["top_p"] = top_p
            if top_k is not None:
                kwargs["top_k"] = top_k

    # Wrap the completion call with retry logic for rate limits
    def _make_request():
        return completion(**kwargs)

    response = retry_with_exponential_backoff(_make_request)

    # Return both content and usage information
    content = response.choices[0].message.content
    usage = response.usage if hasattr(response, 'usage') else None

    # Extract reasoning content if available (for thinking models)
    # Note: GLM-4.6V only generates reasoning for text-only inputs, not for vision/video
    reasoning_content = None
    message = response.choices[0].message

    # Check for reasoning_content in message (LiteLLM/GLM models store it here)
    if hasattr(message, 'reasoning_content'):
        rc = message.reasoning_content
        if rc and isinstance(rc, str) and rc.strip():
            reasoning_content = rc

    # Check for alternative reasoning field (fallback)
    if not reasoning_content and hasattr(message, 'reasoning'):
        r = message.reasoning
        if r and isinstance(r, str) and r.strip():
            reasoning_content = r

    # Check provider_specific_fields (another fallback)
    if not reasoning_content and hasattr(message, 'provider_specific_fields'):
        provider_fields = message.provider_specific_fields
        if isinstance(provider_fields, dict):
            rc = provider_fields.get('reasoning_content') or provider_fields.get('reasoning')
            if rc and isinstance(rc, str) and rc.strip():
                reasoning_content = rc
    
    # Check if reasoning is embedded in content with think tags (for vLLM thinking models)
    if content and '</think>' in content:
        import re
        # Option 1: Both <think> and </think> tags present
        if '<think>' in content:
            think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
            if think_match:
                reasoning_content = think_match.group(1).strip()
                # Remove the <think>...</think> section from content
                content = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL).strip()
        # Option 2: Only </think> tag present (reasoning starts at beginning)
        else:
            think_match = re.search(r'^(.*?)</think>', content, re.DOTALL)
            if think_match:
                reasoning_content = think_match.group(1).strip()
                # Remove everything up to and including </think> from content
                content = re.sub(r'^.*?</think>\s*', '', content, flags=re.DOTALL).strip()

    # Add reasoning_content to usage object for easy access
    if usage and reasoning_content:
        usage.reasoning_content = reasoning_content

    return content, usage


def query_with_images(
    model_name: str,
    prompt: str,
    image_paths: List[Path],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    video_mode: bool = False,
    video_fps: float = 1.0,
    api_base: str = None,
    sampling_fps: float = None,
    system_prompt: str = None,
    mirror: bool = False,
    top_p: float = None,
    top_k: int = None,
    timeout: float = 600.0,
    disable_thinking: bool = False,
    frame_duplication: int = 1
):
    """
    Query a model with images (either as video or image sequence) using LiteLLM.

    Args:
        model_name: Model short name (key from MODELS dict) or direct model path/identifier
        prompt: The user prompt
        image_paths: List of image file paths
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        video_mode: If True, encode as video. If False, send as image sequence
        video_fps: Frame rate for video encoding (determines video temporal density)
        frame_duplication: If > 1, each frame is repeated N times in order before
            encoding/sending (slows perceived motion). Default 1 = no duplication.
        api_base: Base URL for vLLM server (e.g., "http://localhost:8000")
        sampling_fps: FPS for VLM to sample frames at. If None, defaults to video_fps (use all frames).
        system_prompt: Optional system prompt to prepend to messages
        mirror: If True, horizontally flip all frames before encoding
        timeout: Request timeout in seconds (default: 600.0)

    Returns:
        Model response text
    """
    # Optionally repeat each frame in-order (frame_duplication > 1 slows motion).
    if frame_duplication and frame_duplication > 1:
        image_paths = [p for p in image_paths for _ in range(frame_duplication)]

    # Get model config or use direct model path
    model_config = MODELS.get(model_name)

    if model_config:
        # Predefined model from MODELS dict
        litellm_model = model_config["model"]
        provider = model_config["provider"]
        # Check if this model only supports image mode (not video)
        image_mode_only = model_config.get("image_mode_only", False)
    else:
        # Direct model path/identifier - assume vLLM provider
        print(f"Model '{model_name}' not found in MODELS dict, treating as direct model path/identifier")
        litellm_model = model_name
        provider = "vllm"
        image_mode_only = False

    # For vLLM models, set api_base
    if provider == "vllm" and api_base:
        api_base_url = api_base
    else:
        api_base_url = None

    # Force image sequence mode for models that don't support video
    if image_mode_only and video_mode:
        print(f"Note: {model_name} does not support video mode via SGLang/vLLM. Using image sequence mode instead.")
        video_mode = False

    if video_mode:
        # Default sampling_fps to video_fps if not specified
        if sampling_fps is None:
            sampling_fps = video_fps

        # Encode images to video
        print(f"Encoding {len(image_paths)} images to video at {video_fps} FPS...")
        video_path = encode_images_to_video(image_paths, fps=video_fps, mirror=mirror)

        # Check if this is a Gemini model that should use Gen AI SDK
        is_gemini = provider == "vertex_ai" or litellm_model.startswith("vertex_ai/")

        # The encoded mp4 lives in a tempfile.mkdtemp() dir; remove it after the
        # request so video-mode calls don't leak /tmp/tmp*/output.mp4 (a 26GB leak
        # accumulated before this). [[feedback_output_locations]]
        try:
            if is_gemini:
                # Extract model name from "vertex_ai/model-name" format if needed
                gemini_model_name = litellm_model.split("/")[-1] if "/" in litellm_model else litellm_model

                print(f"Querying {model_name} via Gen AI SDK (encoded at {video_fps} FPS, sampling at {sampling_fps} FPS)...")
                return query_gemini_with_genai_sdk(
                    model_name=gemini_model_name,
                    prompt=prompt,
                    video_path=video_path,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    sampling_fps=sampling_fps,
                    project_id="swordhealth-ai-research",
                    location="global",
                    system_prompt=system_prompt
                )
            else:
                # Use LiteLLM for non-Gemini models
                print(f"Querying {model_name} in video mode (encoded at {video_fps} FPS, sampling at {sampling_fps} FPS)...")
                return query_with_litellm(
                    model_name=litellm_model,
                    prompt=prompt,
                    video_path=video_path,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    api_base=api_base_url,
                    sampling_fps=sampling_fps,
                    system_prompt=system_prompt,
                    top_p=top_p,
                    top_k=top_k,
                    timeout=timeout,
                    disable_thinking=disable_thinking
                )
        finally:
            try:
                import shutil
                shutil.rmtree(os.path.dirname(video_path), ignore_errors=True)
            except Exception:
                pass
    else:
        # Send as image sequence
        print(f"Querying {model_name} with {len(image_paths)} images...")
        return query_with_litellm(
            model_name=litellm_model,
            prompt=prompt,
            image_paths=image_paths,
            max_tokens=max_tokens,
            temperature=temperature,
            api_base=api_base_url,
            system_prompt=system_prompt,
            top_p=top_p,
            top_k=top_k,
            timeout=timeout,
            disable_thinking=disable_thinking
        )


def query_with_path(
    model_name: str,
    prompt: str,
    image_path: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    video_mode: bool = False,
    video_fps: float = 1.0,
    api_base: str = None,
    sampling_fps: float = None,
    top_p: float = None,
    top_k: int = None,
    timeout: float = 600.0,
    mirror: bool = False,
):
    """
    Query model with images from a directory path.

    Args:
        model_name: Model short name
        prompt: The user prompt
        image_path: Path to directory containing images
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        video_mode: If True, encode as video. If False, send as image sequence
        video_fps: Frame rate for video encoding (default: 1.0, reads from fps.txt if available)
        api_base: Base URL for vLLM server (optional)
        sampling_fps: FPS for VLM frame sampling (if None, uses all frames)
        timeout: Request timeout in seconds (default: 600.0)

    Returns:
        Model response text
    """
    path_obj = Path(image_path)

    if not path_obj.exists():
        raise ValueError(f"Path not found: {image_path}")

    if not path_obj.is_dir():
        raise ValueError(f"Path is not a directory: {image_path}")

    print(f"Loading images from: {image_path}")

    # Get all image files from directory
    image_extensions = ['.webp', '.jpg', '.jpeg', '.png']
    images = []
    for ext in image_extensions:
        images.extend(path_obj.glob(f"*{ext}"))

    # Sort by filename to maintain temporal order
    images.sort(key=lambda x: x.name)

    if not images:
        raise ValueError(f"No image files found in directory: {image_path}")

    # Check if there's an fps.txt file in the directory
    fps_file = path_obj / "fps.txt"
    if fps_file.exists() and video_fps == 1.0:
        try:
            fps_from_file = float(fps_file.read_text().strip())
            video_fps = fps_from_file
            print(f"Found {len(images)} images")
            print(f"Using encoding FPS from fps.txt: {video_fps:.2f}")
        except (ValueError, IOError):
            print(f"Found {len(images)} images")
            if video_mode:
                print(f"Video encoding FPS: {video_fps}")
    else:
        print(f"Found {len(images)} images")
        if video_mode:
            print(f"Video encoding FPS: {video_fps}")
        else:
            print(f"Image sequence mode")

    print(f"Prompt: {prompt}\n")

    return query_with_images(
        model_name=model_name,
        prompt=prompt,
        image_paths=images,
        max_tokens=max_tokens,
        temperature=temperature,
        video_mode=video_mode,
        video_fps=video_fps,
        api_base=api_base,
        sampling_fps=sampling_fps,
        top_p=top_p,
        top_k=top_k,
        timeout=timeout,
        mirror=mirror,
    )


def query_text(
    prompt: str,
    model: str,
    api_base: str = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    timeout: float = 600.0
) -> Tuple[str, Dict]:
    """
    Query a model with text-only input (no images/video).

    Args:
        prompt: The text prompt
        model: Model short name or full litellm identifier / direct model path
        api_base: Base URL for vLLM server (optional)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        timeout: Request timeout in seconds (default: 600.0)

    Returns:
        Tuple of (response_text, token_usage_dict)
        token_usage_dict contains: {
            'input_tokens': int,
            'output_tokens': int,
            'reasoning_tokens': int (if available)
        }
    """
    # Get model configuration or use direct model path
    model_config = MODELS.get(model)
    if model_config:
        litellm_model = model_config["model"]
        provider = model_config.get("provider", "vllm")
    else:
        # Direct model path/identifier - assume vLLM if api_base provided
        print(f"Model '{model}' not found in MODELS dict, treating as direct model path/identifier")
        litellm_model = model
        provider = "vllm" if api_base else "unknown"

    # For vLLM with api_base, we need to use openai/ prefix for litellm
    # unless the model already has a provider prefix
    if api_base and provider == "vllm" and not litellm_model.startswith("openai/"):
        # Use the direct OpenAI client approach instead of litellm for vLLM
        # This is more reliable for custom vLLM servers
        from openai import OpenAI

        # Ensure api_base ends with /v1
        if not api_base.endswith('/v1'):
            api_base = f"{api_base}/v1"

        client = OpenAI(
            api_key="EMPTY",
            base_url=api_base,
            timeout=timeout,
        )

        def _make_request():
            return client.chat.completions.create(
                model=litellm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )

        response = retry_with_exponential_backoff(_make_request)

        # Extract token usage
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0

        content = response.choices[0].message.content

        # Capture the reasoning trace (text-only path). vLLM exposes it on the
        # message as reasoning_content/reasoning (separate-reasoning mode), or it
        # may be inline as <think>…</think> in content. Mirror the extraction in
        # query_with_litellm so text-only evals don't silently drop the trace
        # (was hardcoded to none here — see visual-obs experiments).
        message = response.choices[0].message
        reasoning_content = None
        for attr in ('reasoning_content', 'reasoning'):
            rc = getattr(message, attr, None)
            if rc and str(rc).strip():
                reasoning_content = str(rc).strip()
                break
        if not reasoning_content and content and '<think>' in content:
            import re as _re
            m = _re.search(r'<think>(.*?)</think>', content, _re.DOTALL)
            if m and m.group(1).strip():
                reasoning_content = m.group(1).strip()

        token_usage = {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'reasoning_tokens': 0,
        }
        if reasoning_content:
            token_usage['reasoning_content'] = reasoning_content

        return content, token_usage

    # For non-vLLM or models without api_base, use litellm
    # Call query_with_litellm (handles text-only when no images/video provided)
    response, usage = query_with_litellm(
        model_name=litellm_model,
        prompt=prompt,
        image_paths=None,
        video_path=None,
        max_tokens=max_tokens,
        temperature=temperature,
        api_base=api_base,
        timeout=timeout
    )

    # Extract token usage
    input_tokens = getattr(usage, 'prompt_tokens', 0)
    output_tokens = getattr(usage, 'completion_tokens', 0)
    reasoning_tokens = 0

    # Check for reasoning tokens (Gemini thinking mode)
    if hasattr(usage, 'completion_tokens_details'):
        details = usage.completion_tokens_details
        if hasattr(details, 'reasoning_tokens'):
            reasoning_tokens = details.reasoning_tokens

    token_usage = {
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'reasoning_tokens': reasoning_tokens
    }

    return response, token_usage


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Query vision models using LiteLLM (supports vLLM servers, Gemini, and other providers)"
    )
    parser.add_argument("--server-url", type=str, default=None,
                       help="vLLM server URL for vLLM models (e.g., http://localhost:8000). Not needed for Gemini.")
    parser.add_argument("--model", type=str, default="glm-4.5v",
                       help="Model to use (predefined name or direct path/identifier). Default: glm-4.5v. "
                            f"Predefined models: {', '.join(MODELS.keys())}")
    parser.add_argument("--max-tokens", type=int, default=4096,
                       help="Maximum tokens to generate (default: 4096)")
    parser.add_argument("--temperature", type=float, default=0,
                       help="Sampling temperature (default: 0)")
    parser.add_argument("--video-mode", action="store_true",
                       help="Encode images as video (default: image sequence mode)")
    parser.add_argument("--video-fps", type=float, default=1.0,
                       help="Frame rate for video encoding (default: 1.0, reads from fps.txt if available)")
    parser.add_argument("--sampling-fps", type=float, default=None,
                       help="FPS for VLM frame sampling (if None, uses all frames in video)")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--prompt", type=str,
                       help="The prompt/question for the model (non-interactive mode)")
    parser.add_argument("--path", type=str,
                       help="Path to directory containing images (non-interactive mode)")
    parser.add_argument("--timeout", type=float, default=600.0,
                       help="Request timeout in seconds (default: 600)")

    args = parser.parse_args()

    try:
        model_config = MODELS.get(args.model)

        if model_config:
            # Predefined model
            if model_config["provider"] == "vllm" and not args.server_url:
                parser.error(f"--server-url is required for vLLM model: {args.model}")

            print(f"Model: {args.model} ({model_config['model']})")
            print(f"Provider: {model_config['provider']}")
        else:
            # Custom model path/identifier
            if not args.server_url:
                parser.error(f"--server-url is required for custom model: {args.model}")

            print(f"Custom Model: {args.model}")
            print(f"Provider: vllm (assumed)")

        if args.server_url:
            print(f"Server URL: {args.server_url}")
        print(f"Mode: {'Video' if args.video_mode else 'Image sequence'}")
        print()

        if args.interactive:
            # Interactive mode
            print("="*80)
            print("Interactive Mode")
            print(f"Mode: {'Video (images as video with FPS)' if args.video_mode else 'Image sequence'}")
            print("Commands:")
            print("  - Enter path (directory with images), FPS (optional), and prompt")
            print("  - Format: [path] [fps] [prompt] OR [path] [prompt]")
            print("  - Type 'quit' or 'exit' to exit")
            print("  - Press Ctrl+C to exit")
            print("="*80)
            print()

            while True:
                try:
                    # Get input from user
                    user_input = input("Enter input: ").strip()

                    if user_input.lower() in ['quit', 'exit', 'q']:
                        print("Exiting...")
                        break

                    # Parse input
                    parts = user_input.split(maxsplit=2)
                    if len(parts) < 2:
                        print("Error: Please provide at least path and prompt")
                        print("Example: /path/to/images/ Describe this exercise")
                        print("Example: /path/to/images/ 2.5 Describe this exercise")
                        continue

                    image_path = parts[0]

                    # Check if second part is a number (video encoding FPS)
                    if len(parts) >= 3:
                        try:
                            video_fps = float(parts[1])
                            prompt = parts[2]
                        except ValueError:
                            # Second part is not a number, treat as part of prompt
                            video_fps = args.video_fps
                            prompt = ' '.join(parts[1:])
                    else:
                        # Only 2 parts: image_path and prompt
                        video_fps = args.video_fps
                        prompt = parts[1]

                    # Query with path
                    print()
                    response, usage = query_with_path(
                        model_name=args.model,
                        prompt=prompt,
                        image_path=image_path,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        video_mode=args.video_mode,
                        video_fps=video_fps,  # Use fps from interactive input as encoding fps
                        api_base=args.server_url,
                        sampling_fps=args.sampling_fps,
                        timeout=args.timeout
                    )

                    # Display reasoning content if available
                    if usage and hasattr(usage, 'reasoning_content') and usage.reasoning_content:
                        print(f"Reasoning:\n{usage.reasoning_content}\n")
                        print("-" * 80)

                    print(f"Response:\n{response}")
                    print("\n" + "="*80 + "\n")

                except KeyboardInterrupt:
                    print("\n\nExiting...")
                    break
                except Exception as e:
                    print(f"Error: {e}\n")

        else:
            # Non-interactive mode
            if not args.prompt or not args.path:
                parser.error("In non-interactive mode, both --prompt and --path are required")

            response, usage = query_with_path(
                model_name=args.model,
                prompt=args.prompt,
                image_path=args.path,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                video_mode=args.video_mode,
                video_fps=args.video_fps,
                api_base=args.server_url,
                sampling_fps=args.sampling_fps,
                timeout=args.timeout
            )

            # Display reasoning content if available
            if usage and hasattr(usage, 'reasoning_content') and usage.reasoning_content:
                print(f"Reasoning:\n{usage.reasoning_content}\n")
                print("-" * 80)

            print(f"Response:\n{response}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
