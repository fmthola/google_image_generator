#!/usr/bin/env python3
"""
Shared utilities for Gemini image generation scripts.
Contains common functions for file handling, API communication, and image processing.
"""

import os
import base64
import mimetypes
import secrets
import threading
import requests
from pathlib import Path
from typing import Optional

# Thread-safe lock for unique filename generation
_filename_lock = threading.Lock()

# Default MIME type for unrecognized image extensions
_DEFAULT_MIME_TYPE = 'image/jpeg'

# Gemini API endpoint
API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-3-pro-image-preview:generateContent?key={}"
)


def get_mime_type(file_path: str) -> str:
    """Get MIME type for an image file."""
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is not None:
        return mime_type
    ext = Path(file_path).suffix.lower()
    mime_map = {
        '.jpg': _DEFAULT_MIME_TYPE,
        '.jpeg': _DEFAULT_MIME_TYPE,
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.tiff': 'image/tiff',
        '.tif': 'image/tiff',
        '.webp': 'image/webp',
    }
    return mime_map.get(ext, _DEFAULT_MIME_TYPE)


def load_image_as_base64(file_path: str) -> tuple[str, str]:
    """Load an image file and return base64 data and MIME type."""
    with open(file_path, 'rb') as f:
        image_data = f.read()
    return base64.b64encode(image_data).decode('utf-8'), get_mime_type(file_path)


def unique_output_path(path_str: str) -> str:
    """
    Thread-safe: If path exists, append a 4-digit hash (0000-9999) before the extension.
    Example: out.png -> out_0427.png
    Uses a lock to prevent race conditions when multiple threads generate filenames.
    """
    with _filename_lock:
        p = Path(path_str)
        if not p.exists():
            p.touch()
            return str(p)

        stem = p.stem
        suffix = p.suffix
        parent = p.parent

        for _ in range(10000):
            h = f"{secrets.randbelow(10000):04d}"
            candidate = parent / f"{stem}_{h}{suffix}"
            if not candidate.exists():
                candidate.touch()
                return str(candidate)

        raise RuntimeError(f"Unable to find a unique filename for {path_str} after many attempts.")


def cleanup_placeholder(output_path: str) -> None:
    """Remove placeholder file created during filename reservation."""
    try:
        Path(output_path).unlink(missing_ok=True)
    except OSError:
        pass


def load_reference(file_path: str, label: str) -> Optional[dict]:
    """Load a reference image and return an inline_data part for the API."""
    if not os.path.exists(file_path):
        print(f"Error: {label} not found: {file_path}")
        return None
    img_data, mime_type = load_image_as_base64(file_path)
    print(f"Loaded {label}: {file_path} ({mime_type})")
    return {"inline_data": {"mime_type": mime_type, "data": img_data}}


def extract_image_from_response(result: dict, output_path: str) -> bool:
    """Extract generated image from API response and save to output_path."""
    candidates = result.get("candidates", [])
    if not candidates:
        return False
    parts = candidates[0].get("content", {}).get("parts", [])
    for part in parts:
        if "inlineData" in part:
            image_data = base64.b64decode(part["inlineData"]["data"])
            with open(output_path, 'wb') as f:
                f.write(image_data)
            return True
    return False


def print_header(prompt, output_path, quality, references=None):
    """Print generation header to stdout."""
    separator = '=' * 60
    print(f"\n{separator}")
    print("Gemini 3 Pro Image Generator (Nano Banana Pro)")
    print(separator)
    print(f"Prompt: {prompt}")
    print(f"Output: {output_path}")
    print(f"Quality: {quality.upper()}")
    if references:
        for label, path in references:
            print(f"{label}: {path}")
    print(f"{separator}\n")


def call_api_and_save(parts: list, quality: str, output_path: str, api_key: str) -> bool:
    """Send the generation request to the Gemini API and save the result."""
    image_size = "4K" if quality.lower() == "4k" else "2K"
    if image_size == "4K":
        print("Using 4K resolution (higher cost: ~$0.24 per image)")
    else:
        print("Using 2K resolution (~$0.134 per image)")
    print("\nGenerating image...")

    url = API_URL.format(api_key)
    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "responseModalities": ["IMAGE", "TEXT"],
            "imageConfig": {
                "aspectRatio": "16:9",
                "imageSize": image_size
            }
        }
    }
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=120)
    except requests.exceptions.Timeout:
        print("\nError: Request timed out after 120 seconds")
        cleanup_placeholder(output_path)
        return False
    except Exception as e:
        print(f"\nError generating image: {e}")
        import traceback
        traceback.print_exc()
        cleanup_placeholder(output_path)
        return False

    if response.status_code != 200:
        print(f"\nError: API request failed with status {response.status_code}")
        print(f"Response: {response.text}")
        cleanup_placeholder(output_path)
        return False

    result = response.json()
    if extract_image_from_response(result, output_path):
        print(f"\nImage saved successfully to: {output_path}")
        print('=' * 60 + '\n')
        return True

    print("\nError: No images generated")
    print(f"Response: {result}")
    cleanup_placeholder(output_path)
    return False


def resolve_api_key(api_key: Optional[str] = None) -> Optional[str]:
    """Resolve API key from argument or environment variable."""
    if api_key is not None:
        return api_key
    key = os.environ.get('GEMINI_API_KEY')
    if not key:
        print("Error: GEMINI_API_KEY environment variable not set and no API key provided")
        return None
    return key


def resolve_output_path(output_path: str) -> str:
    """Resolve output path, finding a unique name if the file already exists."""
    original = output_path
    resolved = unique_output_path(output_path)
    if resolved != original:
        print(f"Output file exists; writing to: {resolved}")
    return resolved
