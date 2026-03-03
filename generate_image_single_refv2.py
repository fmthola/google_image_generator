#!/usr/bin/env python3
"""
Generate images using Google Gemini 3 Pro Image API (Nano Banana Pro).
Supports a single reference image (original) plus a textual target description.
Uses the latest gemini-3-pro-image-preview model with 2K/4K support.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from image_gen_utils import (
    resolve_api_key, resolve_output_path, print_header,
    load_reference, cleanup_placeholder, call_api_and_save,
)


def build_single_reference_prompt(target_description: str) -> str:
    """Wrap a target description with guidance for one-reference generation."""
    target_description = (target_description or "").strip()
    return (
        "Use the attached reference image as the ONLY visual reference.\n"
        "KEEP CONSISTENT: character identity (face/features), art style, and overall quality.\n"
        "CHANGE TO THIS NEW IMAGE (target description):\n"
        f"{target_description}\n\n"
        "Avoid changing the character into a different person. Avoid text/watermarks unless requested."
    )


def generate_image_with_reference(
    prompt: str,
    output_path: str,
    reference_image1: Optional[str] = None,
    quality: str = "2k",
    api_key: Optional[str] = None
) -> bool:
    """
    Generate an image using Google Gemini 3 Pro Image API with an optional reference image.

    This function uses the latest Gemini 3 Pro Image model (gemini-3-pro-image-preview)
    which supports:
    - Reference image guidance (this script supports 1 reference image)
    - 2K and 4K native resolution
    - Image editing and style transfer

    Args:
        prompt: Text description of the image to generate (this is your main creative prompt)
        output_path: Path where the generated image will be saved
        reference_image1: Path to the original reference image (character/style reference)
        quality: Image quality - either "2k" or "4k" (default: "2k")
        api_key: Gemini API key (if not provided, uses GEMINI_API_KEY env var)

    Returns:
        True if successful, False otherwise
    """
    api_key = resolve_api_key(api_key)
    if not api_key:
        return False

    output_path = resolve_output_path(output_path)

    references = []
    if reference_image1:
        references.append(("Reference Image (Original)", reference_image1))
    print_header(prompt, output_path, quality, references or None)

    text = build_single_reference_prompt(prompt) if reference_image1 else prompt
    parts = [{"text": text}]

    if reference_image1:
        ref_part = load_reference(reference_image1, "reference image (original)")
        if ref_part is None:
            cleanup_placeholder(output_path)
            return False
        parts.append(ref_part)

    return call_api_and_save(parts, quality, output_path, api_key)


def main():
    parser = argparse.ArgumentParser(
        description='Generate images using Google Gemini 3 Pro Image API with single-reference support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  1. Generate from text prompt only (2K quality):
     python generate_image.py "A serene mountain landscape at sunset" output.png

  2. Generate from text prompt (4K quality):
     python generate_image.py "A serene mountain landscape at sunset" output.png --quality 4k

  3. One-reference generation (keep character/style from the reference, change to match the description):
     python generate_image.py "Full-body pose, standing confidently, Starfleet command uniform (gold), on a starship bridge, cinematic cool lighting" output.png --ref1 original_reference.jpg

  4. Another one-reference example (simple edit request):
     python generate_image.py "Same character, but wearing a winter coat and scarf, snowy city street background, warm street lights" output.png --ref1 original_reference.jpg

Notes:
  - If you provide --ref1, this script treats it as the ONLY visual reference and wraps your target description
    with guidance to preserve identity and style.
  - Provide your API key via --api-key or set GEMINI_API_KEY in your environment.
"""
    )

    parser.add_argument(
        'prompt',
        help='TARGET DESCRIPTION: Describe what the new image should look like. '
             'If --ref1 is provided, it is treated as the ONLY visual reference and your description is wrapped '
             'with guidance to keep the same character identity and art style.'
    )
    parser.add_argument(
        'output',
        help='Output file path (e.g., output.png, result.jpg)'
    )
    parser.add_argument(
        '--ref1',
        '--reference1',
        dest='reference_image1',
        help='REFERENCE IMAGE: Path to the original reference image. '
             'This should be the character/style reference you want to keep consistent. '
             'Supports PNG, JPG, JPEG, GIF, BMP, TIFF, WebP formats.'
    )
    parser.add_argument(
        '--quality',
        '-q',
        choices=['2k', '4k'],
        default='2k',
        help='Image quality/resolution: "2k" (default, ~$0.134/image) or "4k" (~$0.24/image)'
    )
    parser.add_argument(
        '--api-key',
        help='Gemini API key (defaults to GEMINI_API_KEY env var)'
    )
    parser.add_argument(
        '--quantity',
        '-n',
        type=int,
        default=1,
        help='Number of images to generate (default: 1)'
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.quantity == 1:
        result = generate_image_with_reference(
            prompt=args.prompt,
            output_path=str(output_path),
            reference_image1=args.reference_image1,
            quality=args.quality,
            api_key=args.api_key
        )
        sys.exit(0 if result else 1)

    # Multiple images - run in parallel
    print(f"\n>>> Generating {args.quantity} images in parallel...")
    success_count = 0
    fail_count = 0

    def generate_one(index: int) -> tuple[int, bool]:
        """Generate a single image and return (index, success)."""
        gen_result = generate_image_with_reference(
            prompt=args.prompt,
            output_path=str(output_path),
            reference_image1=args.reference_image1,
            quality=args.quality,
            api_key=args.api_key
        )
        return (index, gen_result)

    # Use a thread pool to generate images in parallel
    with ThreadPoolExecutor(max_workers=args.quantity) as executor:
        futures = [executor.submit(generate_one, i) for i in range(args.quantity)]

        for future in as_completed(futures):
            _index, result = future.result()
            if result:
                success_count += 1
            else:
                fail_count += 1

    print(f"\n{'=' * 60}")
    print(f"Generation complete: {success_count} succeeded, {fail_count} failed")
    print('=' * 60 + '\n')

    sys.exit(0 if success_count > 0 else 1)


if __name__ == '__main__':
    main()
