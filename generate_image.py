#!/usr/bin/env python3
"""
Generate images using Google Gemini 3 Pro Image API (Nano Banana Pro).
Supports reference images for style and content transfer.
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


def generate_image_with_references(
    prompt: str,
    output_path: str,
    reference_image1: Optional[str] = None,
    reference_image2: Optional[str] = None,
    quality: str = "2k",
    api_key: Optional[str] = None
) -> bool:
    """
    Generate an image using Google Gemini 3 Pro Image API with optional reference images.

    This function uses the latest Gemini 3 Pro Image model (gemini-3-pro-image-preview)
    which supports:
    - Up to 14 reference images (we support 2 in this script)
    - 2K and 4K native resolution
    - Image editing and style transfer

    Args:
        prompt: Text description of the image to generate (this is your main creative prompt)
        output_path: Path where the generated image will be saved
        reference_image1: Path to first reference image (use this for content/subject reference)
        reference_image2: Path to second reference image (use this for style reference)
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
        references.append(("Reference Image 1 (Content)", reference_image1))
    if reference_image2:
        references.append(("Reference Image 2 (Style)", reference_image2))
    print_header(prompt, output_path, quality, references or None)

    parts = [{"text": prompt}]
    for ref_path, label in [(reference_image1, "reference image 1"),
                            (reference_image2, "reference image 2")]:
        if ref_path is None:
            continue
        ref_part = load_reference(ref_path, label)
        if ref_part is None:
            cleanup_placeholder(output_path)
            return False
        parts.append(ref_part)

    return call_api_and_save(parts, quality, output_path, api_key)


def main():
    parser = argparse.ArgumentParser(
        description='Generate images using Google Gemini 3 Pro Image API with reference image support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  1. Generate from text prompt only (2K quality):
     python generate_image.py "A serene mountain landscape at sunset" output.png

  2. Generate from text prompt (4K quality):
     python generate_image.py "A serene mountain landscape at sunset" output.png --quality 4k

  3. Use one reference image for content:
     python generate_image.py "Make this more vibrant and colorful" output.png --ref1 original.jpg

  4. Use two reference images (content + style transfer):
     python generate_image.py "Combine these elements with the style of the second image" output.png \\
         --ref1 content_photo.jpg --ref2 style_reference.jpg --quality 4k

  5. Custom prompts for reference images:
     python generate_image.py "Create an illustration using the subject from ref1 in the artistic style of ref2" \\
         output.png --ref1 subject.jpg --ref2 art_style.jpg

Supported input formats:
  - PNG, JPG, JPEG, GIF, BMP, TIFF, WebP (any format supported by PIL/Pillow)

Tips:
  - Use --ref1 for the main content/subject you want to reference
  - Use --ref2 for the style you want to apply
  - Be specific in your prompts about how to use the reference images
  - 4K costs about 1.8x more than 2K but provides higher resolution
  - You can use up to 14 reference images with the API (script supports 2)
        """
    )

    parser.add_argument(
        'prompt',
        help='MAIN CREATIVE PROMPT: Text description of the image to generate. '
             'Be specific about what you want and how to use reference images if provided.'
    )
    parser.add_argument(
        'output',
        help='Output file path (e.g., output.png, result.jpg)'
    )
    parser.add_argument(
        '--ref1',
        '--reference1',
        dest='reference_image1',
        help='REFERENCE IMAGE 1: Use this for content/subject reference. '
             'The model will use this as a visual reference for the main subject or content. '
             'Supports PNG, JPG, JPEG, GIF, BMP, TIFF, WebP formats.'
    )
    parser.add_argument(
        '--ref2',
        '--reference2',
        dest='reference_image2',
        help='REFERENCE IMAGE 2: Use this for style reference. '
             'The model will apply the artistic style, color palette, or aesthetic from this image. '
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
        result = generate_image_with_references(
            prompt=args.prompt,
            output_path=str(output_path),
            reference_image1=args.reference_image1,
            reference_image2=args.reference_image2,
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
        gen_result = generate_image_with_references(
            prompt=args.prompt,
            output_path=str(output_path),
            reference_image1=args.reference_image1,
            reference_image2=args.reference_image2,
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
