"""
Tests for image_gen_utils, generate_image, and generate_image_single_refv2.
Covers utility functions, error handling, API interaction (mocked), and CLI parsing.
"""

import base64
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import image_gen_utils
from image_gen_utils import (
    get_mime_type,
    load_image_as_base64,
    unique_output_path,
    cleanup_placeholder,
    load_reference,
    extract_image_from_response,
    print_header,
    call_api_and_save,
    resolve_api_key,
    resolve_output_path,
    API_URL,
    _DEFAULT_MIME_TYPE,
)


# ── get_mime_type ──────────────────────────────────────────────────────────

class TestGetMimeType:
    def test_known_extensions(self):
        assert get_mime_type("photo.jpg") == "image/jpeg"
        assert get_mime_type("photo.jpeg") == "image/jpeg"
        assert get_mime_type("photo.png") == "image/png"
        assert get_mime_type("photo.gif") == "image/gif"
        assert get_mime_type("photo.bmp") == "image/bmp"
        assert get_mime_type("photo.tiff") == "image/tiff"
        assert get_mime_type("photo.tif") == "image/tiff"
        assert get_mime_type("photo.webp") == "image/webp"

    def test_unknown_extension_returns_default(self):
        assert get_mime_type("file.zzzzunknown") == _DEFAULT_MIME_TYPE

    def test_mimetypes_module_resolution(self):
        result = get_mime_type("test.png")
        assert result == "image/png"

    def test_default_constant_value(self):
        assert _DEFAULT_MIME_TYPE == "image/jpeg"


# ── load_image_as_base64 ──────────────────────────────────────────────────

class TestLoadImageAsBase64:
    def test_loads_and_encodes_file(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
            f.flush()
            path = f.name
        try:
            data, mime = load_image_as_base64(path)
            assert isinstance(data, str)
            decoded = base64.b64decode(data)
            assert decoded.startswith(b"\x89PNG")
            assert mime == "image/png"
        finally:
            os.unlink(path)

    def test_jpeg_mime_type(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"\xff\xd8\xff" + b"\x00" * 50)
            f.flush()
            path = f.name
        try:
            _, mime = load_image_as_base64(path)
            assert mime == "image/jpeg"
        finally:
            os.unlink(path)


# ── unique_output_path ────────────────────────────────────────────────────

class TestUniqueOutputPath:
    def test_returns_original_if_not_exists(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "output.png")
            result = unique_output_path(path)
            assert result == path
            assert os.path.exists(result)

    def test_returns_different_name_if_exists(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "output.png")
            Path(path).touch()
            result = unique_output_path(path)
            assert result != path
            assert result.startswith(os.path.join(d, "output_"))
            assert result.endswith(".png")
            assert os.path.exists(result)

    def test_reserves_file_atomically(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test.png")
            r1 = unique_output_path(path)
            r2 = unique_output_path(path)
            assert r1 != r2
            assert os.path.exists(r1)
            assert os.path.exists(r2)


# ── cleanup_placeholder ──────────────────────────────────────────────────

class TestCleanupPlaceholder:
    def test_removes_existing_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            path = f.name
        assert os.path.exists(path)
        cleanup_placeholder(path)
        assert not os.path.exists(path)

    def test_no_error_if_missing(self):
        cleanup_placeholder("/tmp/nonexistent_placeholder_12345.png")


# ── load_reference ────────────────────────────────────────────────────────

class TestLoadReference:
    def test_returns_none_for_missing_file(self):
        result = load_reference("/nonexistent/file.png", "test ref")
        assert result is None

    def test_returns_inline_data_for_existing_file(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG" + b"\x00" * 50)
            f.flush()
            path = f.name
        try:
            result = load_reference(path, "test ref")
            assert result is not None
            assert "inline_data" in result
            assert result["inline_data"]["mime_type"] == "image/png"
            assert len(result["inline_data"]["data"]) > 0
        finally:
            os.unlink(path)


# ── extract_image_from_response ──────────────────────────────────────────

class TestExtractImageFromResponse:
    def test_empty_response(self):
        assert extract_image_from_response({}, "/tmp/out.png") is False

    def test_empty_candidates(self):
        assert extract_image_from_response({"candidates": []}, "/tmp/out.png") is False

    def test_no_inline_data(self):
        result = {"candidates": [{"content": {"parts": [{"text": "hello"}]}}]}
        assert extract_image_from_response(result, "/tmp/out.png") is False

    def test_extracts_and_saves_image(self):
        img_bytes = b"\x89PNG\r\n\x1a\ntest_image_data"
        encoded = base64.b64encode(img_bytes).decode("utf-8")
        result = {
            "candidates": [{
                "content": {
                    "parts": [{"inlineData": {"data": encoded, "mimeType": "image/png"}}]
                }
            }]
        }
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            assert extract_image_from_response(result, path) is True
            with open(path, "rb") as f:
                assert f.read() == img_bytes
        finally:
            os.unlink(path)

    def test_missing_content_key(self):
        result = {"candidates": [{"finishReason": "STOP"}]}
        assert extract_image_from_response(result, "/tmp/out.png") is False


# ── print_header ──────────────────────────────────────────────────────────

class TestPrintHeader:
    def test_prints_without_references(self, capsys):
        print_header("test prompt", "/tmp/out.png", "2k")
        captured = capsys.readouterr()
        assert "test prompt" in captured.out
        assert "/tmp/out.png" in captured.out
        assert "2K" in captured.out

    def test_prints_with_references(self, capsys):
        refs = [("Ref 1", "image1.png"), ("Ref 2", "image2.png")]
        print_header("prompt", "/tmp/out.png", "4k", refs)
        captured = capsys.readouterr()
        assert "Ref 1: image1.png" in captured.out
        assert "Ref 2: image2.png" in captured.out
        assert "4K" in captured.out


# ── resolve_api_key ───────────────────────────────────────────────────────

class TestResolveApiKey:
    def test_returns_provided_key(self):
        assert resolve_api_key("my-key-123") == "my-key-123"

    def test_returns_env_var(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "env-key-456"}):
            assert resolve_api_key(None) == "env-key-456"

    def test_returns_none_when_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GEMINI_API_KEY", None)
            assert resolve_api_key(None) is None


# ── resolve_output_path ──────────────────────────────────────────────────

class TestResolveOutputPath:
    def test_returns_path_if_not_exists(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "new_file.png")
            result = resolve_output_path(path)
            assert result == path

    def test_returns_unique_if_exists(self, capsys):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "existing.png")
            Path(path).touch()
            result = resolve_output_path(path)
            assert result != path
            captured = capsys.readouterr()
            assert "Output file exists" in captured.out


# ── API_URL ───────────────────────────────────────────────────────────────

class TestApiUrl:
    def test_contains_model_name(self):
        assert "gemini-3-pro-image-preview" in API_URL

    def test_has_key_placeholder(self):
        assert "{}" in API_URL

    def test_format_with_key(self):
        url = API_URL.format("test-key")
        assert "key=test-key" in url


# ── call_api_and_save ─────────────────────────────────────────────────────

class TestCallApiAndSave:
    def _make_success_response(self):
        img_bytes = b"\x89PNG_test"
        encoded = base64.b64encode(img_bytes).decode("utf-8")
        return {
            "candidates": [{
                "content": {
                    "parts": [{"inlineData": {"data": encoded, "mimeType": "image/png"}}]
                }
            }]
        }

    @patch("image_gen_utils.requests.post")
    def test_successful_generation(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = self._make_success_response()
        mock_post.return_value = mock_resp

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            result = call_api_and_save([{"text": "test"}], "2k", path, "fake-key")
            assert result is True
            assert os.path.exists(path)
            with open(path, "rb") as f:
                assert f.read() == b"\x89PNG_test"
        finally:
            if os.path.exists(path):
                os.unlink(path)

    @patch("image_gen_utils.requests.post")
    def test_4k_quality(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = self._make_success_response()
        mock_post.return_value = mock_resp

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            call_api_and_save([{"text": "test"}], "4k", path, "fake-key")
            call_args = mock_post.call_args
            payload = call_args.kwargs.get("json") or call_args[1].get("json")
            assert payload["generationConfig"]["imageConfig"]["imageSize"] == "4K"
        finally:
            if os.path.exists(path):
                os.unlink(path)

    @patch("image_gen_utils.requests.post")
    def test_api_error_status(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.text = '{"error": "bad request"}'
        mock_post.return_value = mock_resp

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        result = call_api_and_save([{"text": "test"}], "2k", path, "fake-key")
        assert result is False
        assert not os.path.exists(path)

    @patch("image_gen_utils.requests.post")
    def test_timeout_error(self, mock_post):
        import requests as req
        mock_post.side_effect = req.exceptions.Timeout("timed out")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        result = call_api_and_save([{"text": "test"}], "2k", path, "fake-key")
        assert result is False
        assert not os.path.exists(path)

    @patch("image_gen_utils.requests.post")
    def test_generic_exception(self, mock_post):
        mock_post.side_effect = ConnectionError("network error")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        result = call_api_and_save([{"text": "test"}], "2k", path, "fake-key")
        assert result is False
        assert not os.path.exists(path)

    @patch("image_gen_utils.requests.post")
    def test_no_image_in_response(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"candidates": [{"content": {"parts": [{"text": "sorry"}]}}]}
        mock_post.return_value = mock_resp

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        result = call_api_and_save([{"text": "test"}], "2k", path, "fake-key")
        assert result is False


# ── generate_image.py integration ────────────────────────────────────────

class TestGenerateImageScript:
    @patch("image_gen_utils.requests.post")
    def test_generate_with_no_refs(self, mock_post):
        from generate_image import generate_image_with_references
        img = base64.b64encode(b"fake_png").decode()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [{"content": {"parts": [{"inlineData": {"data": img, "mimeType": "image/png"}}]}}]
        }
        mock_post.return_value = mock_resp

        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "test_out.png")
            result = generate_image_with_references("test prompt", out, api_key="fake")
            assert result is True
            assert os.path.exists(out)

    @patch("image_gen_utils.requests.post")
    def test_generate_with_ref_images(self, mock_post):
        from generate_image import generate_image_with_references
        img = base64.b64encode(b"fake_png").decode()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [{"content": {"parts": [{"inlineData": {"data": img, "mimeType": "image/png"}}]}}]
        }
        mock_post.return_value = mock_resp

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as ref:
            ref.write(b"\x89PNG" + b"\x00" * 50)
            ref.flush()
            ref_path = ref.name

        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "test_out.png")
            result = generate_image_with_references("test", out, reference_image1=ref_path, api_key="fake")
            assert result is True
        os.unlink(ref_path)

    def test_missing_ref_image(self):
        from generate_image import generate_image_with_references
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "test_out.png")
            result = generate_image_with_references("test", out, reference_image1="/no/such/file.png", api_key="fake")
            assert result is False

    def test_no_api_key(self):
        from generate_image import generate_image_with_references
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GEMINI_API_KEY", None)
            result = generate_image_with_references("test", "/tmp/out.png")
            assert result is False


# ── generate_image_single_refv2.py integration ──────────────────────────

class TestSingleRefScript:
    def test_build_single_reference_prompt(self):
        from generate_image_single_refv2 import build_single_reference_prompt
        result = build_single_reference_prompt("wearing a hat")
        assert "wearing a hat" in result
        assert "KEEP CONSISTENT" in result
        assert "character identity" in result

    def test_build_prompt_empty_input(self):
        from generate_image_single_refv2 import build_single_reference_prompt
        result = build_single_reference_prompt("")
        assert "KEEP CONSISTENT" in result

    def test_build_prompt_none_input(self):
        from generate_image_single_refv2 import build_single_reference_prompt
        result = build_single_reference_prompt(None)
        assert "KEEP CONSISTENT" in result

    @patch("image_gen_utils.requests.post")
    def test_generate_text_only(self, mock_post):
        from generate_image_single_refv2 import generate_image_with_reference
        img = base64.b64encode(b"fake_png").decode()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [{"content": {"parts": [{"inlineData": {"data": img, "mimeType": "image/png"}}]}}]
        }
        mock_post.return_value = mock_resp

        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "test_out.png")
            result = generate_image_with_reference("a sunset", out, api_key="fake")
            assert result is True

    @patch("image_gen_utils.requests.post")
    def test_generate_with_reference(self, mock_post):
        from generate_image_single_refv2 import generate_image_with_reference
        img = base64.b64encode(b"fake_png").decode()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [{"content": {"parts": [{"inlineData": {"data": img, "mimeType": "image/png"}}]}}]
        }
        mock_post.return_value = mock_resp

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as ref:
            ref.write(b"\x89PNG" + b"\x00" * 50)
            ref.flush()
            ref_path = ref.name

        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "test_out.png")
            result = generate_image_with_reference("same but blue", out, reference_image1=ref_path, api_key="fake")
            assert result is True
            # Verify the prompt was wrapped with identity preservation
            call_args = mock_post.call_args
            payload = call_args.kwargs.get("json") or call_args[1].get("json")
            text_part = payload["contents"][0]["parts"][0]["text"]
            assert "KEEP CONSISTENT" in text_part
        os.unlink(ref_path)

    def test_missing_ref_returns_false(self):
        from generate_image_single_refv2 import generate_image_with_reference
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "test_out.png")
            result = generate_image_with_reference("test", out, reference_image1="/no/file.png", api_key="fake")
            assert result is False
