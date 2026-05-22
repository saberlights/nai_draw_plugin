import base64
import importlib.util
import os
import sys
from pathlib import Path


def _load_image_url_helper_module():
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    module_path = Path(__file__).resolve().parents[1] / "core" / "utils" / "image_url_helper.py"
    spec = importlib.util.spec_from_file_location("test_image_url_helper_module", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_save_base64_image_to_file_uses_magic_bytes_for_png_extension(
    monkeypatch, tmp_path: Path
) -> None:
    image_url_helper = _load_image_url_helper_module()
    image_bytes = b"\x89PNG\r\n\x1a\n" + os.urandom(16)

    monkeypatch.setattr(image_url_helper, "_IMAGE_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(image_url_helper, "_last_cleanup_ts", 0.0)

    file_path = image_url_helper.save_base64_image_to_file(base64.b64encode(image_bytes).decode("ascii"))

    assert file_path is not None
    assert file_path.endswith(".png")
    assert Path(file_path).read_bytes() == image_bytes


def test_save_base64_image_to_file_uses_data_uri_type_when_bytes_are_unknown(
    monkeypatch, tmp_path: Path
) -> None:
    image_url_helper = _load_image_url_helper_module()
    image_bytes = b"not-a-real-jpeg-but-header-says-so"
    data_uri = f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('ascii')}"

    monkeypatch.setattr(image_url_helper, "_IMAGE_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(image_url_helper, "_last_cleanup_ts", 0.0)

    file_path = image_url_helper.save_base64_image_to_file(data_uri)

    assert file_path is not None
    assert file_path.endswith(".jpg")
    assert Path(file_path).read_bytes() == image_bytes


def test_save_base64_image_to_file_falls_back_to_png_for_empty_bytes(
    monkeypatch, tmp_path: Path
) -> None:
    image_url_helper = _load_image_url_helper_module()

    monkeypatch.setattr(image_url_helper, "_IMAGE_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(image_url_helper, "_last_cleanup_ts", 0.0)

    file_path = image_url_helper.save_base64_image_to_file("")

    assert file_path is not None
    assert file_path.endswith(".png")
    assert Path(file_path).read_bytes() == b""
