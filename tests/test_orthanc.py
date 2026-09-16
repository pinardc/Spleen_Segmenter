import zipfile
from pathlib import Path

import pytest

from spleen_segmenter.orthanc import safe_case_id, safe_extract_zip


def test_safe_case_id_normalizes_and_rejects_empty_values() -> None:
    assert safe_case_id(" dog 01 / venous ") == "dog_01_venous"
    with pytest.raises(ValueError):
        safe_case_id("../")


def test_safe_extract_zip_rejects_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../../outside.dcm", b"not dicom")

    with pytest.raises(ValueError, match="Unsafe ZIP"):
        safe_extract_zip(archive_path, tmp_path / "output")

    assert not (tmp_path / "outside.dcm").exists()
