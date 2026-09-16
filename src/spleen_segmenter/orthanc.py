"""Privacy-conscious Orthanc CT import helpers."""

from __future__ import annotations

import csv
import json
import re
import shutil
import stat
import subprocess
import tempfile
import zipfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_SAFE_CASE_ID = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True, slots=True)
class SeriesRecord:
    """Non-identifying subset of Orthanc series metadata."""

    orthanc_series_id: str
    modality: str
    series_description: str
    series_number: str
    instance_count: int | None


@dataclass(frozen=True, slots=True)
class ImportRecord:
    case_id: str
    orthanc_series_id: str
    ct_path: str


def safe_case_id(value: str) -> str:
    """Normalize an identifier for filenames without exposing DICOM tags."""
    normalized = _SAFE_CASE_ID.sub("_", value.strip()).strip("._-")
    if not normalized or normalized in {".", ".."}:
        raise ValueError("case_id must contain at least one letter or digit")
    if len(normalized) > 128:
        raise ValueError("case_id must be 128 characters or fewer")
    return normalized


class OrthancClient:
    """Minimal Orthanc REST client that never logs credentials."""

    def __init__(
        self,
        base_url: str,
        *,
        username: str = "",
        password: str = "",
        verify_tls: bool = True,
        timeout_seconds: int = 120,
    ) -> None:
        if bool(username) != bool(password):
            raise ValueError("Orthanc username and password must be supplied together")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        self.session.verify = verify_tls
        if username:
            self.session.auth = (username, password)
        retries = Retry(
            total=3,
            connect=3,
            read=3,
            status=3,
            backoff_factor=0.5,
            status_forcelist=(429, 502, 503, 504),
            allowed_methods=frozenset({"GET", "POST"}),
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        self.session.mount("http://", HTTPAdapter(max_retries=retries))

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        response = self.session.request(
            method,
            f"{self.base_url}/{path.lstrip('/')}",
            timeout=self.timeout_seconds,
            **kwargs,
        )
        response.raise_for_status()
        return response

    def system_info(self) -> dict[str, str]:
        """Return a strict, non-identifying subset of `/system`."""
        payload = self._request("GET", "/system").json()
        return {
            key: str(payload[key])
            for key in ("Name", "Version", "ApiVersion")
            if key in payload
        }

    def find_ct_series(
        self,
        query: dict[str, str] | None = None,
        *,
        limit: int = 100,
    ) -> list[SeriesRecord]:
        """Find CT series while returning only an allow-list of series fields."""
        if limit <= 0:
            raise ValueError("limit must be positive")
        dicom_query = {"Modality": "CT"}
        dicom_query.update(query or {})
        payload = {
            "Level": "Series",
            "Query": dicom_query,
            "Expand": True,
            "Limit": limit,
        }
        matches = self._request("POST", "/tools/find", json=payload).json()
        records: list[SeriesRecord] = []
        for item in matches:
            tags = item.get("MainDicomTags", {})
            count = item.get("CountInstances")
            if count is None and isinstance(item.get("Instances"), list):
                count = len(item["Instances"])
            records.append(
                SeriesRecord(
                    orthanc_series_id=str(item["ID"]),
                    modality=str(tags.get("Modality", "")),
                    series_description=str(tags.get("SeriesDescription", "")),
                    series_number=str(tags.get("SeriesNumber", "")),
                    instance_count=int(count) if count is not None else None,
                )
            )
        return records

    def download_series_archive(
        self,
        series_id: str,
        destination: str | Path,
        *,
        chunk_size: int = 1024 * 1024,
    ) -> Path:
        """Stream a series ZIP archive to disk without buffering it in memory."""
        destination_path = Path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        with self._request(
            "GET",
            f"/series/{series_id}/archive",
            stream=True,
        ) as response, destination_path.open("wb") as output:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    output.write(chunk)
        if not zipfile.is_zipfile(destination_path):
            destination_path.unlink(missing_ok=True)
            raise ValueError("Orthanc response was not a valid ZIP archive")
        return destination_path


def _validated_members(archive: zipfile.ZipFile) -> Iterable[zipfile.ZipInfo]:
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        mode = member.external_attr >> 16
        if (
            path.is_absolute()
            or ".." in path.parts
            or member.flag_bits & 0x1
            or stat.S_ISLNK(mode)
        ):
            raise ValueError(f"Unsafe ZIP member rejected: {member.filename!r}")
        yield member


def safe_extract_zip(zip_path: str | Path, destination: str | Path) -> Path:
    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for member in _validated_members(archive):
            archive.extract(member, destination_path)
    return destination_path


def convert_dicom_directory(
    dicom_dir: str | Path,
    output_path: str | Path,
    *,
    executable: str = "dcm2niix",
) -> Path:
    """Convert exactly one DICOM series to a gzipped NIfTI atomically."""
    binary = shutil.which(executable)
    if not binary:
        raise FileNotFoundError(
            f"{executable!r} was not found; install dcm2niix before importing CTs"
        )
    output = Path(output_path)
    if output.name.endswith(".nii.gz"):
        stem = output.name[:-7]
    elif output.suffix == ".nii":
        stem = output.stem
    else:
        raise ValueError("output_path must end in .nii or .nii.gz")
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=".dcm2niix-", dir=output.parent) as tmp:
        tmp_path = Path(tmp)
        command = [
            binary,
            "-b",
            "n",
            "-f",
            stem,
            "-z",
            "y" if output.name.endswith(".gz") else "n",
            "-o",
            str(tmp_path),
            str(Path(dicom_dir)),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        candidates = sorted(tmp_path.glob(f"{stem}*.nii*"))
        if len(candidates) != 1:
            raise ValueError(
                "Expected one CT volume from the selected Orthanc series; "
                f"dcm2niix produced {len(candidates)}"
            )
        candidates[0].replace(output)
    return output


def append_local_manifest(record: ImportRecord, manifest_path: str | Path) -> None:
    """Append an import record to an ignored local CSV manifest."""
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(record)))
        if not exists:
            writer.writeheader()
        writer.writerow(asdict(record))


def import_series(
    client: OrthancClient,
    series_id: str,
    output_dir: str | Path,
    *,
    case_id: str | None = None,
    manifest_path: str | Path | None = None,
) -> ImportRecord:
    """Download and convert one Orthanc series, deleting temporary DICOM data."""
    normalized_id = safe_case_id(case_id or series_id)
    output_path = Path(output_dir) / f"{normalized_id}.nii.gz"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing CT: {output_path}")

    work_root = Path(output_dir).parent / ".imports"
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{normalized_id}-", dir=work_root) as tmp:
        temp_path = Path(tmp)
        archive_path = client.download_series_archive(
            series_id,
            temp_path / "series.zip",
        )
        dicom_dir = safe_extract_zip(archive_path, temp_path / "dicom")
        convert_dicom_directory(dicom_dir, output_path)

    record = ImportRecord(
        case_id=normalized_id,
        orthanc_series_id=series_id,
        ct_path=str(output_path),
    )
    if manifest_path:
        append_local_manifest(record, manifest_path)
    return record


def write_records_json(records: Iterable[ImportRecord], output: BinaryIO) -> None:
    """Serialize non-identifying records for an approved local workflow."""
    payload = [asdict(record) for record in records]
    output.write(json.dumps(payload, indent=2).encode())
