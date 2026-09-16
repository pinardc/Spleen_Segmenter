"""MONAI Label REST client for deterministic file-based inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class MonaiLabelClient:
    def __init__(self, base_url: str, *, timeout_seconds: int = 1800) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        retries = Retry(
            total=2,
            connect=2,
            read=0,
            status=2,
            backoff_factor=1,
            status_forcelist=(502, 503, 504),
            allowed_methods=frozenset({"GET", "POST"}),
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def info(self) -> dict[str, Any]:
        response = self.session.get(
            f"{self.base_url}/info/",
            timeout=min(self.timeout_seconds, 30),
        )
        response.raise_for_status()
        return response.json()

    def require_model(self, model: str = "segmentation_spleen") -> dict[str, Any]:
        info = self.info()
        models = info.get("models", {})
        available = set(models) if isinstance(models, dict) else set(models or [])
        if model not in available:
            raise RuntimeError(
                f"MONAI Label model {model!r} is unavailable; available: {sorted(available)}"
            )
        return info

    def infer(
        self,
        image_path: str | Path,
        output_path: str | Path,
        *,
        model: str = "segmentation_spleen",
        device: str | None = None,
        overwrite: bool = False,
    ) -> Path:
        image = Path(image_path)
        output = Path(output_path)
        if not image.is_file():
            raise FileNotFoundError(image)
        if output.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite prediction: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_name(f".{output.name}.partial")
        partial.unlink(missing_ok=True)

        params = {"device": device} if device else {}
        try:
            with image.open("rb") as stream:
                response = self.session.post(
                    f"{self.base_url}/infer/{model}",
                    params={"output": "image"},
                    data={"params": json.dumps(params)},
                    files={"file": (image.name, stream, "application/gzip")},
                    timeout=self.timeout_seconds,
                    stream=True,
                )
                response.raise_for_status()
                with partial.open("wb") as destination:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            destination.write(chunk)
            if partial.stat().st_size == 0:
                raise RuntimeError("MONAI Label returned an empty prediction")
            partial.replace(output)
        except Exception:
            partial.unlink(missing_ok=True)
            raise
        return output
