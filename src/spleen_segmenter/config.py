"""Environment-backed configuration without secret logging."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _as_bool(value: str, *, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


def _as_positive_int(value: str, *, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _path(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings. The representation intentionally omits credentials."""

    root: Path
    orthanc_url: str
    orthanc_username: str
    orthanc_password: str
    orthanc_verify_tls: bool
    orthanc_timeout_seconds: int
    monai_label_url: str
    monai_label_timeout_seconds: int
    ct_nifti_dir: Path
    radiologist_mask_dir: Path
    prediction_dir: Path
    report_dir: Path

    @classmethod
    def from_env(
        cls,
        root: str | Path = ".",
        *,
        env_file: str | Path | None = None,
    ) -> Settings:
        root_path = Path(root).expanduser().resolve()
        load_dotenv(env_file or root_path / ".env", override=False)

        orthanc_url = os.getenv("ORTHANC_URL", "").strip().rstrip("/")
        if not orthanc_url:
            raise ValueError("ORTHANC_URL is required; copy .env.example to .env")
        if not orthanc_url.startswith(("https://", "http://")):
            raise ValueError("ORTHANC_URL must begin with https:// or http://")

        return cls(
            root=root_path,
            orthanc_url=orthanc_url,
            orthanc_username=os.getenv("ORTHANC_USERNAME", ""),
            orthanc_password=os.getenv("ORTHANC_PASSWORD", ""),
            orthanc_verify_tls=_as_bool(
                os.getenv("ORTHANC_VERIFY_TLS", "true"),
                name="ORTHANC_VERIFY_TLS",
            ),
            orthanc_timeout_seconds=_as_positive_int(
                os.getenv("ORTHANC_TIMEOUT_SECONDS", "120"),
                name="ORTHANC_TIMEOUT_SECONDS",
            ),
            monai_label_url=os.getenv(
                "MONAI_LABEL_URL", "http://127.0.0.1:8000"
            ).rstrip("/"),
            monai_label_timeout_seconds=_as_positive_int(
                os.getenv("MONAI_LABEL_TIMEOUT_SECONDS", "1800"),
                name="MONAI_LABEL_TIMEOUT_SECONDS",
            ),
            ct_nifti_dir=_path(os.getenv("CT_NIFTI_DIR", "data/ct"), root_path),
            radiologist_mask_dir=_path(
                os.getenv("RADIOLOGIST_MASK_DIR", "data/radiologist_masks"),
                root_path,
            ),
            prediction_dir=_path(
                os.getenv("PREDICTION_DIR", "data/predictions"), root_path
            ),
            report_dir=_path(os.getenv("REPORT_DIR", "reports"), root_path),
        )

    def ensure_local_directories(self) -> None:
        for path in (
            self.ct_nifti_dir,
            self.radiologist_mask_dir,
            self.prediction_dir,
            self.report_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def public_summary(self) -> dict[str, object]:
        """Return safe configuration details suitable for notebook display."""
        return {
            "orthanc_host": self.orthanc_url.split("://", 1)[-1].split("/", 1)[0],
            "orthanc_verify_tls": self.orthanc_verify_tls,
            "monai_label_url": self.monai_label_url,
            "ct_nifti_dir": str(self.ct_nifti_dir),
            "radiologist_mask_dir": str(self.radiologist_mask_dir),
            "prediction_dir": str(self.prediction_dir),
            "report_dir": str(self.report_dir),
        }
