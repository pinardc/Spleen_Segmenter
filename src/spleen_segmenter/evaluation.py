"""NIfTI pairing, geometry validation, and binary segmentation metrics."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy import ndimage


@dataclass(frozen=True, slots=True)
class CasePaths:
    case_id: str
    ct_path: Path
    reference_path: Path
    prediction_path: Path | None = None


@dataclass(frozen=True, slots=True)
class PairingResult:
    cases: tuple[CasePaths, ...]
    missing_references: tuple[str, ...]
    orphan_references: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Geometry:
    shape: tuple[int, ...]
    spacing_mm: tuple[float, ...]
    orientation: tuple[str | None, ...]
    affine: tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class Metrics:
    case_id: str
    dice: float
    jaccard: float
    precision: float
    recall: float
    hausdorff95_mm: float
    assd_mm: float
    reference_volume_ml: float
    prediction_volume_ml: float
    volume_bias_percent: float
    prediction_resampled: bool

    def to_dict(self) -> dict[str, str | float | bool]:
        return asdict(self)


def nifti_stem(path: str | Path) -> str:
    name = Path(path).name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    raise ValueError(f"Not a NIfTI filename: {name}")


def _index_nifti(directory: str | Path) -> dict[str, Path]:
    directory_path = Path(directory)
    paths = sorted((*directory_path.glob("*.nii"), *directory_path.glob("*.nii.gz")))
    indexed: dict[str, Path] = {}
    for path in paths:
        stem = nifti_stem(path)
        if stem in indexed:
            raise ValueError(
                f"Duplicate NIfTI basename {stem!r}: {indexed[stem]} and {path}"
            )
        indexed[stem] = path
    return indexed


def pair_cases(
    ct_dir: str | Path,
    reference_dir: str | Path,
    prediction_dir: str | Path | None = None,
) -> PairingResult:
    """Pair CTs and masks by exact `.nii[.gz]` basename."""
    ct_by_id = _index_nifti(ct_dir)
    reference_by_id = _index_nifti(reference_dir)
    prediction_by_id = _index_nifti(prediction_dir) if prediction_dir else {}
    shared = sorted(ct_by_id.keys() & reference_by_id.keys())
    cases = tuple(
        CasePaths(
            case_id=case_id,
            ct_path=ct_by_id[case_id],
            reference_path=reference_by_id[case_id],
            prediction_path=prediction_by_id.get(case_id),
        )
        for case_id in shared
    )
    return PairingResult(
        cases=cases,
        missing_references=tuple(sorted(ct_by_id.keys() - reference_by_id.keys())),
        orphan_references=tuple(sorted(reference_by_id.keys() - ct_by_id.keys())),
    )


def geometry(image: nib.spatialimages.SpatialImage) -> Geometry:
    ndim = len(image.shape)
    return Geometry(
        shape=tuple(int(value) for value in image.shape),
        spacing_mm=tuple(float(value) for value in image.header.get_zooms()[:ndim]),
        orientation=tuple(nib.aff2axcodes(image.affine)),
        affine=tuple(tuple(float(value) for value in row) for row in image.affine),
    )


def geometry_matches(
    first: nib.spatialimages.SpatialImage,
    second: nib.spatialimages.SpatialImage,
    *,
    affine_tolerance: float = 1e-4,
) -> bool:
    return first.shape == second.shape and np.allclose(
        first.affine,
        second.affine,
        atol=affine_tolerance,
        rtol=0,
    )


def validate_binary_mask(
    image: nib.spatialimages.SpatialImage,
    *,
    name: str = "mask",
) -> np.ndarray:
    data = np.asanyarray(image.dataobj)
    if data.ndim != 3:
        raise ValueError(f"{name} must be 3D, got shape {data.shape}")
    if not np.isfinite(data).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    unique = np.unique(data)
    if not np.all(np.isin(unique, (0, 1))):
        raise ValueError(f"{name} must contain only 0 and 1, got {unique[:10]}")
    return data.astype(bool, copy=False)


def _surface_distances(
    source: np.ndarray,
    target: np.ndarray,
    spacing_mm: tuple[float, float, float],
) -> np.ndarray:
    source_surface = source ^ ndimage.binary_erosion(source)
    target_surface = target ^ ndimage.binary_erosion(target)
    distance_to_target = ndimage.distance_transform_edt(
        ~target_surface,
        sampling=spacing_mm,
    )
    return distance_to_target[source_surface]


def binary_metrics(
    reference: np.ndarray,
    prediction: np.ndarray,
    spacing_mm: Iterable[float],
    *,
    case_id: str,
    prediction_resampled: bool = False,
) -> Metrics:
    """Calculate overlap, surface-distance, and volume metrics."""
    reference = np.asarray(reference, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    if reference.shape != prediction.shape or reference.ndim != 3:
        raise ValueError("Reference and prediction must be same-shape 3D arrays")
    spacing = tuple(float(value) for value in spacing_mm)
    if len(spacing) != 3 or any(value <= 0 for value in spacing):
        raise ValueError("spacing_mm must contain three positive values")

    true_positive = int(np.count_nonzero(reference & prediction))
    reference_count = int(np.count_nonzero(reference))
    prediction_count = int(np.count_nonzero(prediction))
    union_count = int(np.count_nonzero(reference | prediction))

    size_sum = reference_count + prediction_count
    dice = 1.0 if size_sum == 0 else 2.0 * true_positive / size_sum
    jaccard = 1.0 if union_count == 0 else true_positive / union_count
    precision = (
        float("nan") if prediction_count == 0 else true_positive / prediction_count
    )
    recall = float("nan") if reference_count == 0 else true_positive / reference_count

    if reference_count == 0 and prediction_count == 0:
        hausdorff95 = 0.0
        assd = 0.0
    elif reference_count == 0 or prediction_count == 0:
        hausdorff95 = float("inf")
        assd = float("inf")
    else:
        distances = np.concatenate(
            (
                _surface_distances(reference, prediction, spacing),
                _surface_distances(prediction, reference, spacing),
            )
        )
        hausdorff95 = float(np.percentile(distances, 95))
        assd = float(np.mean(distances))

    voxel_volume_ml = float(np.prod(spacing)) / 1000.0
    reference_volume = reference_count * voxel_volume_ml
    prediction_volume = prediction_count * voxel_volume_ml
    volume_bias = (
        float("nan")
        if reference_volume == 0
        else 100.0 * (prediction_volume - reference_volume) / reference_volume
    )
    return Metrics(
        case_id=case_id,
        dice=float(dice),
        jaccard=float(jaccard),
        precision=float(precision),
        recall=float(recall),
        hausdorff95_mm=hausdorff95,
        assd_mm=assd,
        reference_volume_ml=reference_volume,
        prediction_volume_ml=prediction_volume,
        volume_bias_percent=volume_bias,
        prediction_resampled=prediction_resampled,
    )


def evaluate_case(
    case_id: str,
    reference_path: str | Path,
    prediction_path: str | Path,
    *,
    allow_resample: bool = False,
) -> tuple[Metrics, Geometry, Geometry]:
    """Validate and compare one prediction against its radiologist reference."""
    reference_image = nib.load(str(reference_path))
    prediction_image = nib.load(str(prediction_path))
    reference_geometry = geometry(reference_image)
    prediction_geometry = geometry(prediction_image)
    was_resampled = False
    if not geometry_matches(reference_image, prediction_image):
        if not allow_resample:
            raise ValueError(
                f"Geometry mismatch for {case_id}; set allow_resample=True "
                "only after reviewing orientation and registration"
            )
        prediction_image = resample_from_to(
            prediction_image,
            (reference_image.shape, reference_image.affine),
            order=0,
        )
        was_resampled = True

    reference = validate_binary_mask(reference_image, name=f"{case_id} reference")
    prediction = validate_binary_mask(prediction_image, name=f"{case_id} prediction")
    metrics = binary_metrics(
        reference,
        prediction,
        reference_image.header.get_zooms()[:3],
        case_id=case_id,
        prediction_resampled=was_resampled,
    )
    return metrics, reference_geometry, prediction_geometry
