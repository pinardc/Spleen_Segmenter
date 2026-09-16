from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from spleen_segmenter.evaluation import (
    binary_metrics,
    evaluate_case,
    geometry_matches,
    pair_cases,
)


def _save(path: Path, data: np.ndarray, affine: np.ndarray | None = None) -> None:
    image_affine = np.eye(4) if affine is None else affine
    nib.save(nib.Nifti1Image(data.astype(np.uint8), image_affine), path)


def test_pair_cases_matches_nii_and_nii_gz_by_stem(tmp_path: Path) -> None:
    ct_dir = tmp_path / "ct"
    reference_dir = tmp_path / "reference"
    ct_dir.mkdir()
    reference_dir.mkdir()
    _save(ct_dir / "case-a.nii.gz", np.zeros((2, 2, 2)))
    _save(ct_dir / "missing.nii", np.zeros((2, 2, 2)))
    _save(reference_dir / "case-a.nii", np.zeros((2, 2, 2)))
    _save(reference_dir / "orphan.nii.gz", np.zeros((2, 2, 2)))

    result = pair_cases(ct_dir, reference_dir)

    assert [case.case_id for case in result.cases] == ["case-a"]
    assert result.missing_references == ("missing",)
    assert result.orphan_references == ("orphan",)


def test_binary_metrics_identical_masks() -> None:
    mask = np.zeros((8, 8, 8), dtype=bool)
    mask[2:6, 2:6, 2:6] = True

    metrics = binary_metrics(mask, mask, (1.0, 1.0, 2.0), case_id="same")

    assert metrics.dice == pytest.approx(1.0)
    assert metrics.jaccard == pytest.approx(1.0)
    assert metrics.hausdorff95_mm == pytest.approx(0.0)
    assert metrics.assd_mm == pytest.approx(0.0)
    assert metrics.reference_volume_ml == pytest.approx(0.128)
    assert metrics.volume_bias_percent == pytest.approx(0.0)


def test_geometry_mismatch_is_explicit_and_resampling_is_opt_in(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.nii.gz"
    prediction_path = tmp_path / "prediction.nii.gz"
    mask = np.zeros((6, 6, 6), dtype=np.uint8)
    mask[2:4, 2:4, 2:4] = 1
    _save(reference_path, mask)
    translated = np.eye(4)
    translated[0, 3] = 1
    _save(prediction_path, mask, translated)

    reference_image = nib.load(reference_path)
    prediction_image = nib.load(prediction_path)
    assert not geometry_matches(reference_image, prediction_image)

    with pytest.raises(ValueError, match="Geometry mismatch"):
        evaluate_case("shifted", reference_path, prediction_path)

    metrics, _, _ = evaluate_case(
        "shifted",
        reference_path,
        prediction_path,
        allow_resample=True,
    )
    assert metrics.prediction_resampled is True
