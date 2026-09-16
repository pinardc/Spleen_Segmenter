# Canine Spleen Benchmark

This project imports selected canine CT series from an Orthanc server, converts them
to NIfTI, runs the existing MONAI Label `segmentation_spleen` model, and compares its
zero-shot predictions with radiologist-created spleen masks.

> The MONAI Label model was trained on **human** CT data. This repository evaluates
> domain transfer to canine CT; it does not establish clinical validity and must not
> be used for patient care.

## Privacy and data handling

This public repository contains code and configuration templates only. Credentials,
DICOM files, NIfTI volumes, model weights, predictions, reports, and local manifests
are ignored by Git. Do not place protected or identifying data in source files,
notebook outputs, commits, issues, or pull requests. The import code returns only a
small allow-list of non-identifying series metadata and uses Orthanc IDs as case IDs.

## Requirements

- Python 3.10 or newer
- Docker with Compose v2
- NVIDIA Container Toolkit and a CUDA-capable GPU (recommended)
- `dcm2niix` on the notebook host
- Network access to Orthanc, Docker Hub, and the MONAI model source

For WSL, enable the distro under Docker Desktop **Settings → Resources → WSL
Integration**. CPU inference is supported but can be slow.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[notebook,dev]"
```

The notebook prompts for the Orthanc URL, username, and hidden password at runtime,
so credentials are not persisted. For non-notebook scripts, `.env.example` can
optionally be copied to an ignored `.env`. Keep TLS verification enabled unless a
private test endpoint has a documented certificate exception.

Create the expected local directories and place radiologist masks in
`data/radiologist_masks`. Each mask must be named `<case_id>.nii` or
`<case_id>.nii.gz`, matching the imported CT basename.

## Start MONAI Label

GPU:

```bash
docker compose up -d monailabel
docker compose logs -f monailabel
```

CPU fallback:

```bash
docker compose --profile cpu up -d monailabel-cpu
docker compose logs -f monailabel-cpu
```

The first start downloads the MONAI Label radiology sample app and the pretrained
human spleen weights into ignored persistent storage. The service runs in inference
only mode at <http://127.0.0.1:8000>.

## Run the benchmark

Open `canine_spleen_benchmark.ipynb`, select the project environment as its kernel,
set `RUN_LIVE = True`, and run cells in order. The preflight cell asks for the
Hostinger Orthanc URL/IP and login; the password input is hidden and remains only in
kernel memory:

1. Load configuration and run local/service preflight checks.
2. Search Orthanc for CT series or provide explicit Orthanc series IDs.
3. Select series and download/convert them to NIfTI.
4. Pair CTs with same-basename radiologist masks.
5. Call `segmentation_spleen` through the MONAI Label REST API.
6. Validate geometry, calculate overlap/surface/volume metrics, and inspect overlays.

The final notebook section includes an interactive CT viewer with case, plane, slice,
window center/width, and mask-opacity controls. Radiologist-only voxels appear green,
MONAI-only voxels red, and overlapping voxels yellow.

The notebook writes a local manifest under `data/` and results under `reports/`.
Both are ignored. For reproducibility, record the image digest, model metadata, and
software versions in an approved non-identifying study record.

## Case naming

By default, an imported case is named with its Orthanc series ID, normalized to safe
characters. For example:

```text
data/ct/0f84c7...e31.nii.gz
data/radiologist_masks/0f84c7...e31.nii.gz
data/predictions/0f84c7...e31.nii.gz
```

If radiologist masks use another case name, pass that `case_id` explicitly when
importing. Duplicate names and multi-volume conversions fail rather than silently
selecting a file.

## Metrics

The benchmark reports Dice, Jaccard/IoU, precision, recall, Hausdorff distance at the
95th percentile, average symmetric surface distance, reference/predicted volume, and
percent volume bias. Empty masks and geometry mismatches are handled explicitly.
Resampling is opt-in and always uses nearest-neighbor interpolation for masks.

## Development

```bash
pytest
ruff check .
python -m nbclient --help
```

`docker compose config` validates the deployment file when Docker Compose is
available.

## Acknowledgments

- [MONAI Label](https://github.com/Project-MONAI/MONAILabel)
- [Orthanc](https://www.orthanc-server.com/)
- [dcm2niix](https://github.com/rordenlab/dcm2niix)

Licensed under Apache-2.0.
