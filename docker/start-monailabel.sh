#!/usr/bin/env bash
set -euo pipefail

APP_ROOT=/workspace/apps
APP_PATH="${APP_ROOT}/radiology"
STUDIES_PATH=/workspace/studies

mkdir -p "${APP_ROOT}" "${STUDIES_PATH}"

if [[ ! -f "${APP_PATH}/main.py" ]]; then
  echo "Downloading the MONAI Label radiology sample app..."
  rm -rf "${APP_PATH}"
  monailabel apps --download --name radiology --output "${APP_ROOT}"
fi

exec monailabel start_server \
  --app "${APP_PATH}" \
  --studies "${STUDIES_PATH}" \
  --host "${MONAI_LABEL_HOST:-0.0.0.0}" \
  --port "${MONAI_LABEL_PORT:-8000}" \
  --conf models segmentation_spleen \
  --conf skip_trainers true \
  --conf skip_scoring true \
  --conf skip_strategies true
