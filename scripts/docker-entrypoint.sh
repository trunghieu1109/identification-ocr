#!/bin/sh
set -eu

# VietOCR downloads weights into tempfile.gettempdir(). Keep that directory in
# the model cache; uploaded image directories use OCR_UPLOAD_TMP_DIR instead.
mkdir -p "$TMPDIR"
exec "$@"
