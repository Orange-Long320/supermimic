#!/usr/bin/env bash
# Install only into the already-created, dedicated GEM environment.
set -euo pipefail
SM_ROOT=/home/2025-Long.xh/SSD/supermimic
SM_DATA=/home/2025-Long.xh/HHD/data/supermimic
SM_GEM_PY=/home/2025-Long.xh/HHD/venvs/supermimic-gem/bin/python
test -d "$SM_ROOT/vendor/gem-x"
test -x "$SM_GEM_PY"
test "$(cat "$SM_ROOT/vendor/gem-x/SOURCE_COMMIT")" = 20555befe2fbbe5b2015e1fff7ff214dc18ea9d6
cd "$SM_ROOT"
export UV_CACHE_DIR="$SM_DATA/cache/uv"
uv --no-config pip install --python "$SM_GEM_PY" \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  "$SM_DATA/downloads/torch-2.14.1+cu130-cp312-cp312-manylinux_2_28_x86_64.whl" \
  "$SM_DATA/downloads/warp_lang-1.18.0-py3-none-manylinux_2_28_x86_64.whl" \
  -e 'vendor/gem-x[video]'
uv --no-config pip check --python "$SM_GEM_PY"
env -u PYTHONPATH -u PYTHONHOME CUDA_VISIBLE_DEVICES='' "$SM_GEM_PY" - <<'PY'
from importlib.metadata import version
import sys
import torch
import soma
from gem.inference.cli import build_parser
assert soma.__version__ == '0.3.3'
assert version('transformers') == '5.3.0'
print('interpreter', sys.executable)
print('torch', torch.__version__, 'CUDA package', torch.version.cuda)
print('SOMA', soma.__version__)
print('GEM CLI', build_parser().prog)
PY
