
#!/usr/bin/env bash
set -euo pipefail
python -m pyv_npu.cli.main run examples/tinyllama.onnx -v
