
# PyV-NPU — Skeleton v0.1

Minimal, testable skeleton for a cycle-aware NPU simulator, aligned with `PyV-NPU_Simulator_PRD_v0.2.md` (included under `docs/`).

## Quickstart
```bash
python -m pyv_npu.cli.main --help
pyv-npu run examples/tinyllama.onnx --level 1
pyv-npu compile examples/tinyllama.onnx -o out/graph.npu.json
```
## Layout (high level)
- `pyv_npu/ir`: Model IR & importers (ONNX).
- `pyv_npu/isa`: NPU IR ops and RISC-V ext encodings (stubs).
- `pyv_npu/compiler`: Passes (tiling, fusion, quantization) & mapper.
- `pyv_npu/runtime`: Scheduler, simulator, memory, bus, TE/VE execution stubs.
- `pyv_npu/cli`: CLI entry.
- `examples`: Place ONNX here; includes a placeholder.
- `tests`: Smoke tests.
