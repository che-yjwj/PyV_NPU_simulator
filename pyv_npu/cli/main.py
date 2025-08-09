
from __future__ import annotations
import argparse, json, os
from ..ir.onnx_importer import load_onnx_as_model_ir
from ..compiler.passes.tiling import apply_tiling_pass
from ..compiler.passes.fusion import apply_fusion_pass
from ..compiler.passes.quantization import apply_quant_pass
from ..compiler.mapper import map_model_ir_to_npu_program
from ..runtime.simulator import run as run_sim

def cmd_compile(args):
    g = load_onnx_as_model_ir(args.model)
    g = apply_fusion_pass(g)
    g = apply_tiling_pass(g)
    g = apply_quant_pass(g, mode=args.quant)
    prog = map_model_ir_to_npu_program(g)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(prog.to_json(), f, indent=2)
    print(f"[OK] Compiled to {args.output}")

def cmd_run(args):
    g = load_onnx_as_model_ir(args.model)
    prog = map_model_ir_to_npu_program(g)
    rep = run_sim(prog)
    print(json.dumps({
        "total_cycles": rep.total_cycles,
        "engine_util": rep.engine_util,
        "timeline": rep.timeline if args.verbose else f"{len(rep.timeline)} ops"
    }, indent=2))

def build_parser():
    p = argparse.ArgumentParser(prog="pyv-npu", description="PyV-NPU simulator (skeleton)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("compile", help="Compile ONNX -> NPU program (JSON)")
    pc.add_argument("model", help="Path to ONNX model")
    pc.add_argument("-o", "--output", default="out/graph.npu.json")
    pc.add_argument("--quant", default="none", choices=["none","int8","vq"])
    pc.set_defaults(func=cmd_compile)

    pr = sub.add_parser("run", help="Compile and simulate")
    pr.add_argument("model", help="Path to ONNX model")
    pr.add_argument("-v", "--verbose", action="store_true")
    pr.set_defaults(func=cmd_run)
    return p

def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)

if __name__ == "__main__":
    main()
