from __future__ import annotations
import argparse, json, os
from ..ir.onnx_importer import load_onnx_as_model_ir
from ..compiler.passes.tiling import apply_tiling_pass
from ..compiler.passes.fusion import apply_fusion_pass
from ..compiler.passes.quantization import apply_quant_pass
from ..compiler.mapper import map_model_ir_to_npu_program
from ..runtime.simulator import run as run_sim
from ..config import SimConfig

def cmd_compile(args):
    """Handles the 'compile' command."""
    print(f"Compiling model: {args.model} with mode: {args.mode}")
    model_ir = load_onnx_as_model_ir(args.model)

    # In the future, compiler passes could be configured via args
    model_ir = apply_fusion_pass(model_ir)
    model_ir = apply_tiling_pass(model_ir)
    model_ir = apply_quant_pass(model_ir, mode=args.quant)

    npu_prog = map_model_ir_to_npu_program(model_ir, mode=args.mode)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(npu_prog.to_json(), f, indent=2)

    print(f"[OK] Compiled to {args.output}")

def cmd_run(args):
    """Handles the 'run' command."""
    # Create simulator config from args
    config = SimConfig.from_args(args)

    print("--- Simulator Configuration ---")
    print(config)
    print("-----------------------------")

    # --- This part is a simplified pipeline for now ---
    # 1. Load and compile the model
    model_ir = load_onnx_as_model_ir(config.model)
    npu_prog = map_model_ir_to_npu_program(model_ir, mode=config.mode)

    # 2. Run simulation
    # The simulator will now need the config to model behavior
    report = run_sim(npu_prog, config)

    # 3. Save report
    os.makedirs(config.report_dir, exist_ok=True)
    report_path = os.path.join(config.report_dir, "report.json")
    with open(report_path, "w") as f:
        # A more sophisticated report would be generated here
        json.dump({
            "total_cycles": report.total_cycles,
            "engine_util": report.engine_util,
            "timeline": report.timeline,
        }, f, indent=2)

    print(f"[OK] Simulation finished. Report saved to {report_path}")

def build_parser():
    p = argparse.ArgumentParser(
        prog="pyv-npu",
        description="PyV-NPU Simulator (PRD v1.1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # --- Compile Command ---
    pc = sub.add_parser("compile", help="Compile ONNX -> NPU program (JSON)",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    pc.add_argument("model", help="Path to ONNX model")
    pc.add_argument("-o", "--output", default="out/graph.npu.json", help="Output path for NPU program")
    pc.add_argument("--quant", default="none", choices=["none", "int8", "vq"], help="Quantization mode")
    pc.add_argument("--mode", default="loose", choices=["loose", "tight"], help="RISC-V to NPU coupling mode for compilation")
    pc.set_defaults(func=cmd_compile)

    # --- Run Command (based on PRD v1.1) ---
    pr = sub.add_parser("run", help="Compile and run simulation",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Core args
    pr.add_argument("model", help="Path to ONNX model")
    pr.add_argument("--level", default="L1", choices=["L0", "L1", "L2", "L3"], help="Simulation fidelity level")
    pr.add_argument("--report", default="out/default_run", help="Directory to save simulation reports")

    # Mode selection
    pr.add_argument("--mode", default="loose", choices=["loose", "tight"], help="RISC-V to NPU coupling mode")

    # Loose mode args
    loose_group = pr.add_argument_group('Loose-coupled Mode Arguments')
    loose_group.add_argument("--mmio-base", type=int, default=0x40000000, help="MMIO base address for NPU control")
    loose_group.add_argument("--queue-size", type=int, default=1024, help="Size of the command queue")

    # Tight mode args
    tight_group = pr.add_argument_group('Tight-coupled Mode Arguments')
    tight_group.add_argument("--isa", default="enqcmd,twait,tbar,tstat", help="Comma-separated list of enabled TE custom instructions")

    pr.set_defaults(func=cmd_run)

    return p

def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)

if __name__ == "__main__":
    main()