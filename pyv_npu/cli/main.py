from __future__ import annotations
import argparse
import json
import os
from ..ir.onnx_importer import load_onnx_as_model_ir
from ..compiler.passes.tiling import apply_tiling_pass
from ..compiler.passes.fusion import apply_fusion_pass
from ..compiler.passes.quantization import apply_quant_pass
from ..compiler.mapper import map_model_ir_to_npu_program
from ..compiler.allocator import Allocator
from ..runtime.simulator import run as run_sim
from ..config import SimConfig
from ..utils.reporting import generate_report


def cmd_compile(args):
    """Handles the 'compile' command."""
    print(f"Compiling model: {args.model} with mode: {args.mode}")
    model_ir = load_onnx_as_model_ir(args.model)

    # In the future, compiler passes could be configured via args
    model_ir = apply_fusion_pass(model_ir)
    model_ir = apply_tiling_pass(model_ir)
    model_ir = apply_quant_pass(model_ir, mode=args.quant)

    npu_prog = map_model_ir_to_npu_program(model_ir, mode=args.mode)

    # Allocate memory for tensors
    allocator = Allocator()  # Use default base address and page size
    allocator.allocate(npu_prog)

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

    # 1. Load and compile the model
    model_ir = load_onnx_as_model_ir(config.model)
    npu_prog = map_model_ir_to_npu_program(model_ir, mode=config.mode)

    # Allocate memory for tensors
    allocator = Allocator(config.dram_base_address, config.dram_page_size)
    allocator.allocate(npu_prog)

    # 2. Run simulation
    schedule, stats = run_sim(npu_prog, config)

    # 3. Generate all reports
    generate_report(schedule, config, stats)

    print(f"[OK] Simulation finished. Reports are in {config.report_dir}")


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
    pc.add_argument("-o", "--output", default="out/graph.npu.json",
                    help="Output path for NPU program")
    pc.add_argument("--quant", default="none", choices=["none", "int8", "vq"],
                    help="Quantization mode")
    pc.add_argument("--mode", default="loose", choices=["loose", "tight"],
                    help="RISC-V to NPU coupling mode for compilation")
    pc.set_defaults(func=cmd_compile)

    # --- Run Command (based on PRD v1.1) ---
    pr = sub.add_parser("run", help="Compile and run simulation",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Config file
    pr.add_argument("-c", "--config", type=str, default=None,
                    help="Path to YAML config file to override defaults")

    # Core args (set default=None to allow override from YAML)
    pr.add_argument("model", nargs='?', default=None,
                    help="Path to ONNX model (optional if specified in config)")
    pr.add_argument("--sim-level", type=str, default=None, dest="sim_level",
                    choices=["IA", "IA_TIMING", "CA_HYBRID", "CA_FULL"],
                    help="Simulation fidelity level (IA, IA_TIMING, CA_HYBRID, CA_FULL)")
    pr.add_argument("--report", type=str, default=None,
                    help="Directory to save simulation reports")
    pr.add_argument("--gantt", type=str, default=None,
                    help="Path to save Gantt chart HTML file")
    pr.add_argument("--ascii-gantt", action="store_true",
                    help="Print an ASCII Gantt chart to the console")

    # Mode selection
    pr.add_argument("--mode", type=str, default=None,
                    choices=["loose", "tight"], help="RISC-V to NPU coupling mode")

    # Loose mode args
    loose_group = pr.add_argument_group('Loose-coupled Mode Arguments')
    loose_group.add_argument("--mmio-base", type=int, default=None,
                             help="MMIO base address for NPU control")
    loose_group.add_argument("--queue-size", type=int, default=None,
                             help="Size of the command queue")

    # Tight mode args
    tight_group = pr.add_argument_group('Tight-coupled Mode Arguments')
    tight_group.add_argument("--isa", type=str, default=None,
                             help="Comma-separated list of enabled TC custom instructions")

    pr.set_defaults(func=cmd_run)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
