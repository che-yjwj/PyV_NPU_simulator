# PyV-NPU Simulator

PyV-NPU is a cycle-aware NPU (Neural Processing Unit) simulator built on a RISC-V foundation. It is designed to transform ONNX/MLIR inputs into a dedicated NPU-IR (Intermediate Representation), schedule operations, and perform detailed, cycle-level analysis of timing and resource utilization. The project integrates a RISC-V frontend (based on LLVM/MLIR, ONNX conversion, and Custom ISA) with a backend that models the NPU, memory hierarchy, DMA, NoC, and cache coherence.

A key feature is the support for a tightly-coupled custom ISA for the Tensor Engine (TE), enabling low-latency kernel submission without CPU intervention.

## Core Features

- **Multiple Simulation Levels**:
    - **L0 (Functional)**: Fast functional verification of ONNX to NPU-IR conversion.
    - **L1 (Tile-Time)**: Models tile-level latency and overlap with simple contention.
    - **L2 (Event/Resource)**: Event-driven simulation with detailed contention modeling for DMA, NoC, and memory banks.
    - **L3 (Cycle-Accurate)**: Optional, high-fidelity verification using Verilator co-simulation for specific kernels.
- **Dual RISC-V Integration Modes**:
    - **Loose-coupled**: Standard MMIO-based doorbell mechanism for communication.
    - **Tight-coupled**: Custom ISA extensions (`ENQCMD_T`, `TWAIT`, etc.) for direct, low-latency TE control.
- **Comprehensive Performance Analysis**:
    - Generates detailed reports (HTML/SVG/CSV), including Gantt charts, resource utilization graphs, and Roofline models.
    - Identifies top performance bottlenecks.
    - Compares the efficiency of Loose vs. Tight coupling modes.
- **Flexible Architecture Modeling**:
    - Configurable parameters for TE/VE cores, SPM size, memory banks, DMA channels, and NoC/DRAM bandwidth.
    - Support for various data precisions (FP16, BF16, FP8, INT8).

## Architecture Overview

The simulator connects a RISC-V host (Py-V) to a configurable NPU cluster.

```
                         ┌────────────────────────────────────────────────────────────────────┐
                         │                          Host (RISC-V / Py‑V)                      │
                         │                                                                    │
App/Runtime/API ───────► │  pyv_npu.Runtime                                                  │
                         │    ├─ onnx_loader / mlir_frontend                                 │
                         │    ├─ compiler_passes (tiling, fusion, prefetch_hint)             │
                         │    ├─ npu_ir_emitter (LOAD/STORE/GEMM_T/ATTN_T/...)               │
                         │    ├─ descriptor_builder (kernel desc & DMA scripts)              │
                         │    └─ driver                                                      │
                         │         ├─ Loose: MMIO write(doorbell) / read(status/IRQ)         │
                         │         └─ Tight: ENQCMD_T / TWAIT / TBAR / TSTAT (custom ISA)    │
                         └─────────┬──────────────────────────────────────────────────────────┘
                                   │ AXI-Lite (MMIO/CSR) & IRQ (loose)  │  Custom opcode (tight)
                                   │                                     │
                                   ▼                                     ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    SoC Interconnect / NoC / AXI                                        │
│                                                                                                        │
│   ┌──────────────┐      ┌────────────────┐      ┌────────────────────────────────┐                     │
│   │  L2 / CCx    │◄────►│   NPU Bridge   │◄────►│          MMIO/CSR Window       │                     │
│   └──────────────┘      └────────────────┘      │  0x4000_0000.. (queue, doorbell)│                     │
│                                                 └────────────────────────────────┘                     │
│                 ▲                                          ▲                                           │
│   DRAM Ctlr  ───┼──────────────────────────────────────────┼───────────────────────────                │
│                 │                                          │                                           │
│    ┌─────────────────────────────────────────── NPU Cluster (xN) ───────────────────────────────────┐ │
│    │   ┌────────── Scheduler/Dispatcher ──────────┐  ┌──────── DMA (0..C-1) ───────┐               │ │
│    │   │  IR Queue | CP-prio | Bank-aware         │  │ per-chan q | DBB | Prefetch │               │ │
│    │   └──────────────────────────────────────────┘  └──────────────────────────────┘               │ │
│    │   ┌────────────── SPM (banked) ───────────────┐  ┌──────────── Compute ─────────┐             │ │
│    │   │ banks | ping-pong | arbiter | counters    │  │  TE array (GEMM/CONV/ATTN)   │             │ │
│    │   └───────────────────────────────────────────┘  │  VE array (LN/GELU/ELTWISE)  │             │ │
│    │                                                  └───────────────────────────────┘             │ │
│    └────────────────────────────────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘

## Quickstart

### CLI Usage

The simulator provides a powerful command-line interface for running simulations.

**Loose-coupled Mode (MMIO/CSR):**
```bash
pyv-npu run model.onnx \
  --level L2 \
  --mode loose \
  --mmio-base 0x40000000 \
  --queue-size 1024 \
  --report out/loose_run
```

**Tight-coupled Mode (Custom ISA):**
```bash
pyv-npu run model.onnx \
  --level L2 \
  --mode tight \
  --isa enqcmd,twait,tbar,tstat \
  --report out/tight_run
```

### API Usage

The simulator can also be controlled programmatically.

```python
from pyv_npu import Simulator

# Loose-coupled mode
sim_loose = Simulator(
    model="model.onnx",
    level="L2",
    mode="loose",
    mmio_base=0x40000000,
    queue_size=1024
)
sim_loose.run()

# Tight-coupled mode
sim_tight = Simulator(
    model="model.onnx",
    level="L2",
    mode="tight",
    te_isa=["enqcmd", "twait", "tbar", "tstat"]
)
sim_tight.run()
```

## Default Configuration

The simulator starts with a set of default hardware parameters, which can be customized.

| Parameter         | Default Value | Description                               |
|-------------------|---------------|-------------------------------------------|
| TE Cores          | 2             | Tensor Engines for heavy computation      |
| VE Cores          | 4             | Vector Engines for element-wise ops       |
| SPM Capacity      | 2 MiB         | Banked Scratchpad Memory                  |
| SPM Banks         | 8             | Number of banks for memory conflict model |
| DMA Channels      | 2             | To support double-buffering               |
| DRAM Bandwidth    | 102.4 GB/s    | Modeled using a token bucket              |
| NoC Bandwidth     | 256 GB/s      | Link bandwidth for the Network-on-Chip    |
| Frequency         | 1.2 GHz       | Clock frequency for timing calculations   |
| Tile (M,N,K)      | 128,128,64    | Default dimensions for GEMM operations    |

## Project Layout

- `pyv_npu/ir`: Model IR (Intermediate Representation) and ONNX importer.
- `pyv_npu/isa`: NPU-IR operations and RISC-V custom instruction encodings.
- `pyv_npu/compiler`: Compiler passes (tiling, fusion, quantization) and mapper.
- `pyv_npu/runtime`: Scheduler, simulator core, memory models, and execution units (TE/VE).
- `pyv_npu/cli`: Command-line interface entry point.
- `examples`: Example models and scripts.
- `tests`: Unit and integration tests.