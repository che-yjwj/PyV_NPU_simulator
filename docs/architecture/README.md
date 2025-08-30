# System Architecture Documentation

This directory contains documents describing the architecture of the PyV-NPU simulator.

## Master Architecture Diagram

A high-level diagram showing the interaction between all major components of the system should be placed here. This provides a single entry point for understanding the overall structure.

*TODO: Create a master diagram illustrating the relationships between the RISC-V Core (Py-V), Compiler pipeline (ONNX -> IR -> NPU Program), Runtime (Scheduler, Resources), and the NPU hardware models (TC, VC, Memory Hierarchy).*

---

## Key Documents

- **[System Architecture](./system_architecture.md)**: Defines the core concepts, memory hierarchy, and simulation levels.
- **[RISC-V ↔ NPU Coupled Mode PRD](../prd/PRD_RISC-V_NPU_coupled_mode.md)**: Details the `Loose` and `Tight` coupling modes between the CPU and NPU.
- **[HW-Aware IR Specification](./HW_Aware_IR_Spec_Instruction_Level.md)**: Describes the format of the hardware-aware intermediate representation.
- **[Code Review](./code_review.md)**: Provides an analysis of the codebase, highlighting strengths and areas for improvement.
