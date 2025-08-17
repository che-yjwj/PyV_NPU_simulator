# PRD: RISC-V ↔ NPU 연동 모드 지원 (Loosely / Tightly Coupled)

## 1. 목적 (Objective)
본 문서는 RISC-V 기반 SoC 내에서 NPU(Neural Processing Unit)를 연동하는 두 가지 아키텍처 모드(Loosely-coupled / Tightly-coupled)에 대한 요구사항 정의서를 제공한다. 또한 Py-V 시뮬레이터 상에서 Blocking / Non-blocking 실행 모델을 모두 지원하도록 하여 실제 HW 아키텍처 탐색 및 성능 분석에 활용할 수 있도록 한다. 이 PRD는 HW-Software co-design의 trade-off를 체계적으로 평가하고, 구현 리스크를 최소화하기 위한 지침을 포함한다.

## 2. 범위 (Scope)
- 연동 모드: Loosely-coupled (MMIO + DMA + DRAM 기반), Tightly-coupled (Custom ISA + 파이프라인 통합)
- 실행 모델: Blocking (CPU 정지 / busy-wait cyclic), Non-blocking (병렬 실행 / interrupt-driven, ROB 기반)
- 대상 플랫폼: Py-V (Cycle-level RISC-V 시뮬레이터)
- 출력물: 성능 추정(Gantt 차트, latency 분석), HW-Software co-design trade-off 평가
- 추가 범위: 위험 요인 분석, 테스트/검증 방법, 구체적 구현 예시 포함
- 제외 범위: 실제 HW 구현 (시뮬레이션 기반 탐색에 초점)

## 3. 아키텍처 개요

### 3.1 Loosely-coupled 구조
+-------------------+          +---------------------+
|    RISC-V Core    |          |        NPU          |
|                   |   MMIO   |                     |
|  Instr. Dec.      |<-------->| Cmd Decoder         |
|  Software Driver  |          | Tensor Engine       |
+---------^---------+          +---------^-----------+
|                              |
| DMA                          | DMA
|                              |
+-------------------+          +-------------------+
|    Shared DRAM    |<-------->|    Scratchpad     |
+-------------------+          +-------------------+
text### 3.2 Tightly-coupled 구조
+----------------------------------------------------+
|                RISC-V Core (with NPU)              |
|                                                    |
|  IF -> ID -> EX(NPU) -> MEM -> WB                  |
|                |                                   |
|                +-- Custom Opcode --> NPU Unit      |
|                           |                        |
|                +----------+-----------+             |
|                | Shared Scratchpad / RegFile |      |
+----------------------------------------------------+
|
v
+--------------+
| Shared DRAM  |
+--------------+
text### 3.3 메모리 계층 비교
[Loosely-coupled]
CPU <-> MMIO <-> NPU
|                  |
v                  v
Shared DRAM <----> Scratchpad (via DMA)
[Tightly-coupled]
RISC-V Pipeline <-> NPU
|               |
+--- Shared Scratchpad/RegFile ---+
|
Shared DRAM
text## 4. 실행 모델 (Blocking vs Non-blocking)

### 4.1 Tight-coupled 파이프라인

#### Blocking
IF -> ID -> EX(NPU 실행중...) ----> MEM -> WB
(stall)

CPU Timeline: |====NPU (stall)====|--> next ops
NPU Timeline: |====MatMul=========|
text#### Non-blocking
IF -> ID -> EX(NPU launch) -> MEM -> WB
|
+-- Dispatch to NPU
|
+-- Continue executing next instructions

CPU Timeline: |launch|--> inst2 --> inst3 --> (wait if dep)
NPU Timeline: |==========MatMul==========|
text### 4.2 Loosely-coupled 제어 흐름

#### Blocking
CPU ----MMIO/CSR----> NPU
(Start)
[CPU는 "대기"]
CPU Timeline: |==== wait for NPU ====|--> next ops
NPU Timeline: |==== compute =========|
text#### Non-blocking
CPU ----MMIO/CSR----> NPU
(Start)
CPU continues other work...
|
+-- Later: interrupt / status check --> collect result
CPU Timeline: |launch|--> other ops --> [IRQ] --> collect
NPU Timeline: |========== compute ==========|
text## 5. 요구사항 (Requirements)

### 5.1 기능 요구사항
1. Loosely-coupled 지원
   - MMIO 기반 NPU 명령 디스패치: 예를 들어, CSR(Control and Status Register) 맵으로 NPU 명령을 전달. 구체적 예시:
     - CSR 주소 0x800: NPU Start Command (e.g., `csrrw x0, 0x800, x1` where x1 holds command parameters)
     - CSR 주소 0x801: Status Poll (e.g., `csrr x2, 0x801` to check completion)
   - Blocking(polling) & Non-blocking(interrupt) 실행 모델: Polling은 busy-loop 구현, Interrupt는 RISC-V PLIC(Platform-Level Interrupt Controller) 시뮬레이션
   - DMA latency 모델링: Py-V에서 DMA 전송 지연을 cycle 단위로 모델링 (e.g., DMA setup: 10 cycles, transfer: 1 cycle/byte)

2. Tightly-coupled 지원
   - Custom RISC-V ISA 확장 (예: `matmul rd, rs1, rs2` for matrix multiplication, `conv rd, rs1, rs2, rs3` for convolution; opcode format: 7-bit custom opcode + funct3/funct7 for variants)
     - Pseudo-code 예시:
matmul a0, a1, a2  # a0 = result, a1 = tensor1 addr, a2 = tensor2 addr
text- Blocking(stall) & Non-blocking(ROB retire) 실행 모델: Stall은 pipeline bubble 삽입, Non-blocking은 Reorder Buffer(ROB)를 통해 dependency 관리
- Scratchpad/RegFile 공유 모델링: Shared memory access latency (e.g., 1 cycle for regfile, 5 cycles for scratchpad)

3. Py-V 통합
- Cycle-level latency 반영: 모든 연산에 cycle-accurate 모델 적용 (e.g., MatMul: 100 cycles for 4x4 matrix)
- Gantt chart 기반 실행 시각화: Matplotlib 또는 ASCII-based timeline 출력
- HW-Software co-design 시나리오 지원: Parameterizable models (e.g., scratchpad size via config file)

### 5.2 성능 및 평가 기준
- 메트릭: Throughput (ops/cycle), Latency (cycles per operation), Power estimation (via cycle count proxy), CPU/NPU utilization (%)
- 벤치마크 시나리오: 
- 간단 NN 모델 (e.g., MNIST inference with 2-layer MLP)
- 복잡 모델 (e.g., MobileNet convolution layers)
- Trade-off 분석: Scratchpad size (1KB vs 16KB) vs DRAM bandwidth (100MB/s vs 1GB/s) 시뮬레이션
- 기준: Non-blocking 모드에서 latency hiding 효과 >20% 개선 목표

### 5.3 테스트/검증 방법
- Unit Test: 개별 모듈 테스트 (e.g., MMIO dispatch unit test in Py-V script)
- Integration Test: 전체 workflow 검증 (e.g., Blocking/Non-blocking end-to-end execution on sample workloads)
- Edge Case: High-latency DMA (e.g., 1000 cycles delay), Dependency conflicts in Non-blocking
- Validation: Simulation results vs. theoretical model (e.g., cycle count mismatch <5%)

## 6. 위험 요인 (Risks)
- Non-blocking 모델의 Interrupt 오버헤드: 추가 5-10 cycles per IRQ, 병렬성 이득 상쇄 가능성
- Tightly-coupled의 Pipeline stall 리스크: Long NPU ops (e.g., large MatMul) 시 전체 throughput 저하
- Py-V 시뮬레이터 한계: Cycle-level 정확도 vs. simulation 속도 trade-off (e.g., large models 시 simulation time >1 hour)
- 구현 리스크: Custom ISA 확장 호환성 (RISC-V standard compliance), DMA 모델링 오류
- 완화 전략: Iterative prototyping, benchmark-based validation

## 7. 활용 시나리오
- Blocking 모드: baseline 성능 분석, 단순 모델 구현
- Non-blocking 모드: CPU/NPU 병렬성 분석, 실제 HW와 유사한 아키텍처 탐색
- 연구 활용:
- NPU latency hiding 효과 검증
- Scratchpad 크기 / DRAM 대역폭 trade-off 분석
- HW-Software 협력 아키텍처 설계

## 8. 용어 사전 (Glossary)
- MMIO: Memory-Mapped I/O
- DMA: Direct Memory Access
- CSR: Control and Status Register
- ROB: Reorder Buffer
- PLIC: Platform-Level Interrupt Controller
- Py-V: Python-based RISC-V Simulator (cycle-level)

## 9. 문서 관리
- 버전: 1.1 (업데이트 날짜: 2025-08-17)
- 업데이트 계획: 요구사항 변경 시 매월 검토, 피드백 반영
- 작성자: [Your Name/Team]
- 승인자: [Approver]