# PyV-NPU 통합 실행 계획서 (통합 PRD v1.2)
작성일: 2025-08-26 · 작성자: ChatGPT, Gemini

## 0) 개요 (Scope & Goals)
- **목표**  
  RISC-V 기반 **PyV-NPU 시뮬레이터**를 단계별(Level)로 개발하여, ONNX/MLIR 입력을 **NPU-IR**로 변환 → 스케줄링 → 타이밍·자원 모델링 → 타일 수준·이벤트 기반·사이클 수준 분석 및 리포트 생성.  
  RISC-V **프론트엔드(LLVM/MLIR, ONNX 변환, Custom ISA)**와 **백엔드(NPU, 메모리 계층, DMA, NoC, Cache Coherence)** 설계를 통합.  
  **[확장]** Tight-coupled Custom ISA 기반 **Tensor Core(TC)** 제어 명령어 세트를 정의·시뮬레이션·검증하여 CPU 개입 없이 저지연 커널 제출이 가능한 구조를 포함.

- **타깃 워크로드**: LLM 추론(Attention/MLP), Conv/GEMM 기반 비전 네트워크.  
- **비목표(본 버전)**: OS/드라이버 전체 모델, 전면 RTL 구현, 정확한 전력 추정(전력은 추후 모델).

---

## 세부 요구사항 문서 (Detailed Requirement Documents)

본 문서는 프로젝트의 통합 요구사항을 다루며, 각 기능 영역에 대한 상세한 내용은 아래의 세부 PRD에서 정의합니다.

### 📜 `PRD_RISC-V_NPU_coupled_mode.md`
- **요약**: RISC-V 코어와 NPU를 연동하는 두 가지 모드(`Loosely-coupled`, `Tightly-coupled`)의 아키텍처와 요구사항을 정의합니다. 각 모드의 제어 흐름, 실행 모델(Blocking/Non-blocking), 성능 평가 기준을 상세히 기술합니다.

### 📜 `PRD_Tinygrad_to_HW_Aware_IR.md`
- **요약**: 경량 딥러닝 프레임워크인 `Tinygrad`의 계산 그래프를 시뮬레이터의 하드웨어 인지 IR(`HW-Aware IR`)로 변환하는 변환 계층의 요구사항을 정의합니다. 이는 빠른 알고리즘 검증과 정밀 시뮬레이션을 연결하는 역할을 합니다.

### 📜 `pyv_npu_runtime.md`
- **요약**: 시뮬레이터의 런타임 시스템 설계 요구사항을 정의합니다. 명령어 큐, 의존성 분석, 스케줄러, Reorder Buffer(ROB) 등 시뮬레이션 실행의 핵심 구성요소와 동작 시나리오를 다룹니다.

### 📜 `NPU_Simulator_Memory_Hierarchy.md` (현 `system_architecture.md`에 통합됨)
- **요약**: NPU의 다계층 메모리 구조(버퍼, L0~L3 SPM, NoC, DRAM)를 상세히 정의했던 핵심 문서입니다. 현재 이 문서의 내용은 `docs/architecture/system_architecture.md`에 공식 아키텍처로 통합되었습니다.

---

## 1) 사용자 스토리
- **시스템 아키텍트**: 타일 크기·메모리 파라미터 스윕 → PPA 트레이드오프 도출.
- **컴파일러 엔지니어**: ONNX→MLIR→NPU-IR 패스(퓨전/타일링/프리패치/배치) 효과 검증.
- **하드웨어 엔지니어**: DMA 채널 수, SPM bank 수, NoC 대역폭, DRAM 채널 변경 시 처리량/지연 민감도 평가.
- **마이크로아키텍처 엔지니어**: Tight-ISA vs Loose-MMIO 모드의 제출 지연(P99 latency) 비교, TC 큐 활용도 분석.

## 2) 요구사항 (Requirements)

### 기능 요구
1. **입력**: ONNX(우선), (옵션) MLIR, (확장) Tinygrad
2. **출력**:  
   - (a) NPU-IR 트레이스(JSONL)  
   - (b) 실행 타임라인(CSV)  
   - (c) 리포트(HTML/SVG/CSV): Gantt, 자원 사용률, Roofline, 병목 Top-N  
   - (d) 실험 메타데이터(run.yaml)
3. **프론트엔드**: LLVM/MLIR 변환 + Dialect 설계
4. **백엔드 모델링**: TC/VC/DMA/SPM/L2/DRAM/NoC + Cache Coherence 옵션
5. **스케줄링**: List-sched + CP-prio + Bank-aware 배치
6. **메모리 모델**: Banked SPM, Double-Buffer DMA, NoC/DRAM 큐
7. **정밀도/양자화**: FP16/BF16/FP8/INT8 + (옵션) Weight-VQ
8. **RISC-V 연동**:  
   - Loose-coupled: MMIO Doorbell  
   - Tight-coupled: Custom ISA (`ENQCMD_T`, `TWAIT`, `TBAR`, `TSTAT`)로 TC 직접 제어
9. **TC Custom ISA 세부 요구**:  
   - **명령어**  
     - `ENQCMD_T rd, rs1, rs2` : 커널 디스크립터 제출  
     - `TWAIT rs1` : ticket 완료 대기  
     - `TBAR imm` : Barrier (tile/cluster/global)  
     - `TSTAT rd` : 상태/큐 깊이/유휴율 조회  
   - **Descriptor 필드**: op, addr_in/out, tile_shape, bank_mask, prio, prefetch_dist
   - **CSR/MMIO 연계**: TC 큐 주소, Doorbell, IRQ 상태, 성능 카운터

## 3) 개발 레벨(Levels) - `sim_level`
- **`IA` (기능 정확성)**: ONNX→NPU-IR 변환 및 기능 검증(초고속)  
- **`IA_TIMING` (명령어+타이밍)**: 타일 단위 지연/중첩 모델 + 단순 경합  
- **`CA_HYBRID` (하이브리드 CA)**: 이벤트 기반 실행, DMA/NoC/Bank 경합·스케줄러 포함  
- **`CA_FULL` (외부 연동 CA, 옵션)**: **MATLAB/Simulink**나 SystemC 등 외부 전문 도구와 연동하여 완전한 Cycle-Accurate 시뮬레이션 구현.

## 4) NPU-IR 스키마 (확장)
```yaml
# 공통
LOAD(dst=SPM[tile], src=DRAM, ...)
STORE(src=SPM[tile], dst=DRAM, ...)
GEMM_T(tile_m,tile_n,tile_k, quant, affinity={TC})
...
BARRIER(scope)
WAIT(ev)

# TC ISA 확장
op: ENQCMD_T
desc_ptr: 0x...
ticket: 42

op: TWAIT
ticket: 42
scope: tile
```

## 5) 타이밍/성능 모델
- **DMA 중첩(Double Buffer)**:  
  `T_tile = max(T_in + T_comp, T_out)`  
- **자원/경합(CA_HYBRID)**:  
  Bank 충돌, NoC/DRAM 큐 모델, Token Bucket 기반 대역폭 모델  
- **스케줄러**: List-sched + CP-prio, Bank-aware 배치

## 6) 메모리/데이터 이동
- **SPM**: banked, ping-pong 버퍼, bank conflict 모델  
- **Prefetch**: prefetch_distance 힌트 반영  
- **Cache bypass**: 대용량 DMA시 L2 우회

## 7) 검증/리포트
- KPI: Latency, Throughput, Utilization, Roofline, 병목 Top-N  
- **추가 분석**: TC ISA 효율, Loose vs Tight 비교, 큐 활용도 시계열 그래프

## 8) CLI/API 예시
```bash
# Tight-coupled 모드 실행
pyv-npu run model.onnx --sim_level CA_HYBRID --te-mode tight --report out/tight_test
```
```python
from pyv_npu import Simulator
Simulator(model="model.onnx", level="L2", te_mode="tight").run()
```

## 9) 기본 파라미터 (초기값)
| 파라미터 | 기본값 | 설명 |
|---|---:|---|
| TC 수 | 2 | Tensor Core |
| VC 수 | 4 | Vector Core |
| SPM 용량 | 2 MiB | banked scratchpad |
| SPM bank 수 | 8 | bank conflict 모델 |
| DMA 채널 | 2 | double buffer 지원 |
| DRAM BW | 102.4 GB/s | Token bucket 모델 |
| NoC BW | 256 GB/s | 링크 대역폭 |
| 주파수 | 1.2 GHz | 타이밍 계산 |
| 타일(M,N,K) | 128,128,64 | GEMM 기준 |

## 10) 마일스톤
- **M0**: NPU-IR, IA end-to-end  
- **M1**: IA_TIMING 타일 타이밍 + Bank 모델  
- **M2**: CA_HYBRID 이벤트 기반 경합 + Cache Coherence  
- **M3**: TC Custom ISA 설계 + MLIR Dialect/Lowering  
- **M4**: Py-V 연동, Loose/Tight 비교 E2E  
- **M5**: 회귀세트, 문서화, 튜토리얼

## 11) 리스크 & 대응
| 리스크 | 영향 | 대응 |
|---|---|---|
| Custom ISA 툴체인 난이도 | 개발 지연 | 초기 `.insn` asm → 후속 LLVM/binutils 확장 |
| TC ISA 시뮬 오버헤드 | 성능 분석 왜곡 | CA_HYBRID 모델 최적화, CA_FULL 보정 |
| ISA 호환성 | 표준 충돌 가능 | custom-0~3 opcode 예약 |

## 12) 블록 다이어그램
```
RISC-V Core (Py-V)
 ├─ LLVM/MLIR Frontend
 ├─ Loose-coupled MMIO
 ├─ Tight-coupled TC ISA
 └─ Runtime API
        ▼
    NPU Cluster
     ├─ Scheduler
     ├─ DMA
     ├─ SPM (banked)
     ├─ TC Array
     └─ VC Array
```

## 13) RISC-V ↔ NPU 연동 방식 (Py-V 기반, 확정안)
(이하 내용은 기존과 동일하여 생략)
