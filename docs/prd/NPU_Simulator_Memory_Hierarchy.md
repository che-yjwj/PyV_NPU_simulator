# PRD: NPU 시뮬레이터의 메모리 계층 구조 설계 (v2.4)

## 1. 서론 및 배경
### 1.1 개요
본 PRD는 NPU(Neural Processing Unit) 시뮬레이터의 메모리 계층 구조를 정의하여 HW-SW 공동 설계, 성능 최적화, 병목 분석을 지원하는 요구사항을 제시합니다. NPU는 AI 워크로드(LLM 추론, CNN, Transformer) 처리에 특화된 프로세서로, 메모리 계층은 데이터 재사용(Locality), Tile 기반 스트리밍, 저지연 접근을 핵심으로 합니다. Scratchpad Memory(SPM) 중심의 명시적 데이터 관리와 DMA 중첩을 통해 DRAM 대역폭 병목을 최소화합니다.

이 PRD는 이전 버전(v2.3)을 확장하여 더 충실히 작성되었으며, 2025년 트렌드(On-chip SRAM 기반 Hierarchical Memory, HBM3E 고대역폭, Tiling 기법, Low-Power μNPU 최적화)를 반영합니다. RISC-V-NPU 연동(Loosely/Tightly Coupled)과 시뮬레이션 레벨(sim_0 ~ sim_3)을 세부적으로 정의하며, 용어 혼용 문제를 해결하여 명확성을 높였습니다.

### 1.2 목적 및 범위
- **목적**:
  - 메모리 계층 모델링으로 RISC-V-NPU 연동 최적화, 데이터 이동 병목 분석, DNN 워크로드(Attention, GEMM, Conv) 성능 향상.
  - HW-SW co-design 지원: Tile 크기/뱅크 수/대역폭 스윕으로 PPA(Power-Performance-Area) trade-off 평가.
  - 실리콘 수준 예측: Cycle-accurate latency hiding (Double Buffering, Prefetch), Contention (Bank/NoC/DRAM Queue) 시뮬레이션.
- **범위**:
  - 메모리 계층: 버퍼, SPM (L0 ~ L3), NoC, DMA.
  - 시뮬레이션 레벨: sim_0 (IA) ~ sim_3 (CA_FULL).
  - 연동: RISC-V (Loose/Tight), Py-V 기반 구현.
  - 제외: 실제 RTL 합성, OS-level 드라이버 모델링, 세부 전력 분석 (Cycle Proxy로 대체).

### 1.3 2025년 NPU 메모리 계층 트렌드
2025년 NPU 메모리 트렌드는 On-device AI 가속화와 에너지 효율성을 중점으로 합니다:
- **Hierarchical Memory with On-chip SRAM**: AMD Ryzen AI, Google TPU v5에서 registers, caches, DRAM의 다계층 구조 표준화. On-chip SRAM을 컴퓨트 블록 가까이에 배치하여 latency/에너지 소비 감소 (Labelvisor 보고서).
- **Tiling Techniques**: Neural Network를 위한 Tile 단위 메모리 분할과 Double Buffering 필수 (IEEE 논문).
- **High-Bandwidth Memory (HBM3E)**: Mitsui 보고서 기준 HBM3E로 대역폭 102.4 GB/s 이상, DRAM 병목 최소화.
- **Low-Power μNPU Optimization**: arXiv 논문에서 PE 클러스터링, RAM partitioning, High-Bandwidth 인터페이스 강조.
- **On-Device AI Versatility**: Goover 보고서에서 NPU의 on-device 처리로 latency 감소, Number Analytics에서 Efficient Data Access 계획 중요성 강조.
- **Future Trends**: C&T Solution에서 NPU vs GPU 비교로 메모리 최적화가 AI 하드웨어 미래를 주도. Trio Dev에서 2025 AI 트렌드 중 메모리 효율 핵심.

## 2. 메모리 계층 구조 개요
NPU 메모리 계층은 DNN 연산 패턴(Weight Reuse, Activation Locality)에 최적화되어 있습니다. SPM 중심의 명시적 데이터 이동을 사용하며, RISC-V 연동을 통해 SoC 통합성을 높입니다.

### 2.1 주요 구성 요소
- **버퍼(Buffer)**:
  - **정의 및 역할**: 데이터 흐름 제어와 속도 조정을 위한 임시 저장. Tile 스트리밍에서 DRAM과 TE/VE 간 병목 완충.
  - **세부 유형**:
    - **Input Buffer**: DRAM → TE/VE, Tile 단위 적재, DMA prefetch 지원.
    - **Output Buffer**: TE/VE → DRAM, 결과 임시 저장, Double Buffering.
    - **Weight Buffer**: 가중치 재사용 (예: Attention Key/Value).
  - **특징**: FIFO 구조, 크기 MB 단위 (기본 2 MiB), Bank-aware 배치로 Conflict 최소화. HBM3E 연동으로 대역폭 확장.
  - **예시**: Transformer Attention에서 Key/Value Tensor를 Input Buffer에 preload → 연산 중 Output Buffer에 결과 누적.
- **Scratchpad Memory (SPM)**:
  - **정의 및 역할**: 재사용 최적화된 로컬 메모리. 명시적 DMA로 Cache Pollution 방지.
  - **L0 SPM (Tile-level)**: PE/TE 내부, 0.5~1 cycle, MatMul/Conv Tile 저장. Bank 분할 (기본 8 banks)로 Conflict 최소화. μNPU에서 PE 클러스터링 최적화.
  - **L1 SPM (Local)**: Per-Cluster, 32KB~256KB, 2~4 cycles, DMA 제어, Prefetch 힌트 (distance 파라미터), ping-pong 버퍼.
  - **L2 SPM (Shared On-chip)**: Cross-Cluster 공유, 수 MB (기본 2 MiB), 10~20 cycles, CCx Coherence (CCI 프로토콜 옵션), Cache Bypass for large DMA. Distributed SRAM으로 확장.
  - **L3/DRAM**: Off-chip (HBM/DDR), 100~200 cycles, Token Bucket BW 모델 (기본 102.4 GB/s). HBM3E로 대역폭 2배 향상.
- **NoC (Network-on-Chip)**: Cluster 간 데이터 이동, 256 GB/s BW, Queue Depth 모델링. PIM-like Compute-near-Memory 지원.
- **RISC-V 연동**:
  - **Loosely-coupled**: MMIO (0x4000_0000 base) + DMA, Doorbell/IRQ for Non-blocking.
  - **Tightly-coupled**: Custom ISA (ENQCMD_T, TWAIT, TBAR, TSTAT), Shared SPM/RegFile.

### 2.2 버퍼 vs SPM 비교
| 구분       | 버퍼 (Buffer)                  | SPM (Scratchpad)               |
|------------|--------------------------------|--------------------------------|
| 목적      | 흐름 완충, DMA 중첩            | 재사용 최적화, Bank-aware      |
| 위치      | 연산 유닛 ↔ DMA/NoC            | PE/Cluster 가까이             |
| 데이터 특성 | 임시/순차적                   | 재사용/임의, Tile Affinity    |
| 크기      | 큼 (MB, 기본 2 MiB)           | KB~MB (Banked, ping-pong)     |
| 구조      | FIFO, Double Buffering        | Banked (8 banks 기본), LRU 옵션, Prefetch    |
| NPU 최적화 | Tile 스트리밍, DMA 오버랩 (HBM3E) | Weight Reuse, CCx Coherence (CCI)   |
| RISC-V 연동 | Loose: DMA via MMIO           | Tight: Shared RegFile/ISA     |
| 단점      | SW 관리 부담                  | Contention, Miss Penalty (NoC latency)      |
| 2025 트렌드 | High-Bandwidth 인터페이스     | Partitioning, μNPU 클러스터링 |

### 2.3 장단점 분석
- **Scratchpad 기반 (주요 선택)**: Deterministic Latency, Bank Conflict 모델로 효율적. 단점: SW 부담 (DMA 스케줄링, Tiling 최적화). μNPU 표준.
- **Cache 기반 (옵션)**: SW 단순, 자동 Coherence. 단점: HW 복잡, Pollution, 에너지 소비 증가.
- **Loosely-coupled**: 독립성 높음, Multi-NPU Scalable. 단점: Submit P99 Latency >50 cycles, Data Copy Overhead (DMA setup 10 cycles).
- **Tightly-coupled**: 저지연 (ENQCMD_T ~5 cycles), Shared SPM으로 Locality 향상. 단점: Pipeline Stall Risk, ISA 확장 복잡.

## 3. 시뮬레이션 요구사항
### 3.1 추상화 수준 (시뮬레이션 레벨)
- **sim_0 (IA)**: Instruction Accurate. 기능 검증 중심, latency 무시. ONNX → NPU-IR 변환, 기본 명령어 실행. 사용 사례: 컴파일러/런타임 검증. 예시:
  ```python
  def sim_0_execute(instr):
      if instr.op == "GEMM_T":
          return "Functional result"  # No timing
  ```
- **sim_1 (IA_TIMING)**: Instruction Accurate with Timing. 타일 단위 지연 (T_tile = max(T_in + T_comp, T_out)), coarse-grain Contention (Bank Conflict). DMA latency (10 cycles/byte), Double Buffering. 사용 사례: Tile Scheduling 효과 분석.
- **sim_2 (CA_HYBRID)**: Cycle Accurate Hybrid. Event-Driven으로 Bank/NoC Queue, Prefetch, CCx 모델링. Py-V hook + Event Scheduler (Queue Depth 시계열). 사용 사례: Contention 분석 (Bank Conflict % >20% 시 병목).
- **sim_3 (CA_FULL)**: Full Cycle Accurate. Pipeline Stall, Custom ISA/MMIO 전체 사이클 모델링. Verilator co-sim, PLIC IRQ. 사용 사례: 실리콘 예측 (P99 Latency 분포).

### 3.2 Event-Driven 방식
Access/Miss/DMA Complete/IRQ 이벤트 트리거, Bank Conflict Stall 숨김. PIM-like 이벤트 추가 (Compute-near-Memory latency 감소).

### 3.3 Py-V 통합 및 구현 세부
- **MMIO**: 0x4000_0000 base, Doorbell (WO), IRQ_STATUS (RO). 예시:
  ```python
  def mmio_write(addr, value):
      if addr == 0x40000020:  # Doorbell
          enqueue_descriptor(value)  # Push to NPU queue
  ```
- **Custom ISA**: ENQCMD_T (desc_ptr → 큐 push), TWAIT (대기), TBAR (Barrier), TSTAT (Queue Depth/Utilization).
- **GPU 오프로딩**: GEMM → PyTorch/CUDA, Simulink for Gantt Chart (Matplotlib).
- **CLI 예시**:
  ```bash
  pyv-npu run model.onnx --sim_level CA_HYBRID --mode tight --report out/
  ```

### 3.4 성능 메트릭스
- **Throughput**: OPS/cycle, Tile 처리 속도, TE/VE/SPM Utilization (% 시간축 그래프).
- **Latency**: Level별 Access (L0 SPM: 1 cycle, L1 SPM: 2~4 cycles, L2 SPM: 10~20 cycles), P99 Submit (Loose vs Tight), Stall 합계.
- **Bandwidth**: GB/s (Token Bucket), Roofline Plot, Queue Depth 시계열, Bank Conflict %.

### 3.5 테스트 시나리오
- **워크로드**: ONNX Transformer (Attention Reuse), MobileNet Conv (Feature Map Tile), NPU-IR (GEMM_T, LOAD/STORE).
- **벤치마크**: Hit Rate >90% (SPM Reuse), DRAM Traffic ↓50% (Prefetch), Loose vs Tight P99 Latency ↓20%.
- **Edge Case**: Queue Full (EBUSY), Illegal ISA Trap, TWAIT Timeout (1000 cycles), Bank Conflict Overload (8 banks 초과).
- **Validation**: Theoretical Model vs Simulation (Mismatch <5%), Regression Test (Tile Size Sweep).

### 3.6 확장성
- **Multi-Cluster**: NoC (BW 256 GB/s, Latency per Hop 5 cycles).
- **Heterogeneous**: HBM vs DDR (BW 차이 시뮬), vNPU Virtualization.
- **2025 확장**: PIM-like (DRAM 내 Compute), μNPU Low-Power Mode (Partitioning).

## 4. 구조 다이어그램
### 4.1 전체 데이터 흐름
```
[RISC-V Core (Py-V)] 
  ├─ Custom ISA (Tight: ENQCMD_T, TWAIT, TBAR, TSTAT) ─┐
  │                                                     │
  └─ MMIO/CSR (Loose: Doorbell, IRQ_STATUS) ────────────┼─→ [NPU Cluster]
                                                        │
[Shared SPM/RegFile] ← NoC (256 GB/s, Queue Model) ─────┼─→ [Input Buffer (FIFO, DMA Prefetch)]
                                                        │
                                                        └─→ [L2 SPM (Shared, CCx, Cache Bypass)] 
                                                              │
                                                              └─→ [L1 SPM (Local, Banked 8, ping-pong, Prefetch_dist)]
                                                                    │
                                                                    └─→ [L0 SPM (Tile-level, PE-local, 1 cycle, Bank Conflict)]
                                                                          │
                                                                          └─→ [TE/VE Array (GEMM/Conv/Attention)]
                                                                                │
                                                                                └─→ [Output Buffer (Double Buffering)] ─→ DMA → [DRAM/HBM (L3, Token Bucket, HBM3E)]
```
- **Miss Flow**: TE → L0 SPM Miss → L1 SPM Miss → L2 SPM Miss → Buffer → NoC → DRAM.
- **보완**: Double Buffering으로 T_comp와 T_in/out 중첩, PIM-like으로 Miss Penalty 감소.

### 4.2 NPU 메모리 계층 (Cluster 기반)
```
[DRAM/HBM (L3, Off-chip, 100-200 cycles, HBM3E BW 102.4+ GB/s)]
  │
  └─ DMA/NoC (256 GB/s, Queue Depth, PIM-like Option)
      │
      └─ [L2 SPM (Shared On-chip, MB-level, CCx Coherence, 10-20 cycles, Distributed SRAM)]
            │
            └─ Cross-Cluster Sharing (vNPU Virtualization)
                │
                ├─ NPU Cluster #0
                │   │
                │   └─ [L1 SPM (Local, 32-256KB, 2-4 cycles, DMA, ping-pong, Prefetch, Banked 8)]
                │         │
                │         └─ [L0 SPM (Tile-level, PE/TE 내부, 0.5-1 cycle, Bank 분할, μNPU Partitioning)]
                │               │
                │               └─ TE/VE Compute (GEMM/Conv, Tiling Techniques)
                │
                ├─ NPU Cluster #1 (Identical Structure)
                │
                └─ NPU Cluster #N
```

## 5. 구현 및 배포 전략
### 5.1 단계별 로드맵
- **Phase 1 (1-2개월)**: Python Py-V 기반 sim_0 (IA)/sim_1 (IA_TIMING). MMIO/CSR hook, 기본 SPM 모델 (Banked FIFO). 테스트: ONNX 변환.
- **Phase 2 (2-3개월)**: sim_2 (CA_HYBRID). Event-Driven Scheduler, Custom ISA decoder. Simulink/Matplotlib for Gantt Chart/Queue 시계열.
- **Phase 3 (3-4개월)**: sim_3 (CA_FULL) with Verilator co-sim. PIM-like/vNPU 확장, SystemC 전환. Roofline, P99 분석.
- **도구**: Py-V (RISC-V), Verilator (RTL), PyTorch/CUDA (오프로딩), Simulink (Visualization).

### 5.2 리스크 관리
- **리스크 1**: Bank Conflict 왜곡 → 대응: Bank-aware Scheduler 최적화, TPU v5 Calibration (<5% Error).
- **리스크 2**: Custom ISA 호환성 → 대응: Custom-0 Opcode 예약, LLVM/Binutils Iterative Prototyping.
- **리스크 3**: Simulation Speed (sim_3 >1 hour) → 대응: GPU Offload, Parameter Sweep Automation.
- **리스크 4**: 2025 트렌드 변화 (HBM4) → 대응: Modular BW Model, Quarterly Update.

## 6. 결론 및 다음 단계
이 PRD는 메모리 계층을 세부적으로 정의하여 시뮬레이터의 실용성과 정확성을 높입니다. 2025 트렌드(HBM3E, Tiling, μNPU)를 반영하며, sim_0 ~ sim_3으로 점진적 개발 가능. 혜택: 병목 분석 시간 단축, PPA Trade-off 신속 평가.

**다음 단계**:
- PoC: sim_0 기반 구현 (1주 내).
- Review: HW/SW 팀 피드백.
- Integration: 전체 NPU 시뮬레이터 PRD 병합.

작성일: 2025-08-24 | 버전: v2.4 | 작성자: Grok (xAI)