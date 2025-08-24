# PyV-NPU 통합 실행 계획서 (통합 PRD v1.1 완전판)
작성일: 2025-08-09 · 작성자: ChatGPT (with 창훈)

## 0) 개요 (Scope & Goals)
- **목표**  
  RISC-V 기반 **PyV-NPU 시뮬레이터**를 단계별(Level)로 개발하여, ONNX/MLIR 입력을 **NPU-IR**로 변환 → 스케줄링 → 타이밍·자원 모델링 → 타일 수준·이벤트 기반·사이클 수준 분석 및 리포트 생성.  
  RISC-V **프론트엔드(LLVM/MLIR, ONNX 변환, Custom ISA)**와 **백엔드(NPU, 메모리 계층, DMA, NoC, Cache Coherence)** 설계를 통합.  
  **[확장]** Tight-coupled Custom ISA 기반 **Tensor Engine(TE)** 제어 명령어 세트를 정의·시뮬레이션·검증하여 CPU介재 없이 저지연 커널 제출 가능 구조 포함.

- **타깃 워크로드**: LLM 추론(Attention/MLP), Conv/GEMM 기반 비전 네트워크.  
- **비목표(본 버전)**: OS/드라이버 전체 모델, 전면 RTL 구현, 정확한 전력 추정(전력은 추후 모델).

## 1) 사용자 스토리
- **시스템 아키텍트**: 타일 크기·메모리 파라미터 스윕 → PPA 트레이드오프 도출.
- **컴파일러 엔지니어**: ONNX→MLIR→NPU-IR 패스(퓨전/타일링/프리패치/배치) 효과 검증.
- **하드웨어 엔지니어**: DMA 채널 수, SPM bank 수, NoC 대역폭, DRAM 채널 변경 시 처리량/지연 민감도 평가.
- **마이크로아키텍처 엔지니어**: Tight-ISA vs Loose-MMIO 모드의 제출 지연(P99 latency) 비교, TE 큐 활용도 분석.

## 2) 요구사항 (Requirements)

### 기능 요구
1. **입력**: ONNX(우선), (옵션) MLIR  
2. **출력**:  
   - (a) NPU-IR 트레이스(JSONL)  
   - (b) 실행 타임라인(CSV)  
   - (c) 리포트(HTML/SVG/CSV): Gantt, 자원 사용률, Roofline, 병목 Top-N  
   - (d) 실험 메타데이터(run.yaml)
3. **프론트엔드**: LLVM/MLIR 변환 + Dialect 설계
4. **백엔드 모델링**: TE/VE/DMA/SPM/L2/DRAM/NoC + Cache Coherence 옵션
5. **스케줄링**: List-sched + CP-prio + Bank-aware 배치
6. **메모리 모델**: Banked SPM, Double-Buffer DMA, NoC/DRAM 큐
7. **정밀도/양자화**: FP16/BF16/FP8/INT8 + (옵션) Weight-VQ
8. **RISC-V 연동**:  
   - Loose-coupled: MMIO Doorbell  
   - Tight-coupled: Custom ISA (`ENQCMD_T`, `TWAIT`, `TBAR`, `TSTAT`)로 TE 직접 제어
9. **TE Custom ISA 세부 요구**:  
   - **명령어**  
     - `ENQCMD_T rd, rs1, rs2` : 커널 디스크립터 제출  
     - `TWAIT rs1` : ticket 완료 대기  
     - `TBAR imm` : Barrier (tile/cluster/global)  
     - `TSTAT rd` : 상태/큐 깊이/유휴율 조회  
   - **Descriptor 필드**: op, addr_in/out, tile_shape, bank_mask, prio, prefetch_dist
   - **CSR/MMIO 연계**: TE 큐 주소, Doorbell, IRQ 상태, 성능 카운터

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
GEMM_T(tile_m,tile_n,tile_k, quant, affinity={TE})
...
BARRIER(scope)
WAIT(ev)

# TE ISA 확장
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
- **자원/경합(L2)**:  
  Bank 충돌, NoC/DRAM 큐 모델, Token Bucket 기반 대역폭 모델  
- **스케줄러**: List-sched + CP-prio, Bank-aware 배치

## 6) 메모리/데이터 이동
- **SPM**: banked, ping-pong 버퍼, bank conflict 모델  
- **Prefetch**: prefetch_distance 힌트 반영  
- **Cache bypass**: 대용량 DMA시 L2 우회

## 7) 검증/리포트
- KPI: Latency, Throughput, Utilization, Roofline, 병목 Top-N  
- **추가 분석**: TE ISA 효율, Loose vs Tight 비교, 큐 활용도 시계열 그래프

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
| TE 수 | 2 | Tensor Engine |
| VE 수 | 4 | Vector Engine |
| SPM 용량 | 2 MiB | banked scratchpad |
| SPM bank 수 | 8 | bank conflict 모델 |
| DMA 채널 | 2 | double buffer 지원 |
| DRAM BW | 102.4 GB/s | Token bucket 모델 |
| NoC BW | 256 GB/s | 링크 대역폭 |
| 주파수 | 1.2 GHz | 타이밍 계산 |
| 타일(M,N,K) | 128,128,64 | GEMM 기준 |

## 10) 마일스톤
- **M0**: NPU-IR, L0 end-to-end  
- **M1**: L1 타일 타이밍 + Bank 모델  
- **M2**: L2 이벤트 기반 경합 + Cache Coherence  
- **M3**: TE Custom ISA 설계 + MLIR Dialect/Lowering  
- **M4**: Py-V 연동, Loose/Tight 비교 E2E  
- **M5**: 회귀세트, 문서화, 튜토리얼

## 11) 리스크 & 대응
| 리스크 | 영향 | 대응 |
|---|---|---|
| Custom ISA 툴체인 난이도 | 개발 지연 | 초기 `.insn` asm → 후속 LLVM/binutils 확장 |
| TE ISA 시뮬 오버헤드 | 성능 분석 왜곡 | L2 모델 최적화, L3 보정 |
| ISA 호환성 | 표준 충돌 가능 | custom-0~3 opcode 예약 |

## 12) 블록 다이어그램
```
RISC-V Core (Py-V)
 ├─ LLVM/MLIR Frontend
 ├─ Loose-coupled MMIO
 ├─ Tight-coupled TE ISA
 └─ Runtime API
        ▼
    NPU Cluster
     ├─ Scheduler
     ├─ DMA
     ├─ SPM (banked)
     ├─ TE Array
     └─ VE Array
```

## 13) RISC-V ↔ NPU 연동 방식 (Py-V 기반, 확정안)
RISC‑V ↔ NPU 연동 방식 (Py‑V 기반, 확정안)
본 시뮬레이터는 https://github.com/kyaso/py-v를 RISC‑V 실행 프런트엔드로 채택한다. 연동 모드는 Loose‑coupled (MMIO/CSR) 와 Tight‑coupled (Custom ISA) 를 모두 지원한다. 두 모드는 동일한 NPU‑IR / 디스크립터 포맷을 공유하며, 제출 경로(submit path) 만 다르다.

A) Loose‑coupled (MMIO/CSR) 모드
메모리 맵 (예시)

NPU_MMIO_BASE = 0x4000_0000

+0x00: QUEUE_BASE_LO/HI (RW) – 커널 디스크립터 링 버퍼 베이스

+0x10: QUEUE_HEAD/TAIL (RW) – 호스트/디바이스 큐 인덱스

+0x20: DOORBELL (WO) – 새 작업 제출 알림 (head 값 또는 ticket)

+0x24: IRQ_STATUS (RO), IRQ_MASK (RW)

+0x40: PERFCNT_* (RO) – TE/VE/DMA/Bank 활용도, 큐 깊이 등 카운터

디스크립터 (공통 포맷, 64B 정렬)

rust
Copy
Edit
u32 op;             // GEMM_T / CONV2D_T / ATTN_T / ELTWISE / ...
u32 flags;          // prio/stream/bank_mask/prefetch_dist
u64 in0_addr; u64 in1_addr; u64 out_addr;   // SPM/DRAM VA (시뮬 주소)
u32 tile_m; u32 tile_n; u32 tile_k;         // 또는 op별 shape
u32 reserved[5];                            // 확장
u32 ticket;                                 // 제출 식별자
호스트 제출 절차

디스크립터 링에 기록 → QUEUE_TAIL 증가

DOORBELL 레지스터에 Tail(or ticket) 쓰기

필요 시 IRQ enable 후 폴링/IRQ 대기

Py‑V 훅

MMIO write/read 핸들러를 Py‑V의 버스 모델에 등록

IRQ 발생 시 Py‑V의 외부 인터럽트 라인 세트

NPU 사이드

Doorbell 수신 → 새 디스크립터 페치 → 스케줄러/큐로 인입 → 실행

완료 시 IRQ_STATUS 세트 + optional ticket 업데이트

B) Tight‑coupled (Custom ISA) 모드
목표: 문맥 전환/드라이버 오버헤드 없이 저지연 커널 제출

명령어 세트 (초안)

ENQCMD_T rd, rs1, rs2

rs1: desc_ptr (디스크립터 주소)

rs2: flags (prio/stream)

rd: ticket 반환

TWAIT rs1 – rs1의 ticket 완료까지 대기

TBAR imm – 타일/클러스터/글로벌 배리어

TSTAT rd – TE 상태/큐 깊이/유휴율 패킹 반환

인코딩 가이드 (RISC‑V custom‑0 예시)

major opcode: CUSTOM‑0 (0b0001011)

ENQCMD_T: R‑type, funct7=0x2E, funct3=0x0

TWAIT : I‑type, funct3=0x1

TBAR : I‑type, funct3=0x2, imm에 scope 인코딩

TSTAT : I‑type, funct3=0x3

시뮬레이터는 합법/불법 인스트럭션을 소프트 디코더로 판별하며, 실제 하드웨어로의 이전 시 binutils/LLVM 백엔드 확장을 진행.

Py‑V 훅 (필수 포인트)

Decoder hook: 위 인코딩 매치를 추가, 해당 핸들러 호출

Handler:

ENQCMD_T: desc_ptr에서 디스크립터 로드 → 내부 NPU 큐에 push → ticket 생성/rd에 기록

TWAIT: 지정 ticket의 완료 future 대기 (시뮬시간)

TBAR: 스케줄러에 범위 배리어 enqueue

TSTAT: 현재 큐 깊이/유휴율/활용도 스냅샷을 패킹해 rd 반환

CSR map (선택): TE 큐 베이스/마스크/IRQ 상태 등을 CSR로 노출 (MMIO 대체/보완)

C) 공통 실행 타임라인
호스트에서 ONNX/MLIR → NPU‑IR 생성 → 디스크립터 빌드

Loose: 링에 쓰기 + DOORBELL / Tight: ENQCMD_T 실행

스케줄러: bank‑aware + CP‑prio로 DMA/TE/VE에 타일 배치

완료 시 ticket/IRQ 갱신 → Loose: 폴링/IRQ read / Tight: TWAIT 리턴

리포트: 타임라인(Gantt), 자원 활용도, TE 큐 깊이 시계열, P99 제출 지연 비교

D) 에러/예외 처리 (시뮬레이터 규칙)
디스크립터 검증 실패: 해당 ticket ERROR 상태로 즉시 완료 + 원인 로그

큐 full: ENQCMD_T는 SW‑visible 리트라이 코드(예: rd=-EBUSY) 반환

불법 명령어: Tight 모드 off일 때는 Illegal Instruction 트랩 발생(옵션)

타임아웃: TWAIT에 최대 대기 사이클 옵션(디폴트 무한)

E) 성능 카운터 (리포트 연계)
제출/수락 지연(T_submit) 분포, P50/P95/P99

TE/VE/DMA/Bank 활용도(%) 시간축

큐 깊이 시계열, 배리어 대기 시간, bank conflict stall 합계

Loose vs Tight 모드별 타일 크기 변화에 따른 T_submit/T_comp 비율

F) CLI/API (갱신)
bash
Copy
Edit
# Loose 모드 (MMIO/CSR)
pyv-npu run model.onnx \
  --sim_level CA_HYBRID \
  --mode loose \
  --mmio-base 0x40000000 \
  --queue-size 1024 \
  --report out/loose_run

# Tight 모드 (Custom ISA)
pyv-npu run model.onnx \
  --sim_level CA_HYBRID \
  --mode tight \
  --isa enqcmd,twait,tbar,tstat \
  --report out/tight_run
python
Copy
Edit
from pyv_npu import Simulator

# Loose
sim = Simulator(model="model.onnx", level="L2", mode="loose",
                mmio_base=0x4000_0000, queue_size=1024)
sim.run()

# Tight
sim = Simulator(model="model.onnx", sim_level="CA_HYBRID", mode="tight",
                te_isa=["enqcmd","twait","tbar","tstat"])
sim.run()
G) 검증 플랜 (요약)
기능: 동일 NPU‑IR로 Loose/Tight 모드 결과 일치성 검증(타일 완료 순서/결과)

성능: 제출 지연(P99), 큐 깊이, TE 유휴율 비교. 타일 크기/prefetch_dist 스윕 자동화.

내고장성: 큐 full/오류 디스크립터/Illegal‑ISA 경로 유닛 테스트.


RISC‑V ↔ NPU 연동 구조도 (Py‑V 기반)
perl
Copy
Edit
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
Legend: DBB=Double Buffering, CP-prio=Critical-Path priority