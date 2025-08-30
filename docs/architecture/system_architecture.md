# PyV-NPU System Architecture

**Version: 2.0 (2025-08-26 기준)**
**Source PRD: NPU_Simulator_Memory_Hierarchy.md (v2.4)**

본 문서는 PyV-NPU 시뮬레이터의 공식 시스템 아키텍처를 정의합니다.

---

## 1. 시뮬레이션 레벨 (`sim_level`)

시뮬레이터는 다음과 같이 4단계의 명확한 추상화 수준을 지원합니다.

| `sim_level` | 이름          | 설명                                                                 |
|-------------|---------------|----------------------------------------------------------------------|
| **`IA`**        | 기능 정확성   | 타이밍 없이 기능의 정확성만 검증. (구 `L0`)                          |
| **`IA_TIMING`** | 명령어+타이밍 | 타일 단위의 개략적인 지연 시간과 기본적인 자원 충돌을 모델링. (구 `L1`) |
| **`CA_HYBRID`** | 하이브리드 CA | 이벤트 기반으로 NoC, 큐, 메모리 계층 등 주요 병목을 정밀하게 모델링. (구 `L2`) |
| **`CA_FULL`**   | 외부 연동 CA (옵션) | 외부 전문 도구(MATLAB/Simulink, SystemC 등)와 연동하여 완전한 Cycle-Accurate 시뮬레이션을 구현합니다. |

---

## 2. 메모리 아키텍처

본 시뮬레이터의 메모리 아키텍처는 NPU의 예측 가능한 성능 확보를 위해 **다계층 SPM(Scratchpad Memory)을 중심**으로 설계되었습니다. 하드웨어 제어 캐시는 비교 분석을 위한 연구 옵션으로만 고려하며, 기본 모델은 SPM입니다.

### 2.1. 주요 구성 요소

- **버퍼(Buffer)**: 데이터 흐름 제어와 속도 조정을 위한 임시 저장소입니다. (Input/Output/Weight 버퍼)
- **Scratchpad Memory (SPM)**: 재사용 최적화를 위한 명시적으로 제어되는 로컬 메모리입니다.
  - **L0 SPM (Tile-level)**: PE/TC 내부에 위치하며, 현재 처리 중인 데이터 타일을 저장합니다. (0.5~1 cycle)
  - **L1 SPM (Local)**: 클러스터별 로컬 메모리로, DMA로 제어됩니다. (2~4 cycles)
  - **L2 SPM (Shared On-chip)**: 여러 클러스터가 공유하는 메모리입니다. (10-20 cycles)
- **DRAM (L3)**: HBM/DDR과 같은 외부 메인 메모리입니다. (100-200 cycles)
- **NoC (Network-on-Chip)**: DRAM과 클러스터, 또는 클러스터 간 데이터 이동을 담당하는 인터커넥트입니다.

### 2.2. 핵심 설계: SPM vs. 캐시

NPU의 정형화된 작업 특성을 고려하여, 예측 가능성, 효율성, 하드웨어 단순성을 극대화하기 위해 하드웨어 관리형 캐시(Cache) 대신 소프트웨어 관리형 SPM(Scratchpad)을 메모리 계층의 중심으로 채택했습니다.

---

## 3. RISC-V (Py-V) 연동 방식

NPU는 RISC-V CPU(Py-V)의 제어를 받으며, 두 가지 연동 방식을 지원합니다.

- **Loose-Coupled (느슨한 결합):**
  - **방식:** MMIO(Memory-Mapped I/O)를 사용합니다.
  - **동작:** CPU가 특정 메모리 주소(Doorbell)에 NPU 프로그램의 디스크립터 주소를 쓰면, NPU가 이를 인지하고 작업을 시작합니다. 완료 시에는 인터럽트(IRQ)로 CPU에 알립니다.

- **Tight-Coupled (강한 결합):**
  - **방식:** RISC-V 커스텀 명령어를 사용합니다.
  - **동작:** `ENQCMD_T` (작업 제출), `TWAIT` (완료 대기) 같은 NPU 제어용 커스텀 명령어를 CPU가 직접 실행합니다. 데이터는 공유 메모리(SPM)나 레지스터를 통해 전달하여 지연 시간을 최소화합니다.

---

## 4. 전체 아키텍처 다이어그램

### 4.1. NPU 메모리 계층 (Cluster 기반)

```
[DRAM/HBM (L3, Off-chip, 100-200 cycles)]
  │
  └─ DMA/NoC (256 GB/s, Queue Depth)
      │
      └─ [L2 SPM (Shared On-chip, MB-level, 10-20 cycles)]
            │
            └─ NPU Cluster
                │
                ├─ [Input/Output Buffers (FIFO)]
                │
                └─ [L1 SPM (Local, 수백 KB, 2-4 cycles)]
                      │
                      └─ [L0 SPM (Tile-level, 수 KB, 0.5-1 cycle)]
                            │
                            └─ [TC/VC Compute Engines]
```

### 4.2. 전체 데이터 흐름

```
[RISC-V Core (Py-V)]
  ├─ Custom ISA (Tight-Coupled) ─┐
  │                               │
  └─ MMIO/CSR (Loose-Coupled) ────┼─→ [NPU Cluster]
                                  │
[Shared SPM/RegFile] ← NoC ──────┼─→ [Input Buffer]
                                  │
                                  └─→ [L2 SPM] → [L1 SPM] → [L0 SPM]
                                        │
                                        └─→ [TC/VC Array] → [Output Buffer] → DMA → [DRAM/HBM]
```
- **Miss Flow**: TC → L0 Miss → L1 Miss → L2 Miss → Buffer → NoC → DRAM.
- **최적화**: Double Buffering으로 Compute와 I/O를 중첩시켜 Latency Hiding.

---

## 5. 아키텍처 확장 계획 (Architecture Extension Plans)

현재 아키텍처는 SPM(Scratchpad Memory) 기반의 동기식 클럭 모델을 중심으로 설계되었으나, 시뮬레이터의 정확도와 활용성을 높이기 위해 다음과 같은 확장 계획을 고려합니다.

### 5.1. 캐시 시스템 도입 (Cache System Integration)

- **L1/L2/L3 캐시 모델**: 현재의 SPM 중심 설계 외에, 연구 및 비교 분석을 목적으로 하드웨어 관리형 캐시 시스템 도입을 고려할 수 있습니다.
  - **L1 캐시**: `Py-V` RISC-V 코어에 L1 캐시를 추가하여, 특히 `Tight-Coupled` 모드에서의 제어 오버헤드를 더 정확하게 모델링합니다.
  - **L2/L3 공유 캐시**: CPU와 NPU 클러스터가 공유하는 L2 또는 L3 캐시를 구현하여, 불규칙한 메모리 접근 패턴에 대한 대응 및 시스템 레벨의 데이터 흐름을 분석합니다.

### 5.2. 비동기 클럭 도메인 (Asynchronous Clock Domains)

- **독립 클럭 속도**: 현재의 단일 클럭(1.2 GHz) 구조에서 나아가, 실제 SoC와 유사하게 CPU, TC, VC 등 주요 컴포넌트가 각기 다른 클럭 속도를 갖는 비동기 클럭 도메인을 구현합니다. 이를 통해 전력 및 성능 분석의 현실성을 높입니다.

---

## 6. 보안 고려사항 (Security Considerations)

현재 시뮬레이터는 신뢰할 수 있는 로컬 환경에서 단일 사용자가 연구 목적으로 사용하는 것을 가정합니다. 따라서 다음과 같은 보안 모델을 가집니다.

- **입력 신뢰**: 시뮬레이터에 입력되는 ONNX 모델 및 설정 파일들은 악의적인 코드를 포함하지 않는다고 신뢰합니다. 입력값에 대한 엄격한 보안 검증(Sanitization)은 수행하지 않습니다.
- **실행 환경**: 시뮬레이터는 사용자의 시스템 권한으로 실행되며, 별도의 샌드박스 환경을 제공하지 않습니다.
- **네트워크 접근**: 시뮬레이터는 외부 네트워크에 접근하지 않습니다.

향후 다중 사용자 환경이나 웹 기반 서비스로 확장될 경우, 별도의 위협 모델링과 보안 기능(입력 검증, 사용자 인증, 권한 분리 등) 설계가 필요합니다.
