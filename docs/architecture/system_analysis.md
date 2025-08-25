v250817
## 1. `docs/` 문서 종합 분석 요약

- **프로젝트 목표:** ONNX 모델을 입력받아 NPU의 동작을 상세히 시뮬레이션하고, RISC-V CPU와의 연동을 통해 End-to-End 성능 분석을 목표로 함.
- **핵심 아키텍처:** `loose-coupled`(MMIO)와 `tight-coupled`(Custom ISA) 두 가지 CPU-NPU 연동 방식을 지원하며, `CA_HYBRID` 이벤트 기반 시뮬레이션이 핵심 기능으로 구현됨.
- **개발 현황:** L2 시뮬레이션의 핵심 골격은 완성되었으나, 컴파일러 최적화 패스, 코스트 모델 정교화, Py-V와의 완전한 통합 등은 향후 과제로 남아있음. 프로젝트는 칸반 보드와 개발 이력 문서를 통해 체계적으로 관리되고 있음.

## 2. `pyv_npu` 모듈 구조 및 폴더별 기능 분석

- **`cli/` (명령줄 인터페이스):** `compile`/`run` 명령어를 처리하는 최상위 제어 센터. 시뮬레이션 파이프라인 전체를 조율.
- **`ir/` (중간 표현):** ONNX 모델을 시뮬레이터가 사용하는 내부 `Model IR` 형식으로 변환.
- **`compiler/` (컴파일러):** `Model IR`을 NPU가 실행할 `NPU Program`으로 매핑. `loose`/`tight` 모드에 따라 다른 명령어 시퀀스를 생성하는 핵심 로직 포함. 최적화 패스는 현재 뼈대만 구현됨.
- **`isa/` (명령어 세트):** `NPU Program`의 구조와 `tight` 모드용 커스텀 RISC-V 명령어의 데이터 형식을 정의.
- **`runtime/` (런타임):** 시뮬레이터의 심장부. 이벤트 기반 스케줄러, 자원 경합 모델, 연산 비용 모델 등 실제 시뮬레이션 실행 로직이 모두 포함됨.
- **`bridge/` (브릿지):** `py-v` CPU 시뮬레이터와 NPU를 연결하는 MMIO 메모리 모델을 구현하여 `loose` 모드 연동을 담당.
- **`utils/` (유틸리티):** 시뮬레이션 결과를 JSON으로 요약하고, 이를 바탕으로 대화형 HTML 간트 차트 및 텍스트 기반 리포트를 생성.
- **`config.py` (설정):** 시뮬레이션에 필요한 모든 하드웨어 및 실행 파라미터를 중앙에서 관리하며, YAML 및 CLI를 통한 계층적 설정 기능을 제공.

---
## 1. PyV-NPU 시뮬레이터 전체 현황 요약

PyV-NPU 시뮬레이터는 ONNX 모델을 입력받아 NPU-IR로 변환하고, 이를 기반으로 NPU의 동작을 시뮬레이션하며, 상세한
타이밍 및 자원 활용 리포트를 생성하는 것을 목표로 합니다. 궁극적으로 RISC-V 시뮬레이터(Py-V)와의 연동을 통해
End-to-End 시뮬레이션을 구현하고자 합니다.

---

**1. 핵심 파이프라인 및 구조**

*   **데이터 흐름**: ONNX 모델 → `Model IR` → `NPU Program` (NPUOp 시퀀스) → `스케줄 결과` →
`리포트/간트차트/JSON`
*   **주요 모듈**:
    *   `cli/`: 커맨드라인 인터페이스 (compile / run)
    *   `ir/`: ONNX 모델을 내부 `Model IR`로 변환
    *   `compiler/`: `Model IR`을 `NPU Program`으로 매핑 (loose/tight 모드 지원)
    *   `isa/`: `NPU Program` 및 RISC-V 확장 ISA 정의
    *   `runtime/`: 스케줄러, 시뮬레이터, 리소스 모델링 (VC/TC 코스트, 메모리, 버스)
    *   `bridge/`: 메모리 브릿지
    *   `utils/`: 로깅, 리포트, 시각화
    *   `config.py`: 시뮬레이션 파라미터 설정

---

**2. 현재 구현 상태 및 주요 성과**

*   **CLI 및 설정**: `compile`/`run` 명령어와 `loose`/`tight` 모드를 완벽하게 지원하며, YAML 및 CLI 인자를 통한
계층적 설정 로딩이 구현되어 있습니다.
*   **컴파일러**: ONNX → `Model_IR` → `NPU_Program` 파이프라인이 명확하며, `mapper.py`는 `loose`/`tight` 모드에
따른 프로그램 생성을 정확히 구현합니다.
    *   **완료된 작업**: `DRAM 채널/뱅크 매핑 정책` (`M-02`) 및 `TC 코스트 모델 v1` (`C-01`) 구현을 위해
컴파일러에 메모리 할당 단계가 추가되었습니다.
*   **런타임**:
    *   `simulator.py`는 설정된 `sim_level`에 따라 `IA`(단순) 또는 `IA_TIMING`/`CA_HYBRID`(이벤트 기반) 스케줄러를 선택합니다.
    *   `scheduler.py`의 `event_driven_schedule`는 이벤트 큐를 통해 의존성과 자원 상태를 모두 고려하는 핵심
로직을 포함합니다.
    *   `resources.py`는 `BandwidthTracker`와 `BankTracker`를 구현하여 자원 경합을 모델링합니다.
    *   `te.py`는 `fill/drain` 오버헤드를 포함한 Systolic Array 비용 모델을 구현했습니다.
    *   **완료된 작업**: `스케줄러 Stall 계산 로직` (`S-02`)이 수정되어 자원 경합 시 `stall_cycles` 계산 정확도가
향상되었고, `tight` 모드 제어 경로 (`T-01`)가 구체화되어 NPU 작업 큐 관리 및 제어 지연 시간(doorbell, csr, issue
_rate)이 반영되었습니다.
*   **리포팅**: `reporting.py`는 상세한 `report.json`을 생성하며, `viz.py`와 연계하여 대화형 HTML 간트 차트 및
콘솔용 ASCII 간트 차트를 시각화하는 기능까지 갖추고 있습니다.
    *   **완료된 작업**: `리포트 확장` (`R-01`)을 통해 스케줄러에 stall 원인 추적 기능이 추가되었고, 리포트에
stall 정보가 포함되며 간트 차트에 시각화됩니다.
*   **RISC-V 연동 (Loose/Tight 모드)**: `pyv_npu/isa/riscv_ext.py`에 `ENQCMD_T`, `TWAIT` 등 Custom ISA가 정의되어
있으며, `cli/main.py`에서 `--mode loose/tight` 옵션을 제공합니다. `tight` 모드 시뮬레이션 및 리포트 기능이
강화되었습니다.

---

**3. 강점 및 보완 포인트**

*   **강점**:
    *   컴파일러 → 런타임 파이프라인이 명확하게 구축되어 있습니다.
    *   L1/L2 스케줄러 코어가 안정적으로 구현되어 있습니다.
    *   리포팅 기능이 자동화되어 시뮬레이션 결과를 직관적으로 파악할 수 있습니다.
    *   `loose` 및 `tight` 모드 지원을 위한 기본적인 구조가 잘 잡혀 있습니다.
*   **보완 포인트 및 향후 계획**:
    *   **코스트 모델 정교화**: TC/VC 코스트 모델 (타일-사이즈, systolic fill/drain, 벡터 길이 등) 및 메모리/버스
모델 (SPM bank/port 수, DMA burst, NoC 홉 딜레이, DRAM 채널/페이지 정책)을 고도화해야 합니다. (EPIC-COST)
    *   **컴파일러 패스 연결**: 타일링 결정, 연산 Fusion, QAT/INT8 매핑 등 최적화 패스들이 현재 이름만 있거나
스텁 상태이므로, 실제 로직을 구현하고 연결해야 합니다. (EPIC-COMP)
    *   **의존성/큐/레지스터 모델**: `tight` 모드에서 레지스터-마이크로코드 큐/doorbell/CSR latency 반영 등 세부
모델링이 필요합니다. (EPIC-CTRL)
    *   **PPA 훅**: op별 에너지 모델 핸들러 인터페이스를 마련하여 리포트에 nJ/op, Watt 추정치를 추가해야 합니다.
    *   **Py-V 연동**: RISC-V 시뮬레이터(Py-V)와의 완전한 연동이 필요합니다. (M4 마일스톤)
    *   **문서화 및 테스트**: 개발자 가이드, 벤치 스크립트, 테스트 스위트 확장 등이 필요합니다. (EPIC-REP,
EPIC-OPS)

---

## 2. pyv_npu/compiler 모듈 분석

`pyv_npu/compiler` 모듈은 ONNX에서 변환된 상위 레벨의 `Model IR`을 시뮬레이터가 실행할 수 있는 하위 레벨의 `NPU
Program`으로 변환하는 역할을 담당합니다.

**주요 구성 요소 및 기능:**

1.  **`mapper.py`**:
    *   이 모듈은 컴파일러의 핵심으로, `Model IR Graph`를 `NPU Program`으로
매핑합니다.
    *   `loose` 모드에서는 `Model IR` 노드를 직접 `NPUOp`로 변환하며,
스케줄러에 필요한 인자(예: `tile_m`, `tile_n`, `tile_k`)를 추가합니다.
    *   `tight` 모드에서는 각 NPU 연산에 대해 `ENQCMD_T` (명령어 제출) 및
`TWAIT` (완료 대기) 명령어를 쌍으로 생성하여, CPU가 NPU 작업을 제어하는
시나리오를 시뮬레이션할 수 있도록 구현되어 있습니다. 이는 PRD에 명시된
`tight-coupled` 연동 방식을 충실히 따르고 있습니다.

2.  **`allocator.py`**:
    *   `Allocator` 클래스를 정의하며, 프로그램 내의 모든 고유 텐서에
메모리 주소를 할당하는 간단한 범프(bump) 할당자 역할을 합니다. 이는
시뮬레이션에 필요한 기본적인 메모리 관리 기능을 제공합니다.

3.  **`passes/` 디렉토리 (`fusion.py`, `quantization.py`, `tiling.py`)**:
    *   이 디렉토리에는 `fusion`, `quantization`, `tiling`과 같은 컴파일러
최적화 패스들이 정의되어 있습니다.
    *   하지만 현재 이 파일들은 모두 플레이스홀더(placeholder) 함수(`TODO`
주석과 `return g`)로만 구성되어 있으며, 실제 최적화 로직은 아직 구현되지
않았습니다. 이는 `kanban_board.md` 및 `code_review.md` 문서에서 언급된
"보완 포인트" 또는 "Backlog" 항목과 일치합니다.

**분석 요약:**

`pyv_npu/compiler`는 `Model IR`을 `NPU Program`으로 변환하는 핵심 로직과
기본적인 메모리 할당 기능을 견고하게 구현하고 있습니다. 특히 `loose` 및
`tight` 모드에 대한 매핑 지원은 시뮬레이터의 중요한 특징입니다. 반면, 고급
컴파일러 최적화 패스(퓨전, 양자화, 타일링)는 향후 구현될 예정으로, 현재는
구조만 갖춰져 있습니다.

---

## 3. py-v/ 모듈 분석

**Py-V: 파이썬으로 구현된 사이클-정확 RISC-V CPU 시뮬레이터 분석**

Py-V는 파이썬으로 작성된 사이클-정확 RISC-V CPU 시뮬레이터이며, 동시에
범용적인 디지털 하드웨어 모델링 라이브러리 역할도 수행합니다.

---

**1. 핵심 아키텍처 및 구성 요소**

Py-V는 하드웨어 설계를 모듈화하고 시뮬레이션하기 위한 여러 핵심 구성 요소로
이루어져 있습니다.

*   **`pyv/simulator.py`**: 시뮬레이션 커널의 핵심입니다. 이벤트 기반의
시뮬레이션 큐를 관리하고, 클럭 틱(tick)을 처리하며, 시뮬레이션 실행을
제어합니다.
*   **`pyv/module.py`**: 모든 하드웨어 모듈의 추상 기본 클래스입니다.
Verilog 모듈과 유사하게 하드웨어 요소를 캡슐화합니다.
*   **`pyv/port.py`**: 모듈 간 연결을 위한 `Input`, `Output`, `Wire` 및
`Constant` 신호를 정의합니다. 포트의 값 변경 시 민감한 메서드(sensitive
methods)를 트리거하는 메커니즘을 포함합니다.
*   **`pyv/reg.py`**: 레지스터(`Reg`) 및 레지스터 파일(`Regfile`)을
구현합니다. 사이클-정확 시뮬레이션을 위해 클럭 동기화된 동작을 지원합니다.
*   **`pyv/mem.py`**: 메모리 모듈을 구현하며, 2개의 읽기 포트와 1개의 쓰기
포트를 가진 간단한 바이트 단위 메모리 모델을 제공합니다.
*   **`pyv/clocked.py`**: `Clock` 클래스를 통해 레지스터와 메모리 같은 클럭
동기화된 요소들의 동작을 관리합니다.
*   **`pyv/isa.py`**: RISC-V 명령어 세트 아키텍처(ISA) 정의(opcode, 명령어
형식, CSR 주소, 예외 등)를 포함합니다.
*   **`pyv/stages.py`**: RISC-V CPU의 5단계 파이프라인(Instruction Fetch,
Decode, Execute, Memory, Write-Back)을 구현합니다. 또한 `BranchUnit` 및
`ExceptionUnit`과 같은 보조 유닛도 포함합니다.
*   **`pyv/csr.py`**: RISC-V의 제어 및 상태 레지스터(CSR)를 구현합니다.
`misa`, `mepc`, `mcause`, `mtvec` 등 주요 CSR에 대한 접근 및 예외 처리
로직을 포함합니다.
*   **`pyv/models/singlecycle.py`**: `pyv/stages.py`의 구성 요소들을
조합하여 실제 단일 사이클 RISC-V CPU 모델을 구현합니다.
*   **`pyv/util.py`**: 비트 조작, `PyVObj` 객체 관리(`VContainer`, `VMap`,
`VArray`를 통한 계층적 구조화) 등 다양한 유틸리티 함수를 제공합니다.
*   **`main.py`**: 컴파일된 RISC-V 바이너리 프로그램을 Py-V 시뮬레이터에서
실행하는 진입점입니다.
*   **`programs/`**: 테스트용 RISC-V 어셈블리/C 프로그램(예: `loop_acc`,
`fibonacci`, `npu_test`)을 포함합니다.

---

**2. 시뮬레이션 모델 및 동작 방식**

Py-V는 **이벤트 기반(event-driven)** 및 **사이클-정확(cycle-accurate)**
시뮬레이션 모델을 따릅니다.

*   **이벤트 기반**: 모듈 입력의 변화가 이벤트로 간주되어 해당 모듈이
시뮬레이션 큐에 추가됩니다. 시뮬레이터는 큐가 빌 때까지 모듈을 처리하며,
큐가 비면 현재 사이클이 완료된 것으로 간주합니다.
*   **사이클-정확**: `Clock` 클래스를 통해 명시적인 클럭 틱을 처리하며, 각
클럭 틱에서 레지스터와 메모리의 상태가 동기화됩니다.

---

**3. 주요 기능 및 특징**

*   **RTL 수준 모델링**: 모듈, 포트, 레지스터, 메모리 등 하드웨어 구성
요소를 파이썬 객체로 추상화하여 RTL 수준의 설계를 가능하게 합니다.
*   **RISC-V ISA 지원**: 기본적인 RISC-V 명령어 세트, CSR, 예외 처리(
`ECALL`, `MRET`)를 지원합니다.
*   **5단계 파이프라인 구현**: `IF`, `ID`, `EX`, `MEM`, `WB` 스테이지가
명확하게 분리되어 구현되어 있습니다.
*   **NPU 연동 지원**: `pyv/models/singlecycle.py`의 `SingleCycle` CPU
모델은 `pyv_npu.bridge.mem.NPUControlMemory` 및
`pyv_npu.runtime.scheduler.Scheduler`를 직접 통합하여 NPU 시뮬레이터와의
연동을 지원합니다. `programs/npu_test/npu_test.S`는 NPU 도어벨을 트리거하는
예제 코드를 포함하여 Py-V가 NPU 시뮬레이터의 RISC-V CPU 프론트엔드 역할을
수행하도록 설계되었음을 보여줍니다.
*   **디버깅 및 관측성**: `set_probes` 기능을 통해 특정 포트의 값을 로깅할
수 있으며, 상세 로깅(`pyv/log.py`)을 지원합니다.
*   **유연한 하드웨어 모델링**: `VContainer`, `VMap`, `VArray`와 같은
유틸리티 클래스를 통해 `PyVObj` 인스턴스들을 유연하게 그룹화하고 계층적으로
관리할 수 있습니다.
*   **테스트 및 CI/CD**: `pytest` 기반의 단위 테스트가 잘 구축되어 있으며,
GitHub Actions를 통해 CI(`ci.yml`) 및 문서 빌드(`docs.yml`)가 자동화되어
있습니다.

---

**4. 최근 변경 사항 (CHANGELOG.md 기준)**

*   **v0.5.0**: `ECALL` 및 `MRET` 명령어 지원 추가, `VArray`, `VMap`,
`VContainer` 구조 도입.
*   **v0.4.0**: CSR 명령어 구현, "on stable" 메커니즘 추가, 포트 초기화 시
민감한 메서드 등록.
*   **v0.3.0**: 로깅 개선, 포트 연결을 위한 `<<` 연산자 오버로드.
*   **v0.2.0**: `PortX`, `RegX` 제거를 통한 시뮬레이션 커널 단순화, `Clock`
클래스 도입, 예외 처리, 동기식 리셋, 이벤트 기능 추가.

---

## 4. RISC-V 확장 기능 활용 여부

현재 구현된 코드는 RISC-V의 특정 확장 기능을 활용하고 있습니다.

1.  **Custom-0/1 opcode space 활용 여부**:
    *   **예, 활용하고 있습니다.** `tight-coupled` 모드에서 NPU 작업을
제어하기 위해 `ENQCMD_T` 및 `TWAIT`와 같은 커스텀 명령어들을 사용합니다. 이
명령어들은 `docs/prd/README.md` 문서의 "RISC-V ↔ NPU 연동 방식" 섹션에서
`CUSTOM-0` opcode space를 활용하는 것으로 명시되어 있습니다.
`pyv_npu/compiler/mapper.py`의 `_map_to_tight_program` 함수에서 이러한
커스텀 명령어들을 생성합니다.

2.  **Vector Extension (V) 활용 여부**:
    *   **아니요, 표준 RISC-V Vector Extension (RVV) opcode space를
직접적으로 활용하고 있지는 않습니다.**
    *   하지만 시뮬레이터는 `Vector Core (VE)`을 모델링하고 있으며,
`docs/code_review.md` 및 `docs/kanban_board.md`에서 `VE 코스트 모델`의
정교화가 보완 포인트 및 백로그(`C-02`)로 언급되어 있습니다. 이 작업은 RVV와
유사한 제약(벡터 길이, issue rate)을 반영하여 element-wise 및 Reduce 연산의
비용을 모델링하는 것을 목표로 합니다.
    *   즉, NPU 자체의 연산(`GEMM_T`, `CONV2D_T` 등)은 NPU 내부에서
처리되며, 표준 RVV 명령어 세트를 직접 사용하는 방식은 아닙니다.

---

## 5. pyv_npu 시뮬레이션 레벨 (L2 vs L3)

현재 `pyv_npu`는 **L2 이벤트 기반 시뮬레이션**에 중점을 두고 구현되어
있습니다.

PRD(제품 요구사항 문서)에 따르면 `CA_FULL` 사이클-정확 시뮬레이션이 옵션으로
제시되어 있지만, 현재 구현된 핵심 기능은 `CA_HYBRID` 수준이며, 더 정교한 사이클-정확
시뮬레이션을 위한 TC/VC 코스트 모델 및 메모리/버스 모델 등은 '보완
포인트'로 남아있습니다.