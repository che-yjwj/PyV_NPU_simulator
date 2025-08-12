 V250813e: 'tight' 모드 시뮬레이션 및 리포트 기능 강화
  ✦ 완료된 작업:

   1. 'tight' 모드 구현 (M3 마일스톤 일부):
       * 컴파일러 매퍼 수정: 'tight' 모드에서 ENQCMD_T, TWAIT 같은 커스텀 명령어를 포함하는 NPU 프로그램을 생성하도록
         mapper.py를 수정했습니다.
       * 스케줄러 수정: L2 이벤트 기반 스케줄러가 'tight' 모드를 지원하도록 업데이트했습니다. CPU와 NPU 작업 큐를 분리하고,
         티켓 기반으로 NPU 작업 완료를 추적하여 TWAIT 명령을 처리하는 로직을 구현했습니다.
       * 테스트 추가: 'tight' 모드 시뮬레이션을 검증하기 위한 테스트 케이스를 추가하고 모든 테스트 통과를 확인했습니다.

   2. 리포트 기능 강화:
       * HTML 간트 차트: plotly 라이브러리를 사용하여 시뮬레이션 타임라인을 대화형 HTML 간트 차트로 시각화하는 기능을
         구현했습니다. (--gantt 옵션)
       * ASCII 간트 차트: HTML/JS 실행이 제한된 환경을 위해, 터미널에 직접 출력되는 텍스트 기반 간트 차트 생성 기능을
         추가하여 안정성을 높였습니다. (--ascii-gantt 옵션)

  ✦ 남은 작업:

   * M3: MLIR 연동: 'tight' 모드 구현의 다음 단계로, MLIR Dialect/Lowering을 통한 컴파일러 패스 구체화가 필요합니다.
   * M4: Py-V 연동: NPU 시뮬레이터를 Py-V RISC-V 시뮬레이터와 연동하여 End-to-End 시뮬레이션을 완성해야 합니다.
   * 고급 스케줄링 및 자원 모델링: PRD에 명시된 CP-prio, Bank-aware 배치 등 고급 스케줄링 정책과 Cache Coherence, Prefetch 등
     세부 메모리 모델링이 필요합니다.

V250812d
 수행한 작업은 다음과 같습니다.

   1. ENQCMD_T 및 TWAIT 명령을 생성하는 tight 모드를 컴파일러 매퍼에 구현했습니다.
   2. L2 이벤트 기반 스케줄러를 수정하여 다음을 포함한 tight 모드를 처리하도록 했습니다.
       * CPU 및 NPU 작업 큐
       * ENQCMD_T 및 TWAIT 명령 처리 로직
       * NPU 작업 완료를 추적하는 티켓 시스템
   3. tight 모드 L2 시뮬레이션을 확인하기 위한 새로운 테스트 케이스를 추가했습니다.
   4. 여러 버그를 수정했습니다.

  MLIR 통합이 남아있어 M3 마일스톤이 완전히 완료된 것은 아니지만, 시뮬레이터에 사용자 정의 ISA를 구현하는 핵심 작업은
  완료되었습니다. 진행 상황을 알려드리겠습니다.M3의 핵심인 'tight' 모드와 커스텀 ISA의 시뮬레이터 연동을 성공적으로
  마쳤습니다. 모든 관련 테스트가 통과하는 것을 확인했습니다.

  이제 다음 단계로 MLIR 연동을 진행할 수 있지만, 그 전에 현재 시뮬레이션 결과를 더 쉽게 분석할 수 있도록 리포트 기능을
  강화하는 것을 제안합니다.

  현재 시뮬레이션 결과는 JSON 파일로 저장됩니다. PRD에 명시된 대로, 이 데이터를 간트 차트(Gantt chart)와 같은 시각적인
  형태로 표현하면 각 엔진의 타임라인과 병목 현상을 직관적으로 파악하는 데 큰 도움이 될 것입니다.

  이 리포트 기능 강화 작업을 먼저 진행하는 것이 어떨까요? 아니면 바로 MLIR 연동 작업으로 넘어갈까요?
V250812c
✦ 자, 지금까지 다음 작업들을 완료했습니다.

   1. 자원 모델 구현 (`resources.py`): DRAM 대역폭과 SPM Bank 경합을 추적하는 BandwidthTracker와 BankTracker를 추가했습니다.
   2. 스케줄러 개선 (`scheduler.py`): event_driven_schedule이 자원 모델을 사용하여 연산의 시작/종료 시간을 더 정확하게
      계산하도록 업데이트했습니다.
   3. Mapper 업데이트 (`mapper.py`): ONNX 모델의 텐서 정보를 NPU 연산(NPUOp)의 인자(args)로 전달하여 스케줄러가 타이밍 계산에
      사용할 수 있도록 수정했습니다.
   4. IR 및 시뮬레이터 정리 (`isa/npu_ir.py`, `runtime/simulator.py`): IR 클래스를 개선하고 시뮬레이터의 리포트 생성 로직을
      정리했습니다.

  이로써 PRD의 M2: L2 이벤트 기반 경합 모델 구현 마일스톤의 핵심적인 부분을 완료했습니다. 이제 시뮬레이터는 단순한 기능 검증을
  넘어, 하드웨어 자원 경합을 고려한 성능 분석(L2 Level)을 수행할 수 있는 기반을 갖추게 되었습니다.

  다음으로 어떤 작업을 진행할까요? 제안 가능한 다음 단계는 다음과 같습니다.

   * 검증 및 테스트: 수정한 L2 시뮬레이터가 의도대로 동작하는지 확인하기 위해 테스트 케이스를 작성하고 실행합니다.
   * M3: TE Custom ISA 및 MLIR 연동: tight 모드 구현을 구체화하여 ENQCMD_T, TWAIT 같은 커스텀 명령어를 처리하는 로직을
     스케줄러에 본격적으로 연동합니다.
   * 리포트 기능 강화: 현재 JSON으로 출력되는 리포트를 PRD에 명시된 Gantt 차트(HTML/SVG) 등으로 시각화하는 기능을 추가합니다.
   
V250812b
  PRD 문서와 코드베이스를 종합하여 현재까지 구현된 기능과 개발 수준을 정리했습니다.

  결론: M2 마일스톤의 초기 단계, 즉 L2 시뮬레이션의 기본 골격 구현에 집중된 상태입니다.

  PRD의 요구사항과 비교한 상세 내역은 다음과 같습니다.

  1. 입력 및 프론트엔드 (M0, M3)
   * ONNX 입력 (구현 완료): pyv_npu/ir/onnx_importer.py를 통해 ONNX 모델을 로드하여 내부 IR(Model_IR)로 변환하는 기능이
     구현되어 있습니다. examples/tinyllama.onnx 파일이 있는 것으로 보아 실제 모델 테스트도 고려하고 있습니다.
   * NPU-IR (기본 구조 구현): pyv_npu/ir/npu_ir.py에 NPUOp와 Program 클래스가 정의되어 있어, PRD에 명시된 NPU-IR 스키마의 기본
     틀이 잡혀있습니다.
   * 컴파일러 패스 (뼈대만 구현):
       * fusion.py, quantization.py, tiling.py 파일은 존재하지만, 내부 로직은 아직 비어있거나(TODO) 단순 통과(pass-through)
         형태로 되어 있습니다.
       * 이는 M3의 MLIR Dialect/Lowering으로 가기 위한 초기 구조 설계로 보입니다.

  2. RISC-V 연동 및 Custom ISA (M3, M4)
   * Tight-Coupled Custom ISA (설계/정의 완료): pyv_npu/isa/riscv_ext.py 파일에 PRD에서 요구한 ENQCMD_T, TWAIT, TBAR, TSTAT
     명령어의 데이터 구조가 명확히 정의되어 있습니다. 이는 M3의 핵심인 TE Custom ISA 설계가 완료되었음을 의미합니다.
   * Loose/Tight 모드 전환 (CLI 구현): pyv_npu/cli/main.py에서 --mode loose/tight 옵션을 제공하여 두 연동 방식을 선택할 수
     있도록 CLI가 구현되어 있습니다. PRD의 CLI 예시와 거의 일치합니다.
   * Py-V 연동 (진행 중): 실제 py-v 시뮬레이터와의 훅(hook) 연결 로직은 아직 명확하게 보이지 않습니다. 현재는 NPU 단독
     시뮬레이션에 집중하고 있으며, M4에서 본격적인 연동이 진행될 것으로 예상됩니다.

  3. 백엔드 및 시뮬레이터 (M1, M2)
   * 시뮬레이션 레벨 (L0/L1/L2 지원): pyv_npu/runtime/simulator.py 와 cli/main.py에서 --level L0/L1/L2 옵션을 선택할 수
     있습니다.
   * 스케줄러 (L2 기본 구현): simulator.py에서 level 값에 따라 simple_greedy_schedule (L0/L1 용) 또는 event_driven_schedule (L2
     용)을 선택하는 로직이 있습니다. 이는 M2의 핵심인 이벤트 기반 경합 모델의 기초가 구현되었음을 시사합니다.
   * 타이밍/자원 모델 (개념만 존재): 현재 시뮬레이터는 스케줄러가 계산한 사이클을 그대로 사용하며, PRD에 명시된 DRAM 대역폭,
     Bank 충돌 등 상세한 L2 모델은 아직 구현되지 않았습니다. (주석으로 "In a real L2/L3 simulation, timing would be calculated
     here" 라고 명시됨)

  4. 출력 및 리포트
   * 기본 리포트 (JSON 출력): 시뮬레이션 결과로 총 사이클, 엔진 활용도, 타임라인을 포함하는 report.json 파일을 생성합니다.
     PRD에서 요구한 Gantt 차트, Roofline 분석 등 시각적인 리포트는 아직 구현되지 않았습니다.

  PRD 대비 현재 상태 요약


  ┌─────────────────────────────┬──────────────────────────────────────┬───────────────────────────────────────────┐
  │ PRD 요구사항 (마일스톤)     │ 현재 구현 상태                       │ 근거 파일                                 │
  ├─────────────────────────────┼──────────────────────────────────────┼───────────────────────────────────────────┤
  │ ONNX 입력 (M0)              │ 완료                                 │ ir/onnx_importer.py, cli/main.py          │
  │ NPU-IR 정의 (M0)            │ 완료                                 │ ir/npu_ir.py                              │
  │ **L0/L1 기능/타일-타임 (M0... │ 기본 구현                            │ runtime/simulator.py, `runtime/schedul... │
  │ L2 이벤트/자원 모델 (M2)    │ 초기 단계 (스케줄러 선택 로직만 ...  │ runtime/simulator.py                      │
  │ TE Custom ISA 설계 (M3)     │ 완료 (데이터 구조 정의)              │ isa/riscv_ext.py                          │
  │ 컴파일러 패스 (M3)          │ 뼈대만 구현                          │ compiler/passes/*.py                      │
  │ Py-V 연동 (M4)              │ 미구현 (CLI 옵션만 존재)             │ cli/main.py                               │
  │ 상세 리포트 (HTML/SVG)      │ 미구현 (기본 JSON 리포트만 제공)     │ runtime/simulator.py                      │
  └─────────────────────────────┴──────────────────────────────────────┴───────────────────────────────────────────┘


  정리하자면, 프로젝트는 ONNX 모델을 입력받아 NPU-IR로 변환하고, 정의된 Custom ISA를 바탕으로 L2 수준의 이벤트 기반 
  시뮬레이션을 실행하기 위한 핵심 골격을 구축한 단계에 있습니다. 다음 단계는 스케줄러와 시뮬레이터에 실제적인 경합 및 자원
  모델을 구현하고(M2 완료), py-v와의 연동(M4)을 진행하는 것이 될 것으로 보입니다.

V250812a
  기능 구현 현황 분석 

  ✅ 구현된 기능 (Implemented)

   1. 입력 형식 (Input):
       * ONNX 모델 로딩: ir/onnx_importer.py를 통해 ONNX 파일을 읽어 내부 IR(Model_IR)로 변환하는 기능이 구현되어 있습니다. (PRD 요구사항 1)

   2. 컴파일러 패스 (Compiler Passes):
       * Tiling, Fusion, Quantization: compiler/passes/ 내에 각 기능이 모듈로 구현되어 있으며, cli/main.py에서 순차적으로 호출합니다. (PRD 요구사항 2-b, 7)

   3. NPU-IR 및 RISC-V ISA:
       * NPU-IR 정의: isa/npu_ir.py에 NPUOp, Program 등 NPU 명령어 표현이 정의되어 있습니다. (PRD 요구사항 4)
       * Tight-Coupled Custom ISA 정의: isa/riscv_ext.py에 ENQCMD_T, TWAIT 등 PRD의 Custom ISA가 정확히 정의되어 있습니다. (PRD 요구사항 9)

   4. 시뮬레이터 핵심 로직 (Simulator Core):
       * L0/L1/L2 레벨 지원: runtime/scheduler.py의 event_driven_schedule 함수는 이벤트 기반(L2) 스케줄링을 구현합니다. 이는 L0(기능 검증), L1(타일 시간) 모델을 포함하거나 확장할 수 있는 기반이 됩니다. (PRD 요구사항 3)
       * Loose-Coupled (MMIO) 연동: Scheduler.handle_doorbell 메서드는 MMIO Doorbell을 통해 작업을 수신하고, 메모리에서 디스크립터를 읽어 파싱하는 로직을 구현했습니다. (PRD 요구사항 8, 13-A)
       * Tight-Coupled (Custom ISA) 연동: event_driven_schedule 내에 config.mode == 'tight' 분기 처리가 있어, CPU가 ENQCMD_T 같은 명령을 내리는 시나리오를 시뮬레이션할 준비가 되어 있습니다. (PRD 요구사항 8, 13-B)
       * 스케줄링: 이벤트 기반 스케줄러는 다중 엔진(TE/VE/DMA)의 유휴 상태를 추적하며 작업을 할당하는 기본 골격을 갖추고 있습니다. (PRD 요구사항 5)

   5. CLI / API:
       * PRD에 명시된 pyv-npu run, compile 명령어와 대부분의 인자(--mode, --level, --isa 등)가 cli/main.py에 구현되어 있습니다. (PRD 요구사항 8, 13-F)

  ---

  ⏳ 구현 필요 / 검증 필요한 기능 (To Be Implemented / Verified)

   1. Py-V (RISC-V Core) 와의 완전한 연동:
       * 현재 NPU 시뮬레이터(pyv-npu)는 독자적으로 실행 가능한 상태로 보입니다. 하지만 PRD의 최종 목표는 py-v RISC-V 시뮬레이터와의 연동입니다.
       * Loose-Coupled: py-v가 MMIO 주소에 write할 때 Scheduler.handle_doorbell을 호출하는 "훅(hook)" 연결이 필요합니다.
       * Tight-Coupled: py-v의 디코더가 Custom ISA를 인지했을 때 isa/riscv_ext.py에 정의된 동작을 수행하도록 하는 "훅" 연결이 필요합니다. (PRD 마일스톤 M4)

   2. 성능/자원 모델 고도화:
       * scheduler.py의 estimate_op_duration 함수는 현재 간단한 placeholder로 구현되어 있습니다. PRD에 명시된 Bank 충돌, NoC/DRAM 큐 모델, Token Bucket 대역폭 모델 등을 반영한 정교한 타이밍 모델 구현이 필요합니다. (PRD 요구사항 5, 6)
       * Cache Coherence, Prefetch, L2 bypass 등 메모리 계층의 세부 동작 모델링이 필요합니다. (PRD 요구사항 4, 6)

   3. 스케줄러 알고리즘 고도화:
       * 현재 스케줄러는 기본적인 이벤트 기반 할당을 수행합니다. PRD에 언급된 CP-prio (Critical-Path Priority), Bank-aware 배치와 같은 고급 스케줄링 정책 구현이 필요합니다. (PRD 요구사항 5)

   4. 리포팅 기능 구체화:
       * 현재는 시뮬레이션 결과로 report.json에 사이클 수, 활용도 등을 저장합니다. PRD에서 요구하는 Gantt 차트(SVG), Roofline 분석, 병목 Top-N 리포트(HTML/CSV) 등 시각적이고 상세한 리포트 생성 기능 구현이 필요합니다. (utils/reporting.py, viz.py
         확장) (PRD 요구사항 2-c)

   5. MLIR 입력 지원:
       * PRD에서 옵션으로 언급된 MLIR 입력을 처리하는 프론트엔드 구현이 필요합니다.

   6. 검증 및 테스트:
       * tests/test_smoke.py 외에 각 모듈(컴파일러 패스, 스케줄러, 메모리 모델 등)에 대한 상세한 유닛 테스트와, Loose/Tight 모드의 결과 일치성을 검증하는 회귀 테스트 세트 구축이 필요합니다. (PRD 마일스톤 M5, 검증 플랜)