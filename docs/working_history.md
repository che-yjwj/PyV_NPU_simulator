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