 To Do (백로그)

   - M5: 회귀 테스트 및 검증
     - NPU 모듈별 유닛/통합 테스트 프레임워크 구축(done)
     - ONNX 런타임 결과와 상호 검증(Validation) 로직 추가
   - M4: Py-V 연동 및 Custom ISA 구현
     - py-v 시뮬레이터에 Custom ISA (ENQCMD_T 등) 후킹
     - Tight-coupled 모드 E2E 연동 및 테스트
   - M3: MLIR 프론트엔드 (선택)
     - MLIR Dialect 정의 및 Lowering 패스 개발
   - M5: 문서화 및 리포팅 강화
     - 사용자 가이드, 튜토리얼, API 문서 작성
     - utils/reporting.py: Roofline, P99 지연 등 고급 리포트 기능 추가
     - utils/viz.py: 타임라인 시각화(Gantt chart) 기능 강화
   - L3: Cycle-Accurate 모델링 (선택)
     - 주요 커널 Verilator 연동 시뮬레이션

  In Progress (진행 중)

   - M2/M3: L2 시뮬레이터 구현
     - runtime의 이벤트 기반 시뮬레이터 고도화
       - scheduler.py: Bank-aware, CP-prio 스케줄링
       - te.py/ve.py: TE/VE 자원 경합 모델링
       - memory.py: NoC/DRAM 큐 및 대역폭 모델링
   - M1/M3: 컴파일러 Pass 안정화
     - compiler/passes/*.py 기능 구현 및 안정화
       - fusion.py: 연산자 퓨전
       - tiling.py: 타일링 및 메모리 배치
       - quantization.py: 양자화 지원
   - M4: Loose-Coupled 브릿지 구현
     - bridge/mem.py: MMIO 기반 RISC-V ↔ NPU 연동 브릿지 안정화

  Done (완료)

   - M0: 기본 구조 및 IR
     - ir/onnx_importer.py: ONNX 모델 → Model IR 변환
     - isa/npu_ir.py: NPU-IR 스키마 정의
     - pyv_npu/cli/main.py: 기본 CLI 인터페이스
   - M0: 프로젝트 설정
     - 프로젝트 구조, 의존성(pyproject.toml), 로깅(utils/logging.py) 등 초기 설정
     - py-v Core 시뮬레이터 소스코드 포함