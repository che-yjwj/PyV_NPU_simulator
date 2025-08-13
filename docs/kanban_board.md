v250813a
# PyV-NPU Simulator 개선 과제 칸반보드(chatgpt)

## 컬럼: Backlog → Ready → In-Progress → Review → Done

---

## Backlog

### C-01 | TE 코스트 모델 v1
- **요약**: systolic fill/drain, K-blocking, tile reuse 반영한 TE latency/throughput 모델 1차 구현
- **수용기준**: (1) matmul(M,N,K)와 어레이 크기(R×C) 입력 시 사이클 계산 일치(단위테스트 5개), (2) fill/drain 오버헤드 별도 리포팅
- **우선순위**: P0 / **사이즈**: L / **의존성**: — / **태그**: TE, 성능모델

### C-02 | VE 코스트 모델 v1
- **요약**: RVV 유사 제약(벡터 길이/issue rate) 반영한 element-wise/Reduce 코스트 모델
- **수용기준**: op별 사이클 추정 ≥ 단위테스트 5개, issue rate/메모리 대역폭 바인딩 케이스 재현
- **우선순위**: P1 / **사이즈**: M / **의존성**: — / **태그**: VE, 성능모델

### M-01 | 메모리/버스 모델 고도화 v1
- **요약**: SPM bank 포트 수, DMA burst, DRAM 채널/페이지 정책(오픈/클로즈드) 파라미터화
- **수용기준**: bank 충돌 케이스 재현 테스트 3종, burst size 변화에 따른 P95 변동 리포트
- **우선순위**: P0 / **사이즈**: L / **의존성**: — / **태그**: 메모리, DMA, 버스

### S-01 | 스케줄러 L2 이벤트 모델 확장
- **요약**: 현재 L2에 NoC 홉 지연, 큐 대기 지연, backpressure 이벤트 추가
- **수용기준**: 이벤트 로그에 홉/큐 지연 개별 카운터 제공, Gantt에 레이어 표시
- **우선순위**: P1 / **사이즈**: L / **의존성**: M-01 / **태그**: 스케줄러, NoC

### P-01 | 컴파일러 타일링 패스 (auto-tile)
- **요약**: shape/메모리 제약 기반 자동 타일 크기 산출 + mapper에 주입
- **수용기준**: 3개 벤치(TinyLLaMA matmul/Conv/GELU)에서 타일링 적용 후 실행 성공 및 총 사이클 감소
- **우선순위**: P0 / **사이즈**: L / **의존성**: C-01, M-01 / **태그**: 컴파일러, 타일링

### P-02 | 연산 Fusion 패스 (elementwise chain)
- **요약**: MatMul → BiasAdd → GELU 등 연속 elementwise 체인을 VE 패킷으로 병합
- **수용기준**: fusion 전/후 NPUOp 수 감소, 사이클/메모리 트래픽 절감 리포트
- **우선순위**: P1 / **사이즈**: M / **의존성**: P-01 / **태그**: 컴파일러, 최적화

### Q-01 | INT8/QAT 파이프라인 도입 (스텁)
- **요약**: 정수 스케일·제로포인트 메타를 IR에 부착하고 런타임 코스트 보정 훅 추가
- **수용기준**: INT8 경로에서 기능 테스트 통과, 보고서에 정밀도/성능 모드 플래그 표기
- **우선순위**: P2 / **사이즈**: M / **의존성**: P-01 / **태그**: 양자화

### T-01 | Tight 모드 제어 경로 구체화
- **요약**: doorbell/CSR/큐 지연, 마이크로코드 발사율 모델
- **수용기준**: tight vs loose 동일 프로그램에서 제어 오버헤드 차이 리포트(P50/P95)
- **우선순위**: P0 / **사이즈**: M / **의존성**: — / **태그**: tight, 제어경로

### R-01 | 리포트 확장 (자원 바인딩 브레이크다운)
- **요약**: op별 stall 원인(메모리/연산/큐) 비율, 병목 Top-N 표
- **수용기준**: HTML 리포트에 breakdown 섹션과 다운로드 가능한 JSON 필드 추가
- **우선순위**: P1 / **사이즈**: S / **의존성**: S-01 / **태그**: 리포트, 분석

### PPA-01 | 에너지 훅 v0
- **요약**: op별 nJ/op 추정 테이블 주입 포인트와 합산 로직
- **수용기준**: 리포트에 총 에너지/ops 별 기여도 표시(근사치)
- **우선순위**: P2 / **사이즈**: M / **의존성**: C-01, M-01 / **태그**: PPA

### CFG-01 | YAML 하드웨어 설정 외부화
- **요약**: config.py의 파라미터를 YAML 로딩으로 완전 전환(검증 포함)
- **수용기준**: 3개 프로파일(loose/edge-tight/server-tight) 스위칭 테스트
- **우선순위**: P0 / **사이즈**: S / **의존성**: — / **태그**: 설정, DevEx

### CI-01 | 테스트 스위트 확장
- **요약**: 단위/통합/회귀 테스트: 코스트 모델, 스케줄러, 리포트 생성, CLI
- **수용기준**: GitHub Actions(or 로컬)에서 50+ 테스트, 커버리지 70%
- **우선순위**: P0 / **사이즈**: M / **의존성**: CFG-01 / **태그**: 테스트, CI

### DOC-01 | 개발자 가이드 v1
- **요약**: “ONNX→IR→Program→Schedule→Report” 풀 파이프라인 가이드 & 확장 포인트
- **수용기준**: docs/에 튜토리얼 1편 + 아키텍처 다이어그램 1개(SVG)
- **우선순위**: P1 / **사이즈**: S / **의존성**: R-01 / **태그**: 문서

### EX-01 | 벤치 스크립트
- **요약**: TinyLLaMA, MLP, ConvNet 3세트 자동 실행/리포트 수집 스크립트
- **수용기준**: 결과 디렉토리에 리포트 HTML/JSON, 비교 표 자동 생성
- **우선순위**: P1 / **사이즈**: S / **의존성**: R-01, CFG-01 / **태그**: 예제, 자동화

---

## Ready

### C-03 | TE 코스트 모델 단위테스트
- **요약**: C-01을 위한 파라미터 스윕용 테스터(타일/어레이/주기)
- **수용기준**: 최소 20 케이스 자동 검증, 경계값 테스트 포함
- **우선순위**: P0 / **사이즈**: S / **의존성**: C-01 / **태그**: 테스트, TE

### M-02 | DRAM 채널/뱅크 매핑 정책
- **요약**: 어드레스→(채널,뱅크,행,열) 매핑 함수 & 정책 선택(Interleave, RR)
- **수용기준**: 정책별 충돌율 비교 리포트(샘플 3종)
- **우선순위**: P1 / **사이즈**: M / **의존성**: M-01 / **태그**: 메모리

### P-03 | IR 메타 확장(Quant/Layout/Stride)
- **요약**: IR 텐서 메타데이터에 quant/stride/layout 플래그 필드 추가
- **수용기준**: onnx_importer→IR→mapper→runtime 경로에서 유지 검증
- **우선순위**: P2 / **사이즈**: S / **의존성**: Q-01 / **태그**: IR

### R-02 | 간트차트 상호작용 메타
- **요약**: 리포트 HTML에 op hover 시 args/자원/대역폭 등 메타 표시
- **수용기준**: 3개 벤치에서 동작, 파일 하나로 self-contained
- **우선순위**: P2 / **사이즈**: S / **의존성**: R-01 / **태그**: 리포트, UI

---

## In-Progress

### CFG-01 | YAML 하드웨어 설정 외부화
- **진행**: 파서/스키마 초안 작성, 샘플 3프로파일 준비 중
- **차주 목표**: CLI 인자 ↔ YAML 머지 정책 문서화

### CI-01 | 테스트 스위트 확장
- **진행**: 스케줄러 L1/L2 케이스 일부 완료, 리포트 생성 테스트 작성 중
- **차주 목표**: 커버리지 50% → 70%

---

## Review

### R-01 | 리포트 확장 (병목 브레이크다운)
- **상태**: HTML/JSON 필드 추가 완료, 예제 2종 검증
- **리뷰 포인트**: stall 분류 기준(메모리 vs 연산 vs 큐) 경계 케이스

---

## Done

### BAS-01 | 프로젝트 구조 인덱싱 & 코드 맵
- **내용**: 서브패키지/콜체인 문서화 및 핵심 모듈 해설
- **출력물**: docs/architecture_overview.md, 분석 스니펫

---

## Epics

- **EPIC-COST**: TE/VE/메모리/버스 코스트 모델 고도화（C-01, C-02, M-01, M-02, C-03, S-01）
- **EPIC-COMP**: 컴파일러 최적화 패스（P-01, P-02, P-03, Q-01）
- **EPIC-CTRL**: Tight 모드 제어 경로 및 마이크로코드（T-01）
- **EPIC-REP**: 리포트/시각화/벤치 자동화（R-01, R-02, EX-01, DOC-01）
- **EPIC-OPS**: 설정 외부화/테스트/CI（CFG-01, CI-01）

---

## 스프린트 제안

1. CFG-01(P0) → CI-01(P0)
2. C-01(P0) → C-03(P0) → P-01(P0)
3. M-01(P0) → S-01(P1) → R-01(P1)
4. T-01(P0)
5. EX-01(P1), DOC-01(P1)

v250812a
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