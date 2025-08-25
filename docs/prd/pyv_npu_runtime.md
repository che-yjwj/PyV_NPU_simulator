# Py-V NPU 시뮬레이터 런타임 PRD

## 1. 개요
본 문서는 Py-V 기반 NPU 시뮬레이터의 **런타임(Runtime) 설계 요구사항**을 정의한다.  
런타임은 ONNX 모델을 입력으로 받아 RISC-V 명령어 시퀀스로 변환하고, 이를 NPU 실행 엔진에 매핑하여 시뮬레이션한다.  
또한 명령어 디스패치, Reorder Buffer(ROB), 의존성 해석, 파이프라인 타이밍 시각화 등을 지원한다.

---

## 2. 아키텍처 개요

### 전체 블록 다이어그램 (ASCII)

```
+---------------------------------------------------------+
|                       Py-V NPU                          |
|                                                         |
|  +----------------+    +-----------------------------+ |
|  | Frontend       |    | Runtime                    | |
|  | (ONNX Loader)  |--->| - Instr Queue              | |
|  +----------------+    | - Dependency Analyzer      | |
|                        | - Scheduler / Dispatcher   | |
|                        | - Reorder Buffer (ROB)     | |
|                        +-------------+--------------+ |
|                                      |                |
|                                      v                |
|                        +----------------------------+ |
|                        | NPU Execution Engine       | |
|                        | - Tensor Core (TC)       | |
|                        | - Vector Core (VC)       | |
|                        | - Custom Ops (GELU, LN...) | |
|                        +----------------------------+ |
|                                                         |
|  +----------------+    +-----------------------------+ |
|  | Visualization  |<---| Runtime Trace / Gantt Chart | |
|  +----------------+    +-----------------------------+ |
+---------------------------------------------------------+
```

---

## 3. 주요 기능

### (1) Frontend
- ONNX 모델 로딩
- 연산자 스키마 추출 및 IR 변환
- RISC-V 명령어 시퀀스 매핑

### (2) Runtime Core
- **명령어 큐 관리**: 실행 대기열 유지
- **의존성 해석기**: Read-after-Write(RAW), Write-after-Read(WAR), Write-after-Write(WAW) 처리
- **스케줄러**: NPU/CPU 실행 엔진으로 명령어 분배
- **Reorder Buffer (ROB)**: out-of-order 실행 지원, retire 시점 기록

### (3) Execution Engine
- Tensor Core (TC): MatMul, Conv 등 텐서 연산 처리
- Vector Core (VC): SIMD 연산, DyT/Attention 지원
- Custom Ops: GELU, LayerNorm, Attention 등 고급 연산
- Shared Memory 및 Bank Conflict 모델링

### (4) Visualization & Analysis
- 파이프라인 단계별 **Gantt Chart**
- 명령어 **Dependency Graph**
- 실행 시간 통계, Utilization 분석
- `.svg` 및 웹 대시보드 출력

---

## 4. 런타임 동작 시나리오

1. 사용자가 `pyv-npu run model.onnx` 실행
2. Frontend에서 ONNX 모델 파싱 및 IR 변환
3. RISC-V 명령어 시퀀스로 변환 후 Instruction Queue에 push
4. Dependency Analyzer가 data hazard 검사
5. Scheduler가 NPU/CPU에 디스패치
6. Execution Engine에서 연산 실행
7. Reorder Buffer에서 retire 시점 기록
8. 결과를 Trace DB에 저장
9. Visualization 모듈에서 Gantt Chart 및 Graph 출력

---

## 5. 런타임 구성 모듈 (ASCII)

```
+---------------------------------------------------+
| Runtime Core                                      |
|                                                   |
|  +-----------------+    +----------------------+ |
|  | Instr Queue     |--->| Dependency Analyzer  | |
|  +-----------------+    +----------------------+ |
|           |                          |            |
|           v                          v            |
|  +-----------------+    +----------------------+ |
|  | Scheduler       |--->| Reorder Buffer (ROB) | |
|  +-----------------+    +----------------------+ |
|                                                   |
+---------------------------------------------------+
```

---

## 6. 인터페이스 정의

- CLI: `pyv-npu run <onnx_model>`
- Input: ONNX IR / RISC-V Instr List
- Output: 실행 Trace, 성능 Report, SVG 시각화

---

## 7. 확장 계획
- Multi-core NPU 지원
- 버스 컨텐션 / DMA 모델 추가
- 발열 및 전력 소모 모델링
- LLVM/MLIR 기반 최적화 연계
- 클라우드/Web 기반 다중 사용자 지원

---

## 8. 성공 기준 (Success Criteria)
- ONNX 모델을 end-to-end로 시뮬레이션 가능
- Gantt Chart 및 Dependency Graph 자동 생성
- RISC-V 명령어 기반 성능 예측 지원
- 확장 가능한 구조로 유지보수 용이성 확보
