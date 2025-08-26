# PRD: Tinygrad IR → HW-Aware IR 변환 계층

## 1. 문서 목적
이 문서는 Py-V 기반 NPU 시뮬레이터에 `Tinygrad IR`을 하드웨어 친화적 실행 단위(`HW-Aware IR`)로 변환하는 계층(이하 변환기)을 설계·구현하기 위한 제품 요구사항 문서(PRD)이다.  
목표는 연구 단계의 가벼운 알고리즘 검증과 하드웨어 최적화/성능 분석 단계의 정밀 시뮬레이션을 하나의 파이프라인에서 지원하는 것이다.

---

## 2. 배경 및 문제 정의
- Tinygrad는 경량화된 연산 그래프(IR)를 제공하여 알고리즘 실험에 유리하나, 하드웨어 실행에 필요한 타일링, 메모리 매핑, DMA 스케줄 등의 정보가 부족하다.
- NPU 시뮬레이션(TE/VE/DMA/Cache/Bank conflict 등) 수준에서 신뢰성 있는 성능 분석을 수행하려면 HW-Aware 수준의 IR이 필요하다.
- 본 변환기는 Tinygrad IR을 입력으로 받아 HW-Aware IR을 생성하여 NPU Backend Mapper 및 Py-V 시뮬레이터와 연동한다.

---

## 3. 목표(Goal)
- Tinygrad IR 입력에 대해 자동으로 HW-Aware IR을 생성한다.
- 생성된 HW-Aware IR은 Py-V NPU 시뮬레이터의 cycle-level executor에서 실행 가능해야 한다.
- 연구 모드(경량 실험)와 HW 모드(정밀 시뮬레이션)를 동일한 툴체인 내에서 전환 가능하게 한다.

---

## 4. 범위(Scope)
- 입력: Tinygrad IR (forward 및 optional한 backward)
- 출력: HW-Aware IR (타일링, 메모리 매핑, DMA 스케줄, 실행 유닛 매핑 포함)
- 통합 대상: Py-V NPU Simulator의 NPU Backend Mapper 및 시뮬레이터
- 제외: ONNX Direct IR 변환기(별도 경로로 존재하나 호환성 확보 필요)

---

## 5. 기능 요구사항(Functional Requirements)

### 5.1. 타일링 엔진
- 대형 텐서를 하드웨어 친화적 타일로 분해.
- 사용자/타겟 기반 기본 타일 정책 제공(예: 256×256, 128×64 등).
- 언정렬(unaligned) 크기 처리 로직 포함.

### 5.2. 메모리 매퍼
- DRAM ↔ SRAM 매핑 계획 수립.
- SRAM bank 배치, base address, stride 계산.
- bank conflict 예측/회피(heuristic) 지원.

### 5.3. DMA 스케줄러
- 비동기 DMA 명령(LOAD/STORE) 생성.
- double-buffering, prefetch 옵션 지원.
- DMA와 연산 유닛(TE/VE) 간 파이프라이닝 고려.

### 5.4. 실행 유닛 디스패처
- 각 instruction에 대해 preferred unit(TE/VE/DMA/ACC)을 할당.
- 연산의 parallelism/atomicity에 근거한 dispatch 제어.

### 5.5. 의존성 및 동기화
- instruction-level dependency graph 생성.
- barrier/wait semantics와 이벤트 기반 동기화 명세.

### 5.6. HW-Aware IR 포맷 생성
- 각 instruction은 id, opcode, operands, dests, tile, mem, deps, scheduling, attributes 등을 포함.
- JSON/YAML 디버그용 + Protobuf/FlatBuffers 프로덕션 직렬화 지원.

### 5.7. Validation & Round-trip 테스트
- Tinygrad IR → HW-Aware IR → NPU Backend → 결과값이 수치적으로 동일(또는 정의된 오차 허용범위)함을 자동 검증하는 테스트 케이스 제공.

---

## 6. 비기능 요구사항(Non-Functional Requirements)
- **모듈화**: 변환기는 타일러, 메모리 매퍼, DMA 스케줄러, 유닛 디스패처로 분리된 모듈로 구현.
- **확장성**: 새로운 opcode, 사용자 정의 activation, hw target을 쉽게 추가.
- **성능**: 변환기 자체의 오버헤드는 실시간/배치 워크플로우에서 허용 가능한 수준으로 제한.
- **가독성**: 디버깅을 위한 사람이 읽을 수 있는 JSON/YAML 출력 제공.
- **버전관리**: 스키마 버전 필드 및 마이그레이션 가이드 포함.

---

## 7. 아키텍처(High-Level)

```
ONNX Model
   │
   ▼
tinygrad Parser → tinygrad IR
   │
   ▼
[HW-Aware IR 변환기]
   ├─ Tile Engine
   ├─ Memory Mapper
   ├─ DMA Scheduler
   ├─ Unit Dispatcher
   └─ Optimizer (locality, prefetch, double-buffer)
   │
   ▼
HW-Aware IR
   │
   ▼
NPU Backend Mapper → NPU Instr Stream
   │
   ▼
Py-V NPU Simulator (cycle-level)
```

---

## 8. 예제 워크플로우(간단)
1. Tinygrad IR:
```
MatMul(W[1024,1024], X[1024,1])
Add(bias[1024])
ReLU
```
2. 변환기:
   - Tile: 1024→4 tiles of 256
   - Memory map: DRAM → SRAM_BANK0/1
   - Generate LOAD/LOAD/MATMUL/ADD/ACT/STORE sequence (with deps)
3. HW-Aware IR → Backend → 시뮬레이터 실행

---

## 9. 검증 지표(Success Metrics)
- 변환기 유닛 테스트 통과율: 100% (단위 ops)
- 통합 테스트(MLP 1024×1024): Tinygrad 경로 vs Direct 경로 결과 오차 < 1e-5(설정값)
- 시뮬레이션에서 double-buffering 활성화 시 SRAM 대기 시간 감소 확인
- 변환기 모듈별 성능(타이핑 당) 측정 리포트 제공

---

## 10. 문서 및 산출물
- PRD(이 문서)
- HW-Aware IR 상세 스펙 문서(`/docs/specs/hw_aware_ir.md`)
- Protobuf schema(`/schema/hw_aware_ir.proto`)
- 변환기 PoC 코드(base: Python)
- Validator & Unit tests

---

## 11. 우선순위 및 일정(권장)
1. v1 스키마 정의 + JSON 예제 (1주)
2. Validator + 단위 테스트 (1주)
3. 변환기 PoC (Tile + LOAD/STORE + MATMUL 매핑) (2주)
4. Py-V 시뮬레이터 연동 및 통합 테스트 (2주)
5. 확장: double-buffer, prefetch, custom ops (2주)

---

## 12. 운영/유지보수
- 스펙 버전 관리(semver)
- 스키마 변경 시 backward/forward 호환성 문서화
- CI에서 스키마 유효성 검사 및 round-trip numeric equivalence 테스트 자동화

---

# 부록: 빠른 시작(개발자용)
- 레포지토리 구조(권장)
```
/pyv-npu/
  /docs/specs/hw_aware_ir.md
  /schema/hw_aware_ir.proto
  /tools/ir_converter/
    tile_engine.py
    memory_mapper.py
    dma_scheduler.py
    unit_dispatcher.py
  /tests/
    test_matmul_roundtrip.py
```
- 개발 언어: Python (변환기 PoC), Protobuf for schema

