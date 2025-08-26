# HW-Aware IR 포맷 스펙 — Instruction-level Schema (v1.0)

## 개요
이 문서는 HW-Aware IR의 instruction-level 스펙을 기술한다. 본 스펙은 Tinygrad IR → HW-Aware IR 변환 계층에서 생성되는 IR의 필드 구조, opcode semantics, 직렬화 권장사항, 검증 규칙을 포함한다.

---

## 1. 설계 원칙
- 명시적 실행 단위: 모든 연산은 하드웨어에서 실행 가능한 최소 단위로 기술.
- 독립성: 각 instruction은 독립적으로 스케줄링 가능하도록 충분한 메타데이터 포함.
- 가독성 + 직렬화: JSON/YAML(디버그) 및 Protobuf(프로덕션) 지원.
- 확장성: custom opcode 및 hw target 확장 용이.

---

## 2. 최상위 구조 (Top-level)

```
HW_AWARE_IR {
  header
  globals
  functions[]
  metadata
}
```

### header (예)
- version: string (ex. "v1.0")
- model_id: string
- hw_target: string (ex. "pyv-npu-v0")
- created_by: string
- timestamp: int64 (unix epoch)

### globals
- memory_map: { region_name: { base, size } }
- sram_config: { bank_count, bank_size, bank_stride }
- cache_config: { L1_size, L2_size, line_size }

### functions
배열 형태의 logical function/block. 각 function은 instruction 배열을 가진다.

---

## 3. Instruction 공통 스키마

```
Instruction {
  id: string
  opcode: string
  operands: [Operand]
  dests: [Operand]
  tile?: TileDescriptor
  mem?: MemoryDescriptor
  scheduling?: SchedulingHints
  deps?: [string]
  attributes?: { string: any }
  annotations?: [Annotation]
  estimate?: PerformanceEstimate
}
```

설명:
- id: 고유 식별자 (func:block:instN)
- opcode: 연산 코드 (아래 카탈로그 참조)
- operands/dests: 입력/출력 피연산자
- tile: 해당 연산이 작동하는 텐서 서브영역
- deps: 선행 instruction id 목록 (의존성)
- attributes: quantization 파라미터, alpha 등 추가 속성

---

## 4. Operand 스키마

```
Operand {
  name: string
  type: enum { DRAM, SRAM, REG, ACC, IMM }
  region?: string
  base?: uint64
  offset?: int
  dtype: string
  shape: [int]
  layout?: string
  stride?: [int]
}
```

주의:
- region은 globals.memory_map에 정의되어 있어야 함.
- dtype은 표준 문자열(예: "fp32","fp16","ue8m0","i8") 사용.

---

## 5. TileDescriptor

```
TileDescriptor {
  origin: [int]
  size: [int]
  padding?: [int]
  transform?: string
}
```

---

## 6. MemoryDescriptor

```
MemoryDescriptor {
  region: string
  base?: uint64
  stride?: [int]
  bank?: int
}
```

---

## 7. SchedulingHints

```
SchedulingHints {
  preferred_unit?: string  // "TE","VE","DMA","ACC"
  priority?: int
  parallelizable?: bool
  double_buffer?: bool
  prefetch?: bool
  pipeline_stage?: int
}
```

---

## 8. Opcode 카탈로그 (핵심)

### LOAD
- 설명: DRAM → SRAM/REG로 데이터를 전송하는 DMA 명령.
- 필수 필드: operands(source DRAM operand), dests(target SRAM/REG), scheduling.preferred_unit="DMA"
- attributes: double_buffer (bool), prefetch(bool)

### STORE
- 설명: SRAM/REG → DRAM 저장
- 필수 필드: operands(source), dests(target DRAM)

### MATMUL
- 설명: 타일 단위 행렬곱
- 필수 필드: operands(A_tile, B_tile), dests(C_tile), tile (m,n,k)
- attributes: acc_precision, gemm_algo_hint

### ADD / MUL /SCALE
- 설명: elementwise 연산 (broadcast 지원 명시)
- 필수 필드: operands, dests

### ACT
- 설명: activation 함수
- attributes: type ("ReLU","GELU","ChaosActivation"), params

### DMA_PEEK / PREFETCH
- 설명: 비동기 preload hint

### BARRIER / WAIT
- 설명: 동기화 프리미티브

### CUSTOM
- 설명: 사용자 정의 연산자 (registry에 스키마 필요)

---

## 9. Validation Rules
- instruction.id는 function 내에서 고유해야 함.
- deps로 참조된 id는 반드시 존재.
- operand.region은 globals.memory_map에 정의되어야 함.
- operand.shape와 tile.size 차원 일치 여부 확인.
- dtype 호환성 규칙: MATMUL의 경우 A.dtype == B.dtype (혹은 명시적 변환 instruction 필요)
- scheduling.preferred_unit이 없을 경우 default policy 적용.

---

## 10. 직렬화 권고
- 디버깅: JSON/YAML
- 프로덕션: Protobuf (예시 스키마는 하단에 제공)
- Protobuf 사용 시 backward/forward 호환을 위한 field numbering 정책 준수

---

## 11. 예제 (JSON) — MatMul 1024×1024 → 256×256 tiles
(예제는 PRD의 예제를 따라 동일하며 생략 가능 — 실제 파일에 포함됨)

---

## 12. Op Registry & Extension
- `/registry/ops.json`에 opcode별 입력/출력/attribute schema 등록
- 새로운 op 추가 시 registry entry와 validator 확장 필요

---

## 13. Testing Matrix (권장)
- 단위 테스트: LOAD, STORE, MATMUL(tile), ACT
- 통합 테스트: MLP layer full roundtrip
- 스트레스 테스트: bank-conflict heavy scenario, odd stride
- 성능 비교: double-buffer on/off, prefetch on/off

---

## 14. Migration & Versioning
- 스키마 변경 시 major.minor 버전 규칙 적용
- backward compatibility: older schema를 읽을 때 변환기 내 마이그레이션 레이어 포함

---

## 15. Tooling
- Schema file: `/schema/hw_aware_ir.proto`
- Validator: `tools/validator.py` (JSON/Protobuf 검증)
- Visualizer: instruction stream → Gantt / dependency graph export (SVG)

---

## 부록: Protobuf 샘플 (간단)
(실제 `hw_aware_ir.proto` 파일은 별도 제공 권장)
