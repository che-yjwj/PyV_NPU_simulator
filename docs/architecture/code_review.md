v250813a
# PyV-NPU 코드 분석 보고서(chatgpt)

## 전체 구조(요점 정리)

```
pyv_npu/
  ├─ cli/                # 커맨드라인 진입점 (compile / run)
  ├─ ir/                 # 모델 IR (ONNX→내부 IR)
  ├─ compiler/           # IR→NPU Program 매핑 (loose/tight 모드)
  │   └─ passes/         # (지금은 빈 패키지; 최적화 패스용)
  ├─ isa/                # NPU Program/Op, RISC-V 확장 ISA 정의
  ├─ runtime/            # 스케줄러·시뮬레이터·리소스/버스/메모리·VE/TE
  ├─ bridge/             # (초기) 제어/메모리 브릿지, 포트 모델
  ├─ utils/              # 로깅·리포트·시각화
  ├─ config.py           # SimConfig(하드웨어/실행 파라미터 집합)
  └─ __init__.py
```

### 데이터 플로우(상위 레벨)

```
ONNX 모델
   ↓ (ir/onnx_importer.load_onnx_as_model_ir)
Model IR (그래프, 텐서 메타)
   ↓ (compiler/map_model_ir_to_npu_program: loose|tight)
NPU Program (isa/npu_ir.Program: NPUOp 시퀀스)
   ↓ (runtime/run → scheduler.*)
스케줄 결과 (ScheduleItem 타임라인)
   ↓ (utils/reporting.generate_html_report)
리포트/간트차트/JSON
```

---

## 주요 모듈별 분석

### 1) `cli/` — 실행 진입점
- **`cli/main.py`**
  - `build_parser()` : `compile` / `run` 서브커맨드 파서 구성
  - `cmd_compile()` : ONNX → Model IR → NPU Program 변환 후 디스크에 저장
  - `cmd_run()` : NPU Program + `SimConfig` 로드 → `runtime.run()` 호출 → 보고서 생성
  - 실행모드 파라미터:
    - **loose**: MMIO base, 큐 크기
    - **tight**: `te_isa` 목록 등

> 팁: 예제 ONNX(`examples/tinyllama.onnx`)로 compile → run 테스트 가능.

---

### 2) `ir/` — 내부 모델 IR
- **`ir/onnx_importer.py`**
  - `_onnx_dtype_to_numpy()`: dtype 매핑
  - `load_onnx_as_model_ir()`: ONNX 로드 → 텐서 shape/dtype 추출 → Model IR 생성
- **`ir/model_ir.py`**
  - 경량 그래프/노드/텐서 스키마
  - 스케줄러에 필요한 shape/dtype/연결 정보 제공

---

### 3) `compiler/` — IR → NPU Program 매핑
- **`compiler/mapper.py`**
  - `map_model_ir_to_npu_program()` : 핵심 엔트리
    - `_map_to_loose_program()` : 느슨한 결합(MMIO/DMA 중심)
    - `_map_to_tight_program()` : 밀결합(TE 제어 ISA 시퀀스)
  - `_add_op_args()` : 스케줄용 인자 추가
  - 출력: **`isa/npu_ir.py`의 `Program`** 객체

---

### 4) `isa/` — NPU Program & RISC‑V 확장
- **`isa/npu_ir.py`**
  - `Program`: NPUOp 리스트
  - `NPUOp`: 연산 타입 및 인자(shape, 주소, 타일 등)
- **`isa/riscv_ext.py`**
  - RISC‑V 확장 ISA 정의
  - tight 모드 제어 시뮬레이션 지원

---

### 5) `runtime/` — 스케줄러·시뮬레이터·리소스
- **`runtime/scheduler.py`**
  - L1: 단순 그리디 스케줄
  - L2: 이벤트 기반 스케줄 (은행 충돌, DRAM 대역폭 모델)
- **`runtime/simulator.py`**
  - `run()` : 스케줄링 → 타임라인 → 통계 → 보고서 생성
- **`runtime/ve.py` / `runtime/te.py`**
  - VE/TE 코스트 모델 스텁
- **`runtime/memory.py`, `runtime/bus.py`**
  - 간단한 메모리·버스 지연 모델
- **`runtime/resources.py`**
  - 대역폭·은행 리소스 관리

---

### 6) `bridge/mem.py` — 메모리 브리지
- `ReadPort`, `WritePort`, `NPUControlMemory`
  - 2R1W 포트 메모리 모델, 리틀엔디안
  - 주소 범위 체크, 예외 처리

---

### 7) `utils/` — 로깅·리포트·시각화
- **`utils/reporting.py`**
  - HTML 리포트 생성 (간트차트 포함)
- **`utils/viz.py`**
  - 타임라인/리소스 시각화
- **`utils/logging.py`**
  - 경량 로깅

---

### 8) `config.py` — 시뮬레이션 파라미터
- `SimConfig.from_args()`
  - 공통 및 모드별 파라미터
  - YAML 오버라이드 훅 있음

---

## 실행 경로(Call chain)
```
cli.main:main()
  ├─ cmd_compile()
  │    ├─ ir.onnx_importer.load_onnx_as_model_ir()
  │    └─ compiler.mapper.map_model_ir_to_npu_program()
  └─ cmd_run()
       ├─ config.SimConfig.from_args()
       ├─ runtime.simulator.run()
       │    ├─ runtime.scheduler.*
       │    └─ utils.reporting.generate_html_report()
       └─ 결과 저장
```

---

## 잘한 점
- 컴파일러→런타임 파이프라인 명확
- 스케줄러 코어가 안정적 (L1+L2)
- 리포팅 자동화

## 보완 포인트
1. TE/VE 코스트 모델 정교화
  * TE: 타일‑사이즈, systolic fill/drain, K‑blocking, SRAM tile‑reuse 반영
  * VE: 벡터 길이/RVV 유사 제약, 파이프라인/발사율(issue rate) 모델
2. 메모리/버스 모델 업그레이드
  * SPM bank/port 수, DMA burst, NoC 홉 딜레이, DRAM 채널/페이지 정책
3. 컴파일러 패스 연결
  * 타일링 결정, 연산 fusion, QAT/INT8·VQ 매핑, 활성/가중치 layout 변환
4. 의존성/큐/레지스터 모델
  * tight 모드에서 레지스터‑마이크로코드 큐/doorbell/CSR latency 반영
5. PPA 훅
  * op별 energy 모델 핸들러 인터페이스 마련 → 보고서에 nJ/op, Watt 추정 추가

---

## 빠른 사용법
```
# 컴파일
python -m pyv_npu.cli.main compile   --onnx examples/tinyllama.onnx   --mode loose   --out build/llama_loose.json

# 실행
python -m pyv_npu.cli.main run   --program build/llama_loose.json   --mode loose   --report-dir build/report_loose
```
> --mode tight 전환 시 --isa 목록 제공 권장
