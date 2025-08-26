# Tinygrad 프론트엔드 지원 작업 계획

## 1. 개요

이 문서는 `PyV_NPU_simulator` 프로젝트에 `tinygrad` 딥러닝 프레임워크를 새로운 프론트엔드로 지원하기 위한 구체적인 작업 계획을 정의합니다. 최종 목표는 `tinygrad`로 작성된 모델을 프로젝트의 내부 중간 표현(Intermediate Representation, IR)인 `model_ir`로 변환하는 importer를 개발하는 것입니다.

## 2. 핵심 과제

`tinygrad`의 지연 평가(lazy evaluation) 방식에 의해 생성되는 계산 그래프(`LazyOp`의 DAG)를 분석하고, 이를 `model_ir`의 노드와 텐서로 정확하게 매핑합니다. 기존 `ONNX importer`는 이 과정에서 중요한 참고 자료가 될 것입니다.

## 3. 세부 작업 리스트

### Phase 1: 분석 및 준비 (Analysis & Preparation)

- **[분석]** **Target IR 구조 이해:**
  - `pyv_npu/ir/model_ir.py` 파일을 정독하여 `ModelIR`, `Node`, `Tensor` 등 핵심 클래스의 구조와 속성을 파악합니다.
  - 어떤 종류의 `Node`(e.g., `Conv`, `MatMul`, `Add`, `Relu`)가 이미 정의되어 있는지 목록을 만듭니다.

- **[분석]** **기존 ONNX Importer 분석:**
  - `pyv_npu/ir/onnx_importer.py`의 코드를 분석하여 다음 로직을 이해합니다.
    - ONNX 그래프를 순회하는 방식
    - ONNX 노드를 `model_ir` 노드로 변환하는 `OpConverter` 또는 유사 패턴
    - 텐서 정보(shape, dtype) 및 가중치(initializer)를 처리하는 방법
    - 변환된 노드들을 관리하는 방식 (e.g., 딕셔너리 사용)

### Phase 2: Importer 초기 구현 (Initial Importer Scaffolding)

- **[구현]** **Importer 파일 생성:**
  - `pyv_npu/ir/tinygrad_importer.py` 파일을 새로 생성합니다.

- **[구현]** **Importer 클래스 정의:**
  - `onnx_importer.py`를 참고하여 `TinygradImporter` 클래스의 기본 구조를 작성합니다.
  - `tinygrad` 모델을 입력받아 `ModelIR` 객체를 반환하는 `__call__` 또는 `convert` 메소드를 정의합니다.

- **[구현]** **그래프 순회 로직 구현:**
  - `tinygrad`의 출력 `Tensor`에서 시작하여 `.lazydata` 속성을 통해 `LazyOp` 그래프에 접근합니다.
  - 재귀(recursion) 또는 스택(stack) 기반으로 `LazyOp`의 `src` (source buffers)를 따라가며 그래프를 역순으로 순회하는 함수를 구현합니다.
  - 이미 방문한 `LazyBuffer`를 `model_ir`의 `Tensor`로 매핑하여 저장하는 딕셔너리(`memo` 또는 `visited`)를 운영하여 중복 변환을 방지합니다.

### Phase 3: 연산자 매핑 구현 (Operation Mapping)

- **[구현]** **LoadOps 매핑 (데이터 로딩):**
  - `LoadOps.CONST`: `model_ir`에서 상수 `Tensor`를 생성하는 로직을 구현합니다.
  - `LoadOps.FROM_CPU`: 외부(Numpy)에서 들어오는 입력을 `model_ir`의 입력 `Tensor`로 처리하는 로직을 구현합니다. 모델의 `input`으로 정의됩니다.

- **[구현]** **BinaryOps / UnaryOps 매핑 (Element-wise 연산):**
  - `BinaryOps.ADD`, `BinaryOps.MUL` -> `AddNode`, `MulNode`
  - `UnaryOps.RELU`, `UnaryOps.EXP`, `UnaryOps.LOG` -> `ReluNode`, `ExpNode`, `LogNode` 등
  - 지원할 연산자들을 우선순위에 따라 순차적으로 매핑 로직을 추가합니다.

- **[구현]** **ReduceOps 매핑 (차원 축소 연산):**
  - `ReduceOps.SUM`, `ReduceOps.MAX` -> `ReduceSumNode`, `ReduceMaxNode`
  - `LazyOp`의 `arg`에 포함된 `axis` 정보를 `model_ir` 노드의 속성으로 전달해야 합니다.

- **[구현]** **MovementOps 매핑 (데이터 이동/형상 변경):**
  - `MovementOps.RESHAPE` -> `ReshapeNode`
  - `MovementOps.PERMUTE` -> `TransposeNode`
  - `MovementOps.SLICE` -> `SliceNode`
  - 마찬가지로 `LazyOp`의 `arg` (new shape, axes order 등)를 노드 속성으로 변환합니다.

- **[구현]** **FusedOps 매핑 (융합 연산):**
  - `FusedOps.MULACC` (Multiply-Accumulate): `MatMulNode` 또는 `ConvNode`로 매핑될 가능성이 높습니다. `MULACC`가 `tinygrad`에서 정확히 어떤 연산을 나타내는지 추가 분석이 필요하며, 이는 가장 복잡한 매핑이 될 수 있습니다.

### Phase 4: 통합 및 테스트 (Integration & Testing)

- **[테스트]** **단위 테스트 작성:**
  - 간단한 `tinygrad` 모델(e.g., `a * b + c`)을 생성합니다.
  - `TinygradImporter`를 통해 `model_ir`로 변환하고, 생성된 `ModelIR`의 구조(노드 수, 연결 관계 등)가 예상과 일치하는지 검증하는 테스트 코드를 작성합니다.

- **[통합]** **엔드투엔드(E2E) 테스트:**
  - `tinyllama.onnx`와 유사한 작은 모델을 `tinygrad`로 구현합니다.
  - 모델을 성공적으로 `model_ir`로 변환한 후, 기존 시뮬레이터의 백엔드(컴파일러, 런타임)와 연동하여 시뮬레이션까지 완료되는지 확인합니다.
