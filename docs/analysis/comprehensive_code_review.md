# PyV-NPU 시뮬레이터 종합 분석 보고서

**문서 버전**: 1.0  
**분석 일자**: 2025-09-01

## 1. Executive Summary (요약)

본 문서는 PyV-NPU 시뮬레이터 프로젝트에 대한 시스템 아키텍처, 보안, 성능, 코드 품질 관점의 종합 분석 결과를 담고 있습니다.

이 프로젝트는 **학술 연구 및 산업용 NPU 아키텍처 탐색을 위한 매우 높은 수준의 기반**을 갖추고 있습니다. 모듈식 설계, 상세한 시뮬레이션 모델, 우수한 개발 프랙티스는 본 프로젝트의 큰 강점입니다. 분석 결과, 심각한 보안 위협이나 아키텍처 결함은 발견되지 않았으나, **성능 모델의 정교화, 스케줄러의 복잡도 관리, 테스트 커버리지 확대**를 통해 프로젝트의 완성도를 한 단계 더 높일 수 있습니다.

본 보고서는 발견된 사항들을 바탕으로 **구체적인 개선 로드맵과 우선순위**를 제안합니다.

| 분석 영역 | 평가 | 핵심 요약 |
| :--- | :--- | :--- |
| **아키텍처** | 🟢 **Excellent** | 명확한 경계와 높은 확장성을 가진 모듈식 설계. |
| **보안** | 🟢 **Good** | 시뮬레이터 특성상 공격 표면이 적으나, 입력 처리 강화 필요. |
| **성능** | 🟡 **Good** | 현실적인 병목 모델링. 단, 일부 모델(FIFO, 버스) 추상화 수준 높음. |
| **품질** | 🟡 **Good** | 높은 코드 품질과 문서화. 단, 스케줄러 복잡도 및 테스트 커버리지 개선 필요. |

---

## 2. Architecture Assessment (아키텍처 평가)

**System Architect 관점**: 10배 성장을 염두에 둔 확장성과 유지보수성 평가

### 2.1. 강점 (Strengths)

*   **명확한 모듈 경계 (Clear Component Boundaries)**: 프로젝트는 `ir`, `compiler`, `runtime`, `isa` 등 기능적으로 완결된 패키지로 분리되어 있습니다. 이는 각 분야의 전문가가 독립적으로 기여하고 테스트하기에 이상적인 구조입니다.
*   **설정 중심 설계 (Configuration-Driven Design)**: `config.py`의 `SimConfig`는 모든 하드웨어 파라미터와 시뮬레이션 옵션을 중앙에서 관리합니다. YAML과 CLI를 통한 유연한 설정 변경은 다양한 아키텍처 실험을 매우 용이하게 합니다.
*   **분리된 데이터 흐름 (Decoupled Data Flow)**: `ONNX -> Model IR -> NPU Program -> Schedule`로 이어지는 데이터 변환 파이프라인이 `cli/main.py`에 명확하게 정의되어 있습니다. 각 단계는 직교성을 가지므로 새로운 컴파일러 패스나 백엔드를 추가하기 용이합니다.
*   **Probe & Commit 패턴**: `scheduler.py`가 `resources.py`의 리소스 모델을 사용할 때, 가능한 스케줄링 슬롯을 탐색(`probe`)하고 최적의 선택지를 확정(`commit`)하는 패턴은 매우 정교한 설계입니다. 이는 부작용 없이 최적의 결정을 내릴 수 있게 하여 스케줄러의 효율성과 정확성을 보장합니다.

### 2.2. 개선 제안 (Recommendations)

*   **스케줄러 로직 분리 (Decouple Scheduler Logic)**:
    *   **문제**: `scheduler.py`의 `calculate_op_timing` 함수는 모든 Opcode와 리소스의 타이밍 계산 로직이 집중되어 있어 비대하고 복잡합니다. (Cyclomatic Complexity가 높음)
    *   **제안**: **Strategy Pattern**을 적용하여 Opcode별 타이밍 계산 로직을 별도의 `TimingCalculator` 클래스(예: `MatMulTimingCalculator`, `LoadTimingCalculator`)로 분리하는 리팩토링을 제안합니다. 이는 칸반 보드의 `REF-02` 항목과 일치하며, 유지보수성과 확장성을 크게 향상시킬 것입니다.
*   **메모리 모델 구체화 (Refine Memory Models)**:
    *   **문제**: 현재 `IOBufferTracker`는 FIFO 동작을 완벽히 모델링하지 않고, 데이터의 존재 유무만 확인합니다. 이는 `code_review.md`에서도 지적된 사항입니다.
    *   **제안**: `collections.deque`를 사용하여 실제 FIFO와 같이 데이터 순서를 모델링하고, 버퍼 내 데이터의 순서가 중요한 복잡한 시나리오(예: 비동기적 데이터 소비)에서의 정확도를 높여야 합니다. 이는 칸반 보드의 `M-01` (메모리/버스 모델 고도화)의 일부로 진행될 수 있습니다.

---

## 3. Security Audit (보안 감사)

**Security Engineer 관점**: 제로 트러스트 및 공격자 관점에서의 잠재적 위협 분석

본 프로젝트는 외부 네트워크와 직접 통신하지 않는 **내부 연구용 시뮬레이터**이므로, 전통적인 웹 애플리케이션의 OWASP Top 10과 같은 보안 위협은 거의 해당되지 않습니다. 분석은 **입력 데이터 처리**와 **의존성 관리**에 초점을 맞춥니다.

### 3.1. 강점 (Strengths)

*   **제한된 공격 표면 (Limited Attack Surface)**: 외부 API나 사용자 입력을 직접 처리하는 부분이 거의 없어 공격 표면이 매우 제한적입니다.
*   **의존성 분리**: `pyproject.toml`을 통해 프로젝트의 직접적인 의존성을 명확히 관리하고 있으며, `onnx`와 같은 핵심 의존성은 `try-except`로 처리하여 부재 시에도 기본적인 동작이 가능하도록 설계되었습니다.

### 3.2. 개선 제안 (Recommendations)

*   **악의적인 입력 파일 처리 (Malicious Input File Handling)**:
    *   **위협**: 조작된 ONNX 파일(`.onnx`)이나 설정 파일(`.yaml`)을 로드할 때 잠재적인 서비스 거부(DoS)나 예기치 않은 동작이 발생할 수 있습니다. (예: 무한 루프, 과도한 메모리 할당)
    *   **제안**:
        1.  `ir/onnx_importer.py`: ONNX 모델 로드 시, 모델의 크기나 노드/텐서의 개수에 대한 합리적인 제한을 두는 검증 로직을 추가합니다.
        2.  `config.py`: YAML 파일 로드 시, `yaml.safe_load`를 사용하고 있어 코드 실행 취약점은 없으나, 설정값(예: `spm_capacity_kib`)에 비정상적으로 큰 값이 들어올 경우를 대비한 유효성 검사(validation) 로직을 `SimConfig.__post_init__`에 추가하는 것을 고려할 수 있습니다. (칸반 `CHR-01`과 연계)

---

## 4. Performance Profile (성능 프로파일)

**Performance Engineer 관점**: 병목 식별 및 리소스 최적화

### 4.1. 강점 (Strengths)

*   **상세한 병목 분석**: 이벤트 기반 스케줄러는 `stall_breakdown`을 통해 각 작업의 지연 원인(`DEP`, `RESOURCE_DRAM_BANK`, `RESOURCE_SPM` 등)을 명확히 식별합니다. 이는 성능 튜닝에 결정적인 데이터를 제공합니다.
*   **현실적인 리소스 모델링**: `DramBankTracker`, `BandwidthTracker` 등은 단순한 사이클 계산을 넘어, 실제 하드웨어에서 발생하는 리소스 경합을 현실적으로 모델링합니다.
*   **비교 분석 용이성**: `loose`와 `tight` 모드를 모두 지원하여, 두 아키텍처의 제어 오버헤드와 성능을 정량적으로 비교 분석할 수 있는 훌륭한 기반을 갖추고 있습니다.

### 4.2. 개선 제안 (Recommendations)

*   **버스 모델링 추가 (Bus Modeling)**:
    *   **문제**: 현재 시뮬레이터는 DRAM, NoC 등 개별 컴포넌트의 대역폭은 모델링하지만, 이들을 연결하는 시스템 버스(AXI 등)의 경합은 명시적으로 모델링하지 않습니다.
    *   **제안**: 여러 마스터(CPU, DMA)가 동시에 버스를 사용하려 할 때 발생하는 경합을 모델링하는 `BusTracker`를 `resources.py`에 추가할 것을 제안합니다. 이는 칸반 보드의 `M-01` (메모리/버스 모델 고도화)의 일부로 고려될 수 있습니다.
*   **스케줄러 성능 최적화 (Scheduler Performance)**:
    *   **문제**: `run_scheduler_pass`는 매번 모든 대기 중인 Op을 순회하며 최적의 Op을 찾습니다. 시뮬레이션해야 할 Op의 수가 수만 개 이상으로 늘어날 경우, 스케줄러 자체가 병목이 될 수 있습니다.
    *   **제안**: 현재는 문제가 아니지만, 향후 대규모 시뮬레이션을 위해 "준비된(ready)" Op들만 별도의 우선순위 큐에 넣어 관리하는 방식을 고려할 수 있습니다. 이는 매번 전체 Op 큐를 순회하는 비용을 줄여줄 것입니다.

---

## 5. Quality Metrics (품질 지표)

**Root Cause Analyst 관점**: 코드 복잡도, 유지보수성, 테스트 용이성 분석

### 5.1. 강점 (Strengths)

*   **높은 코드 가독성 및 문서화**: 대부분의 코드에 타입 힌트가 명확하게 적용되어 있으며, `README.md`, `code_review.md` 등 문서가 매우 상세하여 코드의 의도를 파악하기 쉽습니다.
*   **견고한 테스트 프랙티스**: `pytest`를 사용한 단위/통합 테스트가 존재하며, `onnx`와 같은 외부 의존성을 Mocking하여 테스트의 안정성과 속도를 높인 점은 매우 훌륭합니다.

### 5.2. 개선 제안 (Recommendations)

*   **테스트 커버리지 확대 (Increase Test Coverage)**:
    *   **문제**: `code_review.md`와 칸반 보드(`TST-01`)에서 지적되었듯이, `mapper.py`의 변환 로직, `reporting.py`의 통계 계산, 다양한 `SimConfig` 조합에 대한 테스트가 부족합니다.
    *   **제안**: `TST-01` 태스크의 우선순위를 높여, 프로젝트의 안정성을 더욱 강화해야 합니다. 특히 다양한 하드웨어 설정값(`SimConfig`)에 따른 경계값(edge case) 테스트를 추가하는 것이 중요합니다.
*   **상수 관리 (Centralize Constants)**:
    *   **문제**: `mapper.py`, `scheduler.py` 등 여러 파일에 `"MatMul"`, `"LOAD"`, `"RESOURCE_SPM"`과 같은 문자열 리터럴이 하드코딩되어 있습니다. 이는 오타에 취약하고 유지보수를 어렵게 만듭니다.
    *   **제안**: 칸반 보드의 `REF-03` (Opcode 상수화)을 즉시 진행할 것을 강력히 권장합니다. `isa/opcodes.py`와 같은 파일을 만들어 모든 Opcode와 리소스 이름을 `Enum`으로 관리하면 코드의 안정성과 일관성이 크게 향상될 것입니다.

---

## 6. Prioritized Action Plan (우선순위 실행 계획)

분석 결과를 바탕으로, 기존 칸반 보드와 연계하여 다음과 같은 실행 계획을 제안합니다.

### 🔴 P0: 즉시 실행 (Critical Issues)
1.  **`REF-03` | Opcode 상수화**: 하드코딩된 문자열을 `Enum`으로 변경하여 코드 안정성을 즉시 확보합니다. (Effort: Small)
2.  **`REF-02` | 스케줄러 가독성 개선**: `calculate_op_timing` 함수를 리팩토링하여 복잡도를 낮추고 유지보수성을 확보합니다. (Effort: Medium)

### 🟡 P1: 다음 마일스톤 (High-Impact Improvements)
1.  **`TST-01` | 테스트 커버리지 확대**: `mapper.py`와 다양한 `SimConfig` 조합에 대한 테스트를 추가하여 신뢰도를 높입니다. (Effort: Medium)
2.  **`M-01` | 메모리/버스 모델 고도화 v1**: `IOBufferTracker`의 FIFO 동작을 개선하고, 시스템 버스 모델을 추가하여 시뮬레이션 정확도를 높입니다. (Effort: Large)

### 🟢 P2: 장기 계획 (Long-term Enhancements)
1.  **`CHR-01` | 예외 처리 및 로깅 개선**: 악의적인 입력 파일에 대한 방어 로직을 추가하고 로깅 시스템을 통일합니다. (Effort: Small)
2.  **`P-01` | 컴파일러 타일링 패스 (auto-tile)**: 현재 수동인 타일링을 자동화하여 사용성을 개선합니다. (Effort: Large)
