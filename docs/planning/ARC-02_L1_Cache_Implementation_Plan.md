# ARC-02: Py-V L1 캐시 구현 상세 계획

## 개요
- **과제**: `ARC-02` | Py-V RISC-V 코어에 L1 캐시를 구현하여 CPU 모델의 현실성을 높입니다.
- **목표**: 시뮬레이터의 사이클 정확성을 높여, 특히 `tight-coupled` 모드의 제어 오버헤드를 정밀하게 측정할 수 있는 기반을 마련합니다.
- **범위**: `py-v/` 디렉토리 내의 RISC-V 코어 시뮬레이터에 대한 수정.

---

### **Phase 1: 기반 설계 및 핵심 로직 구현 (Foundation & Core Logic)**

이 단계에서는 재사용 가능하고 테스트 가능한 캐시의 핵심 컴포넌트를 만듭니다.

1.  **캐시 설정 클래스 정의 (`py-v/pyv/cache_config.py` 생성)**
    - **작업:** 캐시의 동작을 정의하는 `CacheConfig` 데이터 클래스를 생성합니다.
    - **세부 내용:** `size_kb`, `line_size_bytes`, `associativity` (연관성), `hit_latency_cycles`, `miss_penalty_cycles`, `write_policy` (e.g., 'Write-Back') 등 설정 가능한 모든 파라미터를 포함합니다.
    - **목표:** 설정을 코어 로직과 분리하여 유연하고 확장 가능한 구조를 만듭니다.

2.  **L1 캐시 모듈 구현 (`py-v/pyv/cache.py` 생성)**
    - **작업:** 실제 캐시의 모든 동작을 담당하는 `L1Cache` 클래스를 구현합니다.
    - **세부 내용:**
        - 주소(address)를 `tag`, `index`, `offset`으로 분해하는 로직.
        - 캐시 라인(valid, dirty, tag, data blocks) 관리.
        - Read/Write 요청에 대한 Hit/Miss 판정 로직.
        - `Write-Back` 정책에 따른 데이터 쓰기 및 'dirty' 비트 관리.
        - `LRU(Least Recently Used)` 교체 정책 구현.
    - **목표:** CPU 모델과 독립적으로 동작할 수 있는 견고한 캐시 컴포넌트를 완성합니다.

3.  **핵심 로직 단위 테스트 (`py-v/test/test_cache.py` 생성)**
    - **작업:** `L1Cache` 클래스의 정확성을 검증하는 단위 테스트를 작성합니다.
    - **세부 내용:**
        - 주소 분해, 읽기/쓰기, Hit/Miss 시나리오, 데이터 교체(eviction), Write-Back 동작 등 모든 핵심 기능을 철저히 검증합니다.
    - **목표:** TDD(Test-Driven Development) 접근 방식으로 초기 단계부터 코드의 안정성을 확보합니다.

### **Phase 2: CPU 모델 통합 및 타이밍 모델 적용 (Integration & Timing)**

이 단계에서는 구현된 캐시를 CPU 모델에 연결하고, 사이클 정확성을 높이는 타이밍 모델을 적용합니다.

4.  **CPU-캐시-메모리 연동 (`py-v/pyv/models/singlecycle.py` 수정)**
    - **작업:** 기존 `SingleCycle` CPU 모델이 메모리에 직접 접근하는 대신, `L1Cache`를 통해 접근하도록 수정합니다.
    - **세부 내용:**
        - Instruction Fetch, `LOAD`, `STORE` 명령어 실행 시, `L1Cache`의 `read`/`write` 메서드를 호출하도록 변경합니다.
        - 캐시 미스(Miss) 발생 시, `L1Cache`가 하위 메모리(`mem.py`)와 통신하여 데이터 블록을 가져오는 로직을 구현합니다.
    - **목표:** CPU의 메모리 접근 파이프라인에 캐시를 완벽하게 통합합니다.

5.  **사이클 정확성 모델링 (Stall & Latency 적용)**
    - **작업:** 캐시의 Hit/Miss에 따른 사이클 변화를 시뮬레이터에 반영합니다.
    - **세부 내용:**
        - 캐시 Hit 시: `hit_latency_cycles` 만큼 사이클을 추가합니다.
        - 캐시 Miss 시: CPU 파이프라인을 Stall 시키고, `miss_penalty_cycles` 만큼의 지연 시간을 정확히 반영합니다.
    - **목표:** `tight-coupled` 모드의 제어 오버헤드를 정확하게 측정한다는 과제의 핵심 목표를 달성합니다.

### **Phase 3: 최종 검증 및 고도화 (Validation & Refinement)**

이 단계에서는 통합된 시스템을 검증하고, 실제 하드웨어 구조를 더 정확히 모방하도록 리팩토링합니다.

6.  **통합 테스트 (`py-v/test/test_integration_cache.py` 생성)**
    - **작업:** 캐시가 통합된 전체 `Py-V` 시뮬레이터의 동작을 검증하는 테스트를 작성합니다.
    - **세부 내용:**
        - 캐시 Hit/Miss가 의도적으로 발생하는 간단한 RISC-V 어셈블리 프로그램을 작성합니다.
        - 해당 프로그램을 시뮬레이터에서 실행하고, 캐시 동작에 따라 총 실행 사이클이 예상대로 변하는지 검증합니다.
    - **목표:** 기능적 정확성과 타이밍 정확성을 모두 갖춘 최종 결과물을 보장합니다.

7.  **I-Cache / D-Cache 분리 (리팩토링)**
    - **작업:** 현실적인 CPU 설계를 반영하여, 명령어(Instruction)와 데이터(Data) 캐시를 분리합니다.
    - **세부 내용:**
        - `SingleCycle` 모델 내에 `L1Cache` 클래스의 인스턴스를 2개(`l1_icache`, `l1_dcache`) 생성합니다.
        - Instruction Fetch는 `l1_icache`를, `LOAD`/`STORE`는 `l1_dcache`를 통하도록 라우팅합니다.
    - **목표:** 시뮬레이터의 하드웨어 모델링 수준을 한 단계 높입니다.

8.  **문서화 (`README.md` 및 관련 문서 업데이트)**
    - **작업:** L1 캐시 기능 및 관련 설정 방법을 프로젝트 문서에 추가합니다.
    - **목표:** 다른 사용자들이 새로운 기능을 쉽게 이해하고 사용할 수 있도록 합니다.
