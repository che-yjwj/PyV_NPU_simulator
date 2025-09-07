# `test_singlecycle.py` 테스트 실패 상세 디버깅 기록

**작성일**: 2025-09-07
**작성자**: Gemini

## 1. 최초 문제 식별

- **현상**: `pytest py-v/test/test_singlecycle.py` 실행 시, `TestCSR`와 `TestExceptions` 클래스의 모든 테스트(10개)가 실패.
- **실패 로그 (예시 - `test_csrrw`)**:
  ```
  E       assert 0 == 1073742080
  py-v\test\test_singlecycle.py:41: AssertionError
  ```
- **핵심 원인 분석**: 위 로그는 `csrrw` 명령어를 통해 `misa` 레지스터 값을 `x5` 레지스터에 써야 하지만, 실제로는 `0`이 쓰여졌음을 의미. 로그 파일의 상세 내용을 확인한 결과, 명령어 디코딩 단계(`IDStage`)에서 `csrrw` 명령어의 opcode를 `SYSTEM (0x1C)`이 아닌 `OP-IMM (4)` 또는 `0`으로 잘못 디코딩하고 있음을 확인. 이는 CPU가 테스트용 명령어가 아닌, `nop`이나 `0`을 명령어 메모리에서 읽어오고 있음을 시사함.

## 2. 가설 및 검증 과정

여러 가설을 세우고 다음 단계에 따라 검증 및 수정을 시도했습니다.

### 가설 1: 명령어 캐시(I-Cache)와 메인 메모리 간의 데이터 불일치

- **가설 상세**: 테스트 코드가 `mem_write_word` 헬퍼 함수를 통해 메인 메모리에 직접 접근하여 명령어를 덮어쓴다. 하지만 CPU의 명령어 패치 유닛은 L1 명령어 캐시를 통해 명령어를 읽는다. `mem_write_word`가 캐시를 업데이트하지 않아, 캐시는 이전에 채워져 있던 `nop` (또는 0) 값을 계속 CPU에 전달할 것이라는 추론.
- **검증 및 수정**:
    1.  **캐시 무효화 기능 추가**:
        - `py-v/pyv/cache.py`의 `L1Cache` 클래스에 모든 캐시 라인의 `valid` 비트를 `False`로 만드는 `invalidate()` 메소드를 추가.
        - `py-v/pyv/memory_system.py`의 `MemorySystem` 클래스에 `self.l1_icache.invalidate()`를 호출하는 `invalidate_icache()` 메소드를 추가.
    2.  **테스트 코드 수정**: `test_singlecycle.py`의 각 테스트 함수에서 `mem_write_word`로 메모리를 수정한 직후, `core.mem.invalidate_icache()`를 호출하여 캐시를 강제로 비움.
- **결과**: **실패**. 캐시를 무효화했음에도 문제는 동일하게 발생. 여전히 잘못된 명령어를 읽어옴.

### 가설 2: `sim.reset()` 호출로 인한 메모리 내용 초기화

- **가설 상세**: `git`으로 되돌린 이전 버전의 코드에서, 테스트 시작 시 호출되는 `sim.reset()`이 `Memory._reset()`을 호출하여 메모리 전체를 `0`으로 지워버린다는 가설. 이로 인해 `mem_write_word`로 쓴 명령어가 사라진다고 추론.
- **검증 및 수정**:
    1.  `py-v/pyv/mem.py`의 `_reset()` 함수가 내용을 실행하지 않고 즉시 `return`하도록 수정.
- **결과**: **실패**. 이 가설은 여러 번의 파일 되돌리기 과정에서 혼란이 있었으나, 최종적으로 `_reset` 함수는 이미 비활성화되어 있었음을 확인. 이 가설은 문제의 원인이 아니었음.

### 가설 3: 메모리 접근 데이터 타입 불일치

- **가설 상세**: `test_singlecycle.py`의 `core` fixture는 `list[int]`에 정수 값을 직접 할당하여 메모리를 초기화하는 반면, `L1Cache`의 `read_block` 함수는 `bytearray`를 사용하여 메모리 블록을 읽어온다. 이 과정에서 데이터 타입이 섞이면서 예기치 않은 변환 오류가 발생할 것이라는 추론.
- **검증 및 수정**:
    1.  `test_singlecycle.py`의 `mem_write_word` 함수와 `core` fixture의 메모리 초기화 코드를 `val.to_bytes(4, byteorder='little')`를 사용하여 `bytearray` 타입으로 통일.
- **결과**: **실패**. 데이터 타입을 일치시켰음에도 문제는 해결되지 않음.

## 3. 최종 결론 및 후속 조치 제안

- **현재 상태**: 모든 시도 후에도 문제는 해결되지 않았으며, CPU는 여전히 메모리에서 잘못된 명령어(0 또는 nop)를 가져오고 있습니다.
- **추정 원인**: `git`으로 코드를 되돌리는 과정에서 확인했듯이, 이 버그는 `memory_system.py`와 관련 하위 모듈의 리팩토링 과정에서 발생한 것이 거의 확실합니다. 캐시와 메인 메모리 사이의 상호작용, 특히 캐시 미스 발생 시 메인 메모리에서 데이터를 가져와 캐시 라인을 채우는(`fill_line`) 로직에 미묘한 버그가 있을 가능성이 가장 높습니다.
- **향후 방향**:
    1.  리팩토링된 `memory_system.py`, `mem.py`, `cache.py`의 `git diff`를 다시 한번 면밀히 검토하여, 리팩토링 이전과 이후의 로직 차이를 상세히 비교해야 합니다.
    2.  `L1Cache.read()` 함수의 miss 처리 경로에 더 많은 디버깅 로그를 추가하여, `read_block`으로 메인 메모리에서 읽어온 값(`new_data`)과 `fill_line`으로 캐시에 채워지는 값을 사이클별로 추적해야 합니다.
