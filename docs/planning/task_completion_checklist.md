# 작업 완료 체크리스트

새로운 기능 개발이나 수정 작업이 완료되면, 아래 절차에 따라 문서 업데이트 및 커밋을 진행합니다.

### 1. 관련 문서 업데이트

-   **`docs/planning/working_history.md`**:
    -   수행한 작업 내역을 최신순으로 간결하게 추가합니다.
    -   (예: `V250901a - 스케줄러 리뷰 피드백 반영 (fix/scheduler-review-fixes): ...`)

-   **`docs/planning/kanban_board.md`**:
    -   완료된 작업 항목을 `In-Progress` 또는 `Review`에서 `Done` 컬럼으로 이동시킵니다.
    -   필요 시, 백로그의 우선순위를 다시 조정합니다.

-   **기타 아키텍처/PRD 문서**:
    -   `README.md`, `docs/architecture/*.md`, `docs/prd/*.md` 등 이번 작업으로 인해 변경이 필요한 모든 문서를 검토하고 최신 상태로 업데이트합니다.

### 2. 문서 변경사항 커밋

-   아래 명령어를 사용하여 문서 변경사항만 별도로 커밋합니다. 이는 코드 변경과 문서 변경의 히스토리를 분리하여 관리하기 위함입니다.

```bash
# 1. 변경된 문서 파일들을 스테이징합니다.
git add docs/

# 2. 커밋 메시지와 함께 커밋합니다.
git commit -m "feat(docs): <작업 내용>에 대한 문서 업데이트" -m "작업 완료에 따라 working_history, kanban_board 등을 최신 상태로 업데이트합니다."
```
