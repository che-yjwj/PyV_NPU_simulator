# PyV-NPU 시뮬레이터 개발 계획서 (PRD v0.2, 업데이트판)
작성일: 2025-08-08 · 작성자: ChatGPT

---

## A. v0.1 장단점 분석

### 강점 (Pros)
- **레벨 기반 개발 로드맵(IA, IA_TIMING, CA_HYBRID, CA_FULL)**: 기능 → 타일 → 이벤트 → 사이클 단위로 정밀도 향상.
- **NPU-IR 초안 제시**: LOAD/STORE, GEMM_T, CONV2D_T, ATTN_T, BARRIER/WAIT 등 최소 명세.
- **메모리/데이터 이동 모델 반영**: SPM(bank), DMA double-buffer, NoC/DRAM 경합 요소 포함.
- **리포트 아티팩트 정의**: Gantt, 자원 사용률, 병목 Top-N 지표 중심 설계.
- **다양한 사용자 스토리**: 아키텍트, 컴파일러, HW 엔지니어 활용 가능.

### 약점 (Cons) & 보완 포인트
1. **타이밍 모델 단순화**
   - 기존: `T_tile = max(T_in + T_comp, T_out)` → steady-state 파이프라인 반영 부족.
   - **개선**: `T_ss = max(T_load, T_comp, T_store)`, `T_total = T_fill + (N-1)*T_ss + T_drain`.

2. **스케줄러 규칙 미정**
   - List-sched만 제시, bank-aware·prefetch 고려 미흡.
   - **개선**: CP-priority × bank-aware × prefetch-aware 융합 휴리스틱.

3. **IR 필드 불충분**
   - layout/stride/pad/dtype 외 확장 메타 없음.
   - **개선**: dilation, groups, memory view(NCHW/NHWC), quant meta(VQ codebook id/offset) 추가.

4. **경합 모델 미정량**
   - NoC/DRAM/Bank 경합 모델이 정성적.
   - **개선**: token-bucket, bank occupancy 기반 stall 계산.

5. **검증 기준 부재**
   - 교차검증 허용 오차, 회귀 커버리지 미정.
   - **개선**: KPI/AC 정의(IA_TIMING 오차 ±15% 이내).

6. **관측 가능성 부족**
   - 이벤트 로그/카운터 세트 미흡.
   - **개선**: 표준 이벤트·카운터 정의.

7. **설정 표준화 미비**
   - run.yaml 구조만 제시.
   - **개선**: JSON-Schema 포함.

---

## B. v0.2 개선 요약
- **타이밍 모델 교정**: fill/drain 포함 steady-state 수식 반영.
- **스케줄러 명세 강화**: bank-aware·prefetch-aware 규칙 통합.
- **IR 확장**: ATTENTION, CONV 확장 필드, VQ 메타 포함.
- **경합 모델 정량화**: bank 충돌률, NoC 직렬화 지연 포함.
- **KPI/AC 설정**: 정확도·커버리지 기준 명시.
- **관측 표준화**: 이벤트·카운터 정의, 리포트 템플릿 제시.
- **재현성 보강**: run.yaml 스키마 제공.

---

## C. 예시 run.yaml 구조
```yaml
tc: 2
vc: 4
spm_banks: 8
dma_channels: 2
bw_dram: 102.4GBps
bw_noc: 256GBps
tile_m: 128
tile_n: 128
tile_k: 64
model: model.onnx
report_dir: out/exp_001
```

## D. JSON-Schema 예시
```json
{
  "type": "object",
  "properties": {
    "tc": {"type": "integer", "minimum": 1},
    "vc": {"type": "integer", "minimum": 1},
    "spm_banks": {"type": "integer", "minimum": 1}
  },
  "required": ["tc", "vc", "spm_banks", "model"]
}
```

---

## E. KPI / Acceptance Criteria
- IA_TIMING-CA_FULL 타이밍 오차(MAPE) ≤ ±15% (보정 후)
- 회귀 테스트 케이스 ≥ 200, 커버리지 ≥ 90%
- 병목 원인 Top-3 자동 리포트
