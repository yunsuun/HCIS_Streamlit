# HCIS Streamlit

## Why This Project?

금융권 실무에서 설명 가능한 리스크 판단 구조를 직접 설계해보기 위해 시작했습니다.

## 프로젝트 목적

이 프로젝트는 금융 대출 심사 과정에서
HCIS 점수체계를 설계하고,
추가검토 구간의 리스크 해석 구조를 실험하기 위한 개인 연구 프로젝트입니다.


Streamlit 기반 웹 애플리케이션으로,  
업로드된 고객 데이터를 즉시 분석·시각화하여  
정책 변경 시 영향도를 직관적으로 확인할 수 있습니다.

---
## 🚧 Current Status

- Feature Mart 구축 완료
- Airflow 기반 배치 테스트 완료
- 추가검토 구간 시뮬레이션 로직 개선 중
---

## 프로젝트 개요

본 프로젝트는 모델 성능 자체보다 ‘운영 안정성·설명 가능성’에 초점을 둡니다.

- PD 모델은 사전 학습된 모델을 그대로 사용
- 운영 환경에서는 재학습 없이 안정적으로 점수 산출
- 점수 → 구간 → 심사결정의 단계를 명확히 분리
- SHAP 기반 설명을 통해 추가검토/거절 사유를 해석 가능

---
## 데모 및 샘플 데이터 안내
```text
본 프로젝트는 포트폴리오 시연을 목적으로,
일부 정책 파라미터를 데모 환경에 한해 임시 조정하여 사용하고 있습니다.

- 추가검토 구간 시연을 충분히 보여주기 위해
  **HCIS 승인 컷오프를 720점으로 임시 상향 조정**
- 해당 조정은 **샘플 데이터(model_df_default.parquet)에만 적용**
- 실제 운영 환경에서는 원래 정책 컷오프(T_HIGH=707)를 사용

> 본 조정은 UI/시뮬레이션 시연을 위한 것이며  
> 모델 구조, PD 산출 로직, SHAP 계산에는 영향을 주지 않습니다.
```

## 시스템 흐름
```text
고객 데이터 (Parquet)
 → 전처리 / Feature 정렬
 → PD 예측 (사전 학습 모델)
 → HCIS 점수 변환
 → 승인 / 추가검토 / 거절 판단
 → 시각화 및 시뮬레이션
```

### 주요 기능
```text
1. 전체 고객 분포 분석
- HCIS 점수 분포
- 승인 / 추가검토 / 거절 비율
- 평균 PD, 평균 점수 등 KPI 요약

2. 승인 정책 시뮬레이션
- 승인 컷 / 추가검토 컷 변경
- 정책 변경 시 고객 분포 변화 확인
- 운영 기준선 시각화

3. 데이터 업로드 & 처리 (관리자 모드)
- Parquet 파일 업로드
- 즉시 전처리 → 모델 추론 → 점수화
- 결과를 model_df.parquet로 저장
- 업로드 결과를 기반으로 전체 대시보드 자동 갱신
```
### 설계 철학
```text
- 절대 기준 점수 체계
- 분위수 기반 상대평가 미사용
- 동일 점수 = 동일 의미 유지
- Dual Cut-off 구조
- 승인 / 추가검토 / 거절 구간 분리
- 중간 위험군에 대한 정책적 개입 가능
- 설명 가능성 중심
- SHAP 기반 위험 요인 분해
- 심사 판단의 근거 제시 가능
```
### 기술 스택
```text
- Language: Python
- Framework: Streamlit
- Data: pandas, numpy
- Visualization: Altair
- Model Inference: scikit-learn / XGBoost (사전 학습 모델)
- Explainability: SHAP
- Storage: Parquet
- Deployment: Streamlit Community Cloud
```
### 환경 변수 설정
```text
- 본 프로젝트는 API Key를 코드에 포함하지 않습니다.
- 로컬 실행
- GEMINI_API_KEY=YOUR_API_KEY
- Streamlit Cloud
- Settings → Secrets에 등록
- GEMINI_API_KEY = "YOUR_API_KEY"
```
### 실행 방법 (로컬)
```text
pip install -r requirements.txt
streamlit run 홈.py
```
### 프로젝트 구조
```text
HCIS_Streamlit/
├─ 홈.py
├─ config.py
├─ airflow/
│  ├─ dags
│  │  └─ hello_hcis.py
│  └─ docker-compose.yaml
├─ artifacts/
│  ├─ models
│  │  ├─ v1.0.0_XGB_artifact.joblib
│  │  ├─ v1.0.1_XGB_artifact.joblib
│  │  └─ v1.0.2_XGB_artifact.joblib
├─ pages/
│  ├─ 01_개요.py
│  ├─ 02_대출_심사.py
│  └─ 03_추가검토_대상.py
├─ modules/
│  ├─ __init__.py
│  ├─ preprocess.py
│  ├─ inference.py
│  ├─ align.py
│  ├─ calibrators.py
│  ├─ cleaning.py
│  └─ model_loader.py
├─ utils/
│  ├─ __init__.py
│  ├─ behavioral_insights.py
│  ├─ data_loader.py
│  ├─ hcis_core.py
│  ├─ feature_semantic_map.py
│  ├─ llm_gemini.py
│  ├─ llm_report.py
│  ├─ review_simulation.py
│  ├─ risk_types.py
│  ├─ rules.py
│  ├─ shap_reason.py
│  └─ data_loader.py
├─ requirements.txt
└─ .gitignore
```
### 비고
```text
- 본 프로젝트는 운영 시나리오 시뮬레이션 및 포트폴리오 목적으로 제작되었습니다.
- 실제 금융 서비스 적용을 위해서는 추가적인 검증 및 규제 검토가 필요합니다.
```

