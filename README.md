# HCIS Streamlit (Operational) - Drop-in Replacement

## 폴더 구조
- app.py
- config.py
- pages/
  - 01_개요.py
  - 02_고객.py
  - 03_관리자_고객조회.py
  - 04_관리자_유지보수.py
- utils/
  - data_loader.py
  - scoring.py
  - rules.py
  - psi.py
  - policy_log.py
  - shap_reason.py
  - lim_explainer.py

## 실행 방법
1) 이 폴더로 이동
2) 아래 중 하나를 루트에 두기
   - final.parquet
   - app_train_full_type_7.parquet
3) (선택) oof_predictions_top70.parquet 를 루트에 두면 관리자 페이지에서 확인 가능
4) 실행
   streamlit run app.py
