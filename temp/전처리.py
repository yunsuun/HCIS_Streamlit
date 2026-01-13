# =======================================
# 0. 전처리하기 위한 원본 데이터셋 불러오기
# =======================================

import pandas as pd
from pathlib import Path

def clean_data_load():
    """
    함수 설명: 전처리하기 위해서 파일 업로드 필요로 해야 함.
    Args:
        - 없음
        - 단, 제출 파일 구성을 Four_Idot 안에 Dataset에 해당 parquet 파일이 존재하게 해야 함.
    
    Returns:
        - app_df: 각 고객이 존재하는 기본 정보
        - bureau, bureau_bal, pre_app, inst_payments, pos_cash, creditcard: 전처리하기 위해 필요한 원본 정보
        - id_set: 아이디 리스트
    """
    PROJECT_ROOT = Path.cwd().parents[1]  # Four_Idot 하위에 있게 파일 구성 미리 해놔야 한 조심
    DATA_DIR = PROJECT_ROOT / "Dataset"
    # 이 데이터셋이 새로 고객을 받는 데이터셋 파일 (새로 받는 데이터는 app_train이나 app_test여야 하고, 나머지는 steamlit 안에 존재해야 함.)
    # app_test = pd.read_parquet(DATA_DIR / "app_test_4.parquet", engine="fastparquet")
    app_test = pd.read_parquet("C:\\Users\\user\\Documents\\GitHub\\Four_Idot\\Dataset\\application_train.parquet", engine="fastparquet")
    bureau = pd.read_parquet("C:\\Users\\user\\Documents\\GitHub\\Four_Idot\\Dataset\\bureau.parquet", engine="fastparquet")
    bureau_bal = pd.read_parquet("C:\\Users\\user\\Documents\\GitHub\\Four_Idot\\Dataset\\bureau_balance.parquet", engine="fastparquet")
    pre_app = pd.read_parquet("C:\\Users\\user\\Documents\\GitHub\\Four_Idot\\Dataset\\previous_application.parquet", engine="fastparquet")
    inst_payments = pd.read_parquet("C:\\Users\\user\\Documents\\GitHub\\Four_Idot\\Dataset\\installments_payments.parquet", engine="fastparquet")
    pos_cash = pd.read_parquet("C:\\Users\\user\\Documents\\GitHub\\Four_Idot\\Dataset\\POS_CASH_balance.parquet", engine="fastparquet")
    creditcard = pd.read_parquet("C:\\Users\\user\\Documents\\GitHub\\Four_Idot\\Dataset\\credit_card_balance.parquet", engine="fastparquet")
    # 모두 대문자라 소문자로 전처리
    app_test.columns = app_test.columns.str.lower()

    # 새로 심사할 데이터셋 명 변경하기
    app_df = app_test

    # Step 0. 기준 ID 집합
    id_set = set(app_df["sk_id_curr"].unique())

    return app_df, bureau, bureau_bal, pre_app, inst_payments, pos_cash, creditcard, id_set

app_df, bu, bu_bal, pre, inst, pos, cc, id_set = clean_data_load()

import numpy as np

# =================================
# 2. app 관련 컬럼 생성 함수
# =================================

def app_derived_variable(customer_df: pd.DataFrame) -> pd.DataFrame:
    """
    목적
    ----
    - application 데이터(app_train / app_test)에 대해
      다중 고객 기준 app-level 파생변수 생성
    - 계산만 수행하고, merge는 외부에서 처리

    입력
    ----
    customer_df : application 데이터 (여러 sk_id_curr)

    출력
    ----
    app_features : sk_id_curr 기준 파생변수 DataFrame
    """

    df = customer_df.copy()

    # =====================================================
    # 1. days_* 특수값 처리 (365243 → NaN)
    # =====================================================
    days_cols = [c for c in df.columns if c.startswith("days_")]
    df[days_cols] = df[days_cols].replace(365243, np.nan)

    # =====================================================
    # 2. 나이 / 근속 파생변수
    # =====================================================
    if "days_birth" in df.columns:
        df["app_age_years"] = (-df["days_birth"] / 365).round(1)

    if "days_employed" in df.columns:
        df["app_years_employed"] = -df["days_employed"] / 365

    if {"app_age_years", "app_years_employed"}.issubset(df.columns):
        age_denom = df['app_age_years'].replace(0, np.nan)
        df["app_employment_stability_ratio"] = (
            df["app_years_employed"] / age_denom
        )

    # =====================================================
    # 3. 범주형 특수값 정리 (XNA / XAP)
    # =====================================================
    obj_cols = df.select_dtypes(include=["object", "category"]).columns
    df[obj_cols] = df[obj_cols].replace(
        {"XNA": np.nan, "xna": np.nan, "XAP": np.nan, "xap": np.nan}
    )

    # =====================================================
    # 4. 문서 관련 파생변수
    # =====================================================
    doc_cols = [c for c in df.columns if c.startswith("flag_document_")]
    if doc_cols:
        df["app_n_documents"] = df[doc_cols].sum(axis=1)

    if "flag_document_3" in df.columns:
        df["flag_document_3"] = df["flag_document_3"]

    # =====================================================
    # 5. 금액 로그 파생변수
    # =====================================================
    log_targets = ["amt_credit", "amt_annuity", "amt_goods_price"]
    for col in log_targets:
        if col in df.columns:
            df[f"app_{col}_log"] = np.log1p(df[col].clip(lower=0))

    # =====================================================
    # . 소득/부담 비율 파생변수
    # =====================================================
    if 'amt_credit' in df.columns and 'amt_income_total' in df.columns:
        denom = df['amt_income_total'].replace(0, np.nan)
        df['app_credit_income_ratio'] = df['amt_credit'] / denom

    # =====================================================
    # 6. 부담 비율 파생변수
    # =====================================================
    if {"amt_annuity", "amt_income_total"}.issubset(df.columns):
        df["app_annuity_income_ratio"] = (
            df["amt_annuity"] /
            df["amt_income_total"].replace(0, np.nan)
        )

    if {"amt_annuity", "amt_credit"}.issubset(df.columns):
        df["app_payment_rate"] = (
            df["amt_annuity"] /
            df["amt_credit"].replace(0, np.nan)
        )

    # =====================================================
    # . 거주지 품질 지수
    # =====================================================
    if 'region_rating_client' in df.columns and 'region_rating_client_w_city' in df.columns:
        df['app_area_quality_index'] = (
            df['region_rating_client'] +
            df['region_rating_client_w_city']
        )

    # =====================================================
    # 7. EXT_SOURCE 핵심 파생변수
    # =====================================================
    ext_cols = ["ext_source_1", "ext_source_2", "ext_source_3"]
    if set(ext_cols).issubset(df.columns):

        df['app_n_ext_source_available'] = df[ext_cols].notna().sum(axis=1).astype('int8')

        df["app_ext_source_min"] = df[ext_cols].min(axis=1)

        df["app_ext_source_weighted"] = (
            0.5 * df["ext_source_1"] +
            0.3 * df["ext_source_2"] +
            0.2 * df["ext_source_3"]
        )

    # =====================================================
    # 8. Social Circle 클리핑
    # =====================================================
    if "def_30_cnt_social_circle" in df.columns:
        df["app_def_30_cnt_social_circle_clipped"] = (
            df["def_30_cnt_social_circle"].clip(upper=5)
        )

    # =========================
    # 9. Credit Bureau 변수 클리핑 (+ 원본 삭제)
    # =========================
    req_cols_caps = {
        'amt_req_credit_bureau_qrt': 4
    }

    created_req_cols = []
    for col, cap in req_cols_caps.items():
        if col in df.columns:
            new_col = 'app_' + col + '_clipped'
            df[new_col] = df[col].clip(upper=cap)
            created_req_cols.append(col)

    if created_req_cols:
        df = df.drop(columns=created_req_cols)


    # =====================================================
    # 10. 최종 반환 컬럼 정리
    # =====================================================
    keep_cols = [
        "sk_id_curr",

        # days
        "days_birth", "days_id_publish", "days_employed",
        "days_last_phone_change", "days_registration",
        "years_beginexpluatation_medi",

        # age / employment / population
        "app_age_years", "app_years_employed",
        "app_employment_stability_ratio",
        "region_population_relative",

        # amounts
        "amt_credit", "amt_annuity", "amt_goods_price",
        "app_amt_credit_log", "app_amt_annuity_log",
        "app_amt_goods_price_log",

        # ratios
        "app_annuity_income_ratio", "app_payment_rate",
        "app_credit_income_ratio",

        # ext source
        "ext_source_1", "ext_source_2", "ext_source_3",
        "app_ext_source_min", "app_ext_source_weighted",
        "app_n_ext_source_available",

        # documents / social
        "app_n_documents", "flag_document_3",
        "app_def_30_cnt_social_circle_clipped",
        "app_amt_req_credit_bureau_qrt_clipped",

        # categorical originals
        "flag_own_car", "flag_work_phone", "own_car_age",
        "code_gender", "name_family_status",
        "region_rating_client_w_city", "organization_type",
        "name_contract_type", "name_income_type",
        "occupation_type", "name_education_type",
        "reg_city_not_live_city",

        # area quality
        "app_area_quality_index",
    ]


    keep_cols = [c for c in keep_cols if c in df.columns]
    app_fin = df[keep_cols]
    return app_fin


# ==============================================
# 3. Pos_cash 서브 파생변수 제작
# ==============================================

# -------------------------------------------
# 3-1. 통합 파생변수 생성 및 전처리 함수 (최종 변수 생성은 pre_app 내 존재여부에 따라 달라짐)
# -------------------------------------------

def pos_derived_variable(pos, id_set):
    """
    함수 설명: pos_cash에서 sk_id_prev 단위 DEF 발생 여부 플래그 생성
    
    Args:
        id_set: 고객 아이디 집합
        pos: 원본 pos_cash 데이터 프레임
    
    Returns:
        pos_d: ['sk_id_curr', 'sk_id_prev', 'pos_dpd_def_flag'] 컬럼을 가진 DataFrame
    """
    
    # 1. 필요한 컬럼만 필터링
    pos_d = pos.loc[
        pos["sk_id_curr"].isin(id_set),
        ["sk_id_curr", "sk_id_prev", "sk_dpd_def"]
    ].copy()
    
    # 2. sk_id_prev 단위로 DEF > 0인 기록이 하나라도 있는지 확인
    pos_d["has_def"] = (pos_d["sk_dpd_def"] > 0).astype("int8")
    
    # 3. sk_id_prev 단위 집계 (max로 any() 효과)
    pos_d = (
        pos_d
        .groupby(["sk_id_curr", "sk_id_prev"], sort=False)
        .agg(pos_dpd_def_flag=("has_def", "max"))
        .reset_index()
    )
    
    return pos_d[["sk_id_curr", "sk_id_prev", "pos_dpd_def_flag"]]

# ==============================================
# 4. creditcard 서브 파생변수 제작
# ==============================================

# -------------------------------------------
# 4-1. 통합 파생변수 생성 및 전처리 함수 (최종 변수 생성은 pre_app 내 존재여부에 따라 달라짐)
# -------------------------------------------

def cc_derived_variable(cc, id_set):
    """
    함수 설명 : creditcard 데이터셋이 존재하는 고객에 대해서 서브 파생변수를 만드는 함수 


    Args:
        - id_set: 고객 아이디 집합
        - cc: 원본 크레딧카드 데이터 프레임
    
    Returns:
        - creditcard_der: 파생변수만 만든 데이터 프레임 (cc_d)
          (만약 아이디 없다면 아예 안만드는 걸로 하기)
    """
    cc_d = cc[cc["sk_id_curr"].isin(id_set)].copy()


    # 신용한도 전처리 (0 → NaN으로 변환)
    cc_d['credit_limit'] = cc_d['amt_credit_limit_actual'].replace(0, np.nan)
    
    # 잔액 전처리 (음수 = 과납 → 0으로 처리)
    cc_d['balance_clean'] = cc_d['amt_balance'].clip(lower=0)

    # 월별 사용률 계산
    cc_d['utilization'] = cc_d['balance_clean'] / cc_d['credit_limit']
    
    # 한도가 없는 경우(NaN) → -1로 표시 (나중에 집계 시 별도 처리)
    # 0이 아닌 -1로 표시하여 "정보 없음"과 "사용률 0%"를 구분
    cc_d['utilization'] = cc_d['utilization'].fillna(-1)

    # 극단값 클리핑 (활성 계좌만)
    UTILIZATION_CLIP_MAX = 2.0 
    mask_active = cc_d['utilization'] >= 0
    cc_d.loc[mask_active, 'utilization'] = cc_d.loc[mask_active, 'utilization'].clip(
        upper=UTILIZATION_CLIP_MAX
    )

    # 한도 초과 사용 플래그 (utilization > 1)
    cc_d['over_limit_flag'] = ((cc_d['utilization'] > 1) & (cc_d['utilization'] >= 0)).astype(int)

    # ⚠️ Point-in-Time: 대출 신청 전 데이터만 사용
    CUTOFF_MONTH = 0
    df_pit = cc_d[cc_d['months_balance'] < CUTOFF_MONTH].copy()

    if len(df_pit) == 0:
        print("⚠️ 경고: Point-in-Time 필터링 후 데이터가 없습니다.")
        return pd.DataFrame()

    # --- 기본 집계 (sk_id_prev, sk_id_curr 매핑 + over_limit 카운트) ---
    agg_all = df_pit.groupby(['sk_id_prev', 'sk_id_curr']).agg(
        # 한도 초과 횟수
        cc_cnt_over_limit=('over_limit_flag', 'sum'),
    ).reset_index()

    # --- Utilization 별도 집계 (활성 계좌만: utilization >= 0) ---
    df_active = df_pit[df_pit['utilization'] >= 0].copy()

    # util 관련 과거 집계
    if len(df_active) > 0:
        util_agg = df_active.groupby(['sk_id_prev', 'sk_id_curr']).agg(
            cc_utilization_mean=('utilization', 'mean'),
            cc_utilization_max=('utilization', 'max')
        ).reset_index()
        agg_all = agg_all.merge(util_agg, on=['sk_id_prev', 'sk_id_curr'], how='left')
    else:
        agg_all['cc_utilization_mean'] = np.nan
        agg_all['cc_utilization_max'] = np.nan
    
    cc_d = agg_all[['sk_id_curr', 'sk_id_prev', 'cc_utilization_mean', 'cc_cnt_over_limit', 'cc_utilization_max']]

    return cc_d

# ==============================================
# 5. installments 서브 파생변수 제작
# ==============================================

# -------------------------------------------
# 5-1. 통합 파생변수 생성 및 전처리 함수
# -------------------------------------------

def inst_derived_variable(inst, id_set):
    """
    함수 설명 : inst 데이터셋이 존재하는 고객에 대해서 서브 파생변수를 만드는 함수

    Args:
        - id_set: 고객 번호 집합
        - inst: 원본 크레딧카드 데이터 프레임
    
    Returns:
        - inst_der: 파생변수만 만든 데이터 프레임 (inst_d)
          (inst 아예 없으면 안 만들기)
    """
    inst_d = inst.loc[inst["sk_id_curr"].isin(id_set)].copy()

    inst_d['inst_payment_delay'] = inst_d['days_entry_payment'] - inst_d['days_instalment']

    # 2. 상태 플래그: 지연 발생
    inst_d['is_delayed'] = (inst_d['inst_payment_delay'] > 0).astype('int8')

    # 3. 지연일수 (지연 발생시만 값 유지, 미지연시 NaN)
    inst_d['delay_days_value'] = inst_d['inst_payment_delay'].where(
        inst_d['inst_payment_delay'] > 0, np.nan
    )

    # ==================== 회차별 1차 집계 ====================
    # 회차별로 중복 납부 처리 (같은 회차에 여러 번 납부한 경우 통합)
    agg_dict = {
        'is_delayed': 'max',          # 한 번이라도 지연 → 지연으로 처리
        'delay_days_value': 'max'     # 최대 지연일수
    }

    df_inst = inst_d.groupby(
        ['sk_id_curr', 'sk_id_prev', 'num_instalment_number']
    ).agg(agg_dict).reset_index()

    # ==================== 대출별 전체 기간 집계 ====================
    df_all = df_inst.groupby(['sk_id_curr', 'sk_id_prev']).agg(
        delay_cnt_all=('is_delayed', 'sum'),           # 지연 회차 수
        total_cnt_all=('is_delayed', 'count'),         # 전체 회차 수
        delay_days_mean_all=('delay_days_value', 'mean')  # 평균 지연일수
    ).reset_index()

    # 비율 계산
    df_all['delay_rate_all'] = df_all['delay_cnt_all'] / df_all['total_cnt_all']
    
    # ==================== 최종 출력 ====================
    inst_d = df_all[['sk_id_curr', 'sk_id_prev', 'delay_rate_all', 'delay_days_mean_all']]
    
    return inst_d

# =============================
# 6. pre_app 서브 파생변수 제작
# =============================

# -------------------------------------------
# 6-1. 통합 파생변수 생성 및 전처리 함수
# -------------------------------------------

def pre_derived_variable(pre, id_set):
    """
    함수 설명 : pre 데이터셋이 존재하는 고객에 대해서 서브 파생변수를 만드는 함수

    Args:
        - id_set: 고객 번호 집합
        - pre: 원본 크레딧카드 데이터 프레임
    
    Returns:
        - pre_der: 파생변수만 만든 데이터 프레임 (pre_d)
          (pre 아예 없으면 안 만들기)
    """
    pre_d = pre[pre["sk_id_curr"].isin(id_set)].copy()

    # ----------------------------------------
    # pre_app 기본 파생변수
    # ----------------------------------------

    # 시간순서로 정렬
    pre_d = pre_d.sort_values(['sk_id_curr', 'sk_id_prev', 'days_decision'], ascending=[True, True, False])

    pre_d['credit_to_goods_ratio'] = np.where(
        pre_d['amt_goods_price'] > 0,
        pre_d['amt_credit'] / pre_d['amt_goods_price'],
        np.nan
    )  # 승인액 대비 상품가격 비율

    # 신규: 승인율 (신청 대비 승인 비율)
    pre_d['approval_ratio'] = np.where(
        pre_d['amt_application'] > 0,
        pre_d['amt_credit'] / pre_d['amt_application'],
        np.nan
    )  # 승인액/신청액 (1.0 초과 = 신청액보다 더 많이 승인)

    # 2. 시간 관련 파생변수
    pre_d['loan_duration'] = pre_d['days_last_due'] - pre_d['days_first_due']

    # 3. 계약 상태 플래그
    pre_d['is_approved'] = (pre_d['name_contract_status'] == 'Approved').astype('int8')
    
    # 9. 고객 타입 플래그
    pre_d['is_repeater'] = (pre_d['name_client_type'] == 'Repeater').astype('int8')
    pre_d['is_new'] = (pre_d['name_client_type'] == 'New').astype('int8')


    # ----------------------------------------
    # 대출 상환 별 집계 (sk_id_prev)
    # ----------------------------------------

    pre_d = pre_d.groupby(['sk_id_curr', 'sk_id_prev']).agg(
        # ===== 금액 관련 =====
        pre_amt_annuity_mean=('amt_annuity', 'mean'),
        pre_amt_credit_mean=('amt_credit', 'mean'),
        pre_amt_credit_max=('amt_credit', 'max'),
        pre_amt_credit_min=('amt_credit', 'min'),
        pre_credit_to_goods_ratio_mean=('credit_to_goods_ratio', 'mean'),
        pre_approval_ratio_mean=('approval_ratio', 'mean'),

        # ===== 시간 관련 =====
        pre_days_decision_mean=('days_decision', 'mean'),
        pre_loan_duration_mean=('loan_duration', 'mean'),
        pre_loan_duration_max=('loan_duration', 'max'),

        # ===== 상태 플래그 =====
        pre_is_approved_sum=('is_approved', 'sum'),

        # ===== 고객 타입 =====
        pre_is_new_sum=('is_new', 'sum'),
        pre_is_repeater_sum=('is_repeater', 'sum'),

        # ===== 추가 카테고리 정보 =====
        pre_weekday_appr_process=('weekday_appr_process_start', 'first'),

        # ===== 신청 횟수 =====
        pre_application_count=('sk_id_prev', 'count')
    ).reset_index()

    pre_d = pre_d[['sk_id_curr', 'sk_id_prev',
                  'pre_amt_annuity_mean', 'pre_amt_credit_mean', 'pre_amt_credit_max',
                  'pre_amt_credit_min', 'pre_credit_to_goods_ratio_mean', 'pre_approval_ratio_mean',
                  'pre_days_decision_mean', 'pre_loan_duration_mean', 'pre_loan_duration_max',
                  'pre_is_approved_sum', 'pre_is_new_sum', 'pre_is_repeater_sum',
                  'pre_weekday_appr_process', 'pre_application_count']]
    
    return pre_d

# ============================================
# 13. bureau 최종 파생변수 제작
# ============================================

# -------------------------------------------
# 13-1. 통합 파생변수 생성 및 전처리 함수
# -------------------------------------------

def bu_derived_variable(bu: pd.DataFrame, 
                                   bu_bal: pd.DataFrame, 
                                   id_set: set) -> pd.DataFrame:
    """
    Bureau 테이블에서 8개 파생변수를 생성하는 최적화 함수
    
    Args:
        bu: 원본 bureau 데이터셋
        bu_bal: 원본 bureau_balance 데이터셋
        id_set: 대상 sk_id_curr 집합
    
    Returns:
        sk_id_curr 기준 8개 파생변수 DataFrame:
        - bu_cnt_active: 활성 대출 수
        - bu_cnt_closed: 종료 대출 수
        - bu_ratio_active_loans: 활성 대출 비율
        - bu_total_debt_for_ratio: 보정된 총 부채
        - bu_any_over_limit_debt: 한도 초과 부채 존재 여부
        - bu_total_balance_months: 총 이력 월수
        - bu_enddate_diff_avg: 종료일 차이 평균
        - bu_days_credit_update_max: 최근 업데이트 일자
    """
    
    # ==========================
    # 1. 필터링 (메모리 효율화) + bureau에 있는 sk_id_bureau만 필터링
    # ==========================
    bureau_filt = bu[bu["sk_id_curr"].isin(id_set)].copy()
    bureau_list = bureau_filt['sk_id_bureau'].unique().tolist()
    bureau_bal_filt = bu_bal[bu_bal['sk_id_bureau'].isin(bureau_list)].copy()

    # ==========================
    # 2. bureau_balance: 같은 달 중복 row 정리 + bureau_balance: 정렬 (과거 → 최근)
    # ==========================
    status_map = {'X': 0, 'C': 0, '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5}
    bureau_bal_filt["status_score"] = bureau_bal_filt["status"].map(status_map).astype("int8")
    
    idx_worst = (
        bureau_bal_filt
        .groupby(['sk_id_bureau', 'months_balance'])['status_score']
        .idxmax()
    )
    bureau_bal_filt = bureau_bal_filt.loc[idx_worst].copy()
    bureau_bal_filt.drop(columns=['status_score'], inplace=True)  

    bureau_bal_filt = (
        bureau_bal_filt
        .sort_values(['sk_id_bureau', 'months_balance'], ascending=[True, True])
        .reset_index(drop=True)
    )  

    # ==========================
    # 3. C 이후 숫자 status 제거 (벡터화)
    # ==========================
    # C가 처음 등장한 월 찾기
    is_c = (bureau_bal_filt['status'] == 'C')
    first_c_month = (
        bureau_bal_filt[is_c]
        .groupby('sk_id_bureau')['months_balance']
        .min()
    )

    bureau_bal_filt = bureau_bal_filt.merge(
        first_c_month.rename('first_c_month'),
        on='sk_id_bureau',
        how='left'
    )

    num_status = set(['0', '1', '2', '3', '4', '5'])

    drop_mask = (
        bureau_bal_filt['first_c_month'].notna() &
        (bureau_bal_filt['months_balance'] > bureau_bal_filt['first_c_month']) &
        bureau_bal_filt['status'].isin(num_status)
    )

    bureau_bal_filt = bureau_bal_filt[~drop_mask].copy()
    bureau_bal_filt.drop(columns=['first_c_month'], inplace=True)
    
    # ==========================
    # 4. bureau 부채 보정
    # ==========================
    bureau_clean = bureau_filt.copy()

    debt = bureau_clean['amt_credit_sum_debt']
    credit_sum = bureau_clean['amt_credit_sum']

    # 부채가 amt_credit_sum보다 큰 경우 (over-limit)
    bureau_clean['over_limit_debt_flag'] = (debt > credit_sum).astype(int)

    # ==========================
    # 5. ratio/합계 계산용 "보정된 부채" 만들기
    # ==========================
    debt_for_ratio = debt.copy()
    debt_for_ratio = debt_for_ratio.clip(lower=0)
    debt_for_ratio = debt_for_ratio.mask(
        (debt_for_ratio > credit_sum) & (credit_sum > 0),
        credit_sum
    )
    bureau_clean['amt_credit_sum_debt_for_ratio'] = debt_for_ratio
        
    # ==========================
    # 6. 너무 오래된 폐쇄 loan 제거 (8년 ≈ 3000일)
    # ==========================
    bureau_clean['very_old_closed_flag'] = (
        (bureau_clean['credit_active'] == 'Closed') &
        (bureau_clean['days_enddate_fact'] < -3000)
    )

    bureau_for_agg = bureau_clean[~bureau_clean['very_old_closed_flag']].copy()
    
    # ==========================
    # 7. bureau_balance: sk_id_bureau 단위 집계 (cnt_months만 필요)
    # ==========================
    bureau_bal_agg = (
        bureau_bal_filt
        .groupby('sk_id_bureau')
        .agg(
            cnt_months=('months_balance', 'count'),
        )
        .reset_index()
    )

    bureau_bal_agg['has_balance_flag'] = 1
    
    # ==========================
    # 8. bureau + bureau_balance 결합 (bureau_enriched)
    # ==========================
    bureau_enriched = bureau_for_agg.merge(
        bureau_bal_agg,
        on='sk_id_bureau',
        how='left'
    )

    bureau_enriched['enddate_diff'] = (
        bureau_enriched['days_enddate_fact'] - bureau_enriched['days_credit_enddate']
    )


    # ==========================
    # 9. sk_id_curr 단위 집계
    # ==========================
    cur_agg = bureau_enriched.groupby('sk_id_curr').agg(
        # 대출 상태 개수
        n_bureau_loans=('sk_id_bureau', 'count'),
        bu_cnt_active=('credit_active', lambda x: (x == 'Active').sum()),
        bu_cnt_closed=('credit_active', lambda x: (x == 'Closed').sum()),
        
        # 종료일 차이: 평균
        bu_enddate_diff_avg=('enddate_diff', 'mean'),

        # 금액 관련
        bu_total_debt_for_ratio=('amt_credit_sum_debt_for_ratio', 'sum'),

        # balance 이력 길이
        bu_total_balance_months=('cnt_months', 'sum'),

        # 부채 이상치 플래그
        bu_any_over_limit_debt=('over_limit_debt_flag', 'any'),

        # 기간 정보
        bu_days_credit_update_max=('days_credit_update', 'max'),
    ).reset_index()

    # ==========================
    # 12. 비율 파생변수 추가
    # ==========================
    def safe_ratio(num, denom):
        """분모가 0 또는 NaN인 경우 NaN을 반환하는 안전한 비율 계산 함수"""
        return np.where(denom > 0, num / denom, np.nan)

    cur_agg['bu_ratio_active_loans'] = safe_ratio(cur_agg['bu_cnt_active'], cur_agg['n_bureau_loans'])

    # bu_any_over_limit_debt를 int로 변환
    cur_agg['bu_any_over_limit_debt'] = cur_agg['bu_any_over_limit_debt'].astype(int)

    # ==========================
    # 13. 최종 컬럼 선택
    # ==========================
    bu_fin = cur_agg[[
        "sk_id_curr",
        "bu_cnt_active",
        "bu_cnt_closed",
        "bu_ratio_active_loans",
        "bu_total_debt_for_ratio",
        "bu_any_over_limit_debt",
        "bu_total_balance_months",
        "bu_enddate_diff_avg",
        "bu_days_credit_update_max",
    ]]
        
    # 최종 컬럼 선택 및 반환
    return bu_fin

# ---------------
# 2, 3, 4, 5, 6, 7번 실행
# ---------------
import time

def timer(fn, *args):
    t0 = time.time()
    out = fn(*args)
    print(f"{fn.__name__}: {time.time() - t0:.1f}s")
    return out

app_fin = timer(app_derived_variable, app_df)
pos_d   = timer(pos_derived_variable, pos, id_set)
cc_d    = timer(cc_derived_variable, cc, id_set)
inst_d  = timer(inst_derived_variable, inst, id_set)
pre_d   = timer(pre_derived_variable, pre, id_set)
bu_fin  = timer(bu_derived_variable, bu, bu_bal, id_set)


# ============================================
# 8. 데이터셋 통합 파생변수
# ============================================

# -------------------------------------------
# 8-1. 통합 파생변수 생성 및 전처리 함수
# -------------------------------------------

def all_merge_data(app_fin, pos_d, cc_d, inst_d, pre_d, bu_fin):
    """
    함수 설명 : 그동안 만들었던 데이터셋을 합쳐서 하나의 데이터셋으로 표현하는 함수

    Args:
        - app_fin: app 최종 파생변수 제작 완료된 데이터셋
        - pos_d: pos_cash 서브 파생변수 (sk_id_prev) 단위 생성된 데이터셋
        - cc_d: creditcard 서브 파생변수 (sk_id_prev) 단위 생성된 데이터셋
        - inst_d: inst 서브 파생변수 (sk_id_prev) 단위 생성된 데이터셋
        - pre_d: pre_app 서브 파생변수 (sk_id_prev) 단위 생성된 데이터셋
        - bu_fin: bureau 최종 파생변수 제작 완료된 데이터셋 

    Returns:
        - df_final: 최종 파생변수 제작까지 모두 완료된 데이터셋
    """
    # ============================================================
    # pre_app + POS / CC / INST (sk_id_prev 기준)
    # ============================================================
    merge_prev = pre_d.copy()

    # sk_id_curr은 pre_d에만 유지, 나머지는 제거 후 merge
    merge_prev = merge_prev.merge(
        pos_d.drop(columns='sk_id_curr'),
        on='sk_id_prev',
        how='left'
    )

    merge_prev = merge_prev.merge(
        cc_d.drop(columns='sk_id_curr'),
        on='sk_id_prev',
        how='left'
    )

    merge_prev = merge_prev.merge(
        inst_d.drop(columns='sk_id_curr'),
        on='sk_id_prev',
        how='left'
    )

    # ============================================================
    # pre_app 범주형 → 행동 플래그 파생
    # ============================================================
    merge_prev['pre_is_weekend'] = (
        merge_prev['pre_weekday_appr_process']
        .isin(['SATURDAY', 'SUNDAY'])
        .astype('int8')
    )

    # ============================================================
    # STEP 3 | prev → curr 집계
    # ============================================================

    # 신청 횟수 (비율 분모)
    app_cnt = (
        merge_prev
        .groupby('sk_id_curr')['pre_application_count']
        .first()
    )

    pre_curr = (
        merge_prev
        .groupby('sk_id_curr')
        .agg(
            # ===== 요일 성향 =====
            pre_weekend_app_ratio=('pre_is_weekend', 'mean'),
            pre_weekday_variety=('pre_weekday_appr_process', 'nunique'),

            # ===== 계약 구조 =====
            pre_approved_cnt=('pre_is_approved_sum', 'sum'),
            pre_new_cnt=('pre_is_new_sum', 'sum'),
            pre_repeat_cnt=('pre_is_repeater_sum', 'sum'),

            # ===== 금액 / 기간 =====
            pre_credit_mean=('pre_amt_credit_mean', 'mean'),
            pre_credit_max=('pre_amt_credit_max', 'max'),
            pre_credit_min=('pre_amt_credit_min', 'min'),
            pre_annuity_mean=('pre_amt_annuity_mean', 'mean'),
            pre_credit_to_goods_mean=('pre_credit_to_goods_ratio_mean', 'mean'),
            pre_approval_ratio=('pre_approval_ratio_mean', 'mean'),
            pre_loan_duration_mean=('pre_loan_duration_mean', 'mean'),
            pre_loan_duration_max=('pre_loan_duration_max', 'max'),
            pre_days_decision_mean=('pre_days_decision_mean', 'mean'),

            # ===== POS =====
            pos_def_flag=('pos_dpd_def_flag', 'max'),

            # ===== CREDIT CARD =====
            cc_util_mean=('cc_utilization_mean', 'mean'),
            cc_util_max=('cc_utilization_max', 'max'),
            cc_over_limit=('cc_cnt_over_limit', 'sum'),

            # ===== INSTALLMENTS =====
            inst_delay_rate=('delay_rate_all', 'mean'),
            inst_delay_days_mean=('delay_days_mean_all', 'mean'),
        )
    )

    # count → ratio 변환
    ratio_cols = ['pre_approved_cnt', 'pre_new_cnt', 'pre_repeat_cnt']
    pre_curr[ratio_cols] = pre_curr[ratio_cols].div(app_cnt, axis=0)
    pre_curr = pre_curr.reset_index()

    # ============================================================
    # app_train + bureau + pre_curr
    # ============================================================  
    df_final = (
        app_fin
        .merge(bu_fin, on='sk_id_curr', how='left')
        .merge(pre_curr, on='sk_id_curr', how='left')
    )  

    # ==========================================================
    # 데이터 타입 정리
    # ==========================================================
    float_cols = df_final.select_dtypes(include=['float64']).columns
    df_final[float_cols] = df_final[float_cols].astype('float32')

    int_cols = df_final.select_dtypes(include=['int64']).columns
    df_final[int_cols] = df_final[int_cols].astype('int32')

    bool_cols = df_final.select_dtypes(include=['bool']).columns
    df_final[bool_cols] = df_final[bool_cols].astype('int8')

    return df_final

# ---------------------------
# 8번 실행
# ---------------------------

df_final = all_merge_data(app_fin, pos_d, cc_d, inst_d, pre_d, bu_fin)

# ==============================================
# 9. 모델에 넣기 위한 전처리 작업
# ==============================================

# -------------------------------------------
# 9-1. 통합 전처리 함수
# -------------------------------------------

import pandas as pd
from typing import Tuple

def preprocess_full_minimal(
    df: pd.DataFrame,
    clip_q: Tuple[float, float] = (0.001, 0.999),
    min_ratio: float = 0.01,
    max_cardinality: int = 30
) -> pd.DataFrame:
    """
    목적
    ----
    - int 다운캐스팅
    - flag / count / days 명시 처리
    - numeric: median fill + soft clipping
    - categorical: MISSING / rare / cardinality cap

    ※ 기존 코드와 결과 의미 100% 동일
    """

    df = df.copy()
    df_id = df['sk_id_curr']
    df = df.drop(columns='sk_id_curr', axis=1)

    
    # =========================
    # 1. int dtype 다운캐스팅
    # =========================
    int_cols = df.select_dtypes(include=["int"]).columns
    for c in int_cols:
        mn, mx = df[c].min(), df[c].max()
        if mn >= -128 and mx <= 127:
            df[c] = df[c].astype("int8")
        elif mn >= -32768 and mx <= 32767:
            df[c] = df[c].astype("int16")
        elif mn >= -2_147_483_648 and mx <= 2_147_483_647:
            df[c] = df[c].astype("int32")
        else:
            df[c] = df[c].astype("int64")

    # =========================
    # 2. flag 컬럼
    # =========================
    flag_cols = ["pos_def_flag", "bu_any_over_limit_debt"]
    exist_flags = [c for c in flag_cols if c in df.columns]

    if exist_flags:
        df[exist_flags] = (
            df[exist_flags]
            .fillna(0)
            .astype("int8")
        )

    # =========================
    # 3. count 계열
    # =========================
    cnt_zero_cols = [
    'app_def_30_cnt_social_circle_clipped',
    'app_amt_req_credit_bureau_qrt_clipped',
    'pre_approved_cnt',
    'pre_new_cnt',
    'bu_cnt_active'
    ]
    exist_cnts = [c for c in cnt_zero_cols if c in df.columns]

    if exist_cnts:
        df[exist_cnts] = df[exist_cnts].fillna(0).astype("int16")

    # =========================
    # 4. days 계열
    # =========================
    days_cols = [
        'days_employed',
        'days_last_phone_change',
        'own_car_age',
        "pre_weekday_variety",
        'pre_loan_duration_max',
        'cc_over_limit',
        'bu_days_credit_update_max',
        'bu_total_balance_months',
    ]
    exist_days = [c for c in days_cols if c in df.columns]

    for col in exist_days:
        df[col] = df[col].fillna(df[col].median())

    if exist_days:
        df[exist_days] = df[exist_days].astype("int16")

    # =========================
    # 5. numeric 처리
    # =========================
    num_cols = df.select_dtypes(include=["int", "float"]).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
        ql, qh = df[col].quantile(list(clip_q))
        df[col] = df[col].clip(ql, qh)

    # 여기서 다시 float32로 통일
    float_cols = df.select_dtypes(include=["float"]).columns
    df[float_cols] = df[float_cols].astype("float32")

    # =========================
    # 6. categorical 처리
    # =========================
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        df[col] = df[col].astype(str).fillna("MISSING")

        vc = df[col].value_counts(normalize=True)
        rare = vc[vc < min_ratio].index
        df[col] = df[col].replace(rare, "OTHER")

        if df[col].nunique() > max_cardinality:
            top = df[col].value_counts().head(max_cardinality).index
            df[col] = df[col].where(df[col].isin(top), "OTHER")

    return df, df_id

# ----------------
# 9번 실행
# ----------------
df_final, df_id = preprocess_full_minimal(df_final)


