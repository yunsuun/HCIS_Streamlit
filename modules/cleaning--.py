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
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = PROJECT_ROOT / "st_data"

    # 이 데이터셋이 새로 고객을 받는 데이터셋 파일 (새로 받는 데이터는 app_train이나 app_test여야 하고, 나머지는 steamlit 안에 존재해야 함.)

    bureau = pd.read_parquet(DATA_DIR / "bureau.parquet")
    bureau_bal = pd.read_parquet(DATA_DIR / "bureau_balance.parquet")
    pre_app = pd.read_parquet(DATA_DIR / "previous_application.parquet")
    inst_payments = pd.read_parquet(DATA_DIR / "installments_payments.parquet")
    pos_cash = pd.read_parquet(DATA_DIR / "POS_CASH_balance.parquet")
    creditcard = pd.read_parquet(DATA_DIR / "credit_card_balance.parquet")

    return bureau, bureau_bal, pre_app, inst_payments, pos_cash, creditcard



# ==================================================
# 1. Home Credit 기업 내 존재하는 신청자 정보에 따라 유형 분류
# ==================================================

# ---------------------------------------------------
# 1-1. 유형 관련 dict 만들기
# ---------------------------------------------------

def setting_train(app_all, 
                 bureau, bureau_bal, 
                 pre_app, inst_payments, 
                 pos_cash, creditcard):
    
    """ 
    함수 설명: Home Credit 기업 내 존재하는 신청자 정보에 따라 유형 분류하기 전 딕셔너리 활용해서 반복문 만들기
    Args:
        - 0단계에서 불러오기한 데이터셋들
    
    Returns:
        - train_dict: train 데이터셋에 존재하는 유형
    """
    # 데이터셋 명과 데이터셋을 동시에 추출하고자 하는 딕셔너리
    df_dict = {"pre_app" : pre_app
            , "bureau" : bureau
            , "creditcard" : creditcard
            , "pos_cash" : pos_cash
            , "inst_payments" : inst_payments}

    # 이 아이디가 성립되는 아이디만 대입하려고 만듬
    app_all_list = app_all["sk_id_curr"].unique().tolist()
    print(f"신청 데이터 셋 전체 ID 수 : {len(app_all_list)}")

    # train 내 sk_id_curr과 일치하게 만드는 함수
    def trans_train_id(df):
        df_set = df[df["sk_id_curr"].isin(app_all_list)]
        return df_set

    # 딕셔너리 반복을 통한 변수 창출 및 개수 파악
    all_dict = {}
    for df_name, df in df_dict.items():
        all_dict[f"{df_name}_all"] = trans_train_id(df)

    # bureau와 bureau_bal 겹치는 신청자를 모아둔 데이터셋
    bureau_bureau_bal = pd.merge(all_dict["bureau_all"], bureau_bal, how='inner', on='sk_id_bureau', validate="one_to_many")

    # 딕셔너리에 있는 bureau_all 꺼내서 작업하기
    bureau_all = all_dict["bureau_all"]

    # bureau랑 bureau_bal이 겹치는 아이디 집합
    bureau_bureau_bal_set = set(bureau_bureau_bal["sk_id_curr"]) 

    # bureau만 있는 데이터셋
    bureau_all_origin = bureau_all[~bureau_all["sk_id_curr"].isin(bureau_bureau_bal_set)]

    # 기존 bureau_all 딕셔너리 제거하기
    all_dict.pop("bureau_all")

    # bureau_train_origin_train으로 두어 bureau만 있는 신청자가 모인 데이터셋을 넣기
    all_dict["bureau_all_origin_all"] = bureau_all_origin

    # bureau_bureau_bal_train으로 두어 bureau와 bureau_bal 겹치는 신청자를 모아둔 데이터셋을 넣기
    all_dict["bureau_bureau_bal_all"] = bureau_bureau_bal

    return all_dict

# --------------------------------------
# 1-2. 신청자 유형 분류하기
# --------------------------------------
class ApplicantTypeClassifier:
    """
    Home Credit 신청자 유형 분류기
    - pre_app, bureau, pos_cash, creditcard, installments 존재 여부로 고객 유형을 분류
    - 총 5가지 정보 → 2^5 = 최대 32개 유형
    """

    def __init__(self, all_dict):
        """
        ApplicantTypeClassifier 객체가 생성되는 순간 자동 실행되는 초기 세팅 함수
        
        self : 객체 자체(데이터 저장 공간)
        train_dict : train 기준으로 필터링된 5개 테이블 딕셔너리
        """
        self.pre_set  = set(all_dict["pre_app_all"]["sk_id_curr"].unique())
        self.bur_ori_set  = set(all_dict["bureau_all_origin_all"]["sk_id_curr"].unique())
        self.bur_bal_set  = set(all_dict["bureau_bureau_bal_all"]["sk_id_curr"].unique())
        self.pos_set  = set(all_dict["pos_cash_all"]["sk_id_curr"].unique())
        self.cc_set   = set(all_dict["creditcard_all"]["sk_id_curr"].unique())
        self.inst_set = set(all_dict["inst_payments_all"]["sk_id_curr"].unique())

        # 유형 이름 정의 (32개)
        self.type_map = self._build_type_map()

    def _exists(self, sk_id, subset):
        """해당 sk_id_curr가 특정 데이터셋에 존재하면 1, 아니면 0"""
        return 1 if sk_id in subset else 0
    

    def _build_type_map(self):
        """
        유형 코드(0~63)를 사람이 읽을 수 있는 이름으로 매핑
        6비트 조합: pre, bureau, bur_bal, pos_cash, creditcard, installments
        ex) 00000 → 'NO_HISTORY'
        """
        mapping = {}
        for code in range(64):
            b = format(code, "06b")  # 6비트 문자열
            pre, bur, bur_bal, pos, cc, inst = b

            name = []
            if pre == "1":     name.append("PRE")
            if bur == "1":     name.append("BUR")
            if bur_bal == "1":  name.append("BUR_BAL")
            if pos == "1":     name.append("POS")
            if cc  == "1":     name.append("CC")
            if inst == "1":    name.append("INST")

            if len(name) == 0:
                mapping[code] = "NO_HISTORY"
            else:
                mapping[code] = "_".join(name)

        return mapping


    def classify(self, sk_id_curr):
        """특정 신청자 1명의 유형 분류"""
        pre  = self._exists(sk_id_curr, self.pre_set)
        bur  = self._exists(sk_id_curr, self.bur_ori_set)
        bur_bal = self._exists(sk_id_curr, self.bur_bal_set)
        pos  = self._exists(sk_id_curr, self.pos_set)
        cc   = self._exists(sk_id_curr, self.cc_set)
        inst = self._exists(sk_id_curr, self.inst_set)

        # bit 조합 → type_code
        # type_code = (pre << 5) + (bur << 4) + (bur_bal << 3) + (pos << 2) + (cc << 1) + inst
        # type_name = self.type_map[type_code]

        return {
            "sk_id_curr": sk_id_curr,
            "pre_app": pre,
            "bureau": bur,
            "bureau_bureau_bal": bur_bal,
            "pos_cash": pos,
            "creditcard": cc,
            "installments": inst,
            # "type_code": type_code,
            # "type_name": type_name
        }


    def classify_all(self, app_all):
        """train 데이터 전체 고객 유형을 분류하여 DataFrame으로 반환"""
        results = []

        for sk in app_all["sk_id_curr"].unique():
            results.append(self.classify(sk))

        classify_result = pd.DataFrame(results)

        return classify_result




import numpy as np
import pandas as pd

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
        df["app_employment_stability_ratio"] = (
            df["app_years_employed"] /
            df["app_age_years"].replace(0, np.nan)
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
    # 7. EXT_SOURCE 핵심 파생변수
    # =====================================================
    ext_cols = ["ext_source_1", "ext_source_2", "ext_source_3"]
    if set(ext_cols).issubset(df.columns):
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

    # =====================================================
    # 9. 최종 반환 컬럼 정리
    # =====================================================
    keep_cols = [
        "sk_id_curr",

        # days
        "days_birth", "days_id_publish", "days_employed", "days_last_phone_change",

        # age / employment
        "app_age_years", "app_years_employed", "app_employment_stability_ratio",

        # amounts
        "amt_credit", "amt_annuity", "amt_goods_price",
        "app_amt_credit_log", "app_amt_annuity_log", "app_amt_goods_price_log",

        # ratios
        "app_annuity_income_ratio", "app_payment_rate",

        # ext source
        "ext_source_1", "ext_source_2", "ext_source_3",
        "app_ext_source_min", "app_ext_source_weighted",

        # documents / social
        "app_n_documents", "flag_document_3",
        "app_def_30_cnt_social_circle_clipped",

        # categorical originals
        "own_car_age", "code_gender", "name_family_status",
        "region_rating_client_w_city", "organization_type",
        "name_income_type", "occupation_type", "name_education_type"
    ]

    keep_cols = [c for c in keep_cols if c in df.columns]

    return df[keep_cols]



# ==============================================
# 3. creditcard 서브 파생변수 제작
# ==============================================

# -------------------------------------------
# 3-1. 통합 파생변수 생성 및 전처리 함수 (최종 변수 생성은 pre_app 내 존재여부에 따라 달라짐)
# -------------------------------------------

def cc_derived_variable(cc, id_set):
    """
    함수 설명 : creditcard 데이터셋이 존재하는 고객에 대해서 서브 파생변수를 만드는 함수 
              (pre_app 없으면 완전 파생변수 생성)

    Args:
        - id_set: 고객 아이디 집합
        - cc: 원본 크레딧카드 데이터 프레임
    
    Returns:
        - creditcard_der: 파생변수만 만든 데이터 프레임 (cc_d)
          (만약 아이디 없다면 아예 안만드는 걸로 하기)
    """
    cc_d = cc[cc["sk_id_curr"].isin(id_set)].copy()

    # ⚠️ Point-in-Time: 대출 신청 전 데이터만 사용
    CUTOFF_MONTH = 0
    cc_d = cc_d[cc_d['months_balance'] < CUTOFF_MONTH].copy()

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


    # 0이 아닌 -1로 표시하여 "정보 없음"과 "사용률 0%"를 구분
    cc_d = cc_d[cc_d['utilization'] >= 0].copy()

    # util 관련 과거 집계
    if len(cc_d) > 0:
        util_agg = cc_d.groupby('sk_id_prev').agg(
            utilization_mean=('utilization', 'mean'),
            utilization_max=('utilization', 'max')
        ).reset_index()

        cc_d = cc_d.merge(util_agg, on='sk_id_prev', how='left')

    else:
        cc_d['utilization_mean'] = np.nan
        cc_d['utilization_max'] = np.nan
    
    cc_d = cc_d[['sk_id_curr', 'sk_id_prev', 'utilization_mean', 'utilization_max']]

    return cc_d

# ==============================================
# 4. installments 서브 파생변수 제작
# ==============================================

# -------------------------------------------
# 4-1. 통합 파생변수 생성 및 전처리 함수
# -------------------------------------------

def inst_derived_variable(inst, id_set):
    """
    함수 설명 : inst 데이터셋이 존재하는 고객에 대해서 서브 파생변수를 만드는 함수
              (pre_app 존재 안하면 최종 파생변수 제작)
    Args:
        - id_set: 고객 번호 집합
        - inst: 원본 크레딧카드 데이터 프레임
    
    Returns:
        - inst_der: 파생변수만 만든 데이터 프레임 (inst_d)
          (inst 아예 없으면 안 만들기)
    """
    inst_d = inst[inst["sk_id_curr"].isin(id_set)].copy()

    # 지연일수 = 실제납부일 - 할부예정일
    inst_d['inst_payment_delay'] = inst_d['days_entry_payment'] - inst_d['days_instalment']

    # 지연 발생
    inst_d['is_delayed'] = (inst_d['inst_payment_delay'] > 0).astype('int8')

    #지연일수 (지연 발생시만 값 유지)
    inst_d['delay_days_value'] = (inst_d['inst_payment_delay'].
                                where(inst_d['inst_payment_delay'] > 0, 
                                        np.nan))

    # 회차별로 중복 납부 처리 (같은 회차에 여러 번 납부한 경우 통합)
    inst_prev = (inst_d.
            groupby(['sk_id_curr', 'sk_id_prev', 'num_instalment_number']).
            agg({'is_delayed': 'max', 'delay_days_value': 'max'}).
            reset_index())
    
    # ===== 전체 기간 =====
    inst_prev = inst_prev.groupby(['sk_id_curr', 'sk_id_prev']).agg(
        # 횟수
        delay_cnt_all=('is_delayed', 'sum'),
        total_cnt_all=('is_delayed', 'count'),
        # 지연일수
        delay_days_mean_all=('delay_days_value', 'mean')
    ).reset_index()

    # 비율 계산
    inst_prev['delay_rate_all'] = inst_prev['delay_cnt_all'] / inst_prev['total_cnt_all']

    inst_d = inst_prev[['sk_id_curr','sk_id_prev', 'delay_rate_all', 'delay_days_mean_all']]
    
    return inst_d




# =============================
# 5. 아이디 포함 딕셔너리 제작
# =============================

def build_id_sets(app_df, pre_app, cc_d, inst_d):
    """
    함수 설명: 각 아이디마다 테이블이 속해있는 지 여부 파악하기 위함

    Args:
        - app_df: 고객정보가 들어가있는 데이터셋
        - pre_app: pre_app 데이터셋
        - cc_d: cc에 서브 파생변수 제작된 데이터셋
        - inst_d: inst에 서브 파생변수 제작된 데이터셋
    
    Returns:
        - id_sets: 각 데이터셋이 어디에 포함됐는 지 알 수 있는 딕셔너리
    """
    base_ids = set(app_df["sk_id_curr"].unique())

    pre_ids  = set(pre_app["sk_id_curr"].unique())
    cc_ids   = set(cc_d["sk_id_curr"].unique())
    inst_ids = set(inst_d["sk_id_curr"].unique())

    id_sets = {
        "base": base_ids,
        "pre": pre_ids,
        "cc": cc_ids,
        "inst": inst_ids,
    }
    return id_sets



# =====================
# 6. 각 케이스 정립된 딕셔너리 제작
# =====================

def split_case_ids(id_sets):
    """
    함수 설명: 각 아이디 존재하는 데이터셋에 따라 케이스 만든 것

    Args:
        - id_sets: 5번에서 만들었던 아이디 딕셔너리
    
    Returns:
        - cases: 각 케이스를 집어넣은 딕셔너리
    """
    base = id_sets["base"]
    pre  = id_sets["pre"]
    cc   = id_sets["cc"]
    inst = id_sets["inst"]

    return {
        # 1. pre + cc + inst
        "case_1": base & pre & cc & inst,

        # 2. pre + cc
        "case_2": (base & pre & cc) - inst,

        # 3. pre + inst
        "case_3": (base & pre & inst) - cc,

        # 4. cc + inst
        "case_4": (base & cc & inst) - pre,

        # 5. pre only
        "case_5": (base & pre) - cc - inst,

        # 6. cc only
        "case_6": (base & cc) - pre - inst,

        # 7. inst only
        "case_7": (base & inst) - pre - cc,

        # 8. none
        "case_8": base - pre - cc - inst,
    }


# ============================================
# 7. pre_app 통합 파생변수
# ============================================

# -------------------------------------------
# 7-1. 통합 파생변수 생성 및 전처리 함수
# -------------------------------------------

def pre_derived_variable(id, df, pre_app):
    """
    함수 설명 : pre_app 존재한다면 파생변수를 제작하기 + cc, inst 최종 파생변수 제작 과정

    Args:
        - df: df_cleaning_final (파생변수 삽입해야 할 데이터셋)
        - customer_df: customer_app (고객 데이터셋 중 pre__app 존재여부 확인용)
        - pre_app: sk_id_curr이 동일한 데이터셋

    Returns:
        - df: df_cleaning_final (pre_app 파생변수 삽입이 완료된 데이터셋)
    """

    # ---------------------------------
    # pre_app에 해당 ID가 존재하면 pre_app 데이터셋 사용 없음 함수 종료
    # -------------------------------- 
    if pre_app.loc[pre_app["sk_id_curr"] == id].empty:
        return None
    
    else:
        pre_app = pre_app[pre_app['sk_id_curr'] == id].copy()

        merge_prev = df.copy()

    # ----------------------------------------
    # pre_app 기본 파생변수
    # ----------------------------------------

    # 시간순서로 정렬
    merge_prev = merge_prev.sort_values(['sk_id_curr', 'sk_id_prev', 'days_decision'], ascending=[True, True, False])

    merge_prev['credit_to_goods_ratio'] = np.where(
        merge_prev['amt_goods_price'] > 0,
        merge_prev['amt_credit'] / merge_prev['amt_goods_price'],
        np.nan
    )  # 승인액 대비 상품가격 비율

    # 신규: 승인율 (신청 대비 승인 비율)
    merge_prev['approval_ratio'] = np.where(
        merge_prev['amt_application'] > 0,
        merge_prev['amt_credit'] / merge_prev['amt_application'],
        np.nan
    )  # 승인액/신청액 (1.0 초과 = 신청액보다 더 많이 승인)

    # 2. 시간 관련 파생변수
    merge_prev['loan_duration'] = merge_prev['days_last_due'] - merge_prev['days_first_due']

    # 3. 계약 상태 플래그
    merge_prev['is_approved'] = (merge_prev['name_contract_status'] == 'Approved').astype('int8')
    
    # 9. 고객 타입 플래그
    merge_prev['is_new'] = (merge_prev['name_client_type'] == 'New').astype('int8')

    # ----------------------------------------
    # 대출 상환 별 집계 (sk_id_prev)
    # ----------------------------------------

    pre_prev = merge_prev.groupby(['sk_id_curr', 'sk_id_prev']).agg(
        # ===== 금액 관련 =====

        # 평균값
        pre_amt_annuity_mean=('amt_annuity', 'mean'),
        pre_amt_credit_mean=('amt_credit', 'mean'),

        # 변동성 (std, range)
        pre_amt_credit_max=('amt_credit', 'max'),

        # 금액 차이/비율
        pre_credit_to_goods_ratio_mean=('credit_to_goods_ratio', 'mean'),
        pre_approval_ratio_mean=('approval_ratio', 'mean'),
        
        # ===== 시간 관련 =====
        pre_days_decision_mean=('days_decision', 'mean'),

        # ===== 상태 플래그 =====
        pre_is_approved_sum=('is_approved', 'sum'),

        # ===== 보험/고객 타입 =====
        pre_is_new_sum=('is_new', 'sum'),

         # ===== 시간 관련 =====
        # 파생 시간 변수
        pre_loan_duration_max=('loan_duration', 'max'),

        # 변동성 (std, range)
        pre_amt_credit_min=('amt_credit', 'min'),

        # ===== 신청 횟수 =====
        pre_application_count=('sk_id_prev', 'count'),  # 이 대출에서 신청 시도 횟수

        # ===== inst (어차피 값 같음) =====
        delay_rate_all=('delay_rate_all', 'mean'),
        delay_days_mean_all=('delay_days_mean_all', 'mean'),

        # ===== creditcard (어차피 값 같음) =====
        utilization_mean=('utilization_mean', 'mean'),
        utilization_max=('utilization_max', 'max')

    ).reset_index()
    
    # -----------------------------------
    # prev → curr 집계
    # -----------------------------------

    # 현재 고객 카운트
    app_cnt = (
        pre_prev
        .groupby('sk_id_curr')['pre_application_count']
        .sum()
        )

    pre_curr = pre_prev.groupby('sk_id_curr').agg(

            # ===== 금액 / 기간 =====
            pre_credit_mean=('pre_amt_credit_mean', 'mean'),

            pre_annuity_mean=('pre_amt_annuity_mean', 'mean'),

            pre_credit_max=('pre_amt_credit_max', 'max'),

            pre_credit_to_goods_mean=('pre_credit_to_goods_ratio_mean', 'mean'),

            pre_approval_ratio=('pre_approval_ratio_mean', 'mean'),

            pre_days_decision_mean=('pre_days_decision_mean', 'mean'),

            pre_credit_min=('pre_amt_credit_min', 'min'),

            # ===== 계약 구조 =====

            pre_approved_cnt=('pre_is_approved_sum', 'sum'),

            pre_new_cnt=('pre_is_new_sum', 'sum'),

            pre_loan_duration_max=('pre_loan_duration_max', 'max'),

            # ===== 신청 횟수 =====

            pre_application_count=('sk_id_prev', 'count'),

            # ===== INSTALLMENTS =====
            inst_delay_rate=('delay_rate_all', 'mean'),
            inst_delay_days_mean=('delay_days_mean_all', 'mean'),

            # ===== CREDIT CARD =====
            cc_util_mean=('utilization_mean', 'mean'),
            cc_util_max=('utilization_max', 'max')
        )
    
    # -----------------------------------
    # cnt → ratio 집계
    # -----------------------------------

    # count → ratio 변환
    ratio_cols = [
        'pre_approved_cnt','pre_new_cnt'
    ]

    pre_curr[ratio_cols] = pre_curr[ratio_cols].div(app_cnt, axis=0)

    pre_curr = pre_curr.reset_index()

    pre_cols = ['sk_id_curr','pre_annuity_mean', 'pre_credit_max', 
                'pre_days_decision_mean', 'pre_credit_to_goods_mean', 
                'pre_approval_ratio', 'pre_credit_mean', 
                'pre_approved_cnt', 'pre_new_cnt', 
                'pre_loan_duration_max', 'pre_credit_min', 
                'inst_delay_rate', 'inst_delay_days_mean',
                'cc_util_mean', 'cc_util_max']

    df = pre_curr[pre_cols]

    return df

# =======================
# 8. pre_app 존재 여부에 따른 pre_derived_variable 함수 호출 함수
# =======================

def run_pre_block(case_ids, pre_app, cc_d, inst_d):
    """
    함수 설명: pre_app 존재하면 자동으로 pre_derived_variable 함수 진행하는 함수
    Args:
        - case_ids: 존재하는 케이스에 대한 아이디 집합
        - pre_app: sk_id_curr이 동일한 데이터셋
        - cc_d: cc 서브 파생변수 생성된 데이터셋
        - inst_d: inst 서브 파생변수 생성된 데이터셋
    
    Returns:
        - 7번에서 제작된 데이터셋
    """
    outs = []

    for sk_id in case_ids:
        pre_curr = pre_app[pre_app["sk_id_curr"] == sk_id].copy()
        if pre_curr.empty:
            continue

        # cc / inst 서브 파생변수 붙이기
        pre_curr = pre_curr.merge(
            cc_d,
            on=["sk_id_curr", "sk_id_prev"],
            how="left"
        )

        pre_curr = pre_curr.merge(
            inst_d,
            on=["sk_id_curr", "sk_id_prev"],
            how="left"
        )

        # pre_derived_variable 호출
        out = pre_derived_variable(
            sk_id,
            pre_curr,
            pre_app
        )

        if out is not None:
            outs.append(out)

    return pd.concat(outs, axis=0) if outs else pd.DataFrame()



# =======================================
# 9. pre_app 아이디 없을 경우의 CC 최종 파생변수
# =======================================

def cc_only_features(cc_d, target_ids):
    """
    함수 설명: pre_app 아이디 없을 경우의 CC만으로 최종 파생변수 제작

    Args:
        - cc_d: cc 서브 전처리된 데이터셋
        - target_ids: pre_app 아이디 없을 경우에 cc만 존재하는 경우에 대한 아이디 딕셔너리
    
    Returns:
        - 최종 전처리된 CC 데이터셋
    """
    return (
        cc_d[cc_d["sk_id_curr"].isin(target_ids)]
        .groupby("sk_id_curr")
        .agg(
            cc_util_mean=("utilization_mean", "mean"),
            cc_util_max=("utilization_max", "max")
        )
        .reset_index()
    )

# ======================================
# 10. pre_app 아이디 없을 경우의 Inst 최종 파생변수
# ======================================

def inst_only_features(inst_d, target_ids):
    """
    함수 설명: pre_app 아이디 없을 경우의 inst만으로 최종 파생변수 제작

    Args:
        - inst_d: inst 서브 전처리된 데이터셋
        - target_ids: pre_app 아이디 없을 경우에 inst만 존재하는 경우에 대한 아이디 딕셔너리
    
    Returns:
        - 최종 전처리된 inst 데이터셋
    """
    return (
        inst_d[inst_d["sk_id_curr"].isin(target_ids)]
        .groupby("sk_id_curr")
        .agg(
            inst_delay_rate=("delay_rate_all", "mean"),
            inst_delay_days_mean=("delay_days_mean_all", "mean")
        )
        .reset_index()
    )


# ============================================
# 11. bureau 최종 파생변수 제작
# ============================================

# -------------------------------------------
# 11-1. 통합 파생변수 생성 및 전처리 함수
# -------------------------------------------

def bu_derived_variable(id, df, bureau):
    """
    함수 설명: bureau가 존재하는 아이디라면 파생변수를 만드는 함수
    Args:
        - id: 데이터셋 내 아이디 집합
        - df: 신청 고객 데이터셋
        - bureau: 원본 bureau 데이터셋
    Returns:
        - df: 파생변수 삽입된 bureau 최종 파생변수
        
    """

    # ---------------------------------
    # pre_app에 해당 ID가 존재하면 pre_app 데이터셋 사용 없음 함수 종료
    # -------------------------------- 
    if bureau.loc[bureau["sk_id_curr"] == id].empty:
        return None
    
    else:
        bureau = bureau[bureau["sk_id_curr"] == id]

    # bureau sk_id_curr 필터링
    bureau_clean = bureau.copy()

    credit_sum = bureau_clean['amt_credit_sum']
    debt = bureau_clean['amt_credit_sum_debt']

    debt_for_ratio = debt.copy()

    # 음수를 0으로 (하한)
    debt_for_ratio = debt_for_ratio.clip(lower=0)

    # credit_sum보다 크면 cap (상한)
    debt_for_ratio = debt_for_ratio.mask(
        (debt_for_ratio > credit_sum) & (credit_sum > 0),
        credit_sum
    )

    # 결과 저장: 이후 집계에서 이 컬럼을 사용
    bureau_clean['amt_credit_sum_debt_for_ratio'] = debt_for_ratio

    # 부채가 amt_credit_sum보다 큰 경우 (over-limit)
    bureau_clean['over_limit_debt_flag'] = (
    debt > credit_sum
    ).astype(int)

    # sk_id_curr 단위 집계 (cur_agg)
    cur_agg = bureau_clean.groupby('sk_id_curr').agg(
    # 대출 상태 개수
    n_bureau_loans=('sk_id_bureau', 'count'),

    bu_cnt_active=('credit_active', lambda x: (x == 'Active').sum()),

    # 부채 이상치 플래그: any_over_limit_debt만 사용
    bu_any_over_limit_debt=('over_limit_debt_flag', 'any'),

    # 금액 관련: total_debt_for_ratio만 유지
    bu_total_debt_for_ratio=('amt_credit_sum_debt_for_ratio', 'sum'),

    # 기간 정보: update_max만
    bu_days_credit_update_max=('days_credit_update', 'max')
    ).reset_index()

    cur_agg = cur_agg.assign(
    # active loan 비율
    bu_ratio_active_loans=lambda df: np.where(df['n_bureau_loans'] > 0, 
                                           df['bu_cnt_active']/df['n_bureau_loans'], 
                                           np.nan))
    
    bu_cols = ['sk_id_curr', 'bu_any_over_limit_debt', 'bu_total_debt_for_ratio',
               'bu_ratio_active_loans', 'bu_days_credit_update_max',
               'bu_cnt_active']
    df = cur_agg[bu_cols]
    
    return df

# =========================
# 12. 11번 코드 실행 함수
# =========================
def run_bureau_block(case_ids, base_df, bureau):
    """
    함수 설명: bureau 실행하는 코드
    Args:
        - case_ids: 각 케이스 집단.
        - df: 고객 신청 아이디들
        - bureau: 원본 bureau 데이터셋
    Returns:
        - df: 11번 데이터셋 반환
        
    """
    outs = []

    for sk_id in case_ids:
        df_tmp = base_df[base_df["sk_id_curr"] == sk_id].copy()
        out = bu_derived_variable(sk_id, df_tmp, bureau)
        if out is not None:
            outs.append(out)

    return pd.concat(outs, axis=0) if outs else pd.DataFrame()

# -------------------------------------------
# 14-1. 통합 전처리 함수
# -------------------------------------------

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

