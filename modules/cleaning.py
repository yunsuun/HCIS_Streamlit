# =======================================
# 0. 전처리하기 위한 원본 데이터셋 불러오기
# =======================================
import re
import pandas as pd
import numpy as np
from pathlib import Path

# Module-level cache for bureau_balance (set in clean_data_load)
_BU_BAL_CACHE = None


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
    global _BU_BAL_CACHE
    _BU_BAL_CACHE = bureau_bal

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

# pre에 없어도 merge
def cc_curr_features(cc_d: pd.DataFrame, id_set: set) -> pd.DataFrame:
    import numpy as np
    import pandas as pd

    if cc_d is None or len(cc_d) == 0:
        return pd.DataFrame({"sk_id_curr": list(id_set),
                             "cc_util_mean": np.nan, "cc_util_max": np.nan, "cc_over_limit": 0})

    out = (
        cc_d.groupby("sk_id_curr", as_index=False)
            .agg(
                cc_util_mean=("cc_utilization_mean", "mean"),
                cc_util_max=("cc_utilization_max", "max"),
                cc_over_limit=("cc_cnt_over_limit", "sum"),
            )
    )
    return out

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

# pre 없어도 inst merge
def inst_curr_features(inst_d: pd.DataFrame, id_set: set) -> pd.DataFrame:
    import numpy as np
    import pandas as pd

    if inst_d is None or len(inst_d) == 0:
        return pd.DataFrame({"sk_id_curr": list(id_set),
                             "inst_delay_rate": np.nan, "inst_delay_days_mean": np.nan})

    out = (
        inst_d.groupby("sk_id_curr", as_index=False)
              .agg(
                  inst_delay_rate=("delay_rate_all", "mean"),
                  inst_delay_days_mean=("delay_days_mean_all", "mean"),
              )
    )
    return out

# 여기서부터 pre_ 전처리
# =============================
# 6. pre_app 서브 파생변수 제작
# =============================

# -------------------------------------------
# 6-1. 통합 파생변수 생성 및 전처리 함수
# -------------------------------------------

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

# 여기서부터 pos_ 전처리
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

# pre 없어도 pos merge
def pos_curr_features(pos_d: pd.DataFrame, id_set: set) -> pd.DataFrame:
    import pandas as pd

    if pos_d is None or len(pos_d) == 0:
        return pd.DataFrame({"sk_id_curr": list(id_set), "pos_def_flag": 0})

    out = (
        pos_d.groupby("sk_id_curr", as_index=False)
             .agg(pos_def_flag=("pos_dpd_def_flag", "max"))
    )
    return out

# 여기서부터 cc_ 전처리
# ==============================================
# 4. creditcard 서브 파생변수 제작
# ==============================================

# -------------------------------------------
# 4-1. 통합 파생변수 생성 및 전처리 함수 (최종 변수 생성은 pre_app 내 존재여부에 따라 달라짐)
# -------------------------------------------



def run_pre_block(case_ids, pre_app, pos_d = None, cc_d=None, inst_d=None):
    """
    [FINAL-aligned + FIX]
    - pre_derived_variable()가 (sk_id_curr, sk_id_prev) 단위로 만들어진 데이터를 반환할 수 있음
    - 이를 전처리.py의 STEP3처럼 sk_id_curr 기준으로 집계(pre_curr)해서 1행/고객으로 만든 뒤 반환
    - 이러면 preprocess_features_only()의 merge(pre_features, on="sk_id_curr")에서 행이 절대 늘지 않음
    """


    id_set = set(case_ids)

    # ------------------------------------------------------------
    # 1) pre (prev 단위) 생성
    # ------------------------------------------------------------
    pre_feat = pre_derived_variable(pre_app, id_set).copy()

    # alias (기존 컬럼명 호환)
    rename_map = {
        "pre_amt_credit_mean": "pre_credit_mean",
        "pre_amt_credit_max": "pre_credit_max",
        "pre_amt_credit_min": "pre_credit_min",
        "pre_credit_to_goods_ratio_mean": "pre_credit_to_goods_mean",
        "pre_approval_ratio_mean": "pre_approval_ratio",
        "pre_is_approved_sum": "pre_approved_cnt",
        "pre_is_new_sum": "pre_new_cnt",
        "pre_is_repeater_sum": "pre_repeat_cnt",
    }
    for src, dst in rename_map.items():
        if src in pre_feat.columns and dst not in pre_feat.columns:
            pre_feat[dst] = pre_feat[src]

    # weekday variety
    if "pre_weekday_variety" not in pre_feat.columns and "pre_weekday_appr_process" in pre_feat.columns:
        def _weekday_variety(x):
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return np.nan
            if isinstance(x, (list, tuple, set)):
                return len(set(x))
            s = str(x)
            if s.lower() in ("nan", "none", "missing"):
                return np.nan
            toks = [t for t in re.split(r"[\s,;|/]+", s) if t]
            return len(set(toks)) if toks else np.nan
        pre_feat["pre_weekday_variety"] = pre_feat["pre_weekday_appr_process"].apply(_weekday_variety)

    # ------------------------------------------------------------
    # 2) (선택) CC/INST를 prev 단위로 붙이기  (sk_id_prev 기준)
    #    - 전처리.py는 merge_prev를 만든 다음 curr로 집계함
    # ------------------------------------------------------------
    merge_prev = pre_feat.copy()

    # cc_d / inst_d는 통상 ['sk_id_curr','sk_id_prev', ...] 형태 (너희 cleaning.py 정의상)
    if cc_d is not None and len(cc_d) > 0:
        merge_prev = merge_prev.merge(
            cc_d.drop_duplicates(subset=["sk_id_curr", "sk_id_prev"]),
            on=["sk_id_curr", "sk_id_prev"],
            how="left",
        )
    else:
        # 집계 때 필요할 컬럼이 없으면 NaN으로 맞춰둠
        merge_prev["cc_utilization_mean"] = np.nan
        merge_prev["cc_utilization_max"] = np.nan
        merge_prev["cc_cnt_over_limit"] = np.nan

    if inst_d is not None and len(inst_d) > 0:
        merge_prev = merge_prev.merge(
            inst_d.drop_duplicates(subset=["sk_id_curr", "sk_id_prev"]),
            on=["sk_id_curr", "sk_id_prev"],
            how="left",
        )
    else:
        merge_prev["delay_rate_all"] = np.nan
        merge_prev["delay_days_mean_all"] = np.nan

    # ------------------------------------------------------------
    # 3) weekend ratio (원본 pre_app에서 sk_id_curr 기준으로 계산)
    # ------------------------------------------------------------
    tmp = pre_app[pre_app["sk_id_curr"].isin(id_set)][["sk_id_curr", "weekday_appr_process_start"]].copy()
    if len(tmp) > 0 and "weekday_appr_process_start" in tmp.columns:
        weekend = {"SATURDAY", "SUNDAY"}
        tmp["_is_weekend"] = tmp["weekday_appr_process_start"].isin(weekend).astype("int8")
        wk = tmp.groupby("sk_id_curr").agg(
            _wk_sum=("_is_weekend", "sum"),
            _wk_cnt=("_is_weekend", "count")
        ).reset_index()
        wk["pre_weekend_app_ratio"] = wk["_wk_sum"] / wk["_wk_cnt"]
        merge_prev = merge_prev.merge(wk[["sk_id_curr", "pre_weekend_app_ratio"]], on="sk_id_curr", how="left")
    else:
        merge_prev["pre_weekend_app_ratio"] = np.nan

    # ------------------------------------------------------------
    # 4) ★핵심 FIX: prev → curr 집계 (1행/고객 만들기)
    #    - 전처리.py STEP3와 동일한 아이디어
    # ------------------------------------------------------------
    # 분모: 신청횟수(있으면 사용)
    if "pre_application_count" in merge_prev.columns:
        app_cnt = merge_prev.groupby("sk_id_curr")["pre_application_count"].first()
    else:
        app_cnt = merge_prev.groupby("sk_id_curr").size()  # fallback

    pre_curr = (
        merge_prev
        .groupby("sk_id_curr")
        .agg(
            pre_weekend_app_ratio=("pre_weekend_app_ratio", "mean"),
            pre_weekday_variety=("pre_weekday_appr_process", "nunique") if "pre_weekday_appr_process" in merge_prev.columns else ("pre_weekday_variety", "mean"),

            pre_approved_cnt=("pre_is_approved_sum", "sum") if "pre_is_approved_sum" in merge_prev.columns else ("pre_approved_cnt", "sum"),
            pre_new_cnt=("pre_is_new_sum", "sum") if "pre_is_new_sum" in merge_prev.columns else ("pre_new_cnt", "sum"),
            pre_repeat_cnt=("pre_is_repeater_sum", "sum") if "pre_is_repeater_sum" in merge_prev.columns else ("pre_repeat_cnt", "sum"),

            pre_credit_mean=("pre_amt_credit_mean", "mean") if "pre_amt_credit_mean" in merge_prev.columns else ("pre_credit_mean", "mean"),
            pre_credit_max=("pre_amt_credit_max", "max") if "pre_amt_credit_max" in merge_prev.columns else ("pre_credit_max", "max"),
            pre_credit_min=("pre_amt_credit_min", "min") if "pre_amt_credit_min" in merge_prev.columns else ("pre_credit_min", "min"),
            pre_annuity_mean=("pre_amt_annuity_mean", "mean") if "pre_amt_annuity_mean" in merge_prev.columns else ("pre_annuity_mean", "mean"),
            pre_credit_to_goods_mean=("pre_credit_to_goods_ratio_mean", "mean") if "pre_credit_to_goods_ratio_mean" in merge_prev.columns else ("pre_credit_to_goods_mean", "mean"),
            pre_approval_ratio=("pre_approval_ratio_mean", "mean") if "pre_approval_ratio_mean" in merge_prev.columns else ("pre_approval_ratio", "mean"),
            pre_loan_duration_mean=("pre_loan_duration_mean", "mean") if "pre_loan_duration_mean" in merge_prev.columns else ("pre_loan_duration_mean", "mean"),
            pre_loan_duration_max=("pre_loan_duration_max", "max") if "pre_loan_duration_max" in merge_prev.columns else ("pre_loan_duration_max", "max"),
            pre_days_decision_mean=("pre_days_decision_mean", "mean") if "pre_days_decision_mean" in merge_prev.columns else ("pre_days_decision_mean", "mean"),

            # CC (curr 집계)
            cc_util_mean=("cc_utilization_mean", "mean"),
            cc_util_max=("cc_utilization_max", "max"),
            cc_over_limit=("cc_cnt_over_limit", "sum"),

            # INST (curr 집계)
            inst_delay_rate=("delay_rate_all", "mean"),
            inst_delay_days_mean=("delay_days_mean_all", "mean"),
        )
        .reset_index()
    )

    # count → ratio 변환 (전처리.py와 동일 컨셉)
    for c in ["pre_approved_cnt", "pre_new_cnt", "pre_repeat_cnt"]:
        if c in pre_curr.columns:
            pre_curr[c] = pre_curr[c] / pre_curr["sk_id_curr"].map(app_cnt).astype(float)

    return pre_curr


# ============================================
# 11. bureau 최종 파생변수 제작
# ============================================

# -------------------------------------------
# 11-1. 통합 파생변수 생성 및 전처리 함수
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

# 여기서부터 inst_ 전처리
# ==============================================
# 5. installments 서브 파생변수 제작
# ==============================================

# -------------------------------------------
# 5-1. 통합 파생변수 생성 및 전처리 함수
# -------------------------------------------

def run_bureau_block(id_set, base_df, bureau, bu_bal=None):
    """[FINAL-aligned]
    final.py 기준 bu_derived_variable 정의를 사용.
    - id_set: 대상 sk_id_curr 집합
    - base_df: (호환성 위해 유지) 사용하지 않음
    - bureau: bureau 원본
    - bu_bal: bureau_balance 원본 (None이면 clean_data_load에서 캐시된 값을 사용)
    """
    global _BU_BAL_CACHE
    if bu_bal is None:
        if _BU_BAL_CACHE is None:
            raise ValueError("bu_bal이 필요합니다. clean_data_load()를 먼저 호출하거나 bu_bal을 인자로 넘기세요.")
        bu_bal = _BU_BAL_CACHE
    return bu_derived_variable(bureau, bu_bal, set(id_set))

def preprocess_full_minimal(
    df: pd.DataFrame,
    clip_q: tuple[float, float] = (0.001, 0.999),
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
