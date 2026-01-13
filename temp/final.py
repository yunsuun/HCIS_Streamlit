#  여기서 부터 bureau 전처리
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

# 여기서부터 inst_ 전처리
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

# 여기서부터 pre_ 전처리
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

# 여기서부터 cc_ 전처리
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