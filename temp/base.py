# 여기서 부터 bureau 전처리
# ==========================
# 1. app_train에 있는 sk_id_curr만 필터링
# ==========================

app_train_list = app_train["sk_id_curr"].unique().tolist()

# bureau sk_id_curr 필터링
bureau_filt = bureau[bureau["sk_id_curr"].isin(app_train_list)].copy()

# ==========================
# 1-1. 통화 필터링: 가장 주된 통화(예: currency 1)만 사용
# ==========================
# credit_currency에서 가장 많이 등장하는 값을 main 통화로 사용
main_currency = bureau_filt['credit_currency'].value_counts().idxmax()

print("▶ 사용 통화:", main_currency)

# main 통화가 아닌 행은 모두 제거
bureau_filt = bureau_filt[bureau_filt['credit_currency'] == main_currency].copy()


# ==========================
# 2. bureau에 있는 sk_id_bureau만 필터링
# ==========================

# bureau_filt에 존재하는 sk_id_bureau만 사용
bureau_list = bureau_filt['sk_id_bureau'].unique().tolist()

# bureau_bal sk_id_bureau 필터링
bureau_bal_filt = bureau_bal[bureau_bal['sk_id_bureau'].isin(bureau_list)].copy()

# ==========================
# 3. bureau_balance: 같은 달 중복 row 정리
#    - 같은 sk_id_bureau + months_balance에 여러 row가 있을 때,
#      가장 "나쁜" status(연체 레벨이 높은 것)를 선택
# ==========================
status_map = {'X': 0, 'C': 0, '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5}

# status를 점수로 매핑 (연체 심각도)
bureau_bal_filt['status_score'] = bureau_bal_filt['status'].map(status_map).astype('int8')

# 같은 (sk_id_bureau, months_balance) 안에서 status_score가 가장 큰(가장 나쁜) row만 남기기
idx_worst = (
    bureau_bal_filt
    .groupby(['sk_id_bureau', 'months_balance'])['status_score']
    .idxmax()
)

bureau_bal_filt = bureau_bal_filt.loc[idx_worst].copy()
bureau_bal_filt.drop(columns=['status_score'], inplace=True)

# ==========================
# 4. bureau_balance: 정렬 (과거 → 최근)
# ==========================
# bureau_balance는 months_balance가 -n, ..., -1, 0일 때
# 값이 클수록 최근이므로 과거→최근 순으로 정렬
bureau_bal_filt = (
    bureau_bal_filt
    .sort_values(['sk_id_bureau', 'months_balance'], ascending=[True, True])
    .reset_index(drop=True)
)

# ==========================
# 5. bureau_balance: C 이후 숫자 STATUS 삭제 (벡터화 버전)
# ==========================
# 종료된 대출(sk_id_bureau)이 C(Closed) 후에 다시 숫자 연체(0~5)가 나오는 것은 비논리적이라 제거

# 각 대출에서 C가 처음 등장한 months_balance 찾기
is_c = (bureau_bal_filt['status'] == 'C')
first_c_month = (
    bureau_bal_filt[is_c]
    .groupby('sk_id_bureau')['months_balance']
    .min()  # 가장 과거의 C
)

# 원본에 merge해서 "이 대출은 C가 언제 처음 나왔는지" 정보 추가
bureau_bal_filt = bureau_bal_filt.merge(
    first_c_month.rename('first_c_month'),
    on='sk_id_bureau',
    how='left'
)

# C가 존재하고(first_c_month notna),
# 그 이후(months_balance > first_c_month)에
# status가 숫자(0~5)이면 삭제
num_status = set(['0', '1', '2', '3', '4', '5'])

drop_mask = (
    bureau_bal_filt['first_c_month'].notna() &
    (bureau_bal_filt['months_balance'] > bureau_bal_filt['first_c_month']) &
    bureau_bal_filt['status'].isin(num_status)
)

bureau_bal_filt = bureau_bal_filt[~drop_mask].copy()

# 더 이상 필요 없는 컬럼 제거
bureau_bal_filt.drop(columns=['first_c_month'], inplace=True)
import pandas as pd
import numpy as np

# ==========================
# safe_ratio: 분모 0/NaN 방지용 공통 함수
# ==========================
def safe_ratio(num, denom):
    """분모가 0 또는 NaN인 경우 NaN을 반환하는 안전한 비율 계산 함수"""
    return np.where(denom > 0, num / denom, np.nan)


# ==========================
# 6. bureau: 부채 관련 이상치/보정
# ==========================
bureau_clean = bureau_filt.copy()

debt       = bureau_clean['amt_credit_sum_debt']      # 부채 원본
credit_sum = bureau_clean['amt_credit_sum']           # "기준 크레딧(원금/한도 비슷한 개념)"
ctype      = bureau_clean['credit_type']

# Active인데 부채 0인 케이스
bureau_clean['active_zero_debt_flag'] = (
    (bureau_clean['credit_active'] == 'Active') &
    (debt == 0)
).astype(int)

# ⚠ 부채가 음수인 케이스 flag 제거 (요청사항 반영)
# bureau_clean['negative_debt_flag'] = (debt < 0).astype(int)

# 부채가 amt_credit_sum보다 큰 경우 (over-limit)
bureau_clean['over_limit_debt_flag'] = (
    debt > credit_sum
).astype(int)

# 분모로 쓰기 애매한 크레딧(0 또는 음수) 플래그
bureau_clean['zero_or_negative_credit_sum_flag'] = (credit_sum <= 0).astype(int)


# ==========================
# 7. ratio/합계 계산용 "보정된 부채" 만들기
#  - 이후 원본 amt_credit_sum_debt는 삭제
# ==========================
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

# 원본 부채 컬럼은 더 이상 사용하지 않으므로 삭제
if 'amt_credit_sum_debt' in bureau_clean.columns:
    bureau_clean.drop(columns=['amt_credit_sum_debt'], inplace=True)


# --------------------------------
# 8. 너무 오래전에 끝난 폐쇄 계정 제외 (8년 ≈ 3000일)
# --------------------------------
bureau_clean['very_old_closed_flag'] = (
    (bureau_clean['credit_active'] == 'Closed') &
    (bureau_clean['days_enddate_fact'] < -3000)
)

bureau_for_agg = bureau_clean[~bureau_clean['very_old_closed_flag']].copy()


# ==========================
# 9. bureau_balance: sk_id_bureau 단위 집계
# ==========================

# 9-1) 최대 연체 레벨 (X/C는 0, 숫자만 등급으로 처리)
def get_max_late_level(s):
    s_obj = s.astype(object)
    numeric = pd.to_numeric(
        s_obj.replace({'X': '0', 'C': '0'}),
        errors='coerce'
    )
    return numeric.max()

# 9-2) 가장 최근 status (months_balance가 가장 큰 row)
# → 최근 status 기반 feature는 사용하지 않으므로 last_status는 계산하지 않아도 무방하지만,
#   필요하면 분석용으로 남겨둘 수 있음. 여기서는 완전히 제거.
# last_status_df = ...

# 9-3) 기본 집계
bureau_bal_agg = (
    bureau_bal_filt
    .groupby('sk_id_bureau')
    .agg(
        cnt_months=('months_balance', 'count'),
        cnt_0=('status', lambda x: (x == '0').sum()),
        cnt_1=('status', lambda x: (x == '1').sum()),
        cnt_2=('status', lambda x: (x == '2').sum()),
        cnt_3=('status', lambda x: (x == '3').sum()),
        cnt_4=('status', lambda x: (x == '4').sum()),
        cnt_5=('status', lambda x: (x == '5').sum()),
        cnt_c=('status', lambda x: (x == 'C').sum()),
        cnt_x=('status', lambda x: (x == 'X').sum()),
        has_month0=('months_balance', lambda x: (x == 0).any()),
    )
    .reset_index()
)

# 9-4) 마지막 연체가 발생한 month (loan 단위)
late_mask = bureau_bal_filt['status'].isin(['1', '2', '3', '4', '5'])

last_late_month_df = (
    bureau_bal_filt[late_mask]
    .groupby('sk_id_bureau')['months_balance']
    .max()
    .reset_index(name='last_late_month')
)

# 9.5) 최근 6/12개월 연체 비율 (loan 단위)
tmp = bureau_bal_filt.copy()
tmp['is_late'] = tmp['status'].isin(['1', '2', '3', '4', '5']).astype(int)

tmp['recent_6m'] = (tmp['months_balance'] >= -6).astype(int)
tmp['recent_12m'] = (tmp['months_balance'] >= -12).astype(int)

tmp['late_recent_6m'] = tmp['is_late'] * tmp['recent_6m']
tmp['late_recent_12m'] = tmp['is_late'] * tmp['recent_12m']

recent_bal_agg = (
    tmp.groupby('sk_id_bureau')
    .agg(
        recent_6m_months=('recent_6m', 'sum'),
        recent_6m_late=('late_recent_6m', 'sum'),
        recent_12m_months=('recent_12m', 'sum'),
        recent_12m_late=('late_recent_12m', 'sum'),
    )
    .reset_index()
)

recent_bal_agg['overdue_ratio_6m'] = safe_ratio(
    recent_bal_agg['recent_6m_late'], recent_bal_agg['recent_6m_months']
)
recent_bal_agg['overdue_ratio_12m'] = safe_ratio(
    recent_bal_agg['recent_12m_late'], recent_bal_agg['recent_12m_months']
)

bureau_bal_agg = bureau_bal_agg.merge(
    recent_bal_agg[['sk_id_bureau', 'overdue_ratio_6m', 'overdue_ratio_12m']],
    on='sk_id_bureau',
    how='left'
)

bureau_bal_agg['has_balance_flag'] = 1

# 짧은 히스토리 플래그: 3개월짜리만
bureau_bal_agg['short_history_3m_flag'] = (bureau_bal_agg['cnt_months'] < 3).astype(int)
# short_history_6m_flag는 제거

# 연체 횟수 및 비율
late_cols = ['cnt_1', 'cnt_2', 'cnt_3', 'cnt_4', 'cnt_5']
bureau_bal_agg['cnt_late'] = bureau_bal_agg[late_cols].sum(axis=1)
bureau_bal_agg['overdue_ratio'] = bureau_bal_agg['cnt_late'] / bureau_bal_agg['cnt_months']
bureau_bal_agg['x_ratio'] = bureau_bal_agg['cnt_x'] / bureau_bal_agg['cnt_months']

bureau_bal_agg['overdue_ratio'] = bureau_bal_agg['overdue_ratio'].fillna(0)
bureau_bal_agg['x_ratio'] = bureau_bal_agg['x_ratio'].fillna(0)

# max_late_level / last_late_month 붙이기
max_late_level_df = (
    bureau_bal_filt
    .groupby('sk_id_bureau')['status']
    .apply(get_max_late_level)
    .reset_index(name='max_late_level')
)

bureau_bal_agg = (
    bureau_bal_agg
    .merge(max_late_level_df, on='sk_id_bureau', how='left')
    .merge(last_late_month_df, on='sk_id_bureau', how='left')
)

# 연체 경험 플래그: has_any_late 제거, severe → heavy로 변경
bureau_bal_agg['has_heavy_late'] = (bureau_bal_agg['max_late_level'] >= 3).astype(int)
# bureau_bal_agg['has_any_late'] = (bureau_bal_agg['cnt_late'] > 0).astype(int)  # 제거


# ==========================
# 10. bureau + bureau_balance 결합 (bureau_enriched)
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
# 11. sk_id_curr 단위 집계 (cur_agg)
# ==========================
cur_agg = bureau_enriched.groupby('sk_id_curr').agg(

    # 대출 상태 개수
    n_bureau_loans=('sk_id_bureau', 'count'),
    cnt_active=('credit_active', lambda x: (x == 'Active').sum()),
    cnt_closed=('credit_active', lambda x: (x == 'Closed').sum()),

    # 부정적인 대출 기록: 개수만 (flag 제외)
    n_bad_debt=('credit_active', lambda x: (x == 'Bad debt').sum()),
    n_sold=('credit_active', lambda x: (x == 'Sold').sum()),
    
    # 종료일 차이: 평균만 = 평균 신용 이용 기간
    enddate_diff_avg=('enddate_diff', 'mean'),

    # 금액 관련: total_debt_for_ratio만 유지
    total_debt_for_ratio=('amt_credit_sum_debt_for_ratio', 'sum'),

    # balance 이력 길이 / 존재 여부
    total_balance_months=('cnt_months', 'sum'),
    n_balance_loans=('has_balance_flag', 'sum'),
    any_short_history_3m=('short_history_3m_flag', 'any'),
    any_has_month0=('has_month0', 'any'),

    # 부채 이상치 플래그: any_over_limit_debt만 사용
    any_over_limit_debt=('over_limit_debt_flag', 'any'),

    # 기간 정보: update_max만
    days_credit_update_max=('days_credit_update', 'max'),
    
    # balance 기반 연체 정보: max_overdue_ratio + has_balance_any
    max_overdue_ratio=('overdue_ratio', 'max'),
    has_balance_any=('has_balance_flag', lambda x: (x == 1).any()),


    # balance 정보 부실률: mean만
    avg_x_ratio=('x_ratio', 'mean'),

    # 연체일수/연장: avg_credit_day_overdue만
    avg_credit_day_overdue=('credit_day_overdue', 'mean'),

    # 심각한 연체 경험자
    max_late_level=('max_late_level', 'max'),
    any_late_level_3plus=('max_late_level', lambda x: (x >= 3).any()),

    # 연체 경험 대출 수
    n_late_loans=('cnt_late', 'sum'),
    n_heavy_late_loans=('has_heavy_late', 'sum'),

    # 마지막 연체 시점
    last_late_month_overall=('last_late_month', 'max'),
    
    # 최근 6/12개월 연체 비율: 평균 + 최대 둘 다
    recent_overdue_ratio_6m_mean=('overdue_ratio_6m', 'mean'),
    recent_overdue_ratio_6m_max=('overdue_ratio_6m', 'max'),
    recent_overdue_ratio_12m_mean=('overdue_ratio_12m', 'mean'),
    recent_overdue_ratio_12m_max=('overdue_ratio_12m', 'max'),
).reset_index()


# ==========================
# 12. 비율/파생변수 추가 (필요한 것만 유지)
# ==========================
cur_agg = cur_agg.assign(
    # active loan 비율
    ratio_active_loans=lambda df: safe_ratio(df['cnt_active'], df['n_bureau_loans']),

    # balance 있는 대출 비율
    balance_ratio=lambda df: safe_ratio(df['n_balance_loans'], df['n_bureau_loans']),
    
    # bureau 기록 존재 여부
    has_bureau_flag=lambda df: (df['n_bureau_loans'] > 0).astype(int),

    # bad_debt / sold 비율
    ratio_bad_debt_loans=lambda df: safe_ratio(df['n_bad_debt'], df['n_bureau_loans']),
    ratio_sold_loans=lambda df: safe_ratio(df['n_sold'], df['n_bureau_loans']),

    # 연체 대출 비율
    ratio_late_loans=lambda df: safe_ratio(df['n_late_loans'], df['n_bureau_loans']),

    # 마지막 연체 이후 경과 개월 수
    months_since_last_late=lambda df: np.where(
        df['last_late_month_overall'].notna(),
        -df['last_late_month_overall'],
        np.nan
    ),

    # 최근 6개월 최대 연체 비율 - 과거 전체 최대 연체비율
    overdue_ratio_gap_6m=lambda df: df['recent_overdue_ratio_6m_max'] - df['max_overdue_ratio'],

    # 최근 12개월 최대 연체 비율 - 과거 전체 최대 연체비율
    overdue_ratio_gap_12m=lambda df: df['recent_overdue_ratio_12m_max'] - df['max_overdue_ratio'],

    # active loan 중 balance가 있는 비율
    active_balance_ratio=lambda df: safe_ratio(df['n_balance_loans'], df['cnt_active']),
)



# 여기서부터 inst_ 전처리
def installments_payments_preprocessed(df: pd.DataFrame):
    """
    installments_payments 테이블 전처리 및 파생변수 생성
    sk_id_curr, sk_id_prev별 최종 집계 (대출별 상환 행태 요약)

    주요 개선사항:
    1. 기간별 분석: 전체 / 6개월 / 3개월
    2. Trend 변수: 최근 vs 전체 비교로 행태 변화 포착
    3. Pattern 변수: 연속 지연, 첫 발생 시점, 집중도
    4. 할부조건 변경 전후 비교
    """

    print("=" * 60)
    print("Installments Payments 전처리 시작")
    print("=" * 60)

    # ==================== 기본 파생변수 생성 ====================
    print("\n▶ 기본 파생변수 생성 중...")

    # 1. 지연일수 = 실제납부일 - 할부예정일
    df['inst_payment_delay'] = df['days_entry_payment'] - df['days_instalment']

    # 2. 과소납부액 = 할부예정액 - 실제납부액
    df['inst_payment_diff'] = df['amt_instalment'] - df['amt_payment']

    # 3. 기간 플래그
    df['is_last3m'] = (df['days_instalment'] >= -90).astype('int8')   # 최근 3개월
    df['is_last6m'] = (df['days_instalment'] >= -180).astype('int8')  # 최근 6개월

    # 4. 상태 플래그
    df['is_delayed'] = (df['inst_payment_delay'] > 0).astype('int8')      # 지연 발생
    df['is_ontime'] = (df['inst_payment_delay'] == 0).astype('int8')      # 정시 납부
    df['is_early'] = (df['inst_payment_delay'] < 0).astype('int8')        # 조기 납부
    df['is_underpay'] = (df['inst_payment_diff'] > 0).astype('int8')      # 과소 납부

    # 5. 지연일수 (지연 발생시만 값 유지)
    df['delay_days_value'] = df['inst_payment_delay'].where(df['inst_payment_delay'] > 0, np.nan)

    # 6. 과소납부 비율
    df['underpay_ratio'] = np.where(
        (df['amt_instalment'] > 0) & (df['inst_payment_diff'] > 0),
        df['inst_payment_diff'] / df['amt_instalment'],
        np.nan
    )

    print("✓ 기본 파생변수 생성 완료")


    # ==================== 회차별 1차 집계 ====================
    print("\n▶ 회차별 1차 집계 중...")

    # 회차별로 중복 납부 처리 (같은 회차에 여러 번 납부한 경우 통합)
    agg_dict = {
        'amt_instalment': 'max',           # 예정액
        'amt_payment': 'sum',              # 실제 납부액 합계
        'is_delayed': 'max',               # 한 번이라도 지연
        'is_ontime': 'max',                # 한 번이라도 정시
        'is_early': 'max',                 # 한 번이라도 조기
        'is_underpay': 'max',              # 한 번이라도 과소납부
        'delay_days_value': 'max',         # 최대 지연일수
        'underpay_ratio': 'max',           # 최대 과소납부 비율
        'is_last3m': 'max',                # 최근 3개월 여부
        'is_last6m': 'max',                # 최근 6개월 여부
        'days_instalment': 'first',        # 할부 예정일 (패턴 분석용)
        'num_instalment_version': 'first'  # 할부 버전
    }

    df_inst = df.groupby(['sk_id_curr', 'sk_id_prev', 'num_instalment_number']).agg(agg_dict).reset_index()

    print(f"✓ 회차별 집계 완료: {len(df_inst):,}건")



    # ==================== 연속 지연 패턴 분석 (개선) ====================
    print("\n▶ 연속 지연 패턴 분석 중...")

    # 대출별로 정렬
    df_inst_sorted = df_inst.sort_values(
        ['sk_id_curr', 'sk_id_prev', 'num_instalment_number']
    ).copy()

    # NaN 방어: is_delayed에 NaN이 있으면 0으로 처리
    df_inst_sorted['is_delayed'] = (
        df_inst_sorted['is_delayed']
        .fillna(0)
        .astype('int8')
    )

    # 1️⃣ 그룹 경계 감지 (sk_id_curr, sk_id_prev가 바뀌는 지점)
    group_change = (
        df_inst_sorted[['sk_id_curr', 'sk_id_prev']]
        .ne(df_inst_sorted[['sk_id_curr', 'sk_id_prev']].shift())
        .any(axis=1)
    )

    # 2️⃣ 지연 상태 변화 감지 (0→1 또는 1→0)
    delay_change = df_inst_sorted['is_delayed'] != df_inst_sorted['is_delayed'].shift()

    # 3️⃣ delay block 식별 (그룹 경계 또는 지연 상태 변화 시 새 block)
    # 첫 행은 shift로 인해 자동으로 새로운 block 시작
    df_inst_sorted['delay_group'] = (group_change | delay_change).cumsum()

    # 4️⃣ 연속 지연 길이 계산 (각 block에서 is_delayed=1인 개수)
    consecutive_delays = (
        df_inst_sorted
        .groupby(['sk_id_curr', 'sk_id_prev', 'delay_group'])['is_delayed']
        .sum()
        .reset_index(name='consecutive_count')
    )

    # 지연이 없는 block 제거 (is_delayed=0인 block)
    consecutive_delays = consecutive_delays[consecutive_delays['consecutive_count'] > 0].copy()

    # 5️⃣ 대출별 최대 연속 지연
    if len(consecutive_delays) > 0:
        max_consecutive = (
            consecutive_delays
            .groupby(['sk_id_curr', 'sk_id_prev'])['consecutive_count']
            .max()
            .reset_index()
        )
        max_consecutive.columns = ['sk_id_curr', 'sk_id_prev', 'max_consecutive_delay']
    else:
        # 연속 지연이 전혀 없는 경우 빈 DataFrame
        max_consecutive = pd.DataFrame(columns=['sk_id_curr', 'sk_id_prev', 'max_consecutive_delay'])

    print("✓ 연속 지연 패턴 분석 완료")


    # ==================== 첫 발생 시점 분석 ====================
    print("\n▶ 첫 발생 시점 분석 중...")

    # 대출별 전체 회차 수
    total_instalments = df_inst.groupby(['sk_id_curr', 'sk_id_prev']).size().reset_index(name='total_inst_count')

    # 첫 지연 발생 회차
    first_delay = df_inst[df_inst['is_delayed'] == 1].groupby(['sk_id_curr', 'sk_id_prev'])['num_instalment_number'].min().reset_index()
    first_delay.columns = ['sk_id_curr', 'sk_id_prev', 'first_delay_inst']

    # 첫 과소납부 발생 회차
    first_underpay = df_inst[df_inst['is_underpay'] == 1].groupby(['sk_id_curr', 'sk_id_prev'])['num_instalment_number'].min().reset_index()
    first_underpay.columns = ['sk_id_curr', 'sk_id_prev', 'first_underpay_inst']

    # 병합
    first_occur = total_instalments.merge(first_delay, on=['sk_id_curr', 'sk_id_prev'], how='left')
    first_occur = first_occur.merge(first_underpay, on=['sk_id_curr', 'sk_id_prev'], how='left')

    # 비율 계산 (첫 발생까지 걸린 비율)
    first_occur['first_delay_gap'] = first_occur['first_delay_inst'] / first_occur['total_inst_count']
    first_occur['first_underpay_gap'] = first_occur['first_underpay_inst'] / first_occur['total_inst_count']

    first_occur = first_occur[['sk_id_curr', 'sk_id_prev', 'first_delay_gap', 'first_underpay_gap']]

    print("✓ 첫 발생 시점 분석 완료")


    # ==================== 할부조건 변경 분석 ====================
    print("\n▶ 할부조건 변경 분석 중...")

    # 버전별로 정렬
    df_ver = df_inst.sort_values(['sk_id_curr', 'sk_id_prev', 'num_instalment_number', 'num_instalment_version'])

    # 이전 버전 추적
    df_ver['prev_version'] = df_ver.groupby(['sk_id_curr', 'sk_id_prev', 'num_instalment_number'])['num_instalment_version'].shift()
    df_ver['ver_changed'] = (
        (df_ver['num_instalment_version'] != df_ver['prev_version']) &
        (df_ver['prev_version'].notna())
    ).astype(int)

    # 변경 시점 식별
    df_ver['has_change'] = df_ver.groupby(['sk_id_curr', 'sk_id_prev'])['ver_changed'].transform('max')

    # 변경 발생 대출만 필터링
    df_changed = df_ver[df_ver['has_change'] == 1].copy()

    # 변경 시점 이전/이후 구분
    df_changed['change_point'] = df_changed.groupby(['sk_id_curr', 'sk_id_prev'])['ver_changed'].transform('idxmax')
    df_changed['is_after_change'] = df_changed.index >= df_changed['change_point']

    # 변경 이후 데이터만 집계
    df_after_change = df_changed[df_changed['is_after_change']].copy()

    after_change_agg = df_after_change.groupby(['sk_id_curr', 'sk_id_prev']).agg(
        delay_cnt_after=('is_delayed', 'sum'),
        total_cnt_after=('is_delayed', 'count'),
        delay_days_after=('delay_days_value', 'mean')
    ).reset_index()

    after_change_agg['delay_rate_after_change'] = after_change_agg['delay_cnt_after'] / after_change_agg['total_cnt_after']
    after_change_agg = after_change_agg[['sk_id_curr', 'sk_id_prev', 'delay_rate_after_change', 'delay_days_after']]
    after_change_agg.columns = ['sk_id_curr', 'sk_id_prev', 'delay_rate_after_ver_change', 'delay_days_after_ver_change']

    print("✓ 할부조건 변경 분석 완료")


    # ==================== 대출별 기간별 집계 ====================
    print("\n▶ 대출별 기간별 집계 중...")

    # ===== 전체 기간 =====
    df_all = df_inst.groupby(['sk_id_curr', 'sk_id_prev']).agg(
        # 횟수
        delay_cnt_all=('is_delayed', 'sum'),
        ontime_cnt_all=('is_ontime', 'sum'),
        early_cnt_all=('is_early', 'sum'),
        underpay_cnt_all=('is_underpay', 'sum'),
        total_cnt_all=('is_delayed', 'count'),

        # 지연일수
        delay_days_mean_all=('delay_days_value', 'mean'),
        delay_days_max_all=('delay_days_value', 'max'),
        delay_days_std_all=('delay_days_value', 'std'),

        # 과소납부
        underpay_ratio_mean_all=('underpay_ratio', 'mean'),
        underpay_ratio_max_all=('underpay_ratio', 'max'),
        underpay_ratio_std_all=('underpay_ratio', 'std')
    ).reset_index()

    # 비율 계산
    df_all['delay_rate_all'] = df_all['delay_cnt_all'] / df_all['total_cnt_all']
    df_all['ontime_rate_all'] = df_all['ontime_cnt_all'] / df_all['total_cnt_all']
    df_all['early_rate_all'] = df_all['early_cnt_all'] / df_all['total_cnt_all']


    # ===== 최근 6개월 =====
    df_6m = df_inst[df_inst['is_last6m'] == 1].copy()

    if len(df_6m) > 0:
        df_6m_agg = df_6m.groupby(['sk_id_curr', 'sk_id_prev']).agg(
            delay_cnt_6m=('is_delayed', 'sum'),
            total_cnt_6m=('is_delayed', 'count'),
            delay_days_mean_6m=('delay_days_value', 'mean'),
            underpay_ratio_mean_6m=('underpay_ratio', 'mean')
        ).reset_index()

        df_6m_agg['delay_rate_6m'] = df_6m_agg['delay_cnt_6m'] / df_6m_agg['total_cnt_6m']
        df_6m_agg = df_6m_agg[['sk_id_curr', 'sk_id_prev', 'delay_rate_6m', 'delay_days_mean_6m', 'underpay_ratio_mean_6m']]
    else:
        df_6m_agg = pd.DataFrame(columns=['sk_id_curr', 'sk_id_prev', 'delay_rate_6m', 'delay_days_mean_6m', 'underpay_ratio_mean_6m'])


    # ===== 최근 3개월 =====
    df_3m = df_inst[df_inst['is_last3m'] == 1].copy()

    if len(df_3m) > 0:
        df_3m_agg = df_3m.groupby(['sk_id_curr', 'sk_id_prev']).agg(
            delay_cnt_3m=('is_delayed', 'sum'),
            total_cnt_3m=('is_delayed', 'count'),
            delay_days_mean_3m=('delay_days_value', 'mean'),
            underpay_ratio_mean_3m=('underpay_ratio', 'mean')
        ).reset_index()

        df_3m_agg['delay_rate_3m'] = df_3m_agg['delay_cnt_3m'] / df_3m_agg['total_cnt_3m']
        df_3m_agg = df_3m_agg[['sk_id_curr', 'sk_id_prev', 'delay_rate_3m', 'delay_days_mean_3m', 'underpay_ratio_mean_3m']]
    else:
        df_3m_agg = pd.DataFrame(columns=['sk_id_curr', 'sk_id_prev', 'delay_rate_3m', 'delay_days_mean_3m', 'underpay_ratio_mean_3m'])

    print("✓ 기간별 집계 완료")


    # ==================== 대출별 최종 병합 ====================
    print("\n▶ 대출별 최종 병합 중...")

    # 전체 기간 기준
    df_loan = df_all.copy()

    # 6개월, 3개월 병합
    df_loan = df_loan.merge(df_6m_agg, on=['sk_id_curr', 'sk_id_prev'], how='left')
    df_loan = df_loan.merge(df_3m_agg, on=['sk_id_curr', 'sk_id_prev'], how='left')

    # 패턴 변수 병합
    df_loan = df_loan.merge(max_consecutive, on=['sk_id_curr', 'sk_id_prev'], how='left')
    df_loan = df_loan.merge(first_occur, on=['sk_id_curr', 'sk_id_prev'], how='left')
    df_loan = df_loan.merge(after_change_agg, on=['sk_id_curr', 'sk_id_prev'], how='left')

    # 연속 지연 없으면 0
    df_loan['max_consecutive_delay'] = df_loan['max_consecutive_delay'].fillna(0)


    # ==================== Trend 변수 생성 ====================
    print("\n▶ Trend 변수 생성 중...")

    # 1. 지연 비율 추세 (최근 3개월 - 전체)
    df_loan['delay_rate_trend'] = df_loan['delay_rate_3m'] - df_loan['delay_rate_all']

    # 2. 지연일수 추세 (최근 3개월 - 전체)
    df_loan['delay_days_trend'] = df_loan['delay_days_mean_3m'] - df_loan['delay_days_mean_all']

    # 3. 과소납부 추세 (최근 3개월 - 전체)
    df_loan['underpay_trend'] = df_loan['underpay_ratio_mean_3m'] - df_loan['underpay_ratio_mean_all']

    print("✓ Trend 변수 생성 완료")


    # ==================== 추가 파생변수 ====================
    print("\n▶ 추가 파생변수 생성 중...")

    # 1. 지연 집중도 (CV: Coefficient of Variation)
    df_loan['delay_concentration'] = np.where(
        df_loan['delay_days_mean_all'] > 0,
        df_loan['delay_days_std_all'] / df_loan['delay_days_mean_all'],
        np.nan
    )

    # 2. 상환 행태 종합 점수 (가중합)
    # 조기납부(+2), 정시(+1), 지연(-2), 과소납부(-1)
    df_loan['behavior_score'] = (
        df_loan['early_rate_all'] * 2 +
        df_loan['ontime_rate_all'] * 1 -
        df_loan['delay_rate_all'] * 2 -
        (df_loan['underpay_cnt_all'] / df_loan['total_cnt_all']) * 1
    )

    print("✓ 추가 파생변수 생성 완료")


    # ==================== 최종 변수 선택 및 정리 ====================
    print("\n▶ 최종 변수 정리 중...")

    final_cols = [
        'sk_id_curr', 'sk_id_prev',

        # ===== 전체 기간 Summary =====
        'delay_rate_all',              # 전체 지연 비율
        'ontime_rate_all',             # 전체 정시 비율
        'early_rate_all',              # 전체 조기 비율
        'delay_days_mean_all',         # 전체 평균 지연일수
        'delay_days_max_all',          # 전체 최대 지연일수
        'delay_days_std_all',          # 전체 지연일수 표준편차
        'underpay_ratio_mean_all',     # 전체 평균 과소납부 비율
        'underpay_ratio_max_all',      # 전체 최대 과소납부 비율
        'underpay_ratio_std_all',      # 전체 과소납부 표준편차

        # ===== 최근 6개월 =====
        'delay_rate_6m',               # 6개월 지연 비율
        'delay_days_mean_6m',          # 6개월 평균 지연일수
        'underpay_ratio_mean_6m',      # 6개월 평균 과소납부 비율

        # ===== 최근 3개월 =====
        'delay_rate_3m',               # 3개월 지연 비율
        'delay_days_mean_3m',          # 3개월 평균 지연일수
        'underpay_ratio_mean_3m',      # 3개월 평균 과소납부 비율

        # ===== Trend (최근 변화) =====
        'delay_rate_trend',            # 지연비율 추세 (3m - all)
        'delay_days_trend',            # 지연일수 추세 (3m - all)
        'underpay_trend',              # 과소납부 추세 (3m - all)

        # ===== Pattern =====
        'max_consecutive_delay',       # 최대 연속 지연 회차
        'first_delay_gap',             # 첫 지연까지 비율
        'first_underpay_gap',          # 첫 과소납부까지 비율
        'delay_concentration',         # 지연 집중도 (CV)

        # ===== 할부조건 변경 효과 =====
        'delay_rate_after_ver_change', # 조건 변경 후 지연 비율
        'delay_days_after_ver_change', # 조건 변경 후 평균 지연일수

        # ===== 종합 점수 =====
        'behavior_score',              # 상환 행태 종합 점수

        # ===== 횟수 (참고용) =====
        'delay_cnt_all',               # 전체 지연 횟수
        'underpay_cnt_all',            # 전체 과소납부 횟수
        'total_cnt_all'                # 전체 회차 수
    ]

    df_final = df_loan[final_cols].copy()

    print("✓ 최종 변수 정리 완료")


    # ==================== 최종 출력 ====================
    print("\n" + "=" * 60)
    print("Installments Payments 전처리 완료!")
    print("=" * 60)
    print(f"✓ 최종 대출 건수: {len(df_final):,}")
    print(f"✓ 생성된 변수 수: {len(df_final.columns) - 2}개")
    print(f"\n✓ 변수 카테고리:")
    print(f"  - 전체 기간 Summary: 9개")
    print(f"  - 최근 6개월: 3개")
    print(f"  - 최근 3개월: 3개")
    print(f"  - Trend (변화량): 3개")
    print(f"  - Pattern (패턴): 4개")
    print(f"  - 할부조건 변경 효과: 2개")
    print(f"  - 종합 점수: 1개")
    print(f"  - 횟수 (참고): 3개")
    print("=" * 60 + "\n")

    return df_final

# 여기서부터 pre_ 전처리
def previous_application_preprocessed(df: pd.DataFrame):
    """
    previous_application 테이블 전처리 및 파생변수 생성
    sk_id_curr, sk_id_prev별 최종 집계 (대출별 특성 요약)

    주요 개선사항:
    1. 집계 기준 명확화: days_decision 기준 최신 데이터 우선
    2. 변동성 지표 추가: std, range 등
    3. 비즈니스 해석 가능한 파생변수 추가
    """

    print("=" * 60)
    print("Previous Application 전처리 시작")
    print("=" * 60)

    # ==================== 정렬: 최신순 정렬 ====================
    # days_decision이 0에 가까울수록 최근 신청
    df = df.sort_values(['sk_id_curr', 'sk_id_prev', 'days_decision'], ascending=[True, True, False])

    print("\n▶ 데이터 정렬 완료 (최신 신청 우선)")


    # ==================== 기본 파생변수 생성 ====================
    print("\n▶ 기본 파생변수 생성 중...")

    # 1. 금액 관련 파생변수
    df['amt_diff'] = df['amt_credit'] - df['amt_application']  # 승인액 - 신청액
    df['amt_diff_ratio'] = np.where(
        df['amt_application'] > 0,
        df['amt_diff'] / df['amt_application'],
        np.nan
    )  # 금액 차이 비율

    df['credit_to_goods_ratio'] = np.where(
        df['amt_goods_price'] > 0,
        df['amt_credit'] / df['amt_goods_price'],
        np.nan
    )  # 승인액 대비 상품가격 비율

    # 신규: 승인율 (신청 대비 승인 비율)
    df['approval_ratio'] = np.where(
        df['amt_application'] > 0,
        df['amt_credit'] / df['amt_application'],
        np.nan
    )  # 승인액/신청액 (1.0 초과 = 신청액보다 더 많이 승인)

    # 2. 시간 관련 파생변수
    df['loan_duration'] = df['days_last_due'] - df['days_first_due']  # 대출 기간

    df['decision_to_first_due'] = df['days_first_due'] - df['days_decision']  # 결정일~첫만기일 간격

    df['is_early_termination'] = (
        (df['days_termination'].notna()) &
        (df['days_termination'] < df['days_last_due'])
    ).astype('int8')  # 조기상환 플래그

    df['termination_gap'] = np.where(
        df['days_termination'].notna(),
        df['days_termination'] - df['days_last_due'],
        np.nan
    )  # 종료일 - 마지막만기일 (음수면 조기상환)

    # 3. 계약 상태 플래그
    df['is_approved'] = (df['name_contract_status'] == 'Approved').astype('int8')
    df['is_refused'] = (df['name_contract_status'] == 'Refused').astype('int8')
    df['is_canceled'] = (df['name_contract_status'] == 'Canceled').astype('int8')
    df['is_unused'] = (df['name_contract_status'] == 'Unused offer').astype('int8')

    # 4. 상품/계약 타입 플래그
    df['is_cash_loan'] = (df['name_contract_type'] == 'Cash loans').astype('int8')
    df['is_consumer_loan'] = (df['name_contract_type'] == 'Consumer loans').astype('int8')
    df['is_revolving_loan'] = (df['name_contract_type'] == 'Revolving loans').astype('int8')

    # 5. 채널 타입 플래그
    df['is_mobile'] = (df['channel_type'].str.contains('Mobile', na=False)).astype('int8')
    df['is_credit_office'] = (df['channel_type'].str.contains('Credit', na=False)).astype('int8')
    df['is_regional'] = (df['channel_type'].str.contains('Regional', na=False)).astype('int8')

    # 6. 리스크 관련 플래그
    df['is_high_yield'] = (df['name_yield_group'] == 'high').astype('int8')
    df['is_middle_yield'] = (df['name_yield_group'] == 'middle').astype('int8')
    df['is_low_yield'] = (df['name_yield_group'].isin(['low_action', 'low_normal'])).astype('int8')

    # 7. 신청 행동 플래그
    df['is_same_day_app'] = (df['nflag_last_appl_in_day'] == 0).astype('int8')
    df['is_last_contract'] = (df['flag_last_appl_per_contract'] == 'Y').astype('int8')

    # 8. 보험 가입 플래그
    df['has_insurance'] = df['nflag_insured_on_approval'].fillna(0).astype('int8')

    # 9. 고객 타입 플래그
    df['is_repeater'] = (df['name_client_type'] == 'Repeater').astype('int8')
    df['is_refreshed'] = (df['name_client_type'] == 'Refreshed').astype('int8')
    df['is_new'] = (df['name_client_type'] == 'New').astype('int8')

    # 10. 이자율 차이
    df['interest_rate_diff'] = df['rate_interest_primary'] - df['rate_interest_privileged']

    print("✓ 기본 파생변수 생성 완료 (21개)")


    # ==================== 대출별 집계 (sk_id_curr, sk_id_prev) ====================
    print("\n▶ 대출별(sk_id_curr, sk_id_prev) 집계 중...")

    df_agg = df.groupby(['sk_id_curr', 'sk_id_prev']).agg(
        # ===== 1. 기본 정보 (최신 기준) =====
        pre_contract_type=('name_contract_type', 'first'),  # 최신 계약 타입
        pre_contract_status=('name_contract_status', 'first'),  # 최신 계약 상태
        pre_client_type=('name_client_type', 'first'),  # 최신 고객 타입

        # ===== 2. 금액 관련 =====
        # 평균값
        pre_amt_application_mean=('amt_application', 'mean'),
        pre_amt_credit_mean=('amt_credit', 'mean'),
        pre_amt_annuity_mean=('amt_annuity', 'mean'),
        pre_amt_goods_price_mean=('amt_goods_price', 'mean'),
        pre_amt_down_payment_mean=('amt_down_payment', 'mean'),

        # 변동성 (std, range)
        pre_amt_credit_std=('amt_credit', 'std'),  # 승인액 변동성
        pre_amt_credit_max=('amt_credit', 'max'),
        pre_amt_credit_min=('amt_credit', 'min'),

        # 금액 차이/비율
        pre_amt_diff_mean=('amt_diff', 'mean'),
        pre_amt_diff_ratio_mean=('amt_diff_ratio', 'mean'),
        pre_credit_to_goods_ratio_mean=('credit_to_goods_ratio', 'mean'),
        pre_approval_ratio_mean=('approval_ratio', 'mean'),  # 신규: 승인율

        # ===== 3. 이자율/계약금 =====
        pre_rate_down_payment_mean=('rate_down_payment', 'mean'),
        pre_rate_interest_primary_mean=('rate_interest_primary', 'mean'),
        pre_rate_interest_privileged_mean=('rate_interest_privileged', 'mean'),
        pre_interest_rate_diff_mean=('interest_rate_diff', 'mean'),

        # ===== 4. 시간 관련 =====
        pre_days_decision_mean=('days_decision', 'mean'),
        pre_days_first_due_mean=('days_first_due', 'mean'),
        pre_days_last_due_mean=('days_last_due', 'mean'),
        pre_days_termination_mean=('days_termination', 'mean'),
        pre_days_first_drawing_mean=('days_first_drawing', 'mean'),

        # 파생 시간 변수
        pre_loan_duration_mean=('loan_duration', 'mean'),
        pre_loan_duration_std=('loan_duration', 'std'),  # 신규: 대출기간 변동성
        pre_loan_duration_max=('loan_duration', 'max'),
        pre_loan_duration_min=('loan_duration', 'min'),

        pre_decision_to_first_due_mean=('decision_to_first_due', 'mean'),
        pre_termination_gap_mean=('termination_gap', 'mean'),

        # ===== 5. 할부/기간 =====
        pre_cnt_payment_mean=('cnt_payment', 'mean'),

        # ===== 6. 상태 플래그 =====
        pre_is_approved_sum=('is_approved', 'sum'),  # 승인 건수
        pre_is_refused_sum=('is_refused', 'sum'),   # 거절 건수
        pre_is_canceled_sum=('is_canceled', 'sum'),
        pre_is_unused_sum=('is_unused', 'sum'),
        pre_is_early_termination_sum=('is_early_termination', 'sum'),

        # ===== 7. 상품 타입 =====
        pre_is_cash_loan_sum=('is_cash_loan', 'sum'),
        pre_is_consumer_loan_sum=('is_consumer_loan', 'sum'),
        pre_is_revolving_loan_sum=('is_revolving_loan', 'sum'),

        # ===== 8. 채널 타입 =====
        pre_is_mobile_sum=('is_mobile', 'sum'),
        pre_is_credit_office_sum=('is_credit_office', 'sum'),
        pre_is_regional_sum=('is_regional', 'sum'),

        # ===== 9. 수익률 그룹 =====
        pre_is_high_yield_sum=('is_high_yield', 'sum'),
        pre_is_middle_yield_sum=('is_middle_yield', 'sum'),
        pre_is_low_yield_sum=('is_low_yield', 'sum'),

        # ===== 10. 신청 행동 =====
        pre_is_same_day_app_sum=('is_same_day_app', 'sum'),
        pre_is_last_contract_sum=('is_last_contract', 'sum'),

        # ===== 11. 보험/고객 타입 =====
        pre_has_insurance_sum=('has_insurance', 'sum'),
        pre_is_repeater_sum=('is_repeater', 'sum'),
        pre_is_refreshed_sum=('is_refreshed', 'sum'),
        pre_is_new_sum=('is_new', 'sum'),

        # ===== 12. 카테고리 다양성 =====
        pre_goods_category_variety=('name_goods_category', 'nunique'),
        pre_portfolio_variety=('name_portfolio', 'nunique'),
        pre_product_type_variety=('name_product_type', 'nunique'),
        pre_payment_type_variety=('name_payment_type', 'nunique'),

        # ===== 13. 추가 카테고리 정보 =====
        pre_seller_place_area_mean=('sellerplace_area', 'mean'),
        pre_weekday_appr_process=('weekday_appr_process_start', 'first'),
        pre_hour_appr_process_mean=('hour_appr_process_start', 'mean'),

        # ===== 14. 신청 횟수 =====
        pre_application_count=('sk_id_prev', 'count')  # 이 대출에서 신청 시도 횟수

    ).reset_index()

    print(f"✓ 대출별 집계 완료: {len(df_agg):,}건")


    # ==================== 추가 파생변수 (집계 후) ====================
    print("\n▶ 집계 후 파생변수 생성 중...")

    # 1. 승인 여부에 따른 플래그 (명확한 분류)
    df_agg['pre_final_status_approved'] = (df_agg['pre_contract_status'] == 'Approved').astype('int8')
    df_agg['pre_final_status_refused'] = (df_agg['pre_contract_status'] == 'Refused').astype('int8')

    # 2. 신규: 승인+거절 둘 다 경험 (조건 바꿔가며 여러 번 시도한 패턴)
    df_agg['pre_has_both_approved_refused'] = (
        (df_agg['pre_is_approved_sum'] > 0) &
        (df_agg['pre_is_refused_sum'] > 0)
    ).astype('int8')

    # 3. 금액 대비 월 납입액 비율
    df_agg['pre_annuity_to_credit_ratio'] = np.where(
        df_agg['pre_amt_credit_mean'] > 0,
        df_agg['pre_amt_annuity_mean'] / df_agg['pre_amt_credit_mean'],
        np.nan
    )

    # 4. 계약금 관련
    df_agg['pre_has_down_payment'] = (df_agg['pre_rate_down_payment_mean'] > 0).astype('int8')

    # 5. 대출 기간 카테고리
    df_agg['pre_is_short_term'] = (df_agg['pre_loan_duration_mean'] <= 180).astype('int8')  # 6개월 이하
    df_agg['pre_is_long_term'] = (df_agg['pre_loan_duration_mean'] >= 730).astype('int8')   # 2년 이상

    # 6. 조기상환 분석
    # 해석: 양수 = 몇 일 일찍 상환했는지 (ex: 30 = 30일 일찍 상환)
    df_agg['pre_early_repay_days'] = np.where(
        df_agg['pre_is_early_termination_sum'] > 0,
        -df_agg['pre_termination_gap_mean'],  # termination_gap이 음수이므로 부호 반전
        np.nan
    )

    # 7. 승인액 변동성 (range)
    df_agg['pre_amt_credit_range'] = df_agg['pre_amt_credit_max'] - df_agg['pre_amt_credit_min']

    # 8. 대출 기간 변동성 (range)
    df_agg['pre_loan_duration_range'] = df_agg['pre_loan_duration_max'] - df_agg['pre_loan_duration_min']

    # 9. 승인율 (승인 건수 / 전체 신청 건수)
    df_agg['pre_approval_rate'] = np.where(
        df_agg['pre_application_count'] > 0,
        df_agg['pre_is_approved_sum'] / df_agg['pre_application_count'],
        np.nan
    )

    # 10. 거절율 (거절 건수 / 전체 신청 건수)
    df_agg['pre_refusal_rate'] = np.where(
        df_agg['pre_application_count'] > 0,
        df_agg['pre_is_refused_sum'] / df_agg['pre_application_count'],
        np.nan
    )

    print("✓ 추가 파생변수 생성 완료 (10개)")


    # ==================== 최종 정리 ====================
    print("\n" + "=" * 60)
    print("Previous Application 전처리 완료!")
    print("=" * 60)
    print(f"✓ 최종 대출 건수: {len(df_agg):,}")
    print(f"✓ 생성된 변수 수: {len(df_agg.columns) - 2}개")
    print(f"\n✓ 변수 카테고리:")
    print(f"  - 기본 정보: 3개 (계약타입, 상태, 고객타입)")
    print(f"  - 금액 관련: 16개 (mean, std, max, min, range, 비율)")
    print(f"  - 이자율/계약금: 4개")
    print(f"  - 시간 관련: 14개 (mean, std, max, min, range)")
    print(f"  - 할부/기간: 1개")
    print(f"  - 상태 플래그: 5개 (sum)")
    print(f"  - 상품/채널 타입: 9개 (sum)")
    print(f"  - 리스크/행동: 7개 (sum)")
    print(f"  - 카테고리 다양성: 4개")
    print(f"  - 기타: 4개 (seller_place, weekday, hour, count)")
    print(f"  - 집계 후 파생: 10개")
    print("=" * 60 + "\n")

    return df_agg

# 여기서부터 pos_ 전처리
# =====================================================================================================
# POS_CASH_balance 파생변수 생성 함수 v6 — (sk_id_prev 기준 집계 버전)
# =====================================================================================================
# ⭐ 목적:
#     - 이 단계에서는 "POS 계약 단위(sk_id_prev)"의 파생변수만 생성한다.
#     - 즉, 고객(sk_id_curr) 기준 집계는 하지 않는다.
#       → 이유: 이후 previous_application(pre_app) 단계에서 sk_id_prev를 기준으로 다시 집계하고
#               그 다음 app_train(sk_id_curr)로 최종 집계를 수행해야 하기 때문.
#
# ⭐ 설계 철학:
#     - POS_CASH는 개별 POS 거래(Contract) 단위이며,
#       고객 하나(sk_id_curr)가 여러 거래(sk_id_prev)를 가질 수 있다.
#
#     - POS_CASH 단계에서 계약 단위의 정보를 충분히 요약해놓으면
#       이후 pre_app, app_train에서 매우 유연하게 활용 가능하다.
#
# ⭐ 포함되는 파생변수 유형:
#     1) 계약 진행도(progress → 상환 진행률)
#     2) 최근 기록 기반 파생 (가장 최근 month)
#     3) 최근 3개 기록 기반 파생
#     4) 일반 연체(DPD) 특성
#     5) 심각 연체(DEF) 특성
#     6) 계약 상태(name_contract_status) 기반 위험도
#
# =====================================================================================================

def build_pos_cash_features_prev(df_pos):
    """
    POS_CASH_balance 원본(df_pos)을 입력받아,
    각 POS 계약(sk_id_prev) 단위의 파생변수를 생성하는 함수.

    [입력 df_pos 컬럼 설명]
        sk_id_prev              : POS 거래 고유 ID (계약 단위)
        sk_id_curr              : 해당 POS 거래가 속한 고객 ID
        months_balance          : 기록 시점 (0 또는 음수. 0=가장 최근, -1=-1개월 전)
        cnt_instalment          : 전체 할부 개수
        cnt_instalment_future   : 해당 시점 기준 남은 할부 개수
        name_contract_status    : 계약 상태 (Completed/Active/Demand 등)
        sk_dpd                  : Days Past Due (일반 연체일)
        sk_dpd_def              : Default 수준 심각 연체일

    [출력]
        prev_features : sk_id_prev 단위의 Feature Vector
                        (POS_CASH에서 가장 중요한 요약값 집합)
    """

    print("\n📌 POS_CASH prev-level 파생변수 v6 생성 시작...")
    df = df_pos.copy()

    # =================================================================================================
    # 0) 계약 상태(name_contract_status) → 정량적 위험도(score)로 변환
    # -------------------------------------------------------------------------------------------------
    # ✔ 왜 필요한가?
    #     - 문자열 상태값은 모델에서 직접 사용하기 어렵다.
    #     - 또한 상태값 자체가 대출 리스크의 중요한 신호이므로
    #       ‘정량적 위험 점수’로 변환해 모델 입력으로 사용 가능하게 한다.
    #
    # ✔ 위험도 설계 원리:
    #     - Completed, Approved, Signed 등 이미 종료/정상 계약 → 위험도 0 (안전)
    #     - Active → 아직 진행 중인 계약 → 위험도 1 (보통 위험)
    #     - Demand, Canceled → 채무 불이행 또는 조치 상태 → 위험도 2 (고위험)
    #
    # ✔ XNA (unknown)은 대체로 Active로 간주하여 위험도 1 부여
    # =================================================================================================
    status_risk_map = {
        "Completed": 0.0,
        "Approved": 0.0,
        "Signed": 0.0,
        "Amortized debt": 0.0,

        "Active": 1.0,
        "Returned to the store": 1.0,

        "Demand": 2.0,
        "Canceled": 2.0,

        "XNA": 1.0
    }

    df["status_risk"] = (
        df["name_contract_status"]
        .astype(str)
        .map(status_risk_map)
        .fillna(1.0)    # 만약 예상치 못한 상태가 나오면 보통 위험(1.0) 처리
        .astype("float32")
    )

    # =================================================================================================
    # 1) 시계열 정렬: months_balance 기준으로 최신 데이터가 가장 위에 오도록
    # -------------------------------------------------------------------------------------------------
    # ✔ 이후 "최근 기록 선택" 또는 "최근 3개 기록 선택" 등에서 중요하므로
    #   정렬을 먼저 수행하는 것이 핵심.
    #
    # ✔ months_balance 값이 클수록 최근 데이터
    #   예: -1 → 최근, -96 → 오래된
    # =================================================================================================
    df = df.sort_values(["sk_id_prev", "months_balance"], ascending=[True, False])

    # =================================================================================================
    # 2) 상환 진행도(progress) 생성
    # -------------------------------------------------------------------------------------------------
    # ✔ 정의:
    #     progress = (전체 할부 - 남은 할부) / 전체 할부
    #
    # ✔ 해석:
    #     - 0에 가까울수록 갓 시작한 계약 (리스크 상대적으로 높음)
    #     - 1에 가까울수록 거의 다 상환된 계약 (리스크 상대적으로 낮음)
    #
    # ✔ 이후 활용:
    #     - sk_id_prev 단위로 평균/최대 progress 생성
    #     - pre_app 또는 app_train으로 넘어갈 때 매우 유용한 지표
    # =================================================================================================
    df["progress"] = (
        (df["cnt_instalment"] - df["cnt_instalment_future"]) /
        (df["cnt_instalment"] + 1e-6)     # 0 나누기 방지
    ).clip(0, 1)

    prev_progress = df.groupby("sk_id_prev")["progress"].agg(
        pos_progress_ratio_mean="mean",    # 계약 기간 전체의 평균 상환 진행도
        pos_progress_ratio_max="max"       # 해당 계약이 가장 많이 상환되었을 때의 진행도
    ).reset_index()

    # =================================================================================================
    # 3) 각 계약(sk_id_prev)의 "가장 최근 기록" 추출
    # -------------------------------------------------------------------------------------------------
    # ✔ 왜 필요한가?
    #     - POS 계약에서 현재 잔여 할부 개수(cnt_instalment_future)는
    #       가장 최근 데이터에서만 의미를 가진다.
    #
    # ✔ 로직:
    #     - months_balance 가장 큰 행(즉, 가장 최근)을 찾는다.
    # =================================================================================================
    recent_idx = df.groupby("sk_id_prev")["months_balance"].idxmax()
    recent = df.loc[recent_idx].copy()

    prev_future = recent.groupby("sk_id_prev")["cnt_instalment_future"].agg(
        pos_future_instalment_mean="mean",   # 거의 1개 값이지만 mean 사용하면 안전
        pos_future_instalment_max="max"      # 혹시라도 예외 상황 고려하여 max도 생성
    ).reset_index()

    # =================================================================================================
    # 4) DPD / DPD_DEF / 계약상태 위험 기반 prev-level 리스크 변수 생성
    # -------------------------------------------------------------------------------------------------
    # ✔ 포함 변수:
    #     - pos_dpd_max                : 최고 연체일수
    #     - pos_dpd_mean               : 평균 연체일수
    #     - pos_dpd_overdue_ratio      : 연체 발생 비율
    #     - pos_dpd_consec_overdue     : 연속 연체 최대 길이
    #     - pos_dpd_def_flag           : 심각 연체 발생 여부
    #     - pos_dpd_def_max            : 심각 연체 최대일수
    #     - pos_status_risk_mean       : 계약 상태 기반 위험 평균
    #     - pos_status_risk_max        : 가장 위험했던 상태
    #
    # ✔ 이유:
    #     - POS 계약의 연체 패턴은 매우 강력한 리스크 신호이며
    #       pre_app → app_train으로 올라갈 때 모델 성능에 큰 도움을 준다.
    # =================================================================================================

    def max_consecutive(arr_bool):
        """True(연체)가 연속된 가장 긴 구간의 길이를 구하는 함수"""
        count, max_count = 0, 0
        for v in arr_bool:
            if v:
                count += 1
                max_count = max(max_count, count)
            else:
                count = 0
        return max_count

    def agg_prev_risk(x):
        """각 sk_id_prev의 시계열을 받아 리스크 요약 변수 생성"""

        dpd = x["sk_dpd"].values
        dpd_def = x["sk_dpd_def"].values
        status_risk = x["status_risk"].values

        overdue = (dpd > 0)
        def_over = (dpd_def > 0)

        return pd.Series({
            "pos_dpd_max": dpd.max() if len(dpd) else 0.0,
            "pos_dpd_mean": dpd.mean() if len(dpd) else 0.0,
            "pos_dpd_overdue_ratio": overdue.mean() if len(overdue) else 0.0,
            "pos_dpd_consec_overdue": max_consecutive(overdue),

            "pos_dpd_def_flag": int(def_over.any()),
            "pos_dpd_def_max": dpd_def.max() if len(dpd_def) else 0.0,
            "pos_dpd_def_mean": dpd_def.mean() if len(dpd_def) else 0.0,

            "pos_status_risk_mean": status_risk.mean() if len(status_risk) else 0.0,
            "pos_status_risk_max": status_risk.max() if len(status_risk) else 0.0,
        })

    prev_risk = (
        df.groupby("sk_id_prev")[["sk_dpd", "sk_dpd_def", "status_risk"]]
          .apply(agg_prev_risk)
          .reset_index()
    )

    # =================================================================================================
    # 5) 최근 3개 기록 기반 파생변수
    # -------------------------------------------------------------------------------------------------
    # ✔ 왜 필요한가?
    #     - months_balance가 반드시 0, -1, -2를 가지지 않기 때문에
    #       "최근 3개의 행"을 직접 선택하는 것이 더 정확하다.
    #
    # ✔ 포함 변수:
    #     - pos_dpd_mean_recent3
    #     - pos_status_risk_recent3
    #
    # =================================================================================================
    recent3 = (
        df.groupby("sk_id_prev", group_keys=False)
          .head(3)
    )

    prev_recent3 = recent3.groupby("sk_id_prev").agg(
        pos_dpd_mean_recent3=("sk_dpd", "mean"),
        pos_status_risk_recent3=("status_risk", "mean")
    ).reset_index()

    # =================================================================================================
    # 6) 모든 prev-level 파생변수 통합
    # -------------------------------------------------------------------------------------------------
    # ✔ 이 단계가 pos_cash 단계의 최종 산출물이 된다.
    # ✔ sk_id_prev 기준으로 merge만 수행하며 sk_id_curr 기준 집계는 절대 하지 않는다.
    # =================================================================================================
    prev_features = (
        prev_progress
            .merge(prev_future, on="sk_id_prev", how="left")
            .merge(prev_risk, on="sk_id_prev", how="left")
            .merge(prev_recent3, on="sk_id_prev", how="left")
    )

    print("✔ POS_CASH prev-level 파생변수 v6 완료:", prev_features.shape)
    return prev_features


# 여기서부터 cc_ 전처리
def create_creditcard_features(creditcard_df: pd.DataFrame, 
                               output_path: str = None) -> pd.DataFrame:
    """
    Credit Card Balance 파생변수 생성 메인 파이프라인 - sk_id_prev 기준 버전
    ===========================================================================
    입력: credit_card_balance 원본 데이터
    출력: sk_id_prev 기준 파생변수 테이블 (각 과거 신용카드 계약 단위)
    
    처리 순서:
    1. 월별 파생변수 생성
    2. Point-in-Time 필터링 (months_balance < 0)
    3. sk_id_prev 단위 집계 (계약 단위)
    ===========================================================================
    """
    print("=" * 70)
    print("Credit Card Feature Engineering Pipeline (개선판, sk_id_prev 기준)")
    print("=" * 70)
    
    # 데이터 정렬
    df = creditcard_df.sort_values(
        ['sk_id_curr', 'sk_id_prev', 'months_balance']
    ).copy()
    
    print(f"\n[1/5] 입력 데이터 크기: {df.shape}")
    print(f"      - 고객 수: {df['sk_id_curr'].nunique():,}")
    print(f"      - 대출/계약 수(sk_id_prev): {df['sk_id_prev'].nunique():,}")
    print(f"      - months_balance 범위: [{df['months_balance'].min()}, {df['months_balance'].max()}]")
    
    # Point-in-Time 확인
    pit_data = df[df['months_balance'] < CUTOFF_MONTH]
    print(f"\n[2/5] Point-in-Time 필터링 (months_balance < {CUTOFF_MONTH})")
    print(f"      - 필터링 후 행 수: {len(pit_data):,} / {len(df):,}")
    print(f"      - 제외된 행 수: {len(df) - len(pit_data):,} (대출 후 데이터)")
    
    # Step 1: 월별 파생변수 생성
    print("\n[3/5] 월별 파생변수 생성 중...")
    df = create_utilization_features(df)
    df = create_dpd_features(df)
    df = create_minpay_miss_features(df)
    df = create_payment_behavior_features(df)
    df = create_contract_status_features(df)
    print("      ✓ Utilization, DPD, MinPay Miss, Payment Behavior, Contract Status 생성 완료")
    
    # Step 2: sk_id_prev 단위 집계
    print("\n[4/5] 과거 대출/계약(sk_id_prev) 단위 집계 중...")
    df_prev = aggregate_to_prev_level(df)
    
    if len(df_prev) == 0:
        print("      ⚠️ 경고: 집계 결과가 없습니다.")
        return pd.DataFrame()
    
    print(f"      ✓ 집계 완료: {len(df_prev):,} 개 sk_id_prev")
    
    # Step 3: Payment Trend 계산 및 병합 (여전히 prev 레벨)
    print("\n[5/5] 납부 추세(Payment Trend) 계산 중...")
    df_pit = df[df['months_balance'] < CUTOFF_MONTH].copy()
    payment_trend = calculate_payment_trend(df_pit)
    df_prev = df_prev.merge(payment_trend, on='sk_id_prev', how='left')
    print("      ✓ Huber Regression 기반 추세 계산 완료")
    
    # 이제 df_prev가 최종 결과 (sk_id_prev 기준)
    df_final = df_prev.copy()
    
    print("\n" + "=" * 70)
    print("파생변수 생성 완료 - (sk_id_prev 기준) 변수 목록")
    print("=" * 70)
    
    # Tier 1: 핵심 변수 (prev 레벨 기준으로 그대로 사용)
    print("\n[Tier 1: 핵심 변수 (Must Have)]")
    tier1_vars = [
        'max_dpd', 'mean_dpd',
        'dpd_30plus_ever',          # 30일+ 경험 여부
        'dpd_severity_max',         # 전체 기간 최악 severity
        'dpd_6m_mean',              # 최근 6개월 평균 DPD
        'dpd_severity_6m_max',      # 최근 6개월 최악 severity
        'weighted_avg_dpd',         # 가중 평균 DPD
        'utilization_mean', 'utilization_max', 'utilization_6m_mean',
        'cnt_minpay_miss', 'minpay_miss_consecutive',
        'min_pay_deficit_ratio_max'
    ]
    for var in tier1_vars:
        if var in df_final.columns:
            null_pct = df_final[var].isnull().mean() * 100
            print(f"  - {var} (NaN: {null_pct:.1f}%)")
    
    print("\n[Tier 2: 중요 변수 (Should Have)]")
    tier2_vars = [
        'pay_vs_use_risk_mean', 'full_payment_rate',
        'dpd_6m_max', 'minpay_miss_6m',
        'is_active_last', 'contract_status_risk_max',
        'cnt_over_limit', 'cnt_inactive_months',
        'high_util_months', 'cnt_months'
    ]
    for var in tier2_vars:
        if var in df_final.columns:
            null_pct = df_final[var].isnull().mean() * 100
            print(f"  - {var} (NaN: {null_pct:.1f}%)")
    
    print("\n[정보성 변수]")
    print("  - sk_id_curr (해당 계약이 속한 고객 ID)")
    print("  - cnt_months (이 계약의 이력 개월 수)")
    
    # 결측치 요약
    null_summary = df_final.isnull().sum()
    cols_with_null = null_summary[null_summary > 0]
    if len(cols_with_null) > 0:
        print("\n⚠️ 결측치가 있는 컬럼:")
        for col, cnt in cols_with_null.items():
            print(f"   - {col}: {cnt:,} ({cnt/len(df_final)*100:.1f}%)")
    
    # ====== 컬럼 prefix 부여 (cc_) ======
    df_out = df_final.copy()
    
    # 이제는 sk_id_prev가 메인 키이므로 둘 다 보존하고 싶으면 둘 다 제외
    exclude_cols = {'sk_id_prev', 'sk_id_curr'}  # prefix 제외할 컬럼들
    rename_map = {
        col: f'cc_{col}'
        for col in df_out.columns
        if col not in exclude_cols
    }
    df_out = df_out.rename(columns=rename_map)
    
    # Parquet 저장 (선택)
    if output_path:
        df_out.to_parquet(output_path, engine="pyarrow", index=False)
        print(f"\n✓ Parquet 파일 저장 완료: {output_path}")
    
    return df_out
