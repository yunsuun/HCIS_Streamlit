import os
import sqlite3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "db", "hcis.db")


# -----------------------------
# Helpers
# -----------------------------
def table_exists(conn, table: str) -> bool:
    r = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?;",
        (table,),
    ).fetchone()
    return r is not None


def get_cols(conn, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
    return {r[1] for r in rows}


def safe_expr(cols: set[str], col: str, cast: str | None = None, default: str = "NULL") -> str:
    """컬럼 있으면 col(캐스팅 optional), 없으면 default"""
    if col in cols:
        return f"CAST({col} AS {cast})" if cast else col
    return default


def drop_table(conn, name: str):
    conn.execute(f"DROP TABLE IF EXISTS {name};")


def create_index_if_col_exists(conn, table: str, col: str, index_name: str):
    cols = get_cols(conn, table)
    if col in cols:
        conn.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table}({col});")


# -----------------------------
# 1) application_train 기반 (외부신용 + 부양가족 대비 수입 중심)
# -----------------------------
def create_feat_application_train(conn):
    src = "application_train"
    out = "feat_application_train"

    if not table_exists(conn, src):
        print(f"❌ {src} 테이블이 없어 {out} 생성 스킵")
        return

    cols = get_cols(conn, src)
    if "sk_id_curr" not in cols:
        print(f"❌ {src}에 sk_id_curr가 없어 {out} 생성 스킵")
        return

    # 원본 컬럼명(홈크레딧 기준)
    ext1 = safe_expr(cols, "ext_source_1", "REAL")
    ext2 = safe_expr(cols, "ext_source_2", "REAL")
    ext3 = safe_expr(cols, "ext_source_3", "REAL")

    income = safe_expr(cols, "amt_income_total", "REAL")
    annuity = safe_expr(cols, "amt_annuity", "REAL")
    credit = safe_expr(cols, "amt_credit", "REAL")

    children = safe_expr(cols, "cnt_children", "REAL")
    fam = safe_expr(cols, "cnt_fam_members", "REAL")

    days_birth = safe_expr(cols, "days_birth", "REAL")
    days_employed = safe_expr(cols, "days_employed", "REAL")

    drop_table(conn, out)
    conn.execute(f"""
    CREATE TABLE {out} AS
    SELECT
      sk_id_curr,

      -- 외부신용: min / mean (NULL-safe)
      CASE
        WHEN {ext1} IS NULL AND {ext2} IS NULL AND {ext3} IS NULL THEN NULL
        ELSE MIN(COALESCE({ext1}, 1e9), COALESCE({ext2}, 1e9), COALESCE({ext3}, 1e9))
      END AS app_ext_min,

      CASE
        WHEN ({ext1} IS NULL AND {ext2} IS NULL AND {ext3} IS NULL) THEN NULL
        ELSE (
          COALESCE({ext1},0) + COALESCE({ext2},0) + COALESCE({ext3},0)
        ) / NULLIF(
          (CASE WHEN {ext1} IS NULL THEN 0 ELSE 1 END)
        + (CASE WHEN {ext2} IS NULL THEN 0 ELSE 1 END)
        + (CASE WHEN {ext3} IS NULL THEN 0 ELSE 1 END)
        , 0)
      END AS app_ext_mean,

      -- 소득/가구 구성
      {income} AS app_income,
      {fam} AS app_fam_members,
      {children} AS app_children,
      ({income} / NULLIF({fam}, 0)) AS app_income_per_member,
      ({children} / NULLIF({fam}, 0)) AS app_children_ratio,

      -- 상환여력 비율
      ({annuity} / NULLIF({income}, 0)) AS app_annuity_income_ratio,
      ({credit} / NULLIF({income}, 0)) AS app_credit_income_ratio,

      -- 나이/근속(일→년)
      (ABS({days_birth}) / 365.0) AS app_age_years,
      (CASE
        WHEN {days_employed} IS NULL THEN NULL
        WHEN {days_employed} > 0 THEN NULL
        ELSE ABS({days_employed}) / 365.0
      END) AS app_employed_years

    FROM {src};
    """)
    create_index_if_col_exists(conn, out, "sk_id_curr", "idx_feat_application_train_sk_id_curr")
    conn.commit()
    print(f"✅ {out} 생성 완료")


# -----------------------------
# 2) bureau + bureau_balance 조합
# -----------------------------
def create_feat_bureau(conn):
    out = "feat_bureau"

    if not table_exists(conn, "bureau"):
        print("❌ bureau 테이블이 없어 feat_bureau 생성 스킵")
        return

    bcols = get_cols(conn, "bureau")
    if "sk_id_curr" not in bcols or "sk_id_bureau" not in bcols:
        print("❌ bureau에 sk_id_curr/sk_id_bureau가 없어 feat_bureau 생성 스킵")
        return

    credit_active = safe_expr(bcols, "credit_active", None, "NULL")
    debt = safe_expr(bcols, "amt_credit_sum_debt", "REAL", "NULL")
    sum_credit = safe_expr(bcols, "amt_credit_sum", "REAL", "NULL")
    overdue = safe_expr(bcols, "amt_credit_max_overdue", "REAL", "NULL")
    day_overdue = safe_expr(bcols, "credit_day_overdue", "REAL", "NULL")
    update = safe_expr(bcols, "days_credit_update", "REAL", "NULL")
    prolong = safe_expr(bcols, "cnt_credit_prolong", "REAL", "NULL")

    has_bb = table_exists(conn, "bureau_balance")
    bb_join = ""
    bb_feats = ""

    if has_bb:
        bb_cols = get_cols(conn, "bureau_balance")
        if "sk_id_bureau" in bb_cols:
            status = safe_expr(bb_cols, "status", None, "NULL")
            months = safe_expr(bb_cols, "months_balance", "REAL", "NULL")

            bb_join = f"""
            LEFT JOIN (
              SELECT
                sk_id_bureau,
                AVG(CASE WHEN {status} IN ('1','2','3','4','5') THEN 1.0 ELSE 0.0 END) AS bb_delinquency_rate,
                MAX(CASE
                      WHEN {status}='5' THEN 5
                      WHEN {status}='4' THEN 4
                      WHEN {status}='3' THEN 3
                      WHEN {status}='2' THEN 2
                      WHEN {status}='1' THEN 1
                      ELSE 0
                    END) AS bb_worst_status,
                MIN({months}) AS bb_oldest_months_balance
              FROM bureau_balance
              GROUP BY sk_id_bureau
            ) bb
            ON b.sk_id_bureau = bb.sk_id_bureau
            """
            bb_feats = """
              , AVG(bb.bb_delinquency_rate) AS bu_bb_delinquency_rate_mean
              , MAX(bb.bb_worst_status)      AS bu_bb_worst_status_max
              , MIN(bb.bb_oldest_months_balance) AS bu_bb_oldest_months_balance
            """

    drop_table(conn, out)
    conn.execute(f"""
    CREATE TABLE {out} AS
    SELECT
      b.sk_id_curr,

      COUNT(*) AS bu_cnt_total,
      SUM(CASE WHEN {credit_active} = 'Active' THEN 1 ELSE 0 END) AS bu_cnt_active,
      SUM(CASE WHEN {credit_active} = 'Closed' THEN 1 ELSE 0 END) AS bu_cnt_closed,

      AVG({debt}) AS bu_debt_mean,
      MAX({debt}) AS bu_debt_max,
      SUM({debt}) AS bu_debt_sum,

      AVG({sum_credit}) AS bu_credit_sum_mean,
      MAX({sum_credit}) AS bu_credit_sum_max,

      MAX({overdue}) AS bu_max_overdue_amt,
      MAX({day_overdue}) AS bu_max_overdue_days,

      MAX({update}) AS bu_days_credit_update_max,
      SUM({prolong}) AS bu_cnt_credit_prolong_sum

      {bb_feats}

    FROM bureau b
    {bb_join}
    GROUP BY b.sk_id_curr;
    """)
    create_index_if_col_exists(conn, out, "sk_id_curr", "idx_feat_bureau_sk_id_curr")
    conn.commit()
    print("✅ feat_bureau 생성 완료")


# -----------------------------
# 3) previous_application + credit_card_balance + POS_CASH_balance + installments_payments
# -----------------------------
def create_feat_behavior(conn):
    out = "feat_behavior"

    # ✅ 로더가 파일명을 lower()로 테이블명 만들었을 가능성이 높음
    t_pre  = "previous_application"     if table_exists(conn, "previous_application") else None
    t_cc   = "credit_card_balance"      if table_exists(conn, "credit_card_balance") else None
    t_pos  = "pos_cash_balance"         if table_exists(conn, "pos_cash_balance") else None
    t_inst = "installments_payments"    if table_exists(conn, "installments_payments") else None

    if not any([t_pre, t_cc, t_pos, t_inst]):
        print("❌ 행동 테이블(previous_application/credit_card_balance/pos_cash_balance/installments_payments) 없음 → feat_behavior 스킵")
        return

    # ✅ 빈 CTE 방어: SELECT에서 참조하는 컬럼을 '항상' 만들어둔다
    pre_sql = """
    SELECT
      NULL AS sk_id_curr,
      NULL AS pre_cnt_total,
      NULL AS pre_approval_rate,
      NULL AS pre_credit_mean,
      NULL AS pre_credit_max,
      NULL AS pre_credit_min,
      NULL AS pre_days_decision_mean
    WHERE 1=0
    """

    cc_sql = """
    SELECT
      NULL AS sk_id_curr,
      NULL AS cc_util_mean,
      NULL AS cc_util_max,
      NULL AS cc_balance_mean,
      NULL AS cc_balance_max
    WHERE 1=0
    """

    pos_sql = """
    SELECT
      NULL AS sk_id_curr,
      NULL AS pos_delay_rate,
      NULL AS pos_dpd_mean,
      NULL AS pos_dpd_max,
      NULL AS pos_dpd_def_max
    WHERE 1=0
    """

    inst_sql = """
    SELECT
      NULL AS sk_id_curr,
      NULL AS inst_delay_rate,
      NULL AS inst_delay_days_mean,
      NULL AS inst_payment_rate
    WHERE 1=0
    """

    # -------------------------
    # previous_application 요약
    # -------------------------
    if t_pre:
        pcols = get_cols(conn, t_pre)
        if "sk_id_curr" in pcols:
            status = safe_expr(pcols, "name_contract_status", None, "NULL")
            amt_credit = safe_expr(pcols, "amt_credit", "REAL", "NULL")
            days_decision = safe_expr(pcols, "days_decision", "REAL", "NULL")

            pre_sql = f"""
            SELECT
              sk_id_curr,
              COUNT(*) AS pre_cnt_total,
              AVG(CASE WHEN {status}='Approved' THEN 1.0 ELSE 0.0 END) AS pre_approval_rate,
              AVG({amt_credit}) AS pre_credit_mean,
              MAX({amt_credit}) AS pre_credit_max,
              MIN({amt_credit}) AS pre_credit_min,
              AVG({days_decision}) AS pre_days_decision_mean
            FROM {t_pre}
            GROUP BY sk_id_curr
            """

    # -------------------------
    # credit_card_balance 요약
    # -------------------------
    if t_cc:
        ccols = get_cols(conn, t_cc)
        if "sk_id_curr" in ccols:
            bal = safe_expr(ccols, "amt_balance", "REAL", "NULL")
            lim = safe_expr(ccols, "amt_credit_limit_actual", "REAL", "NULL")
            util = f"({bal} / NULLIF({lim}, 0))"

            cc_sql = f"""
            SELECT
              sk_id_curr,
              AVG({util}) AS cc_util_mean,
              MAX({util}) AS cc_util_max,
              AVG({bal}) AS cc_balance_mean,
              MAX({bal}) AS cc_balance_max
            FROM {t_cc}
            GROUP BY sk_id_curr
            """

    # -------------------------
    # pos_cash_balance 요약 (사진 컬럼 기준)
    # - SK_DPD / SK_DPD_DEF 사용
    # -------------------------
    if t_pos:
        pos_cols = get_cols(conn, t_pos)
        # ✅ POS_CASH_balance는 sk_id_curr가 있음(사진에도 있음)
        if "sk_id_curr" in pos_cols:
            sk_dpd = safe_expr(pos_cols, "sk_dpd", "REAL", "NULL")
            sk_dpd_def = safe_expr(pos_cols, "sk_dpd_def", "REAL", "NULL")

            pos_sql = f"""
            SELECT
              sk_id_curr,
              AVG(CASE WHEN {sk_dpd} > 0 THEN 1.0 ELSE 0.0 END) AS pos_delay_rate,
              AVG({sk_dpd}) AS pos_dpd_mean,
              MAX({sk_dpd}) AS pos_dpd_max,
              MAX({sk_dpd_def}) AS pos_dpd_def_max
            FROM {t_pos}
            GROUP BY sk_id_curr
            """

    # -------------------------
    # installments_payments 요약
    # -------------------------
    if t_inst:
        icols = get_cols(conn, t_inst)
        if "sk_id_curr" in icols:
            entry = safe_expr(icols, "days_entry_payment", "REAL", "NULL")
            instday = safe_expr(icols, "days_instalment", "REAL", "NULL")
            pay = safe_expr(icols, "amt_payment", "REAL", "NULL")
            instamt = safe_expr(icols, "amt_instalment", "REAL", "NULL")

            delay_days = f"({entry} - {instday})"
            delay_flag = f"CASE WHEN {delay_days} > 0 THEN 1.0 ELSE 0.0 END"
            pay_rate = f"({pay} / NULLIF({instamt}, 0))"

            inst_sql = f"""
            SELECT
              sk_id_curr,
              AVG({delay_flag}) AS inst_delay_rate,
              AVG(CASE WHEN {delay_days} > 0 THEN {delay_days} ELSE NULL END) AS inst_delay_days_mean,
              AVG({pay_rate}) AS inst_payment_rate
            FROM {t_inst}
            GROUP BY sk_id_curr
            """

    # -------------------------
    # 최종 feat_behavior 생성
    # -------------------------
    drop_table(conn, out)
    conn.execute(f"""
    CREATE TABLE {out} AS
    WITH
      pre AS ({pre_sql}),
      cc  AS ({cc_sql}),
      pos AS ({pos_sql}),
      ins AS ({inst_sql}),
      keys AS (
        SELECT sk_id_curr FROM pre
        UNION SELECT sk_id_curr FROM cc
        UNION SELECT sk_id_curr FROM pos
        UNION SELECT sk_id_curr FROM ins
      )
    SELECT
      k.sk_id_curr,

      pre.pre_cnt_total,
      pre.pre_approval_rate,
      pre.pre_credit_mean,
      pre.pre_credit_max,
      pre.pre_credit_min,
      pre.pre_days_decision_mean,

      cc.cc_util_mean,
      cc.cc_util_max,
      cc.cc_balance_mean,
      cc.cc_balance_max,

      pos.pos_delay_rate,
      pos.pos_dpd_mean,
      pos.pos_dpd_max,
      pos.pos_dpd_def_max,

      ins.inst_delay_rate,
      ins.inst_delay_days_mean,
      ins.inst_payment_rate

    FROM keys k
    LEFT JOIN pre ON k.sk_id_curr = pre.sk_id_curr
    LEFT JOIN cc  ON k.sk_id_curr = cc.sk_id_curr
    LEFT JOIN pos ON k.sk_id_curr = pos.sk_id_curr
    LEFT JOIN ins ON k.sk_id_curr = ins.sk_id_curr
    ;
    """)

    create_index_if_col_exists(conn, out, "sk_id_curr", "idx_feat_behavior_sk_id_curr")
    conn.commit()
    print(f"✅ {out} 생성 완료 (pre={t_pre}, cc={t_cc}, pos={t_pos}, inst={t_inst})")


# -----------------------------
# 4) 최종 합치기
# -----------------------------
def create_feat_all(conn):
    base = "feat_application_train"
    out = "feat_all"

    if not table_exists(conn, base):
        print(f"❌ {base}가 없어 {out} 생성 스킵")
        return

    drop_table(conn, out)

    # SQLite는 컬럼 제외 문법이 없으니 명시적으로 붙임(안전)
    has_bu = table_exists(conn, "feat_bureau")
    has_bh = table_exists(conn, "feat_behavior")

    select_cols = ["a.*"]

    join_sql = ""
    if has_bu:
        select_cols += [
            "b.bu_cnt_total", "b.bu_cnt_active", "b.bu_cnt_closed",
            "b.bu_debt_mean", "b.bu_debt_max", "b.bu_debt_sum",
            "b.bu_credit_sum_mean", "b.bu_credit_sum_max",
            "b.bu_max_overdue_amt", "b.bu_max_overdue_days",
            "b.bu_days_credit_update_max", "b.bu_cnt_credit_prolong_sum",
            "b.bu_bb_delinquency_rate_mean", "b.bu_bb_worst_status_max", "b.bu_bb_oldest_months_balance",
        ]
        join_sql += "LEFT JOIN feat_bureau b ON a.sk_id_curr = b.sk_id_curr\n"

    if has_bh:
        select_cols += [
            "v.pre_cnt_total", "v.pre_approval_rate", "v.pre_credit_mean", "v.pre_credit_max", "v.pre_credit_min", "v.pre_days_decision_mean",
            "v.cc_util_mean", "v.cc_util_max", "v.cc_balance_mean", "v.cc_balance_max",
            "v.pos_delay_rate", "v.pos_dpd_mean", "v.pos_dpd_max", "v.pos_dpd_def_max",
            "v.inst_delay_rate", "v.inst_delay_days_mean", "v.inst_payment_rate",
        ]
        join_sql += "LEFT JOIN feat_behavior v ON a.sk_id_curr = v.sk_id_curr\n"

    conn.execute(f"""
    CREATE TABLE {out} AS
    SELECT
      {", ".join(select_cols)}
    FROM {base} a
    {join_sql}
    ;
    """)

    create_index_if_col_exists(conn, out, "sk_id_curr", "idx_feat_all_sk_id_curr")
    conn.commit()
    print("✅ feat_all 생성 완료")


def show_counts(conn, tables: list[str]):
    for t in tables:
        if table_exists(conn, t):
            cnt = conn.execute(f"SELECT COUNT(*) FROM {t};").fetchone()[0]
            print(f"📌 {t}: {cnt:,} rows")
        else:
            print(f"⚠️ {t}: 없음")


def main():
    print(f"DB: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    try:
        create_feat_application_train(conn)
        create_feat_bureau(conn)
        create_feat_behavior(conn)
        create_feat_all(conn)

        print("\n--- row counts ---")
        show_counts(conn, ["feat_application_train", "feat_bureau", "feat_behavior", "feat_all"])
    finally:
        conn.close()


if __name__ == "__main__":
    main()
