import pandas as pd

from modules.cleaning import clean_data_load, setting_train, ApplicantTypeClassifier, app_derived_variable, pos_derived_variable, cc_derived_variable, inst_derived_variable, build_id_sets, split_case_ids
from modules.cleaning import pos_curr_features, cc_curr_features, inst_curr_features, run_pre_block, run_bureau_block, preprocess_full_minimal

def preprocess_features_only(df: pd.DataFrame):
    """
    [FINAL]
    - raw app_df를 받아
    - 모든 파생변수 생성
    - 케이스 분기 처리
    - ML 입력용 X, id 반환

    Returns
    -------
    X : pd.DataFrame
        모델 입력용 feature matrix
    id_series : pd.Series
        sk_id_curr (추론 결과 병합용)
    """
    app_df = df.copy()
    app_df.columns = app_df.columns.str.lower()
    # =====================================================
    # 0. 원본 데이터 로드 (외부 테이블)
    # =====================================================
    bu, bu_bal, pre, inst, pos, cc = clean_data_load()
    id_set = set(app_df["sk_id_curr"].unique())
    # =====================================================
    # 1. 고객 유형 분류
    # =====================================================
    all_dict = setting_train(app_df, bu, bu_bal, pre, inst, pos, cc)
    type_clf = ApplicantTypeClassifier(all_dict)
    df_customer_types = type_clf.classify_all(app_df)

    # =====================================================
    # 2. app 파생변수
    # =====================================================
    app_features = app_derived_variable(app_df)

    # =====================================================
    # 3. pos / cc / inst 서브 파생
    # =====================================================
    pos_d = pos_derived_variable(pos, id_set)
    cc_d = cc_derived_variable(cc, id_set)
    inst_d = inst_derived_variable(inst, id_set)

    # curr 단위 집계 (항상 실행)
    pos_curr  = pos_curr_features(pos_d, id_set)
    cc_curr   = cc_curr_features(cc_d, id_set)
    inst_curr = inst_curr_features(inst_d, id_set)

    # =====================================================
    # 4. 케이스 분기
    # =====================================================
    id_sets = build_id_sets(app_df, pre, cc_d, inst_d)
    cases = split_case_ids(id_sets)

    # =====================================================
    # 5. pre_app 기반 파생
    # =====================================================
    pre_case_ids = (
        cases["case_1"] |
        cases["case_2"] |
        cases["case_3"] |
        cases["case_5"]
    )

    pre_features = run_pre_block(
        pre_case_ids,
        pre,
        pos_d,
        cc_d,
        inst_d
    )

    # # =====================================================
    # # 6. pre 없는 케이스 처리
    # # =====================================================
    # no_pre_ids = cases["case_4"] | cases["case_6"] | cases["case_7"]

    # cc_only_feat = cc_only_features(cc_d, no_pre_ids)
    # inst_only_feat = inst_only_features(inst_d, no_pre_ids)

    # =====================================================
    # 7. bureau 파생
    # =====================================================
    bureau_features = run_bureau_block(
        id_set,
        app_features[["sk_id_curr"]],
        bu
    )

    # =====================================================
    # 8. 최종 병합
    # =====================================================
    df_final = (
        df_customer_types
        .merge(app_features, on="sk_id_curr", how="left")
        .merge(pos_curr, on="sk_id_curr", how="left")
        .merge(cc_curr, on="sk_id_curr", how="left")
        .merge(inst_curr, on="sk_id_curr", how="left")
        .merge(pre_features, on="sk_id_curr", how="left")
        .merge(bureau_features, on="sk_id_curr", how="left")
    )

    # 디버깅용
    assert (
        df_final["sk_id_curr"].nunique() == len(df_final)
    ), "❌ 고객당 1행 규칙이 깨졌습니다 (merge 증식 발생)"

    dup_cols = df_final.columns[df_final.columns.duplicated()].tolist()
    assert not dup_cols, f"❌ 중복 컬럼 발생: {dup_cols}"

    # df_final = df_final.set_index("sk_id_curr")

    # df_final.update(
    #     cc_only_feat.set_index("sk_id_curr"),
    #     overwrite=False
    # )

    # df_final.update(
    #     inst_only_feat.set_index("sk_id_curr"),
    #     overwrite=False
    # )

    # df_final = df_final.reset_index()

    # =====================================================
    # 9. ML 최종 클리닝
    # =====================================================
    X, id_series = preprocess_full_minimal(df_final)

    return X, id_series
