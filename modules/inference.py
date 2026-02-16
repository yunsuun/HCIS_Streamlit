import numpy as np
import shap
import xgboost as xgb
from packaging import version

def predict_pd_only(model, calibrator, model_type: str, X):
    # 1) PD raw
    if model_type == "XGB":
        X_np = X.to_numpy(np.float32)
        pd_raw = model.predict_proba(X_np)[:, 1]
    else:
        pd_raw = model.predict_proba(X)[:, 1]

    # 2) calibration
    pd_hat = calibrator.predict(pd_raw)
    return pd_hat

def predict_pd_upload_with_shap(
    model,
    calibrator,
    model_type: str,
    X,
    top_n: int = 10
):
    # =========================
    # 1) PD (기존 그대로)
    # =========================
    if model_type == "XGB":
        X_np = X.to_numpy(np.float32)
        pd_raw = model.predict_proba(X_np)[:, 1]
    else:
        X_np = X.to_numpy()
        pd_raw = model.predict_proba(X)[:, 1]

    pd_hat = calibrator.predict(pd_raw)

    # =========================
    # 2) SHAP 분기
    # =========================
    shap_features = None
    shap_values = None

    if model_type == "XGB":
        xgb_ver = version.parse(xgb.__version__)

        # -------------------------
        # (A) xgboost < 3.1 → 기존 TreeExplainer
        # -------------------------
        if xgb_ver < version.parse("3.1.0"):
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_np)

            if isinstance(sv, list):  # binary
                sv = sv[1]

        # -------------------------
        # (B) xgboost >= 3.1 → pred_contribs 경로
        # -------------------------
        else:
            booster = model.get_booster()
            sv = booster.predict(
                xgb.DMatrix(X_np),
                pred_contribs=True
            )
            # 마지막 컬럼 = bias → 제거
            sv = sv[:, :-1]

        # -------------------------
        # top-N 정리
        # -------------------------
        feat_names = np.array(X.columns)
        shap_features = []
        shap_values = []

        for i in range(sv.shape[0]):
            row = sv[i]
            top_idx = np.argsort(np.abs(row))[::-1][:top_n]
            shap_features.append(feat_names[top_idx].tolist())
            shap_values.append(row[top_idx].astype(float).tolist())

    return pd_hat, shap_features, shap_values



# def predict_pd_single(model, calibrator, model_type, X):
#     if model_type == "XGB":
#         pd_raw = model.predict_proba(X.to_numpy(np.float32))[:, 1]
#     else:
#         pd_raw = model.predict_proba(X)[:, 1]
#     return float(calibrator.predict(pd_raw)[0])


# def predict_pd_upload(model, calibrator, model_type, X):
#     if model_type == "XGB":
#         pd_raw = model.predict_proba(X.to_numpy(np.float32))[:, 1]
#     else:
#         pd_raw = model.predict_proba(X)[:, 1]
#     return calibrator.predict(pd_raw)


# # ============================================================
# # NEW: PD + SHAP(top-N) together
# # ============================================================
# def predict_pd_upload_with_shap(
#     model,
#     calibrator,
#     model_type: str,
#     X,                         # pandas DataFrame (already aligned!)
#     top_n: int = 10
# ):
#     """
#     Returns
#     -------
#     pd_hat: np.ndarray shape (n,)
#     shap_features: list[list[str]]   # per row top-N feature names
#     shap_values: list[list[float]]   # per row top-N shap values (signed)
#     """
#     # 1) PD
#     if model_type == "XGB":
#         X_np = X.to_numpy(np.float32)
#         pd_raw = model.predict_proba(X_np)[:, 1]
#     else:
#         X_np = X.to_numpy()
#         pd_raw = model.predict_proba(X)[:, 1]

#     pd_hat = calibrator.predict(pd_raw)

#     # 2) SHAP
#     try:
#         import shap
#     except Exception as e:
#         raise RuntimeError(
#             "SHAP 계산을 위해 'shap' 라이브러리가 필요합니다. "
#             "pip install shap 로 설치 후 다시 실행해주세요."
#         ) from e

#     # XGB 기준 (TreeExplainer가 가장 안정적/빠름)
#     if model_type == "XGB":
#         explainer = shap.TreeExplainer(model)
#         sv = explainer.shap_values(X_np)

#         # binary일 때 shap 버전에 따라 list로 올 수도 있음
#         if isinstance(sv, list):
#             # 보통 [class0, class1] 형태 → class1 사용
#             sv = sv[1]
#     else:
#         # (여기서는 XGB만 쓸 거라면 else는 막아도 됨)
#         explainer = shap.Explainer(model, X_np)
#         sv = explainer(X_np).values

#     # 3) top-N 추출 (절대값으로 순위만, 값은 signed 그대로 저장)
#     feat_names = np.array(list(X.columns), dtype=object)

#     shap_features: list[list[str]] = []
#     shap_values: list[list[float]] = []

#     sv = np.asarray(sv, dtype=float)
#     n_rows, n_cols = sv.shape
#     use_n = min(top_n, n_cols)

#     for i in range(n_rows):
#         row_sv = sv[i].copy()  # signed

#         # 순위는 abs 기준
#         top_idx = np.argsort(np.abs(row_sv))[::-1][:use_n]

#         # 저장은 signed 그대로
#         top_feats = feat_names[top_idx].tolist()
#         top_vals  = row_sv[top_idx].astype(float).tolist()

#         # sanity check: abs로 저장되는 실수를 잡아냄
#         # top_idx 중 row_sv에 음수가 있는데 top_vals가 전부 양수면 뭔가 잘못된 것
#         if np.any(row_sv[top_idx] < 0) and all(v >= 0 for v in top_vals):
#             raise RuntimeError("SHAP 부호가 손실되었습니다. abs()가 저장 단계에 적용된 것으로 보입니다.")

#         shap_features.append(top_feats)
#         shap_values.append(top_vals)


#     return pd_hat, shap_features, shap_values
