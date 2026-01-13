# utils/shap_reason.py
from __future__ import annotations
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd

def _coerce_listlike(x) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, float) and np.isnan(x):
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, str):
        # parquet/serialize 과정에서 문자열로 들어왔을 가능성
        try:
            import ast
            v = ast.literal_eval(x)
            return v if isinstance(v, list) else []
        except Exception:
            return []
    if hasattr(x, "__iter__"):
        try:
            return list(x)
        except Exception:
            return []
    return []

def get_top_reasons_from_shap_row(
    row: pd.Series,
    map_dict: Dict[str, Dict[str, str]],
    *,
    top_features_col: str = "shap_features",
    top_values_col: str = "shap_values",
    top_k: int = 10,
    only_risk_positive: bool = True,
) -> List[str]:
    """
    row에 저장된 shap_features/shap_values를 기반으로
    UI에서 보여줄 '주요 참고 요인' 문구(top_k개)를 생성.
    - only_risk_positive=True면 +SHAP(부도 위험을 올린 방향)만 우선 표시
    - map_dict: feature -> {"reason_label":..., "super_group":...} 형태를 가정
    """
    feats = _coerce_listlike(row.get(top_features_col))
    vals  = _coerce_listlike(row.get(top_values_col))

    if not feats or not vals or len(feats) != len(vals):
        return []

    # (feature, shap) 묶기 + 숫자 변환
    pairs: List[Tuple[str, float]] = []
    for f, v in zip(feats, vals):
        if f is None:
            continue
        try:
            pairs.append((str(f), float(v)))
        except Exception:
            continue

    if not pairs:
        return []

    # 위험 방향 우선 (+SHAP)만 보고 싶으면 필터
    if only_risk_positive:
        pairs_pos = [(f, v) for f, v in pairs if v > 0]
        if pairs_pos:
            pairs = pairs_pos  # 양수 SHAP이 하나라도 있으면 그걸 우선 사용

    # 절대값 기준 Top-K
    pairs = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:top_k]

    # 문구 생성
    out: List[str] = []
    for f, v in pairs:
        m = map_dict.get(f, {}) if isinstance(map_dict, dict) else {}
        reason = m.get("reason_label") or m.get("reason_label_ko") or "기타"
        group  = m.get("super_group") or m.get("group") or m.get("bin") or ""

        direction = "위험↑" if v > 0 else "위험↓"
        if group:
            out.append(f"[{group}] {reason} ({direction})")
        else:
            out.append(f"{reason} ({direction})")

    return out
