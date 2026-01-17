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


def get_top_reason_items_from_shap_row(
    row: pd.Series,
    map_dict: Dict[str, Dict[str, str]],
    *,
    top_features_col: str = "shap_features",
    top_values_col: str = "shap_values",
    top_k: int = 10,
    only_risk_positive: bool = True,
) -> List[Dict[str, Any]]:
    """
    get_top_reasons_from_shap_row의 확장판.
    UI에서 색 진하기 조절을 위해 shap 값까지 포함해 반환한다.
    반환 예:
      [{"text": "[그룹] 이유 (위험↑)", "shap": 0.12, "abs_shap": 0.12, "dir": "up"}, ...]
    """
    feats = _coerce_listlike(row.get(top_features_col))
    vals  = _coerce_listlike(row.get(top_values_col))

    if not feats or not vals or len(feats) != len(vals):
        return []

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

    if only_risk_positive:
        pairs_pos = [(f, v) for f, v in pairs if v > 0]
        if pairs_pos:
            pairs = pairs_pos

    pairs = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:top_k]

    out: List[Dict[str, Any]] = []
    for f, v in pairs:
        m = map_dict.get(f, {}) if isinstance(map_dict, dict) else {}
        reason = m.get("reason_label") or m.get("reason_label_ko") or "기타"
        group  = m.get("super_group") or m.get("group") or m.get("bin") or ""

        direction = "위험↑" if v > 0 else "위험↓"
        text = f"[{group}] {reason} ({direction})" if group else f"{reason} ({direction})"

        out.append({
            "text": text,
            "shap": v,
            "abs_shap": abs(v),
            "dir": "up" if v > 0 else "down",
        })

    return out
