# utils/behavioral_insights.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .feature_semantic_map import FEATURE_SEMANTIC_MAP

def _safe_float(x) -> Optional[float]:
    try:
        if x is None: return None
        if isinstance(x, (float, int, np.floating, np.integer)):
            return float(x)
        # 문자열 숫자 처리
        s = str(x).strip()
        if s == "" or s.lower() in ("nan", "none"): return None
        return float(s)
    except Exception:
        return None

def _percentile_of_value(ref: pd.Series, v: float) -> Optional[float]:
    """ref 분포 내에서 v의 분위(0~1). ref가 없으면 None."""
    try:
        s = pd.to_numeric(ref, errors="coerce").dropna()
        if s.empty: return None
        return float((s < v).mean())
    except Exception:
        return None

def _highlow_from_percentile(p: Optional[float], hi: float = 0.7, lo: float = 0.3) -> Optional[str]:
    if p is None: 
        return None
    if p >= hi: return "high"
    if p <= lo: return "low"
    return "mid"

def estimate_ead_from_row(row: pd.Series) -> Optional[float]:
    """
    amt_credit이 없을 때, (가능하면) app_payment_rate & amt_annuity로 대출금(credit)을 역산
    - payment_rate = annuity / credit 라는 전형적 정의를 전제로 함
    """
    ann = _safe_float(row.get("amt_annuity"))
    pr  = _safe_float(row.get("app_payment_rate"))
    if ann is None or pr is None or pr <= 0:
        return None
    credit = ann / pr
    if not np.isfinite(credit) or credit <= 0:
        return None
    return float(credit)

def generate_behavioral_insights(
    row: pd.Series,
    *,
    shap_features: Optional[List[str]] = None,
    shap_values: Optional[List[float]] = None,
    shap_top_10: Optional[List[Dict[str, Any]]] = None,
    ref_df: Optional[pd.DataFrame] = None,
    top_k: int = 5,
) -> List[str]:
    """
    결과: '행태/맥락' 문장 리스트 (최대 top_k)
    - 판단 기준:
      1) ref_df가 있으면 분위(상/하)를 사용해 high/low 템플릿 선택
      2) ref_df가 없거나 mid면, SHAP 부호를 보조로 사용(위험↑면 high쪽, 위험↓면 low쪽)하는 fallback
    """
    # 입력 정리: shap_top_10 우선(이미 정렬되어 있을 확률 높음)
    feats: List[str] = []
    vals: List[float] = []

    if isinstance(shap_top_10, list) and shap_top_10:
        for it in shap_top_10:
            f = it.get("feature")
            v = it.get("shap")
            if f is None: 
                continue
            vf = _safe_float(v)
            if vf is None:
                continue
            feats.append(str(f))
            vals.append(vf)
    else:
        feats = list(shap_features or [])
        vals = [v for v in (shap_values or []) if isinstance(v, (int, float, np.floating, np.integer))]

    # 길이 맞추기
    n = min(len(feats), len(vals))
    feats, vals = feats[:n], vals[:n]
    if n == 0:
        return []

    out: List[str] = []
    used = set()

    for f, sv in zip(feats, vals):
        if f in used:
            continue
        used.add(f)

        tpl = FEATURE_SEMANTIC_MAP.get(f)
        if not tpl:
            continue  # 사전에 없는 feature는 스킵 (원하면 default 문장 넣어도 됨)

        # 분위 기반 high/low 판정
        v = _safe_float(row.get(f))
        hl: Optional[str] = None
        if ref_df is not None and v is not None and f in ref_df.columns:
            p = _percentile_of_value(ref_df[f], v)
            hl = _highlow_from_percentile(p)

        # mid거나 ref_df가 없으면, SHAP 부호를 보조로 사용
        # - 위험 증가(sv>0)면 high쪽 문장, 위험 감소(sv<0)면 low쪽 문장
        if hl is None or hl == "mid":
            hl = "high" if sv > 0 else "low"

        sent = tpl.get(hl)
        if sent:
            out.append(sent)

        if len(out) >= top_k:
            break

    # (보너스) EAD 추정 기반 맥락 문장 추가(가능한 경우)
    ead = estimate_ead_from_row(row)
    inc = _safe_float(row.get("amt_income_total"))
    if ead is not None and inc is not None and inc > 0:
        ratio = ead / inc
        # 과장 방지: 거칠게 0.3/0.6 기준만
        if ratio <= 0.3:
            out.insert(0, "추정 대출 규모가 소득 대비 낮은 편으로, 보수적인 신청 성향이 관측됩니다.")
        elif ratio >= 0.6:
            out.insert(0, "추정 대출 규모가 소득 대비 큰 편이라, 상환 여력 점검이 도움이 될 수 있습니다.")

    # 중복 제거
    dedup: List[str] = []
    seen = set()
    for s in out:
        if s not in seen:
            dedup.append(s)
            seen.add(s)

    return dedup[:top_k]
