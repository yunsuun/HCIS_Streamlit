from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd


@dataclass
class SimParams:
    ead: float = 5_000_000.0          # 건당 대출원금(EAD) 가정
    apr: float = 0.12                 # 연이자율
    tenor_months: int = 12            # 기간(개월)
    lgd: float = 0.6                  # 손실률
    review_cost_per_case: float = 10_000.0  # 추가검토 비용(건당)
    target_col: Optional[str] = "target"    # 있으면 실제 위험률 계산


def simulate_type_based_conversion(
    df_review: pd.DataFrame,
    *,
    include_types: List[str],
    conv_rates: List[float],
    params: SimParams,
    pd_col: str = "pd_hat",
    type_col: str = "risk_type_key",
) -> pd.DataFrame:
    """
    추가검토 중 특정 Risk Type들을 '검증/확인 성공 시 승인 전환'으로 가정해 시뮬레이션.
    - include_types: 승인 전환 후보 타입 키 리스트 (예: TYPE2/TYPE3/TYPE4)
    - conv_rates: 확인 성공률 시나리오 (예: [0.3, 0.5, 0.7])
    """
    req = {pd_col, type_col}
    miss = req - set(df_review.columns)
    if miss:
        raise ValueError(f"df_review에 필수 컬럼 누락: {sorted(miss)}")

    df = df_review.copy()
    df[pd_col] = pd.to_numeric(df[pd_col], errors="coerce")

    cand = df[df[type_col].isin(include_types)].copy()
    n_review = len(df)
    n_cand = len(cand)

    # 후보가 없으면 빈 결과
    if n_cand == 0:
        return pd.DataFrame([{
            "scenario": "no_candidates",
            "n_review": n_review,
            "n_candidates": 0,
            "n_converted": 0,
            "interest_income": 0.0,
            "expected_loss": 0.0,
            "review_cost": 0.0,
            "net_profit": 0.0,
            "avg_pd_converted": np.nan,
            "actual_default_rate_converted": np.nan,
        }])

    # 후보 집단에서 PD 분포 기반으로 평균 손실 계산을 간단히 하기 위해, 기대손실은 '전환된 수'에 비례하도록 처리
    avg_pd_cand = float(np.nanmean(cand[pd_col].to_numpy()))

    rows = []
    for r in conv_rates:
        r = float(r)
        r = min(max(r, 0.0), 1.0)

        n_conv = int(round(n_cand * r))

        # 이자수익(기간 비례)
        interest_income = n_conv * params.ead * params.apr * (params.tenor_months / 12)

        # 기대손실(EL) = PD * LGD * EAD
        expected_loss = n_conv * avg_pd_cand * params.lgd * params.ead

        # 운영비용: 후보 전체를 확인한다고 보면 후보 수 기준이 더 현실적일 때가 많음
        # (전환된 수만큼 비용이 드는 게 아니라 '검증'을 한 만큼 비용이 들기 때문)
        review_cost = n_cand * params.review_cost_per_case

        net_profit = interest_income - expected_loss - review_cost

        # 실제 위험률(옵션): target이 있으면 후보 중 상위 n_conv를 뽑는 게 아니라
        # "확인 성공은 랜덤" 가정이므로, 후보 집단 default rate을 그대로 사용
        adr = np.nan
        if params.target_col and (params.target_col in cand.columns):
            t = pd.to_numeric(cand[params.target_col], errors="coerce")
            if t.notna().any():
                adr = float(t.mean())

        rows.append({
            "scenario": f"conv_rate_{int(r*100)}pct",
            "n_review": n_review,
            "n_candidates": n_cand,
            "n_converted": n_conv,
            "interest_income": float(interest_income),
            "expected_loss": float(expected_loss),
            "review_cost": float(review_cost),
            "net_profit": float(net_profit),
            "avg_pd_converted": float(avg_pd_cand),
            "actual_default_rate_converted": adr,
        })

    out = pd.DataFrame(rows)
    out["profit_per_converted"] = out["net_profit"] / out["n_converted"].replace(0, np.nan)
    out["el_per_converted"] = out["expected_loss"] / out["n_converted"].replace(0, np.nan)
    out["income_per_converted"] = out["interest_income"] / out["n_converted"].replace(0, np.nan)
    return out


def summarize_candidates_by_type(df_review: pd.DataFrame, type_col: str = "risk_type_key", pd_col: str = "pd_hat") -> pd.DataFrame:
    df = df_review.copy()
    df[pd_col] = pd.to_numeric(df[pd_col], errors="coerce")
    g = df.groupby(type_col, dropna=False).agg(
        n=("sk_id_curr", "count"),
        avg_pd=(pd_col, "mean"),
    ).reset_index()
    return g.sort_values("n", ascending=False)
