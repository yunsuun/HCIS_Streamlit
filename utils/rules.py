import numpy as np
from config import (
    PD_GRADE_CUTS, GRADE_ORDER,
    SCORE_APPROVE, SCORE_COND,
    PD_APPROVE, PD_COND
)

def pd_to_grade(pd_value: float) -> str:
    if pd_value is None or (isinstance(pd_value, float) and np.isnan(pd_value)):
        return "N/A"
    for g in GRADE_ORDER:
        if pd_value <= PD_GRADE_CUTS[g]:
            return g
    return "E"

def underwriting_decision_dual(score: float, pd_value: float) -> str:
    """운영용 듀얼 컷오프 심사.
    - 승인: score>=SCORE_APPROVE AND pd<=PD_APPROVE
    - 조건부: score>=SCORE_COND AND pd<=PD_COND (단, 승인 제외)
    - 위험: 그 외
    """
    if score is None or pd_value is None:
        return "판단불가"
    if np.isnan(score) or np.isnan(pd_value):
        return "판단불가"

    if (score >= SCORE_APPROVE) and (pd_value <= PD_APPROVE):
        return "승인"
    if (score >= SCORE_COND) and (pd_value <= PD_COND):
        return "조건부"
    return "위험"

def apply_conditional_terms(decision: str, pd_value: float, score: float) -> dict:
    """조건부일 때 부가조건(예: 한도 축소/금리 가산)을 '예시'로 부여.
    프로젝트 데모 목적: 정책을 눈에 보이게 표현.
    """
    if decision != "조건부":
        return {"조건": None, "권고": None}

    # 예시 규칙 (필요 시 정책화)
    terms = []
    if pd_value > 0.10:
        terms.append("한도 30% 축소 권고")
    if score < 620:
        terms.append("추가 서류 확인(소득/재직) 권고")
    if not terms:
        terms.append("조건부 승인(표준 조건)")

    return {"조건": " / ".join(terms), "권고": "리스크 완화 조건 충족 시 승인 전환 가능"}
