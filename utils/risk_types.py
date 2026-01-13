from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple



@dataclass(frozen=True)
class RiskTypeSpec:
    """추가검토(Review) 고객 2차 평가를 위한 Risk Type 정의."""
    key: str
    name: str
    short_desc: str
    checklist_questions: List[str]
    suggested_actions: List[str]


# ------------------------------------------------------------
# Default risk types
# ------------------------------------------------------------

RISK_TYPES: Dict[str, RiskTypeSpec] = {
    "TYPE1_STRUCTURAL_CREDIT": RiskTypeSpec(
        key="TYPE1_STRUCTURAL_CREDIT",
        name="1.구조적 신용/상환 리스크",
        short_desc="연체/다중부채/신용이력 축에서 위험 신호가 집중된 유형",
        checklist_questions=[
            "최근 6~12개월 내 연체가 반복되는 패턴이 있는가?",
            "동시에 유지 중인 부채/상환 의무가 과도한가?",
            "한도 초과/급격한 부채 증가 신호가 있는가?",
        ],
        suggested_actions=[
            "보수적 심사(조건 완화보다 리스크 제한) 또는 거절 유지 검토",
            "가능하면 부채 정리/상환 개선 후 재신청 안내",
        ],
    ),
    "TYPE2_DOCS_UNCERTAINTY": RiskTypeSpec(
        key="TYPE2_DOCS_UNCERTAINTY",
        name="2.서류/정보 불확실성",
        short_desc="정보 불일치·결측·신청 변동성 등 '확인으로 해소' 가능한 리스크",
        checklist_questions=[
            "신청 정보(직장/소득/연락처/거주) 일치 여부가 확인되는가?",
            "최근 신청/심사 과정에서 처리 지연/보완 요청이 반복되었는가?",
            "핵심 서류(재직/소득/거주)로 불확실성을 해소할 수 있는가?",
        ],
        suggested_actions=[
            "핵심 서류 1~2개로 불확실성 해소(과도한 서류 요구는 지양)",
            "확인 전화(근무/소득/연락가능성) + 내부 QC 체크리스트",
            "확인 완료 시 승인 전환(또는 조건부 승인) 시나리오",
        ],
    ),
    "TYPE3_SPENDING_IMBALANCE": RiskTypeSpec(
        key="TYPE3_SPENDING_IMBALANCE",
        name="3.소비/부채-소득 불균형",
        short_desc="소비성 지표/상환부담이 튀지만 신용이력 자체는 치명적이지 않은 유형",
        checklist_questions=[
            "최근 소비/카드 사용이 급증한 일시 이벤트가 있는가?",
            "소득 대비 월 상환 부담(또는 부채 부담)이 과도하지 않은가?",
            "대출 목적/사용처가 명확하며 상환 계획이 현실적인가?",
        ],
        suggested_actions=[
            "조건부 승인: 한도/금액 조정, 상환 기간 조정 등으로 부담 완화",
            "단기 모니터링(초기 1~3개월) 및 자동이체/상환관리 가이드",
        ],
    ),
    "TYPE4_EMPLOYMENT_LIFECYCLE": RiskTypeSpec(
        key="TYPE4_EMPLOYMENT_LIFECYCLE",
        name="4.고용/라이프사이클 리스크",
        short_desc="근속/직업 안정성·라이프 이벤트로 변동성이 있으나 확인 가능성이 높은 유형",
        checklist_questions=[
            "최근 이직/근속기간 변동 등으로 소득 안정성이 흔들렸는가?",
            "현재 소득 흐름(최근 입금/재직 상태)이 확인되는가?",
            "가족구성/거주 변화 등 일시적 이벤트가 있는가?",
        ],
        suggested_actions=[
            "재직/소득 흐름 확인 후 승인 전환 또는 조건부 승인",
            "상환 시작 시점/분할 구조 조정 등으로 초기 리스크 완화",
        ],
    ),
    "TYPE5_MIXED": RiskTypeSpec(
        key="TYPE5_MIXED",
        name="5.혼합형(추가 분해 필요)",
        short_desc="여러 축이 동시에 영향을 주는 케이스(개인 심사 페이지에서 상세 확인 권장)",
        checklist_questions=[
            "리스크가 특정 축 1~2개로 수렴하는가, 아니면 다축 분산인가?",
            "확인으로 해소 가능한 요소(정보/서류)와 구조적 요소(신용/부채)를 구분했는가?",
        ],
        suggested_actions=[
            "상위 리스크 축 1~2개를 중심으로 2차 평가 질문을 재정의",
            "개인 심사 페이지에서 SHAP 근거(상위 10개)를 확인 후 결정",
        ],
    ),
}


# ------------------------------------------------------------
# Classification rules
# ------------------------------------------------------------

# 기본 그룹명(너희 매핑에서 사용 중인 라벨을 우선)
GROUP_ALIASES = {
    "CREDIT": {"신용/상환이력", "연체이력", "타사대출이력", "카드/리볼빙", "할부상환"},
    "DOCS": {"서류/운영", "서류", "연락가능성"},
    "CAPACITY": {"부채·소득·상환여력", "부채부담", "소득여력", "신청규모"},
    "EMP": {"고용·직업 안정성", "고용안정성", "직군/업종"},
    "ASSET_REGION": {"거주/자산/지역", "거주형태", "자산보유", "지역/이동", "주거품질"},
}

# feature 키워드로 타입을 보조 판정(매핑이 완벽하지 않을 때 안전망)
FEATURE_KEYWORDS = {
    "SPENDING": {"cc_util", "revolving", "credit_to_goods", "goods_price", "amt_goods", "pre_credit_to_goods"},
    "CAPACITY": {"annuity", "payment_rate", "income", "debt", "dti", "ratio"},
    "EMP": {"days_employed", "years_employed", "occupation", "org", "employment"},
    "DOCS": {"flag_document", "document", "phone", "email", "contact", "process_start", "days_last_phone"},
}


def _group_pct_map(group_contribution_summary: Any) -> Dict[str, float]:
    """payload.group_contribution_summary -> {group: pct}"""
    out: Dict[str, float] = {}
    if isinstance(group_contribution_summary, list):
        for it in group_contribution_summary:
            if not isinstance(it, dict):
                continue
            g = it.get("super_group")
            p = it.get("risk_pct_of_top10")
            if g is None or p is None:
                continue
            try:
                out[str(g)] = float(p)
            except Exception:
                pass
    return out


def _match_group(name: str, alias_set: set) -> bool:
    return str(name) in alias_set


def _count_positive_drivers_in_groups(shap_top_10: Any, groups: set) -> int:
    n = 0
    if not isinstance(shap_top_10, list):
        return 0
    for it in shap_top_10:
        if not isinstance(it, dict):
            continue
        g = it.get("reason_group")
        v = it.get("shap")
        if g is None or v is None:
            continue
        try:
            if (str(g) in groups) and float(v) > 0:
                n += 1
        except Exception:
            continue
    return n


def _count_keyword_hits(shap_top_10: Any, keywords: set) -> int:
    n = 0
    if not isinstance(shap_top_10, list):
        return 0
    for it in shap_top_10:
        if not isinstance(it, dict):
            continue
        f = it.get("feature")
        if f is None:
            continue
        ff = str(f).lower()
        if any(k in ff for k in keywords):
            n += 1
    return n


def classify_review_payload(
    payload: Dict[str, Any],
    *,
    credit_dom_threshold: float = 45.0,
    docs_dom_threshold: float = 30.0,
    emp_dom_threshold: float = 25.0,
    capacity_dom_threshold: float = 35.0,
) -> Tuple[str, Dict[str, Any]]:
    """추가검토 고객 payload를 Risk Type으로 분류.

    반환:
      - risk_type_key
      - debug dict(우세 그룹/근거 수치)
    """
    group_summary = payload.get("group_contribution_summary") or payload.get("reason_contribution_summary") or []
    shap_top_10 = payload.get("shap_top_10") or payload.get("top_reasons") or []

    gp = _group_pct_map(group_summary)
    # dominant group
    dom_g, dom_p = (None, 0.0)
    if gp:
        dom_g, dom_p = max(gp.items(), key=lambda x: x[1])

    credit_alias = GROUP_ALIASES["CREDIT"]
    docs_alias = GROUP_ALIASES["DOCS"]
    emp_alias = GROUP_ALIASES["EMP"]
    cap_alias = GROUP_ALIASES["CAPACITY"]

    credit_p = max([gp.get(g, 0.0) for g in credit_alias] or [0.0])
    docs_p = max([gp.get(g, 0.0) for g in docs_alias] or [0.0])
    emp_p = max([gp.get(g, 0.0) for g in emp_alias] or [0.0])
    cap_p = max([gp.get(g, 0.0) for g in cap_alias] or [0.0])

    # 보조 signal: 키워드 히트
    kw_docs = _count_keyword_hits(shap_top_10, FEATURE_KEYWORDS["DOCS"])
    kw_spend = _count_keyword_hits(shap_top_10, FEATURE_KEYWORDS["SPENDING"])
    kw_cap = _count_keyword_hits(shap_top_10, FEATURE_KEYWORDS["CAPACITY"])
    kw_emp = _count_keyword_hits(shap_top_10, FEATURE_KEYWORDS["EMP"])

    # positive driver count (신용 그룹)
    pos_credit_cnt = _count_positive_drivers_in_groups(shap_top_10, credit_alias)

    # Rule 1: 구조적 신용/상환 리스크 (신용이 우세 + 위험↑ driver가 다수)
    if (credit_p >= credit_dom_threshold) and (pos_credit_cnt >= 2):
        rt = "TYPE1_STRUCTURAL_CREDIT"

    # Rule 2: 서류/정보 불확실성 (서류 우세 or 서류 키워드 다수)
    elif (docs_p >= docs_dom_threshold) or (kw_docs >= 2):
        rt = "TYPE2_DOCS_UNCERTAINTY"

    # Rule 3: 소비/부채-소득 불균형 (상환여력/소비 관련 signal)
    elif (cap_p >= capacity_dom_threshold) or (kw_spend >= 2) or (kw_cap >= 3):
        rt = "TYPE3_SPENDING_IMBALANCE"

    # Rule 4: 고용/라이프사이클
    elif (emp_p >= emp_dom_threshold) or (kw_emp >= 2):
        rt = "TYPE4_EMPLOYMENT_LIFECYCLE"

    else:
        rt = "TYPE5_MIXED"

    debug = {
        "dominant_group": dom_g,
        "dominant_pct": float(dom_p) if dom_g is not None else None,
        "credit_pct": float(credit_p),
        "docs_pct": float(docs_p),
        "capacity_pct": float(cap_p),
        "emp_pct": float(emp_p),
        "kw_docs": int(kw_docs),
        "kw_spending": int(kw_spend),
        "kw_capacity": int(kw_cap),
        "kw_emp": int(kw_emp),
        "pos_credit_cnt": int(pos_credit_cnt),
    }
    return rt, debug


def risk_type_display(rt_key: str) -> str:
    spec = RISK_TYPES.get(rt_key)
    return spec.name if spec else rt_key


def risk_type_guidance(rt_key: str) -> Dict[str, List[str]]:
    spec = RISK_TYPES.get(rt_key)
    if not spec:
        return {"checklist_questions": [], "suggested_actions": []}
    return {
        "checklist_questions": spec.checklist_questions,
        "suggested_actions": spec.suggested_actions,
    }
