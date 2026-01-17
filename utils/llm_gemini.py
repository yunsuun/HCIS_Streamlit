from __future__ import annotations
import os, json, time
from dotenv import load_dotenv

from typing import Any, Dict, List, Optional, Callable, Tuple, Type
from pathlib import Path
from pydantic import BaseModel, Field

USE_LLM = bool(os.getenv("GEMINI_API_KEY"))
if USE_LLM:
    from google import genai
    from google.genai import types

# .env 로딩
_CURRENT = Path(__file__).resolve()
ENV_PATH = _CURRENT.parent.parent / ".env" 
load_dotenv(dotenv_path=ENV_PATH)
# =========================================================
# LLM Output Schema + System Instructions (Underwriter/Customer)
# - 심사용: SHAP/값/Top10기여%까지 상세 노출 OK
# - 고객용: feature명/SHAP/내부 QC 등 내부정보 노출 금지
# =========================================================


# ---------------------------------------------------------
# 1) Response schema
#    - UnderwriterResponse: 심사팀용(내부 상세 근거 OK)
#    - CustomerResponse: 고객용(민감 수치/내부 로직 노출 금지)
# ---------------------------------------------------------

class UnderwriterResponse(BaseModel):
    # "확정" 표현을 피하고, "정책 기준상 ~구간" 톤으로 요약하도록 유도
    summary: str = Field(
        description=(
            "정책 기준상 band(승인/추가검토/거절) 구간 + 컷오프 대비 margin을 1문장으로 요약. "
            "확정 표현(예: '승인되었습니다/거절되었습니다') 금지."
        ),
        max_length=240,
    )

    # 그룹별 기여도는 payload.group_contribution_summary 기준
    reason_contributions: List[str] = Field(
        description=(
            "payload.group_contribution_summary(또는 reason_contribution_summary) 기반 상위 3~6개. "
            "형식: '<그룹명>: <pct>%' (pct는 risk_pct_of_top10 사용)"
        ),
        min_items=1,
        max_items=6,
    )

    # 리스크 요인은 반드시 top10 근거로(그룹만 보고 쓰지 않게)
    risk_drivers: List[str] = Field(
        description=(
            "리스크 요인 1~3개. 반드시 shap_top_10(=top_reasons) 상위 3개 feature 근거로 작성."
        ),
        min_items=1,
        max_items=3,
    )

    # 심사팀은 '기여도(값)'이 핵심이므로, 포맷을 강하게 고정(흔들림 방지)
    top_feature_rationales: List[str] = Field(
        description=(
            "shap_top_10(=top_reasons)에서만 3~5개 작성. "
            "반드시 아래 포맷을 지켜라:\n"
            "- <feature>(<reason_label>/<reason_group>) | 값=<value> | SHAP=<shap> | Top10기여=<risk_pct>% | 해석=<한줄>\n"
            "예: cc_util_mean(카드/리볼빙/신용/상환이력) | 값=0.73 | SHAP=+0.21 | Top10기여=18.40% | 카드 사용률이 높아 위험도 상승"
        ),
        min_items=3,
        max_items=5,
    )

    # 승인+마진 충분이면 빈 배열 허용(중요)
    verification_questions: List[str] = Field(
        description=(
            "심사팀 확인 질문 0~5개. "
            "policy.band='승인' AND policy.margin_score>=80이면 기본적으로 빈 배열([]). "
            "단, shap_top_10 상위에 '서류/운영' 관련이 직접 있을 때만 QC 체크 1~2개 허용."
        ),
        min_items=0,
        max_items=5,
    )

    suggested_actions_for_review: List[str] = Field(
        description="추가검토(REVIEW/추가검토 band)일 때만 제안. 그 외에는 빈 배열([]).",
        min_items=0,
        max_items=5,
    )

    customer_message_draft: str = Field(
        description=(
            "고객 안내문 초안(과장 금지). "
            "심사 내부용어(feature명, SHAP, QC, 내부 그룹명 등)는 직접 노출하지 말고, "
            "쉽게 설명하되 단정 금지."
        ),
        max_length=700,
    )

    disclaimer: str = Field(
        description="면책 문구(예: 제공된 정보 기준이며 추가 확인에 따라 변동 가능)",
        max_length=220,
    )


class CustomerResponse(BaseModel):
    summary: str = Field(
        description="현재 상태를 쉬운 말로 1문장(확정 표현 금지)",
        max_length=200,
    )

    reason_contributions: List[str] = Field(
        description=(
            "payload.group_contribution_summary 기준 상위 2~3개만 %로 표현. "
            "형식: '<사유(쉬운말)>: <pct>%'"
        ),
        min_items=2,
        max_items=3,
    )

    main_reasons: List[str] = Field(
        description=(
            "주요 사유 2~3개(쉬운 표현). "
            "feature명/SHAP/내부 그룹명/서류 QC 같은 내부 용어 금지."
        ),
        min_items=2,
        max_items=3,
    )

    # 고객용은 수치/feature명 노출 금지. '의미 중심'으로만.
    top_feature_rationales: List[str] = Field(
        description=(
            "근거 3~5개. "
            "숫자(값/SHAP/%), feature명, 내부 그룹명(서류/운영 등) 직접 언급 금지. "
            "대신 의미를 자연어로만 설명."
        ),
        min_items=3,
        max_items=5,
    )

    what_to_improve: List[str] = Field(
        description="개선 행동 1~3개(고객이 실행 가능한 수준)",
        min_items=1,
        max_items=3,
    )

    what_to_prepare: List[str] = Field(
        description="준비할 자료/확인사항 1~3개(내부 QC 표현 금지, 고객 요청 톤 과장 금지)",
        min_items=1,
        max_items=3,
    )

    disclaimer: str = Field(
        description="면책 문구(확정 금지, 추가 확인에 따라 변동 가능)",
        max_length=220,
    )


# ---------------------------------------------------------
# 2) System instructions
#    - 심사용/고객용을 반드시 분리
#    - payload 키 경로: policy.band, policy.margin_score / group_contribution_summary / shap_top_10
# ---------------------------------------------------------

SYSTEM_UNDERWRITER = """
당신은 금융 대출 심사팀을 보조하는 AI 어시스턴트입니다.
입력 JSON(payload)에 포함된 정보만 근거로 사용하세요.

[공통 원칙]
- 승인/거절을 확정적으로 표현하지 마세요. (예: "승인되었습니다/거절되었습니다" 금지)
- 입력에 없는 사실/숫자/원인을 만들지 마세요.
- 출력은 반드시 JSON만 반환하세요. (추가 텍스트/설명 금지)

[기여도 산출 원칙]
- reason_contributions(그룹별 기여도)는 아래 중 존재하는 것을 사용하세요:
  1) payload.group_contribution_summary  (권장, 위험도%: risk_pct_of_top10)
  2) payload.reason_contribution_summary (alias가 있을 경우)
- 퍼센트는 risk_pct_of_top10 값을 그대로 사용하세요. (임의 계산/추정 금지)

[승인/추가검토/거절 표현 규칙]
- summary는 아래 템플릿 중 하나를 따르세요(확정 표현 금지):
  - 승인: "정책 기준상 승인 구간(컷오프 {cutoff}점)이며 +{margin}점으로 마진이 충분해 승인 진행이 가능합니다."
  - 추가검토: "정책 기준상 추가검토 구간이며 컷오프 대비 {margin}점입니다. 추가 확인 후 진행을 권고드립니다."
  - 거절: "정책 기준상 거절 구간이며 컷오프 대비 {margin}점입니다. 재평가를 위해 추가 정보 확인이 필요할 수 있습니다."
  (cutoff=policy.cutoff_score, margin=policy.margin_score)

[승인 & 마진 충분 시 확인 질문 규칙]
- policy.band가 "승인"이고 policy.margin_score >= 80 이면,
  verification_questions는 기본적으로 빈 배열([])로 출력하세요.
- 단, shap_top_10(=top_reasons) 상위 항목에 '서류/운영' 관련 피처가 직접적으로 포함될 때만
  내부 QC(운영점검) 체크리스트 형태로 1~2개를 허용합니다.
  (고객에게 추가자료를 요구하는 톤 금지)

[SHAP 활용 규칙]
- risk_drivers는 shap_top_10의 상위 3개 feature 기반으로만 작성하세요.
- reason_contributions는 참고용이며, risk_drivers/verification_questions의 근거로 단독 사용하지 마세요.
- top_feature_rationales는 shap_top_10의 value/shap/risk_pct_of_top10을 반드시 포함해,
  스키마 설명의 고정 포맷을 지키세요.
  
[행태 기반 해석]
- payload.behavioral_insights가 존재하면,
  risk_drivers 또는 customer_message_draft에
  이를 바탕으로 고객의 금융/소비/상환 행태를 2~4문장으로 설명하세요.
- 단, behavioral_insights에 없는 사실을 새로 만들지 마세요.

[고객 안내문]
- customer_message_draft에는 feature명/SHAP/내부 QC/내부 그룹명 같은 내부 용어를 직접 노출하지 마세요.
""".strip()

# ---------------------------------------------------------
# 3-1) Underwriter band별 Prompt (중복 제거 + 역할 분리)
# - SYSTEM_UNDERWRITER는 "규칙"
# - 여기 PROMPT는 "요약/해석/액션을 어떻게 쓸지" 지시 (band별로 톤/초점 다르게)
# ---------------------------------------------------------

UNDERWRITER_PROMPT_APPROVE = """
당신은 심사팀용 Underwriter AI입니다. 출력은 UnderwriterResponse 스키마(JSON)만 반환하세요.

[중복 제거 원칙]
- summary는 1문장 결론으로만 작성하고, 아래 필드에서 같은 말을 반복하지 마세요.
- reason_contributions에는 그룹명: pct% 목록만 간결하게.
- risk_drivers / top_feature_rationales에서는 shap_top_10 상위 근거로만 작성.

[승인(Approve) 톤/초점]
- 해석은 "왜 승인 구간인지"를 짧게 정리하되, 과장/단정 금지.
- verification_questions는 규칙에 따라 기본적으로 비워두세요(승인 & 마진 충분 시).
- suggested_actions_for_review는 반드시 빈 배열([]).
""".strip()

UNDERWRITER_PROMPT_REVIEW = """
당신은 심사팀용 Underwriter AI입니다. 출력은 UnderwriterResponse 스키마(JSON)만 반환하세요.

[중복 제거 원칙]
- summary는 1문장으로만. 숫자/퍼센트/컷오프를 risk_drivers에서 반복 금지.
- reason_contributions에는 그룹명: pct% 목록만 간결하게.
- risk_drivers / top_feature_rationales는 반드시 shap_top_10 상위 근거로만.

[추가검토(Review) 톤/초점: 가장 중요]
- 이 케이스는 "즉시 승인도 거절도 아닌 경계"이므로,
  '확인하면 승인 가능'한 포인트와 '확인해도 위험한 포인트'를 구분해 서술하세요.
- verification_questions는 승인/거절을 가르는 핵심 질문 위주로 2~5개.
- suggested_actions_for_review에는 "추가서류/확인전화/조건부 승인 조건"을 2~5개로 구체적으로 제시하세요.
- customer_message_draft는 고객에게 부담을 주지 않는 톤으로, 단정 금지.
""".strip()

UNDERWRITER_PROMPT_REJECT = """
당신은 심사팀용 Underwriter AI입니다. 출력은 UnderwriterResponse 스키마(JSON)만 반환하세요.

[중복 제거 원칙]
- summary는 1문장 결론으로만.
- reason_contributions는 그룹명: pct% 목록만 간결하게.
- risk_drivers / top_feature_rationales는 shap_top_10 상위 근거로만 작성.

[거절(Reject) 톤/초점]
- 단정/비난 금지. 리스크 관리 관점에서 "왜 지금은 어렵다"를 설명.
- verification_questions는 '재검토 가능성'이 있는 항목만 제한적으로.
- suggested_actions_for_review는 빈 배열([]).
- customer_message_draft에는 내부 용어/feature/SHAP 노출 금지.
""".strip()

UNDERWRITER_PROMPT_BY_BAND = {
    "승인": UNDERWRITER_PROMPT_APPROVE,
    "추가검토": UNDERWRITER_PROMPT_REVIEW,
    "거절": UNDERWRITER_PROMPT_REJECT,
}


SYSTEM_CUSTOMER = """
당신은 금융 대출 심사 결과를 고객에게 쉽게 설명하는 안내 AI입니다.
입력 JSON(payload)에 포함된 정보만 근거로 사용하세요.

[공통 원칙]
- 승인/거절을 확정적으로 표현하지 마세요. (예: "승인되었습니다/거절되었습니다" 금지)
- 입력에 없는 사실/숫자/원인을 만들지 마세요.
- 출력은 반드시 JSON만 반환하세요. (추가 텍스트/설명 금지)

[민감 정보/내부 로직 노출 금지]
- 절대 금지: feature명, SHAP 값, risk_pct_of_top10 수치, 내부 QC, 내부 그룹명(예: 서류/운영 등) 직접 언급
- 숫자는 고객이 이해 가능한 범위에서만 최소 사용하고, 내부 지표/모델 수치는 숨기세요.

[설명 방식]
- summary: 고객이 이해하기 쉬운 말로 현재 상태를 1문장(단정 금지)
- reason_contributions: payload.group_contribution_summary 기반 상위 2~3개만 선택하여 %로 표현하되,
  그룹명을 그대로 쓰지 말고 고객 친화적으로 바꿔서 작성하세요.
- main_reasons / top_feature_rationales: 수치/컬럼명 없이 의미 중심으로 설명하세요.
- what_to_improve / what_to_prepare: 고객이 실행 가능한 행동/준비물로 작성하세요.
""".strip()




# Gemini client
def get_gemini_client() -> Tuple[genai.Client, str]:
    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    if not api_key:
        return None, model
    return genai.Client(api_key=api_key), model

# 기본값으로 사용될 MOCK 데모모드
def mock_underwriter_response(payload_llm: dict) -> dict:
    policy = payload_llm.get("policy", {}) or {}
    band = policy.get("band", "추가검토")
    margin = policy.get("margin_score", 0)

    # 그룹 기여도(있으면 상위 3~5개 사용)
    contrib = payload_llm.get("group_contribution_summary") or payload_llm.get("reason_contribution_summary") or []
    reason_contributions = []
    if isinstance(contrib, list):
        reason_contributions = [str(x) for x in contrib[:5]]

    # shap_top_10 있으면 상위 2~3개를 리스크 요인으로
    top10 = payload_llm.get("shap_top_10") or []
    risk_drivers = []
    top_feature_rationales = []
    if isinstance(top10, list) and top10:
        for item in top10[:3]:
            f = item.get("feature")
            v = item.get("value")
            s = item.get("shap")
            g = item.get("reason_group")
            r = item.get("reason_label")
            pct = item.get("risk_pct_of_top10")
            risk_drivers.append(f"{r or f} 관련 요인이 상대적으로 크게 작용했습니다.")

            # 스키마 포맷과 유사하게(심사용)
            top_feature_rationales.append(
                f"{f}({r}/{g}) | 값={v} | SHAP={s} | Top10기여={pct}% | 해석=해당 요인이 위험도에 기여"
            )

    # band별 확인 질문/액션(간단 버전)
    verification_questions = []
    suggested_actions_for_review = []

    if band == "추가검토":
        verification_questions = [
            "최근 소득/재직 변동 여부를 확인할 수 있을까요?",
            "현재 부채 및 월 상환 부담에 대한 추가 확인이 필요합니다.",
        ]
        suggested_actions_for_review = [
            "추가 소득증빙/재직증빙 확인",
            "부채현황 재확인 후 조건부 승인 검토",
        ]
    elif band == "거절":
        verification_questions = [
            "최근 연체/상환 이력에 대한 추가 확인이 필요할 수 있습니다."
        ]

    # llm_report.py가 읽는 summary/reason_contributions/verification_questions 키를 맞춰줌 :contentReference[oaicite:3]{index=3}
    return {
        "_mode": "demo",  # UI에서 배지 띄우기 용도
        "summary": f"🧪 데모 모드: 정책 기준상 {band} 구간이며 컷오프 대비 {margin:+.1f}점입니다. (API Key 미설정)",
        "reason_contributions": reason_contributions or ["(데모) 위험 기여도 요약을 표시합니다."],
        "risk_drivers": risk_drivers or ["(데모) 주요 요인 요약"],
        "top_feature_rationales": top_feature_rationales or ["(데모) 상세 근거는 API 연결 시 제공됩니다."],
        "verification_questions": verification_questions,
        "suggested_actions_for_review": suggested_actions_for_review,
        "customer_message_draft": "현재는 데모 모드로 운영되어 안내 문구가 간략히 표시됩니다.",
        "disclaimer": "본 내용은 데모 출력이며, 실제 심사 결과는 추가 확인에 따라 달라질 수 있습니다.",
    }


# core runner
def run_gemini_structured(
    case_payload: Dict[str, Any],
    schema: Type[BaseModel],                    
    system_instruction: str,                      
    prompt: str,                                 
    client: Optional[genai.Client] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_output_tokens: int = 900,
) -> Dict[str, Any]:
    if client is None or model_name is None:
        client, model_name = get_gemini_client()

    generation_config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
        response_mime_type="application/json",
        response_schema=schema,             
        system_instruction=system_instruction,
    )

    contents = f"""{prompt}

<INPUT_JSON>
{json.dumps(case_payload, ensure_ascii=False)}
</INPUT_JSON>
"""

    resp = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=generation_config,
    )

    parsed = resp.parsed
    if parsed is None:
        raise RuntimeError("Gemini 응답 파싱 실패: response.parsed is None")
    return parsed.model_dump()

# llm을 위한 shap_bundle 정규화
def normalize_payload_for_llm(payload: dict) -> dict:
    """
    hcis_core payload(**shap_bundle 포함)을 LLM이 쓰기 쉬운 형태로 표준화

    표준 shap_top_10 형식:
      [{feature, shap, value, reason_label, reason_group, risk_pct_of_top10}]
    """
    p = dict(payload)
    top10 = None  # ✅ NameError 방지

    # ------------------------------------------------------------
    # 1) (최우선) hcis_core의 top_reasons -> shap_top_10
    #    top_reasons 원본 키(권장):
    #      feature, feature_value, shap_value, reason_label, super_group, risk_pct_of_top10
    # ------------------------------------------------------------
    if isinstance(p.get("top_reasons"), list) and p["top_reasons"]:
        normalized = []
        for item in p["top_reasons"][:10]:
            if not isinstance(item, dict):
                continue
            normalized.append({
                "feature": item.get("feature"),
                "shap": item.get("shap_value"),               # FIX
                "value": item.get("feature_value"),           # FIX
                "reason_label": item.get("reason_label"),
                "reason_group": item.get("super_group"),
                "risk_pct_of_top10": item.get("risk_pct_of_top10")
            })
        p["shap_top_10"] = normalized

        # alias 정리(둘 중 뭐가 오든 프롬프트에서 쓰기 쉽게)
        if "reason_contribution_summary" not in p and isinstance(p.get("group_contribution_summary"), list):
            p["reason_contribution_summary"] = p["group_contribution_summary"]

        return p

    # ------------------------------------------------------------
    # 2) fallback A: 업로드 결과(너희 방식) - shap_features/shap_values 리스트
    #    (row 기반 payload에 그대로 들어오는 경우)
    # ------------------------------------------------------------
    feats = p.get("shap_features")
    vals  = p.get("shap_values")

    if isinstance(feats, list) and isinstance(vals, list) and len(feats) == len(vals) and len(feats) > 0:
        top10 = [{"feature": f, "shap": v} for f, v in zip(feats[:10], vals[:10])]

    # ------------------------------------------------------------
    # 3) fallback B: 다른 키 후보(프로젝트/팀 산출물 다양성 대응)
    # ------------------------------------------------------------
    if top10 is None:
        feats = p.get("top_features") or p.get("features_top") or p.get("shap_top_features")
        vals  = p.get("top_values")   or p.get("values_top")   or p.get("shap_top_values")
        if isinstance(feats, list) and isinstance(vals, list) and len(feats) == len(vals) and len(feats) > 0:
            top10 = [{"feature": f, "shap": v} for f, v in zip(feats[:10], vals[:10])]

    # ------------------------------------------------------------
    # 4) reason/그룹/값 매핑이 있으면 enrich
    # ------------------------------------------------------------
    feat_to_reason = p.get("feature_reason_map") or {}
    feat_to_group  = p.get("feature_group_map") or {}
    feat_to_value  = p.get("feature_value_map") or {}

    if isinstance(top10, list):
        normalized = []
        for item in top10[:10]:
            f = item.get("feature") if isinstance(item, dict) else None
            normalized.append({
                "feature": f,
                "shap": item.get("shap") if isinstance(item, dict) else None,
                "value": item.get("value") if isinstance(item, dict) else feat_to_value.get(f),
                "reason_label": item.get("reason_label") if isinstance(item, dict) else feat_to_reason.get(f),
                "reason_group": item.get("reason_group") if isinstance(item, dict) else feat_to_group.get(f),
                "risk_pct_of_top10": item.get("risk_pct_of_top10") if isinstance(item, dict) else None,
            })
        p["shap_top_10"] = normalized

    return p



# 503 retry wrapper
def run_with_retry(
    fn: Callable[[], Dict[str, Any]],
    max_retries: int = 5,
    base_delay: float = 1.5,
) -> Dict[str, Any]:
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as e:
            msg = str(e)
            if any(k in msg for k in ["503", "Service Unavailable", "temporarily unavailable"]):
                wait = base_delay * (2 ** (attempt - 1))
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("Gemini 호출 실패: 재시도 횟수 초과")


# 심사용 실행
def ask_underwriter(payload: dict) -> dict:

    # api key 없으면 Mock 실행
    if not USE_LLM:
        return mock_underwriter_response(payload_llm)
    
    client, model = get_gemini_client()
    payload_llm = normalize_payload_for_llm(payload)

    
    # shap 확인여부
    if not payload_llm.get("shap_top_10"):
        raise RuntimeError("SHAP(top10) 정보가 payload에 없습니다. 업로드/추론 단계에서 shap_features/shap_values 저장 여부를 확인하세요.")
    
    band = (payload_llm.get("policy", {}) or {}).get("band", "")
    prompt_band = UNDERWRITER_PROMPT_BY_BAND.get(band, UNDERWRITER_PROMPT_REVIEW)

    def _call():
        return run_gemini_structured(
            case_payload=payload_llm,
            schema=UnderwriterResponse,
            system_instruction=SYSTEM_UNDERWRITER,
            prompt=prompt_band,
            client=client,
            model_name=model,
            temperature=0.3,        # 보통 0.3까지가 규칙적인 답변
            top_p=0.9,              # 0.3이면 매우 보수적, 같은 단어만 반복. 0.9면 자연스러움 추가
            max_output_tokens=1000, # 단순하게 글자 제한이라 보고 우리가 원하는거 다 출력 가능한 수치로 지정
        )
    return run_with_retry(_call, max_retries=4, base_delay=1.2)

