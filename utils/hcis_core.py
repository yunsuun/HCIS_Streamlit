import numpy as np
import pandas as pd
import ast

from config import OFFSET, FACTOR, T_LOW, T_HIGH, PD_CEIL, PD_FLOOR, TOP_N, SCORE_MAX, SCORE_MIN
from pathlib import Path
from typing import Dict, Any
from typing import Dict, Any
import pandas as pd

# supergroup 선언
SUPER_GROUP_MAP = {
    "외부평점":"신용/상환이력",
    "연체이력":"신용/상환이력",
    "카드/리볼빙":"신용/상환이력",
    "할부상환":"신용/상환이력",
    "소액/소비대출":"신용/상환이력",
    "타사대출이력":"신용/상환이력",
    "과거신청이력":"신용/상환이력",
    "부채부담":"부채·소득·상환여력",
    "소득여력":"부채·소득·상환여력",
    "신청규모":"부채·소득·상환여력",
    "고용안정성":"고용·직업 안정성",
    "직군/업종":"고용·직업 안정성",
    "신분안정성":"고용·직업 안정성",
    "인구통계":"인구통계·가구·교육",
    "학력":"인구통계·가구·교육",
    "가구구성":"인구통계·가구·교육",
    "거주형태":"거주/자산/지역",
    "주거품질":"거주/자산/지역",
    "자산보유":"거주/자산/지역",
    "지역/이동":"거주/자산/지역",
    "서류":"서류/운영",
    "연락가능성":"서류/운영",
    "신청시점":"서류/운영",
    "기타":"서류/운영",
}

# supergroup 매핑 함수
def load_mapping_enriched(mapping_path: Path) -> pd.DataFrame:
    if mapping_path.suffix.lower() == ".parquet":
        m = pd.read_parquet(mapping_path)                               # 매핑용 파케이 파일 만들어 둚.
    elif mapping_path.suffix.lower() in [".xlsx", ".xls"]:              # 혹시나 엑셀로 저장해둔 경우 문제 없이 실행하기 위함
        m = pd.read_excel(mapping_path)
    else:
        raise ValueError(f"지원하지 않는 mapping 형식: {mapping_path}")

    # 혹시 모를 공백 제거
    m.columns = m.columns.map(lambda x: str(x).strip())
    
    # 컬럼명 새로 지정
    rename = {}
    if "컬럼명" in m.columns: rename["컬럼명"] = "feature"
    if "reason_label_ko" in m.columns: rename["reason_label_ko"] = "reason_label"
    m = m.rename(columns=rename)                                       

    # 필수 컬럼 없으면 에러 코드
    if "feature" not in m.columns or "reason_label" not in m.columns:
        raise KeyError(f"mapping 파일에 feature/reason_label 컬럼 필요. 현재: {m.columns.tolist()}")

    m["feature"] = m["feature"].astype(str)
    m["reason_label"] = m["reason_label"].astype(str)
    
    # 매핑할게 없다면 일단 '기타'가 포함된 '서류/운영'으로 지정
    m["super_group"] = m["reason_label"].map(SUPER_GROUP_MAP).fillna("서류/운영")

    # 컬럼명, 그룹, 슈퍼그룹 반환
    return m[["feature","reason_label","super_group"]].drop_duplicates("feature").reset_index(drop=True)

#  supergroup 매핑 실행
def build_map_dict(mapping_path: Path) -> Dict[str, Dict[str, str]]:
    df = load_mapping_enriched(mapping_path)
    return df.set_index("feature")[["reason_label","super_group"]].to_dict("index")

# pd_hat -> hcis 점수 계산
def pd_to_hcis(pd_hat: float, offset: float, factor: float) -> float:
    pd_hat = float(pd_hat)
    pd_hat = min(max(pd_hat, 1e-6), 1-1e-6)
    odds = (1 - pd_hat) / pd_hat
    return float(offset + factor * np.log(odds))

# hcis 점수 기준 승인, 추가검토, 거절로 구분
def hcis_band(score: float, t_low: float, t_high: float) -> str:
    if score < t_low: return "거절"
    if score < t_high: return "추가검토"
    return "승인"

# 혹시나 shap이 정상적이지 않더라도 정상작동하도록 안전장치

def compute_hcis_columns(
    df: pd.DataFrame,
    *,
    pd_col: str = "pd_hat",
    out_score_col: str = "hcis_score",
    out_band_col: str = "band",
    out_cutoff_col: str = "cutoff_score",
    out_margin_col: str = "margin_score",
    t_low: float = T_LOW,
    t_high: float = T_HIGH,
    offset: float = OFFSET,
    factor: float = FACTOR,
    pd_floor: float = PD_FLOOR,
    pd_ceil: float = PD_CEIL,
    score_min: float = SCORE_MIN,
    score_max: float = SCORE_MAX,
) -> pd.DataFrame:
    """배치용 HCIS 계산 컬럼 생성.

    - PD_hat 클리핑 (pd_floor~pd_ceil) 후 score 산출
    - band: '거절'/'추가검토'/'승인'
    - cutoff_score: band에 따라 t_low 또는 t_high
    - margin_score: hcis_score - cutoff_score

    반환: 원본 df 복사본에 컬럼을 추가한 DataFrame
    """
    if pd_col not in df.columns:
        raise KeyError(f"'{pd_col}' 컬럼이 없습니다.")

    out = df.copy()
    pd_series = out[pd_col].astype(float).copy()

    # 정책 클리핑
    pd_series = pd_series.clip(lower=pd_floor, upper=pd_ceil)

    # 점수 계산
    out[out_score_col] = pd_series.apply(lambda p: pd_to_hcis(p, offset, factor)).astype(float)
    
    # 점수 클리핑
    out[out_score_col] = out[out_score_col].clip(
        lower=score_min,
        upper=score_max
    )
    # band/컷/마진
    out[out_band_col] = out[out_score_col].apply(lambda s: hcis_band(float(s), t_low, t_high))
    out[out_cutoff_col] = out[out_band_col].apply(lambda b: float(t_high) if b == "승인" else float(t_low))
    out[out_margin_col] = (out[out_score_col] - out[out_cutoff_col]).round(2)

    return out

def _coerce_listlike(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    if isinstance(x, str):
        try:
            v = ast.literal_eval(x)
            if isinstance(v, (list, tuple, np.ndarray)):
                return list(v)
        except Exception:
            return None
    return None

# shap_top10, payload로
def build_top10_shap_bundle(
    row: pd.Series,                         # (고객 한 명의 데이터)
    map_dict: Dict[str, Dict[str, str]],    # 예시) ext_source_2, ['외부평점', '신용이력']
    top_features_col: str,                  # 예시) [app_income, bu_total_dept, ...]
    top_values_col: str,                    # 예시) [-0.31, -0.21, 0.15, ...]
    top_n_use: int = TOP_N                  # shap 개수가 10개
) -> Dict[str, Any]:
    
    # 아예 없으면 빈 결과 반환
    feats_raw = row.get(top_features_col, None)
    vals_raw  = row.get(top_values_col, None)
    if feats_raw is None or vals_raw is None:
        return {
            "top_reasons": [],
            "group_contribution_summary": [],
            "shap_top_10": [],                 # ✅ 추가 (LLM 표준 키)
            "top_reasons_public": [],          # ✅ 추가 (고객용)
            "other_note": f"{top_features_col}/{top_values_col} missing -> no SHAP bundle"
        }

    feats = _coerce_listlike(feats_raw)         # 예시) ["ext_source_2", "app_income_total", ...]
    vals  = _coerce_listlike(vals_raw)          # 예시) [-0.31, -0.21, ...] => 둘다 순서대로 되어있음. 1등~10등순

    # 변환 결과도 비정상이면 빈 결과 반환
    if feats is None or vals is None:
        return {
            "top_reasons": [],
            "group_contribution_summary": [],
            "shap_top_10": [],
            "top_reasons_public": [],
            "other_note": "coerce failed -> no SHAP bundle"
        }
    
    # 타입 검증 (리스트가 맞고 갯수가 맞냐)
    if not isinstance(feats, (list, tuple)) or not isinstance(vals, (list, tuple, np.ndarray)):
        raise TypeError(f"{top_features_col}/{top_values_col} must be list-like")
    if len(feats) != len(vals):
        raise ValueError(f"len(feats)={len(feats)} != len(vals)={len(vals)}")

    # list로 된 두 변수 df로 전환
    df = pd.DataFrame({
        "feature": list(map(str, feats)),
        "shap_value": np.array(vals, dtype=float)
    })

    # 절대값으로 계산하기 위한 컬럼 추가
    df["shap_abs"] = df["shap_value"].abs()

    # feature마다 그에 맞는 값 매핑 (shap이 아닌 실제 컬럼의 값)
    df["feature_value"] = df["feature"].map(lambda f: row.get(f, None))

    # label 분리
    df["reason_label"] = df["feature"].map(
        lambda f: map_dict.get(f, {}).get("reason_label", "기타")
    )

    # supergroup 분리
    df["super_group"] = df["feature"].map(
        lambda f: map_dict.get(f, {}).get("super_group", "서류/운영")
    )

    # 정렬되어 있었겠지만 최종적으로 함 더 정렬
    df = df.sort_values("shap_abs", ascending=False).reset_index(drop=True)

    # 나누기 할 때 0으로 나뉘어지는 것 방지
    total_abs = float(df["shap_abs"].sum()) or 1.0

    # shap 기여도를 퍼센트로 변경
    df_top = df.head(top_n_use).copy()   # 위에서 shap_abs로 역순 정렬하였으니 head(10)이 shap top10.
    df_top["risk_pct_of_top10"] = df_top["shap_abs"] / total_abs * 100

    # ===========================
    # (추가) numpy → python scalar 변환 (LLM/JSON 안정성)
    # ===========================
    def _to_py(x):
        try:
            if pd.isna(x):
                return None
        except Exception:
            pass
        if hasattr(x, "item"):
            try:
                return x.item()
            except Exception:
                return x
        return x

    df_top["feature_value"] = df_top["feature_value"].map(_to_py)
    df_top["shap_value"] = df_top["shap_value"].map(lambda x: float(x))
    df_top["risk_pct_of_top10"] = df_top["risk_pct_of_top10"].map(lambda x: round(float(x), 2))

    # ===========================
    # 반환값
    # ===========================

    # 기존 반환 (유지)
    top_reasons = df_top[[
        "feature",              # 예시) "ext_source_2"
        "reason_label",         # 예시) "외부평점"
        "super_group",          # 예시) "신용/상환이력"
        "feature_value",        # 예시) 0.43  => 해당 고객의 ext_source_2 값
        "shap_value",           # 예시) -0.31 => shap 값
        "shap_abs",             # 예시) 0.31
        "risk_pct_of_top10"     # 예시) 18.4
    ]].to_dict("records")

    # ===========================
    # (추가) LLM 표준 키 (심사용)
    # ===========================
    shap_top_10 = []
    for r in top_reasons:
        shap_top_10.append({
            "feature": r.get("feature"),
            "value": r.get("feature_value"),
            "shap": r.get("shap_value"),
            "risk_pct_of_top10": r.get("risk_pct_of_top10"),
            "reason_label": r.get("reason_label"),
            "reason_group": r.get("super_group"),
        })

    # ===========================
    # (추가) 고객용 bundle (값/SHAP 수치 제거)
    # ===========================
    top_reasons_public = []
    for r in top_reasons:
        top_reasons_public.append({
            "reason_label": r.get("reason_label"),
            "reason_group": r.get("super_group"),
            # 고객에게 feature명을 숨기고 싶으면 이 줄 제거 가능
            "summary_hint": r.get("feature"),
        })

    # ===========================
    # shap top 10 요약 (기존 로직 유지)
    # ===========================
    g = df_top.groupby("super_group")["shap_abs"].sum().reset_index()
    g["risk_pct_of_top10"] = g["shap_abs"] / float(df_top["shap_abs"].sum() or 1.0) * 100
    g = g.sort_values("risk_pct_of_top10", ascending=False)
    g["risk_pct_of_top10"] = g["risk_pct_of_top10"].map(lambda x: round(float(x), 2))
    group_summary = g[["super_group", "risk_pct_of_top10"]].to_dict("records")

    return {
        "top_reasons": top_reasons,                 # 기존 유지
        "group_contribution_summary": group_summary,
        "shap_top_10": shap_top_10,                 # 심사용 (값 + SHAP + %)
        "top_reasons_public": top_reasons_public,   # 고객용 (수치 제거)
        "other_note": "group_contribution_summary는 SHAP Top10 범위 내 절대값 합 기준"
    }


# ai에 필요한 payload 빌드 실행
def build_payload_from_team_row(
    row: pd.Series,
    map_dict: Dict[str, Dict[str, str]],
    *,
    # 컬럼명은 여전히 외부에서 바꿀 수 있게 유지
    id_col: str = "sk_id_curr",
    pd_col: str = "pd_hat",
    top_features_col: str = "shap_features",   
    top_values_col: str = "shap_values",       
    top_n_use: int = TOP_N,

    # 정책값은 config를 기본값으로
    t_low: float = T_LOW,
    t_high: float = T_HIGH,
    offset: float = OFFSET,
    factor: float = FACTOR,

    # PD 클리핑도 config 기반
    pd_floor: float = PD_FLOOR,
    pd_ceil: float = PD_CEIL,
) -> Dict[str, Any]:

    sk = int(row[id_col])
    pd_hat = float(row[pd_col])

    # PD 안정화(클리핑) - config 기준
    if pd_hat < pd_floor:
        pd_hat = pd_floor
    elif pd_hat > pd_ceil:
        pd_hat = pd_ceil

    # hcis 점수 계산
    hcis = pd_to_hcis(pd_hat, offset=offset, factor=factor)

    # ✅ 점수 정책 클리핑
    hcis = min(max(hcis, SCORE_MIN), SCORE_MAX)

    # 점수 기반 그룹 나누기
    band = hcis_band(hcis, t_low=t_low, t_high=t_high)

    # 속한 그룹의 컷오프 (안전한 그룹이면 high 컷, 아니면 low 컷)
    cutoff = t_high if band == "승인" else t_low
    # 컷오프와 차이 계산 ('거절' 고객은 마이너스로 나옴)
    margin = hcis - cutoff

    # shap top_10 번들 실행
    shap_bundle = build_top10_shap_bundle(
        row=row,
        map_dict=map_dict,
        top_features_col=top_features_col,
        top_values_col=top_values_col,
        top_n_use=top_n_use,
    )

    # =======================
    # 반환값
    # =======================
    payload = {
        "sk_id_curr": sk,
        "pd_hat": pd_hat,
        "hcis_score": hcis,
        "policy": {
            "T_LOW": t_low,
            "T_HIGH": t_high,
            "band": band,
            "cutoff_score": cutoff,
            "margin_score": round(margin, 2),
        },
        **shap_bundle,
    }

    return payload
