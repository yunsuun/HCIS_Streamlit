# ===========================================================
# 02_간편_조회.py — 고객 단건 심사 (성능 개선 + 시각화 강화)
# 캐싱 최적화
# Score, Grade, Decision 시각화
# 불필요한 재계산 제거
# (추가) st_data/model_df.parquet 우선 로드
# (추가) hcis_score/band 없으면 compute_hcis_columns로 생성
# ===========================================================

import streamlit as st
import numpy as np
import pandas as pd
import streamlit.components.v1 as components

from pathlib import Path
from config import (
    APP_TITLE, ID_COL, OFFSET, FACTOR, T_LOW, T_HIGH,
    MODEL_DF_PARQUET, MAPPING_PATH, SCORE_MIN, SCORE_MAX, TOP_N
)
from utils.llm_report import render_underwriter_report
from utils.shap_reason import get_top_reasons_from_shap_row
from utils.hcis_core import build_map_dict, build_payload_from_team_row, compute_hcis_columns
from utils.behavioral_insights import generate_behavioral_insights
from utils.llm_gemini import ask_underwriter

st.markdown("""
<style>
.block-container {
    padding-top: 2rem !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# Page config
# -----------------------------------------------------------
st.set_page_config(page_title=f"{APP_TITLE} | 고객 심사", layout="wide")

# -----------------------------------------------------------
# 캐싱된 데이터 로드 및 전처리
# -----------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="데이터 로딩 중...")
def load_df_work(data_path):
    df = pd.read_parquet(data_path)
    df[ID_COL] = df[ID_COL].astype(str)  # 검색 안정화
    return df

# -----------------------------------------------------------
# 운영 테이블 우선 로드 (개요 Tab4 업로드 결과: st_data/model_df.parquet)
# -----------------------------------------------------------
ST_DATA_DF_PARQUET = Path("st_data") / "model_df.parquet"
DATA_SRC = None

if ST_DATA_DF_PARQUET.exists():
    DATA_SRC = f"st_data ({ST_DATA_DF_PARQUET.as_posix()})"
    df_work = load_df_work(ST_DATA_DF_PARQUET)
else:
    DATA_SRC = f"config ({Path(MODEL_DF_PARQUET).as_posix()})"
    df_work = load_df_work(MODEL_DF_PARQUET)

# -----------------------------------------------------------
# HCIS 컬럼이 없으면 공통 로직으로 생성 (개요/대출심사 일관성 보장)
# -----------------------------------------------------------
if ("hcis_score" not in df_work.columns) or ("band" not in df_work.columns):
    # pd_hat 컬럼명이 다르면 여기만 바꾸면 됨
    df_work = compute_hcis_columns(df_work, pd_col="pd_hat")

# -----------------------------------------------------------
# UI
# -----------------------------------------------------------
st.title("👤 대출 심사 조회")
st.caption("본 화면은 고객 간 상대 비교가 아닌, 내부 점수 체계 기준 화면입니다.")
st.caption(f"데이터 소스: `{DATA_SRC}`")

# -----------------------------------------------------------
# 고객 선택 (사이드바)
# -----------------------------------------------------------
id_list = df_work[ID_COL].dropna().astype(str).unique().tolist()

with st.sidebar:
    st.subheader("🔍 고객 검색")
    selected_id = st.text_input("고객 ID를 입력하세요 (6자리)", max_chars=6)

    if selected_id:
        if not selected_id.isdigit() or len(selected_id) != 6:
            st.warning("❌ 6자리 숫자만 입력 가능합니다")
            selected_id = None
        elif selected_id not in id_list:
            st.info("⚠️ 해당 ID가 존재하지 않습니다")
            selected_id = None
    else:
        selected_id = None

if selected_id is None:
    st.info("👆 사이드바에서 고객 ID를 입력해주세요")
    st.stop()

# -----------------------------------------------------------
# 고객 데이터 추출 및 계산 (캐싱)
# -----------------------------------------------------------
@st.cache_resource
def get_map_dict(mapping_path: str):
    return build_map_dict(Path(mapping_path))

@st.cache_data(show_spinner=False)
def get_customer_analysis(df: pd.DataFrame, cid, mapping_path):
    """
    고객 데이터 추출 + HCIS payload 생성까지 한 번에 처리 (캐싱)
    반환:
      - row_dict: 고객 row (dict)
      - payload: hcis_core payload (dict)
      - score: hcis_score
      - band: 거절/추가검토/승인
      - action: UI용 텍스트
      - pos_pct: SCORE_MIN~MAX 기준 위치(%)
      - margin: cutoff 대비 마진
    """
    df_idx = df.copy()
    df_idx[ID_COL] = df_idx[ID_COL].astype(str)
    cid = str(cid)

    matched = df_idx[df_idx[ID_COL] == cid]
    if matched.empty:
        raise KeyError(f"{ID_COL}={cid} 고객을 찾지 못했습니다.")
    if len(matched) > 1:
        matched = matched.iloc[[0]]

    row_series = matched.iloc[0]
    row_dict = row_series.to_dict()

    map_dict = build_map_dict(mapping_path)

    payload = build_payload_from_team_row(
        row=row_series,
        map_dict=map_dict,
        id_col=ID_COL,
        pd_col="pd_hat",
        top_features_col="shap_features",
        top_values_col="shap_values",
        t_low=T_LOW,
        t_high=T_HIGH,
        offset=OFFSET,
        factor=FACTOR,
        top_n_use=TOP_N,
    )

    score = float(payload["hcis_score"])

    # band는 payload 안 band를 그대로 써도 되지만,
    # 페이지용 표시를 확실히 하려고 점수 기준으로 한 번 더 확정
    if score >= T_HIGH:
        band = "승인"
    elif score >= T_LOW:
        band = "추가검토"
    else:
        band = "거절"

    margin = float(payload["policy"]["margin_score"])

    if band == "승인":
        action = "통과"
    elif band == "추가검토":
        action = "검토 필요"
    else:
        action = "고위험, 거절"

    pos_pct = (score - SCORE_MIN) / (SCORE_MAX - SCORE_MIN)
    pos_pct = float(np.clip(pos_pct, 0, 1) * 100)

    return row_series, row_dict, payload, score, band, action, pos_pct, margin, map_dict

row_series, row, payload, score, band, action, pos_pct, margin, map_dict = get_customer_analysis(
    df=df_work,
    cid=selected_id,
    mapping_path=MAPPING_PATH,
)

payload["behavioral_insights"] = generate_behavioral_insights(
    row_series,
    shap_top_10=payload.get("shap_top_10"),
    ref_df=df_work,   # model_df 전체를 넣어 분위(높은편/낮은편) 판별에 사용
    top_k=5
)

under = ask_underwriter(payload)

# -----------------------------------------------------------
# 심사 결과 (상단)
# -----------------------------------------------------------
st.markdown("#### ⚖️ 고객님 심사 판단 결과")

decision_styles = {
    "승인": ("success", "🟢 승인"),
    "추가검토": ("warning", "🟡 추가검토"),
    "거절": ("error", "🔴 거절")
}
msg_type, msg_text = decision_styles[band]
# getattr(st, msg_type)(msg_text)  # 필요 시 박스 표시

# -----------------------------------------------------------
# 핵심 수치 시각화 (Score, Grade, Decision)
# -----------------------------------------------------------
with st.container():
    c1, c2, c3 = st.columns(3)

    with c1:
        score_color = "#e74c3c" if band == "거절" else ("#f1c40f" if band == "추가검토" else "#2ecc71")
        score_pct = (score - SCORE_MIN) / (SCORE_MAX - SCORE_MIN) * 100
        st.markdown(f"""
        <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white; height: 180px; display: flex; flex-direction: column; justify-content: space-between;'>
            <div style='font-size: 14px; opacity: 0.9;'>HCIS 점수</div>
            <div style='font-size: 40px; font-weight: bold; margin: 10px 0;'>{score:.0f}</div>
            <div>
                <div style='background: rgba(255,255,255,0.2); border-radius: 10px; height: 10px; margin-bottom: 8px;'>
                    <div style='background: {score_color}; width: {score_pct:.1f}%; height: 100%; border-radius: 10px; transition: width 0.3s;'></div>
                </div>
                <div style='font-size: 11px; opacity: 0.8;'>{SCORE_MIN} ~ {SCORE_MAX}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        grade_colors = {"승인": "#2ecc71", "추가검토": "#f39c12", "거절": "#e74c3c"}
        grade_descriptions = {"승인": "우수", "추가검토": "주의", "거절": "위험"}
        grade_color = grade_colors.get(band, "#95a5a6")
        grade_desc = grade_descriptions.get(band, "")

        st.markdown(f"""
        <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); border-radius: 10px; color: white; height: 180px; display: flex; flex-direction: column; justify-content: space-between;'>
            <div style='font-size: 14px; opacity: 0.9;'>신용등급</div>
            <div style="flex: 1; display: flex; align-items: center; justify-content: center;">
                <div style='font-size: 60px; font-weight: bold; color: {grade_color}; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); line-height: 1;'>{grade_desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        decision_visual = {
            "승인": ("#2ecc71", "🟢", "linear-gradient(135deg, #11998e 0%, #38ef7d 100%)", "통과"),
            "추가검토": ("#f1c40f", "🟡", "linear-gradient(135deg, #f2994a 0%, #f2c94c 100%)", "검토 필요"),
            "거절": ("#e74c3c", "🔴", "linear-gradient(135deg, #eb3349 0%, #f45c43 100%)", "주의")
        }
        d_color, d_icon, d_gradient, d_status = decision_visual[band]

        st.markdown(f"""
        <div style='text-align: center; padding: 20px; background: {d_gradient}; border-radius: 10px; color: white; height: 180px; display: flex; flex-direction: column; justify-content: space-between;'>
            <div style='font-size: 14px; opacity: 0.9;'>심사 결과</div>
            <div>
                <div style='font-size: 50px; line-height: 1;'>{d_icon}</div>
                <div style='font-size: 22px; font-weight: bold; margin-top: 5px;'>{band}</div>
            </div>
            <div style='font-size: 11px; opacity: 0.8;'>{d_status}</div>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------
# 점수 바 시각화
# -----------------------------------------------------------
st.divider()

with st.container():
    col_left, col_right = st.columns([2, 1])

    with col_left:
        DISPLAY_MIN = 500
        DISPLAY_MAX = 900

        score_disp = max(DISPLAY_MIN, min(DISPLAY_MAX, score))
        tlow_disp = max(DISPLAY_MIN, min(DISPLAY_MAX, T_LOW))
        thigh_disp = max(DISPLAY_MIN, min(DISPLAY_MAX, T_HIGH))

        def to_pct(x):
            return (x - DISPLAY_MIN) / (DISPLAY_MAX - DISPLAY_MIN) * 100

        pos_pct = to_pct(score_disp)
        cond_pct = to_pct(tlow_disp)
        appr_pct = to_pct(thigh_disp)

        cond_pct, appr_pct = sorted([cond_pct, appr_pct])

        if score >= T_HIGH:
            bar_color = "#2ecc71"
            section_label = "승인 구간"
        elif score >= T_LOW:
            bar_color = "#f1c40f"
            section_label = "추가 검토 구간"
        else:
            bar_color = "#e74c3c"
            section_label = "위험 구간"

        score_bar_html = f"""
        <div style="background:#ffffff; border-radius:16px; padding:18px 20px 20px 20px; box-shadow: 0 4px 14px rgba(0,0,0,0.06);">
            <div style="font-size:17px; font-weight:700; margin-bottom:14px; display:flex; align-items:center; gap:8px;">
                🎯 점수 기준 내 위치
            </div>

            <div style="position:relative; height:120px;">
                <div style="position:absolute; left:{cond_pct:.1f}%; top:0; transform:translateX(-50%); font-size:12px;">
                    <div style="background:#f39c12; color:white; padding:2px 8px; border-radius:4px;">추가검토 기준</div>
                    <div style="text-align:center;">{T_LOW}</div>
                </div>

                <div style="position:absolute; left:{appr_pct:.1f}%; top:0; transform:translateX(-50%); font-size:12px;">
                    <div style="background:#2ecc71; color:white; padding:2px 8px; border-radius:4px;">승인 기준</div>
                    <div style="text-align:center;">{T_HIGH}</div>
                </div>

                <div style="position:absolute; top:45px; width:100%; height:30px;
                    background:linear-gradient(
                        90deg,
                        #e74c3c 0%,
                        #f39c12 {cond_pct:.1f}%,
                        #f1c40f {appr_pct:.1f}%,
                        #2ecc71 100%
                    );
                    border-radius:15px;
                ">
                    <div style="position:absolute; left:{pos_pct:.1f}%; top:50%;
                        transform:translate(-50%, -50%);
                        width:36px; height:36px;
                        background:white; border:3px solid {bar_color}; border-radius:50%;">
                    </div>
                </div>

                <div style="position:absolute; left:{pos_pct:.1f}%; top:88px; transform:translateX(-50%);">
                    <div style="background:{bar_color}; color:white; padding:4px 12px; border-radius:6px; font-weight:bold; font-size:15px;">
                        {score:.0f}점
                    </div>
                </div>
            </div>

            <div style="text-align:center; margin-top:16px; font-weight:600; color:{bar_color};">
                현재 고객은 「{section_label}」에 해당합니다.
            </div>
        </div>
        """
        components.html(score_bar_html, height=260)

    # -----------------------------------------------------------
    # 주요 참고 요인
    # -----------------------------------------------------------
    with col_right:
        st.markdown("""
        <div style="background: #ffffff; border-radius: 14px;
                    padding: 18px 18px 14px 18px;
                    margin-bottom: 22px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
                    border: 1px solid #eee; color:#111;">
            <div style="font-size: 17px; font-weight: 700; color: #111;
                        margin-bottom: 12px;
                        display: flex; align-items: center; gap: 8px;">
                🔎 주요 참고 요인
            </div>
        """, unsafe_allow_html=True)

        # SHAP 기반 Top3 문구 생성
        reasons = get_top_reasons_from_shap_row(
            row_series,
            map_dict,
            top_k=3,
            top_features_col="shap_features",
            top_values_col="shap_values",
            only_risk_positive=False
        )

        if reasons:
            for i, r in enumerate(reasons, 1):
                st.markdown(f"""
                <div style="margin-bottom: 10px;
                            padding: 10px 12px;
                            background: #f9fafb;
                            border-radius: 10px;
                            font-size: 14px;
                            color: #111;
                            box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
                    <b>{i}.</b> {r}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="padding: 12px;
                        background: #f1f3f5;
                        border-radius: 10px;
                        color: #333;
                        font-size: 14px;">
                특이사항 없음
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------
# LLM 코멘트
# -----------------------------------------------------------
with st.container():
    st.subheader("🧠 심사팀 AI 코멘트")

    if st.button("심사팀 코멘트 생성", type="primary"):
        with st.spinner("Gemini 생성 중..."):
            under = ask_underwriter(payload)
            render_underwriter_report(
                under=under,
                band=band,
                score=score,
                margin=margin
            )
        st.success("완료")
        with st.expander("🔧 원본 JSON 보기(디버깅/로그용)", expanded=False):
            st.json(under)

        st.markdown("### 🧠 고객 행태 기반 해석")
        for s in payload.get("behavioral_insights", []):
            st.markdown(f"- {s}")
