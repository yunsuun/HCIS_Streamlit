# -----------------------------------------------------------
# 라이브러리 import
# -----------------------------------------------------------
# streamlit       : 대시보드 UI
# pandas          : 데이터 처리
# altair          : 분포 시각화
# components.html : HTML/CSS 기반 커스텀 UI
# -----------------------------------------------------------

import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import streamlit.components.v1 as components
from pathlib import Path
# -----------------------------------------------------------
# 내부 설정 및 유틸 함수
# -----------------------------------------------------------
# APP_TITLE      : 앱 공통 타이틀
# SCORE_APPROVE  : 기본 승인 컷
# SCORE_COND     : 기본 조건부 컷
# -----------------------------------------------------------

from config import (
    APP_TITLE,
    T_LOW,
    T_HIGH,
    MODEL_DF_PARQUET,
    ST_DATA_DIR
)

# 데이터 로드 / 전처리 / 점수화 관련 공통 함수
from utils.data_loader import load_base_df, ensure_id, pick_pd_column
# (removed) score/grade/decision utilities (HCIS band 기반으로 통일)

# 업로드 데이터 전처리, 모델링, 추출 함수
from modules.model_loader import load_artifact
from modules.preprocess import preprocess_features_only
from modules.align import sanitize_and_align
from modules.inference import predict_pd_upload_with_shap
from utils.hcis_core import compute_hcis_columns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# -----------------------------------------------------------
# 전체 레이아웃 여백 조정 (상단 패딩 축소)
# -----------------------------------------------------------

st.markdown("""
<style>
.block-container {
    padding-top: 2rem !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# 페이지 기본 설정
# -----------------------------------------------------------
# wide layout 사용
# 타이틀은 개요 페이지임을 명확히 표시
# -----------------------------------------------------------


st.set_page_config(page_title=f"{APP_TITLE} | 개요", layout="wide")

# ===========================================================
# Session state (이번 세션에서만 유효)
# ===========================================================
if "data_ready" not in st.session_state:
    st.session_state["data_ready"] = False   # 처음 실행은 항상 비어있게

if "data_version" not in st.session_state:
    st.session_state["data_version"] = 0     # 캐시 갱신 키


# ===========================================================
# Lazy model loader (업로드 탭에서만 사용)
# ===========================================================
@st.cache_resource(show_spinner="모델 로딩 중...")
def get_model_artifact():
    # modules/model_loader.py 의 load_artifact()를 사용한다고 가정
    return load_artifact()  # (model, calibrator, model_type, feature_names)

# ===========================================================
# 데이터 로드 및 분포 계산 (캐싱) - 단일 정의로 통일
# ===========================================================

@st.cache_data(ttl=3600, show_spinner="데이터 로딩 중...")
def load_and_compute_distributions(data_ready: bool, data_version: int):
    """
    - 원천 데이터 로드
    - PD → Score → Grade → Decision 일괄 계산
    - KPI 산출에 필요한 통계값 사전 계산
    반환값을 (pd, score, grade, decision, stats)로 통일
    """
    # ✅ "이번 세션에서 업로드로 활성화"되기 전까지는 무조건 비움
    if not data_ready:
        return None

    # 파일은 있을 수도/없을 수도 있는데, 어쨌든 로더 결과로 판단
    df, src = load_base_df(data_version)
    if df is None or len(df) == 0:
        return None

    df = ensure_id(df)
    pd_col = pick_pd_column(df)
    if pd_col is None:
        return None

    pd_series = df[pd_col].dropna().astype(float)

    # HCIS 운영 정책(클리핑/컷오프/마진 포함)으로 통일
    tmp = pd.DataFrame({"pd_hat": pd_series})
    tmp = compute_hcis_columns(tmp, pd_col="pd_hat")
    score_series = tmp["hcis_score"]   # 기존 변수명 유지(하위 코드 수정 최소화)
    decision_series = tmp["band"]      # '승인'/'추가검토'/'거절'
    grade_series = decision_series     # 개요에서는 grade 대신 band를 동일하게 사용


    stats = {
        "src": src,
        "total_customers": len(score_series),
        "score_min": float(score_series.min()),
        "score_max": float(score_series.max()),
        "score_mean": float(score_series.mean()),
        "pd_mean": float(pd_series.mean()),
    }

    return pd_series, score_series, grade_series, decision_series, stats


# -----------------------------------------------------------
# 캐싱된 데이터 호출 (단일 호출)
# -----------------------------------------------------------
# data = load_and_compute_distributions()
# if data is None:
#     st.error("분포 시각화를 위한 PD 컬럼을 찾지 못했습니다.")
#     st.stop()

# pd_s, score_s, grade_s, decision_s, stats = data




# -----------------------------------------------------------
# 분포 차트용 데이터 준비
# -----------------------------------------------------------

@st.cache_data(show_spinner=False)
def prepare_distribution_data(grade_s, decision_s):
    """분포 데이터 준비 (HCIS band 기준)"""
    band_dist = (
        decision_s.value_counts()
        .reindex(["승인", "추가검토", "거절"], fill_value=0)
        .reset_index()
    )
    band_dist.columns = ["Band", "Count"]
    # grade_dist는 기존 코드 호환을 위해 더미로 반환
    grade_dist = band_dist.copy()
    grade_dist.columns = ["Grade", "Count"]
    decision_dist = band_dist.copy()
    decision_dist.columns = ["Decision", "Count"]
    return grade_dist, decision_dist



# ===========================================================
# 화면 영역 시작
# ===========================================================

st.title("📌 HCIS 운영 개요")

# -----------------------------------------------------------
# 탭 구성
# -----------------------------------------------------------
# Tab1 : 시스템 흐름 / 철학
# Tab2 : 전체 고객 KPI & 분포
# Tab3 : 승인 컷 시뮬레이션
# -----------------------------------------------------------

with st.container():

    tab1, tab2, tab3, tab4 = st.tabs([
        "📌 시스템 흐름",
        "📊 전체 고객 분포",
        "🧪 승인/추가검토 컷",
        "📌 데이터 관리(고객추가/업로드)"
    ])

# ===========================================================
# 흐름 카드 (시각화 개선) - Tab1

# Tab1 설명

# - HCIS가 '어떻게 운영되는지'
# - 왜 이런 구조를 택했는지
# - '운영 논리' 설명
# ===========================================================

    with st.container():
        with tab1:
            
            st.markdown("#### 🧭 HCIS 운영 흐름")
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.markdown("""
                <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white; height: 180px; display: flex; flex-direction: column; justify-content: space-between;'>
                    <div style='font-size: 38px;'>👤</div>
                    <div>
                        <div style='font-size: 16px; font-weight: bold; margin-bottom: 10px;'>고객 선택</div>
                        <div style='font-size: 10px; opacity: 0.9;'>• ID 입력<br>• 단건 조회<br>• 위치 확인</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                st.markdown("""
                <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); border-radius: 10px; color: white; height: 180px; display: flex; flex-direction: column; justify-content: space-between;'>
                    <div style='font-size: 40px;'>📊</div>
                    <div>
                        <div style='font-size: 16px; font-weight: bold; margin-bottom: 10px;'>운영 PD</div>
                        <div style='font-size: 10px; opacity: 0.9;'>• 사전 계산<br>• 재학습 ❌<br>• 안정성 보장</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with c3:
                st.markdown("""
                <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); border-radius: 10px; color: white; height: 180px; display: flex; flex-direction: column; justify-content: space-between;'>
                    <div style='font-size: 38px;'>🧮</div>
                    <div>
                        <div style='font-size: 16px; font-weight: bold; margin-bottom: 10px;'>점수·등급</div>
                        <div style='font-size: 10px; opacity: 0.9;'>• PD→Score<br>• 절대 기준<br>• A~E 등급</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with c4:
                st.markdown("""
                <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); border-radius: 10px; color: white; height: 180px; display: flex; flex-direction: column; justify-content: space-between;'>
                    <div style='font-size: 38px;'>⚖️</div>
                    <div>
                        <div style='font-size: 16px; font-weight: bold; margin-bottom: 10px;'>심사결정</div>
                        <div style='font-size: 10px; opacity: 0.9;'>• Dual Cut-off<br>• 승인/조건부<br>• 위험 판정</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.divider()

            components.html("""
            <div style="
                width:100%;
                margin:16px 0 0 0;
                padding:14px 18px;
                background:linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius:12px;
                color:white;
                font-size:14px;
                font-weight:500;
            ">
            🎯 <b>HCIS</b>는 사전 계산된 <b>PD</b>를 기반으로 
            <b>Score → Grade → Decision</b>을 일관된 기준으로 산출하는 
            <b>운영 중심 신용평가 시스템</b>입니다.
            </div>
            """, height=80)
            
            c1, c2, c3 = st.columns(3)
                
            with c1:
                with st.expander("⚙️ 운영 구조 원칙"):
                    st.markdown("""
                    - **단계 분리**: PD · Score · Decision 독립 운영  
                    - **절대 기준**: 분위수 미사용 · 일관성 확보  
                    - **Dual Cut-off**: Score + PD 동시 판단  
                    """)


            with c2:
                with st.expander("🤝 판단 철학 (금융 포용성)"):
                    st.markdown("""
                    무조건 승인이 아닌 **리스크 인지 기반**의 포용적 판단  
                    - 조건부 승인  
                    - SHAP 기반 원인 분해  
                    - 설명 가능한 거절  
                    """)

            with c3:
                with st.expander("💼 비즈니스 임팩트"):
                    col_l, col_r = st.columns(2)

                    with col_l:
                        st.markdown("""
                        **🏦 금융사**
                        - 승인율 ↑ (리스크 유지)
                        - 연체 관리 비용 ↓
                        - 규제 대응력 ↑
                        """)

                    with col_r:
                        st.markdown("""
                        **👤 고객**
                        - 이유 있는 판단
                        - 금융 이력 단절 방지
                        - **금융 재진입 경로**
                        """)



# ===========================================================
# 전체 고객 정보 - Tab2

# Tab2 설명

# - 현재 운영 기준 하에서 전체 고객 상태를 한 눈에 확인
# - 정책 변경 전 '기준선 역할'
# ===========================================================

    with st.container():
        with tab2:
            
            data = load_and_compute_distributions(st.session_state["data_ready"], st.session_state["data_version"])
            if data is None:
                st.info("📂 아직 업로드된 결과가 없습니다. Tab4에서 업로드 후 '처리 시작'을 눌러주세요.")
                st.caption("업로드 후 자동으로 st_data/model_df.parquet가 생성됩니다.")
            else:
                pd_s, score_s, grade_s, decision_s, stats = data

                # -----------------------------------------------------------
                # KPI 계산 (UI 전용)
                # -----------------------------------------------------------
                total_customers = stats["total_customers"]
                avg_score = stats["score_mean"]

                approve_rate = (decision_s == "승인").mean() * 100
                avg_pd = stats["pd_mean"] * 100

                st.metric("고객 수", stats["total_customers"])
                st.metric("평균 PD(%)", round(stats["pd_mean"] * 100, 2))
                st.markdown("#### 📈 HCIS 전체 고객 KPI")

                grade_dist, decision_dist = prepare_distribution_data(grade_s, decision_s)
                
                # 차트 생성
                grade_chart = (
                    alt.Chart(grade_dist)
                    .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
                    .encode(
                        x=alt.X("Grade:N", title="내부 등급", axis=alt.Axis(labelFontSize=12)),
                        y=alt.Y("Count:Q", title="고객 수", axis=alt.Axis(labelFontSize=12)),
                        color=alt.Color(
                            "Grade:N",
                            scale=alt.Scale(
                                domain=["승인","추가검토","거절"],
                                range=["#2ecc71", "#f1c40f", "#e74c3c"]
                            ),
                            legend=None
                        ),
                        tooltip=[
                            alt.Tooltip("Grade", title="등급"),
                            alt.Tooltip("Count:Q", title="고객 수", format=",")
                        ]
                    )
                    .properties(height=360, title="신용등급별 분포")
                )

                decision_chart = (
                    alt.Chart(decision_dist)
                    .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
                    .encode(
                        x=alt.X("Decision:N", title="심사 결과", axis=alt.Axis(labelFontSize=12)),
                        y=alt.Y("Count:Q", title="고객 수", axis=alt.Axis(labelFontSize=12)),
                        color=alt.Color(
                            "Decision:N",
                            scale=alt.Scale(
                                domain=["승인","추가검토","거절"],
                                range=["#2ecc71", "#f1c40f", "#e74c3c"]
                            ),
                            legend=None
                        ),
                        tooltip=[
                            alt.Tooltip("Decision", title="심사 결과"),
                            alt.Tooltip("Count:Q", title="고객 수", format=",")
                        ]
                    )
                    .properties(height=380, title="심사 결과별 분포")
                )
                components.html(f"""
                <div style="
                    display:flex;
                    gap:16px;
                    margin-top:8px;
                ">

                <!-- KPI 1 -->
                <div style="
                    flex:1;
                    text-align:center;
                    padding:16px;
                    background:linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius:12px;
                    color:white;
                ">
                    <div style="font-size:12px; opacity:0.9;">총 분석 고객 수</div>
                    <div style="font-size:28px; font-weight:700;">{total_customers:,}명</div>
                </div>

                <!-- KPI 2 -->
                <div style="
                    flex:1;
                    text-align:center;
                    padding:16px;
                    background:linear-gradient(135deg, #43cea2 0%, #185a9d 100%);
                    border-radius:12px;
                    color:white;
                ">
                    <div style="font-size:12px; opacity:0.9;">평균 점수</div>
                    <div style="font-size:28px; font-weight:700;">{avg_score:.1f}</div>
                </div>

                <!-- KPI 3 -->
                <div style="
                    flex:1;
                    text-align:center;
                    padding:16px;
                    background:linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
                    border-radius:12px;
                    color:#333;
                ">
                    <div style="font-size:12px; opacity:0.9;">승인 비율</div>
                    <div style="font-size:28px; font-weight:700;">{approve_rate:.1f}%</div>
                </div>

                <!-- KPI 4 -->
                <div style="
                    flex:1;
                    text-align:center;
                    padding:16px;
                    background:linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
                    border-radius:12px;
                    color:white;
                ">
                    <div style="font-size:12px; opacity:0.9;">평균 PD</div>
                    <div style="font-size:28px; font-weight:700;">{avg_pd:.2f}%</div>
                </div>

                </div>
                """, height=130)

                st.divider()

                st.markdown("#### 📊 HCIS 전체 고객 분포 요약")

                c1, c2 = st.columns(2)
                with c1:
                    st.altair_chart(grade_chart, use_container_width=True)
                with c2:
                    st.altair_chart(decision_chart, use_container_width=True)

# ===========================================================
# 관리자 시뮬레이션 - Tab3

# Tab3 설명

# - 승인, 조건부 컷을 변경했을 때 승인, 조건부, 위험의 분포 변동 확인
# - 시뮬레이션 결과
# ===========================================================
        
    with st.container():
        with tab3:
            data = load_and_compute_distributions(st.session_state["data_ready"], st.session_state["data_version"])
            if data is None:
                st.info("📂 아직 업로드된 결과가 없습니다. Tab4에서 업로드 후 '처리 시작'을 눌러주세요.")
                st.caption("업로드 후 자동으로 st_data/model_df.parquet가 생성됩니다.")
            else:
                pd_s, score_s, grade_s, decision_s, stats = data

                st.markdown("#### 🧮 심사 결과별 고객 수")
                
                sim_score_approve = T_HIGH
                sim_score_cond = T_LOW

                # 시뮬레이션 결과 통계 계산
                sim_approve = (score_s >= sim_score_approve).sum()
                sim_cond = ((score_s >= sim_score_cond) & (score_s < sim_score_approve)).sum()
                sim_reject = (score_s < sim_score_cond).sum()

                # 시뮬레이션 결과 표시
                result_c1, result_c2, result_c3 = st.columns(3)

                with result_c1:
                    st.markdown(f"""
                    <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); border-radius: 10px; color: white;'>
                        <div style='font-size: 12px; opacity: 0.9;'>승인</div>
                        <div style='font-size: 30px; font-weight: bold;'>{sim_approve:,}명</div>
                        <div style='font-size: 10px; opacity: 0.8;'>{sim_approve/len(score_s)*100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                with result_c2:
                    st.markdown(f"""
                    <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #f2994a 0%, #f2c94c 100%); border-radius: 10px; color: white;'>
                        <div style='font-size: 12px; opacity: 0.9;'>추가검토</div>
                        <div style='font-size: 30px; font-weight: bold;'>{sim_cond:,}명</div>
                        <div style='font-size: 10px; opacity: 0.8;'>{sim_cond/len(score_s)*100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                with result_c3:
                    st.markdown(f"""
                    <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); border-radius: 10px; color: white;'>
                        <div style='font-size: 12px; opacity: 0.9;'>거절</div>
                        <div style='font-size: 30px; font-weight: bold;'>{sim_reject:,}명</div>
                        <div style='font-size: 10px; opacity: 0.8;'>{sim_reject/len(score_s)*100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                # ===========================================================
                # 점수 분포 히스토그램
                # ===========================================================
                
                st.divider()

                st.markdown("#### 🧮 점수 분포 및 심사 기준선")

                score_df = pd.DataFrame({"Score": score_s})

                score_hist = (
                    alt.Chart(score_df)
                    .transform_calculate(
                        zone=f"""
                        datum.Score >= {sim_score_approve} ? '승인' :
                        datum.Score >= {sim_score_cond} ? '추가검토' :
                        '거절'
                        """
                    )
                    .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                    .encode(
                        x=alt.X(
                            "Score:Q",
                            bin=alt.Bin(maxbins=30),
                            title="HCIS 점수",
                            axis=alt.Axis(labelFontSize=12)
                        ),
                        y=alt.Y("count()", title="고객 수", axis=alt.Axis(labelFontSize=12)),
                        color=alt.Color(
                            "zone:N",
                            scale=alt.Scale(
                                domain=["승인","추가검토","거절"],
                                range=["#2ecc71", "#f1c40f", "#e74c3c"]
                            ),
                            legend=alt.Legend(title="심사 구간", titleFontSize=12, labelFontSize=11)
                        ),
                        tooltip=[
                            alt.Tooltip("zone:N", title="구간"),
                            alt.Tooltip("count()", title="고객 수", format=",")
                        ]
                    )
                )

                # 기준선 차트
                approve_line = (
                    alt.Chart(pd.DataFrame({"x": [sim_score_approve], "label": ["승인 기준"]}))
                    .mark_rule(color="#2ecc71", strokeWidth=3, opacity=0.8)
                    .encode(
                        x="x:Q",
                        tooltip=[alt.Tooltip("label:N", title=""), alt.Tooltip("x:Q", title="점수")]
                    )
                )

                cond_line = (
                    alt.Chart(pd.DataFrame({"x": [sim_score_cond], "label": ["조건부 기준"]}))
                    .mark_rule(color="#f1c40f", strokeWidth=3, strokeDash=[6, 4], opacity=0.8)
                    .encode(
                        x="x:Q",
                        tooltip=[alt.Tooltip("label:N", title=""), alt.Tooltip("x:Q", title="점수")]
                    )
                )

                # 좌우 2분할 (비율 2:1)
                col1, col2 = st.columns([2, 1])

                with col1:
                    # 히스토그램 + 기준선
                    st.altair_chart(
                        (score_hist + approve_line + cond_line).properties(
                            height=300,
                            title=" "
                        ),
                        use_container_width=True
                    )

                with col2:
                    # 기준 정보 및 시뮬레이션 요약
                    st.markdown("<div style='font-size:16px; font-weight:600; margin-bottom:0.5rem;'>📌 시뮬레이션 결과</div>",unsafe_allow_html=True)
                    st.markdown(f"- 승인 컷: **{sim_score_approve}점** 이상")
                    st.markdown(f"- 조건부 컷: **{sim_score_cond}점** 이상")
                    
                    sim_approve = (score_s >= sim_score_approve).sum()
                    sim_cond = ((score_s >= sim_score_cond) & (score_s < sim_score_approve)).sum()
                    sim_reject = (score_s < sim_score_cond).sum()

                    st.markdown(f"- 승인 고객 수: **{sim_approve:,}명** ({sim_approve/len(score_s)*100:.1f}%)")
                    st.markdown(f"- 조건부 고객 수: **{sim_cond:,}명** ({sim_cond/len(score_s)*100:.1f}%)")
                    st.markdown(f"- 위험 고객 수: **{sim_reject:,}명** ({sim_reject/len(score_s)*100:.1f}%)")

    with st.container():
        with tab4:
            admin_mode = st.toggle("🛠 관리자 모드", value=False)
            st.subheader("📎 데이터 관리")

            if not admin_mode:
                st.info("관리자 모드에서만 업로드 가능합니다.")
                st.stop()

            # ---------------------------
            # 0) 세션 키 초기화
            # ---------------------------
            if "tab4_uploader_key" not in st.session_state:
                st.session_state["tab4_uploader_key"] = 0

            # 결과 저장용(화면 표시용)
            if "tab4_result_df" not in st.session_state:
                st.session_state["tab4_result_df"] = None

            # ---------------------------
            # 1) 업로더 (key로 완전 초기화 가능)
            # ---------------------------
            uploaded_file = st.file_uploader(
                "Parquet 파일 업로드",
                type=["parquet"],
                key=f"tab4_uploader_{st.session_state['tab4_uploader_key']}"
            )

            colA, colB = st.columns([1, 1])
            with colA:
                run = st.button("🚀 처리 시작", type="primary")
            with colB:
                reset = st.button("🧹 결과/업로드 초기화")

            # ---------------------------
            # 2) 수동 초기화 버튼
            # ---------------------------
            if reset:
                # 1) 화면 결과 비우기
                st.session_state["tab4_result_df"] = None
                # ✅ 통계 비활성화 + 캐시 갱신
                st.session_state["data_ready"] = False
                st.session_state["data_version"] += 1
                # 2) ✅ 디스크에 남아있는 결과 파일까지 삭제
                try:
                    if MODEL_DF_PARQUET.exists():
                        MODEL_DF_PARQUET.unlink()
                except Exception as e:
                    st.warning(f"결과 파일 삭제 실패: {e}")

                # 3) 업로더 위젯 리셋 (key 증가)
                st.session_state["tab4_uploader_key"] += 1

                # 4) 캐시 제거 (load_and_compute_distributions() 포함)
                st.cache_data.clear()

                st.success("🧹 결과/업로드 초기화 완료! (파일 삭제 포함)")
                st.rerun()
            
            # ---------------------------
            # 3) 처리 시작 버튼
            # ---------------------------
            if run:
                # (A) 버튼 눌렀을 때: 이전 결과를 먼저 비움
                st.session_state["tab4_result_df"] = None

                # 파일 없으면 안내하고 끝
                if uploaded_file is None:
                    st.warning("먼저 Parquet 파일을 업로드해주세요.")
                    st.stop()

                # (B) 여기부터 새로 처리
                model, calibrator, model_type, feature_names = get_model_artifact()

                try:
                    df_raw = pd.read_parquet(uploaded_file)
                    df_raw.columns = df_raw.columns.str.lower()

                    # 1) 전처리: ids는 preprocess_features_only가 리턴한 것을 그대로 신뢰
                    X, ids = preprocess_features_only(df_raw)
                    ids_arr = np.asarray(ids).reshape(-1).astype(str)

                    # 2) 학습 컬럼 정렬
                    X = sanitize_and_align(X, feature_names)

                    # 3) 추론 + SHAP
                    pd_hat, shap_feats, shap_vals = predict_pd_upload_with_shap(
                        model, calibrator, model_type, X, top_n=10
                    )

                    pd_hat_arr = np.asarray(pd_hat).reshape(-1).astype(float)

                    # 4) 길이 검증
                    if len(ids_arr) != len(pd_hat_arr):
                        raise ValueError(f"Length mismatch: ids={len(ids_arr)}, pd_hat={len(pd_hat_arr)}")

                    pred_df = pd.DataFrame({
                        "sk_id_curr": ids_arr,
                        "pd_hat": pd_hat_arr,
                    })

                    # 5) SHAP 컬럼
                    if shap_feats is not None and shap_vals is not None:
                        if len(shap_feats) != len(pred_df) or len(shap_vals) != len(pred_df):
                            raise ValueError(
                                f"Length mismatch: pred_df={len(pred_df)}, "
                                f"shap_feats={len(shap_feats)}, shap_vals={len(shap_vals)}"
                            )
                        pred_df["shap_features"] = list(shap_feats)
                        pred_df["shap_values"] = list(shap_vals)

                    # 6) HCIS 파생
                    pred_df = compute_hcis_columns(pred_df, pd_col="pd_hat")

                    # 7) 저장
                    result_df = pred_df.copy()
                    result_df["source_file"] = getattr(uploaded_file, "name", "uploaded_parquet")

                    ST_DATA_DIR.mkdir(parents=True, exist_ok=True)
                    result_df.to_parquet(MODEL_DF_PARQUET, index=False)

                    # 통계 활성화 + 캐시 갱신 키 증가
                    st.session_state["data_ready"] = True
                    st.session_state["data_version"] += 1

                    # 캐시 완전 안전빵
                    st.cache_data.clear()

                    # 세션에 "이번 결과만" 저장해서 화면에 보여주기
                    st.session_state["tab4_result_df"] = result_df

                    # 업로더도 비워서 “새로 올렸을 때만” 다시 처리되게 하고 싶다면:
                    st.session_state["tab4_uploader_key"] += 1

                    st.success(f"✅ 처리 완료! 저장됨: {MODEL_DF_PARQUET}")
                    st.rerun()

                except Exception as e:
                    st.exception(e)
                    st.stop()

            # ---------------------------
            # 4) 화면 표시: 세션에 저장된 최신 결과만 보여줌
            # ---------------------------
            if st.session_state["tab4_result_df"] is not None:
                st.caption("✅ 최신 처리 결과 (상위 30행 미리보기)")
                st.dataframe(st.session_state["tab4_result_df"].head(30), use_container_width=True)
            else:
                st.info("업로드 후 '처리 시작'을 누르면 결과가 표시됩니다.")
