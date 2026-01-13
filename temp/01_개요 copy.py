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
    SCORE_APPROVE,
    SCORE_COND
)

# 데이터 로드 / 전처리 / 점수화 관련 공통 함수
from utils.data_loader import load_base_df, ensure_id, pick_pd_column
from utils.scoring import pd_to_score, clip_pd
from utils.rules import pd_to_grade, underwriting_decision_dual

# 업로드 데이터 전처리, 모델링, 추출 함수
from modules.model_loader import load_artifact
from modules.preprocess import preprocess_features_only
from modules.align import sanitize_and_align
from modules.inference import predict_pd_upload
from modules.scoring_copy import pd_to_hcis

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ST_DATA_DIR = PROJECT_ROOT / "st_data"
MODEL_DF_PATH = ST_DATA_DIR / "model_df.parquet"
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

st.set_page_config(
    page_title="HCIS 신용평가 시스템",
    layout="wide"
)

st.set_page_config(page_title=f"{APP_TITLE} | 개요", layout="wide")

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
def load_and_compute_distributions():
    """
    - 원천 데이터 로드
    - PD → Score → Grade → Decision 일괄 계산
    - KPI 산출에 필요한 통계값 사전 계산
    반환값을 (pd, score, grade, decision, stats)로 통일
    """
    df, src = load_base_df()

    # ✅ model_df가 아직 없으면 None 반환 (분포 탭에서 안내문구 띄우게)
    if df is None or len(df) == 0:
        return None
    df = ensure_id(df)
    pd_col = pick_pd_column(df)

    if pd_col is None:
        return None

    pd_series = df[pd_col].dropna().apply(clip_pd)
    score_series = pd_series.apply(pd_to_score)
    grade_series = pd_series.apply(pd_to_grade)
    decision_series = pd.Series([
        underwriting_decision_dual(s, p)
        for s, p in zip(score_series, pd_series)
    ])

    stats = {
        "src": src,
        "total_customers": len(score_series),
        "score_min": int(score_series.min()),
        "score_max": int(score_series.max()),
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
    """분포 데이터 준비"""
    grade_dist = (
        grade_s.value_counts()
        .reindex(["A", "B", "C", "D", "E"], fill_value=0)
        .reset_index()
    )
    grade_dist.columns = ["Grade", "Count"]

    decision_dist = (
        decision_s.value_counts()
        .reindex(["승인", "조건부", "위험"], fill_value=0)
        .reset_index()
    )
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
        "🧪 승인/조건부 컷",
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
            
            data = load_and_compute_distributions()
            if data is None:
                st.info("아직 분포를 그릴 데이터가 없습니다. '데이터 관리(고객추가/업로드)'에서 Parquet를 업로드해 주세요.")
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
                                domain=["A", "B", "C", "D", "E"],
                                range=["#2ecc71", "#8fd19e", "#f1c40f", "#e67e22", "#e74c3c"]
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
                                domain=["승인", "조건부", "위험"],
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
            data = load_and_compute_distributions()
            if data is None:
                st.info("시뮬레이션을 수행할 데이터가 없습니다. 먼저 업로드를 완료해 주세요.")
            else:
                pd_s, score_s, grade_s, decision_s, stats = data

                st.markdown("#### 🧮 심사 결과별 고객 수")
                
                sim_score_approve = SCORE_APPROVE
                sim_score_cond = SCORE_COND

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
                        <div style='font-size: 12px; opacity: 0.9;'>조건부</div>
                        <div style='font-size: 30px; font-weight: bold;'>{sim_cond:,}명</div>
                        <div style='font-size: 10px; opacity: 0.8;'>{sim_cond/len(score_s)*100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                with result_c3:
                    st.markdown(f"""
                    <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); border-radius: 10px; color: white;'>
                        <div style='font-size: 12px; opacity: 0.9;'>위험</div>
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
                        datum.Score >= {sim_score_cond} ? '조건부' :
                        '위험'
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
                                domain=["승인", "조건부", "위험"],
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
            else:
                uploaded_file = st.file_uploader("Parquet 파일 업로드", type=["parquet"])
                if uploaded_file is not None:
                    # 여기서만 모델 로딩
                    model, calibrator, model_type, feature_names = get_model_artifact()
                if uploaded_file is not None:
                    try:
                        df_raw = pd.read_parquet(uploaded_file)

                        # (권장) 컬럼 소문자 통일 - preprocess 내부에서 이미 하면 생략 가능
                        df_raw.columns = df_raw.columns.str.lower()

                        # 1) 전처리
                        X, ids = preprocess_features_only(df_raw)

                        # 2) 학습 컬럼 정렬
                        X = sanitize_and_align(X, feature_names)

                        # 3) 모델 추론 (batch)
                        pd_hat = predict_pd_upload(model, calibrator, model_type, X)
                        score = pd_to_hcis(pd_hat)

                        # 4) 결과 DF 생성
                        pred_df = pd.DataFrame({
                            "sk_id_curr": ids,
                            "pd_hat": pd_hat,
                            "score": score
                        })

                        if "sk_id_curr" in df_raw.columns:
                            result_df = df_raw[["sk_id_curr"]].merge(pred_df, on="sk_id_curr", how="left")
                        else:
                            # id가 없으면 최소 결과만 저장
                            result_df = pred_df.copy()

                        # (선택) 업로드 파일명 기록
                        result_df["source_file"] = getattr(uploaded_file, "name", "uploaded_parquet")

                        # 5) 저장
                        ST_DATA_DIR.mkdir(parents=True, exist_ok=True)
                        result_df.to_parquet(MODEL_DF_PATH, index=False)

                        st.success(f"✅ 업로드 처리 완료! 저장됨: {MODEL_DF_PATH}")
                        st.caption("이제 '분포' 탭에서 확인할 수 있어요.")
                        st.cache_data.clear()
                        st.rerun()

                    except Exception as e:
                        st.exception(e)