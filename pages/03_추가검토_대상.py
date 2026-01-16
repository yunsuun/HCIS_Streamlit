import streamlit as st
import numpy as np
import pandas as pd

from pathlib import Path

from config import (
    APP_TITLE, ID_COL, OFFSET, FACTOR, T_LOW, T_HIGH,
    MODEL_DF_PARQUET, MAPPING_PATH, TOP_N
)

from utils.hcis_core import build_map_dict, build_payload_from_team_row, compute_hcis_columns
from utils.shap_reason import get_top_reasons_from_shap_row
from utils.risk_types import (
    RISK_TYPES,
    classify_review_payload,
    risk_type_display,
    risk_type_guidance,
)
from utils.review_simulation import SimParams, simulate_type_based_conversion, summarize_candidates_by_type



# -----------------------------------------------------------
# Page config
# -----------------------------------------------------------
st.set_page_config(page_title=f"{APP_TITLE} | 추가검토 확인", layout="wide")

st.markdown(
    """
<style>
.block-container { padding-top: 2rem !important; }
.small-muted { color: #666; font-size: 12px; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🟡 추가검토 확인 센터")
st.caption("추가검토 구간 고객을 '점수 줄세우기'가 아니라 '리스크 타입'으로 분류하고, 2차 평가 체크리스트를 제공합니다.")


# -----------------------------------------------------------
# Data load
# -----------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="데이터 로딩 중...")
def load_df_work(data_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(data_path)
    df[ID_COL] = df[ID_COL].astype(str)
    return df

ST_DATA_DF_PARQUET = Path("st_data") / "model_df.parquet"

if ST_DATA_DF_PARQUET.exists():
    data_src = f"st_data ({ST_DATA_DF_PARQUET.as_posix()})"
    df_work = load_df_work(ST_DATA_DF_PARQUET)
else:
    data_src = f"config ({Path(MODEL_DF_PARQUET).as_posix()})"
    df_work = load_df_work(Path(MODEL_DF_PARQUET))

st.caption(f"데이터 소스: `{data_src}`")

# HCIS 컬럼 보정
if ("hcis_score" not in df_work.columns) or ("band" not in df_work.columns):
    df_work = compute_hcis_columns(df_work, pd_col="pd_hat")

# -----------------------------------------------------------
# Filter: Review band
# -----------------------------------------------------------
df_review = df_work[df_work["band"] == "추가검토"].copy()

# 상단 KPI
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("전체 고객", f"{len(df_work):,}")
with c2:
    st.metric("추가검토 고객", f"{len(df_review):,}")
with c3:
    rate = (len(df_review) / max(len(df_work), 1)) * 100
    st.metric("추가검토 비중", f"{rate:.2f}%")
with c4:
    if len(df_review) > 0:
        st.metric("추가검토 평균 HCIS", f"{df_review['hcis_score'].mean():.1f}")
    else:
        st.metric("추가검토 평균 HCIS", "-")

if df_review.empty:
    st.info("추가검토 고객이 없습니다. 개요에서 업로드/추론 후 다시 확인하세요.")
    st.stop()


# -----------------------------------------------------------
# Mapping + Classification (cache)
# -----------------------------------------------------------
@st.cache_resource
def get_map_dict_cached(mapping_path: str):
    return build_map_dict(Path(mapping_path))

@st.cache_data(show_spinner="추가검토 고객 분류 중...")
def classify_review_rows(df: pd.DataFrame, mapping_path: str) -> pd.DataFrame:
    map_dict = get_map_dict_cached(mapping_path)

    rows = []
    for _, r in df.iterrows():
        row_series = r  # pd.Series

        # payload: SHAP bundle + group contribution까지 포함
        payload = build_payload_from_team_row(
            row=row_series,
            map_dict=map_dict,
            id_col=ID_COL,
            t_low=T_LOW,
            t_high=T_HIGH,
            offset=OFFSET,
            factor=FACTOR,
            top_n_use=TOP_N,
            top_features_col="shap_features",
            top_values_col="shap_values",
        )

        rt_key, dbg = classify_review_payload(payload)

        # UI용 top reasons (간단 문장 10개)
        reasons_txt = get_top_reasons_from_shap_row(
            row_series,
            map_dict,
            top_k=TOP_N,
            top_features_col="shap_features",
            top_values_col="shap_values",
            only_risk_positive=True,
        )

        rows.append(
            {
                "sk_id_curr": str(row_series.get(ID_COL)),
                "hcis_score": float(payload.get("hcis_score", np.nan)),
                "margin_score": float(payload.get("policy", {}).get("margin_score", np.nan)),
                "pd_hat": float(payload.get("pd_hat", np.nan)),
                "risk_type_key": rt_key,
                "risk_type": risk_type_display(rt_key),
                "dominant_group": dbg.get("dominant_group"),
                "credit_pct": dbg.get("credit_pct"),
                "docs_pct": dbg.get("docs_pct"),
                "capacity_pct": dbg.get("capacity_pct"),
                "emp_pct": dbg.get("emp_pct"),
                "top_reasons": " / ".join(reasons_txt) if reasons_txt else "",
            }
        )

    out = pd.DataFrame(rows)

    # 정렬: 마진 큰 순(승인에 더 가까운 추가검토) 우선
    if "margin_score" in out.columns:
        out = out.sort_values("margin_score", ascending=False, na_position="last")

    return out


df_classified = classify_review_rows(df_review[[c for c in df_review.columns]].copy(), MAPPING_PATH)

st.markdown("---")
st.subheader("📈 추가검토 승인 전환 시뮬레이션 (Risk Type 기반)")

# df_classified에는 risk_type_key가 있고, pd_hat / (있으면 target) 도 있음
# 후보 타입 기본값: Type2/Type3/Type4
default_types = ["TYPE2_DOCS_UNCERTAINTY", "TYPE3_SPENDING_IMBALANCE", "TYPE4_EMPLOYMENT_LIFECYCLE"]

with st.expander("설정 / 가정", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        ead = st.number_input("EAD(건당 대출원금, 원)", min_value=0, value=5_000_000, step=100_000)
    with c2:
        apr = st.number_input("APR(연 이자율)", min_value=0.0, value=0.12, step=0.01, format="%.2f")
    with c3:
        tenor = st.number_input("기간(개월)", min_value=1, value=12, step=1)

    c4, c5 = st.columns(2)
    with c4:
        lgd = st.number_input("LGD(손실률)", min_value=0.0, value=0.60, step=0.05, format="%.2f")
    with c5:
        review_cost = st.number_input("추가검토 운영비용(후보 1건당, 원)", min_value=0, value=10_000, step=1_000)

    type_options = list(RISK_TYPES.keys())
    include_types = st.multiselect(
        "승인 전환 후보 타입(확인으로 해소 가능한 유형을 선택)",
        options=type_options,
        default=[t for t in default_types if t in type_options],
    )

    conv_rates = st.multiselect(
        "확인 성공률 시나리오(후보 중 승인 전환 비율)",
        options=[0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
        default=[0.3, 0.5, 0.7],
    )

params = SimParams(
    ead=float(ead),
    apr=float(apr),
    tenor_months=int(tenor),
    lgd=float(lgd),
    review_cost_per_case=float(review_cost),
    target_col="target" if "target" in df_review.columns else None
)

# 타입별 후보 현황 요약
st.markdown("#### 후보 타입 현황(추가검토 내)")
cand_summary = summarize_candidates_by_type(df_review.merge(
    df_classified[[ID_COL, "risk_type_key"]], on=ID_COL, how="left"
), type_col="risk_type_key", pd_col="pd_hat")
st.dataframe(cand_summary, use_container_width=True, hide_index=True)

# 시뮬레이션 실행
if st.button("시뮬레이션 실행", type="primary"):
    # df_review(원본) + risk_type_key 조인
    df_for_sim = df_review.merge(
        df_classified[[ID_COL, "risk_type_key"]],
        on=ID_COL,
        how="left"
    )

    res = simulate_type_based_conversion(
        df_for_sim,
        include_types=include_types,
        conv_rates=sorted(conv_rates),
        params=params,
        pd_col="pd_hat",
        type_col="risk_type_key",
    )

    st.markdown("#### 시뮬레이션 결과")
    st.dataframe(res, use_container_width=True, hide_index=True)

    # 핵심 KPI 한 줄 요약
    best = res.sort_values("net_profit", ascending=False).iloc[0]
    st.success(
        f"가장 높은 순이익 시나리오: {best['scenario']} · "
        f"전환 {int(best['n_converted']):,}명 / 후보 {int(best['n_candidates']):,}명 · "
        f"순이익 {best['net_profit']:,.0f}원"
    )

# -----------------------------------------------------------
# Type 분포
# -----------------------------------------------------------
st.markdown("---")
st.subheader("📌 추가검토 리스크 타입 분포")

counts = df_classified["risk_type"].value_counts().reset_index()
counts.columns = ["risk_type", "n"]

# bar chart
import altair as alt

chart = (
    alt.Chart(counts)
    .mark_bar()
    .encode(
        y=alt.Y(
            "risk_type:N",
            title="Risk Type",
            sort="-x",
            axis=alt.Axis(
                labelLimit=300,      # 글자 잘림 방지 (핵심)
                labelFontSize=12
            )
        ),
        x=alt.X("n:Q", title="고객 수"),
        tooltip=["risk_type:N", "n:Q"]
    )
    .properties(
        height=320,
        padding={"left": 30}     # 왼쪽 여백 강제 확보 (핵심)
    )
)

st.altair_chart(chart, use_container_width=True)




# -----------------------------------------------------------
# Filter controls
# -----------------------------------------------------------
st.markdown("---")
st.subheader("🔎 타입별 후보 조회")

col_f1, col_f2, col_f3 = st.columns([2, 2, 2])
with col_f1:
    type_options = ["전체"] + sorted(df_classified["risk_type"].dropna().unique().tolist())
    sel_type = st.selectbox("Risk Type", type_options, index=0)
with col_f2:
    min_hcis = float(df_classified["hcis_score"].min())
    max_hcis = float(df_classified["hcis_score"].max())

    if np.isclose(min_hcis, max_hcis):
        st.info(f"HCIS가 단일 값입니다: {min_hcis:.2f}")
        hcis_range = (min_hcis, max_hcis)
    else:
        hcis_range = st.slider(
            "HCIS 범위",
            min_value=min_hcis,
            max_value=max_hcis,
            value=(min_hcis, max_hcis),
        )

with col_f3:
    # 마진은 음수~양수 섞임
    ms = pd.to_numeric(df_classified["margin_score"], errors="coerce")
    min_m = float(np.nanmin(ms))
    max_m = float(np.nanmax(ms))

    # ms가 전부 NaN이거나, 단일 값이면 slider 대신 고정
    if (not np.isfinite(min_m)) or (not np.isfinite(max_m)):
        st.info("마진 값이 비어있습니다.")
        margin_range = (-np.inf, np.inf)
    elif np.isclose(min_m, max_m):
        st.info(f"마진이 단일 값입니다: {min_m:.2f}")
        margin_range = (min_m, max_m)
    else:
        margin_range = st.slider(
            "마진 범위(cutoff 대비)",
            min_value=min_m,
            max_value=max_m,
            value=(min_m, max_m),
        )


filtered = df_classified.copy()
if sel_type != "전체":
    filtered = filtered[filtered["risk_type"] == sel_type]
filtered = filtered[(filtered["hcis_score"] >= hcis_range[0]) & (filtered["hcis_score"] <= hcis_range[1])]
filtered = filtered[(filtered["margin_score"] >= margin_range[0]) & (filtered["margin_score"] <= margin_range[1])]

st.caption(f"필터 결과: {len(filtered):,}명")

# -----------------------------------------------------------
# Candidate table + drilldown
# -----------------------------------------------------------
show_cols = [
    ID_COL,
    "hcis_score",
    "margin_score",
    "pd_hat",
    "risk_type",
    "dominant_group",
    "top_reasons",
]

st.dataframe(
    filtered[show_cols],
    use_container_width=True,
    hide_index=True,
)

st.markdown("<div class='small-muted'>Tip: 아래에서 고객 ID를 선택하면, 개인 심사 페이지에서 해당 ID로 바로 조회할 수 있습니다.</div>", unsafe_allow_html=True)

# -----------------------------------------------------------
# Drilldown: select one customer
# -----------------------------------------------------------
left, right = st.columns([2, 3])

with left:
    st.markdown("#### 👤 고객 선택")
    ids = filtered[ID_COL].astype(str).unique().tolist()
    sel_id = st.selectbox("고객 ID", ids) if ids else None

    if sel_id:
        st.code(f"선택 고객 ID: {sel_id}")
        st.markdown("- 좌측 사이드바에서 동일 ID를 입력하면 **'대출 심사 조회'** 페이지에서 바로 확인할 수 있어요.")

with right:
    st.markdown("#### 🧭 2차 평가 가이드")
    if not sel_id:
        st.info("왼쪽에서 고객을 선택해주세요")
    else:
        row_sel = filtered[filtered[ID_COL] == str(sel_id)].iloc[0]
        rt_key = row_sel["risk_type_key"]
        spec = RISK_TYPES.get(rt_key)
        guide = risk_type_guidance(rt_key)

        st.markdown(f"**{spec.name if spec else rt_key}**")
        st.write(spec.short_desc if spec else "")

        cqa, cact = st.columns(2)
        with cqa:
            st.markdown("**✅ 확인 질문(체크리스트)**")
            if guide["checklist_questions"]:
                for q in guide["checklist_questions"]:
                    st.write(f"- {q}")
            else:
                st.write("- (정의된 질문이 없습니다)")

        with cact:
            st.markdown("**🛠️ 권장 액션(심사/운영)**")
            if guide["suggested_actions"]:
                for a in guide["suggested_actions"]:
                    st.write(f"- {a}")
            else:
                st.write("- (정의된 액션이 없습니다)")

        st.markdown("---")
        st.markdown("**📍 선택 고객 핵심 지표**")
        k1, k2, k3 = st.columns(3)
        k1.metric("HCIS", f"{row_sel['hcis_score']:.0f}")
        k2.metric("마진", f"{row_sel['margin_score']:+.1f}")
        k3.metric("PD_hat", f"{row_sel['pd_hat']:.4f}")

        if row_sel.get("top_reasons"):
            st.markdown("**🔎 주요 참고 요인(Top10)**")
            for i, t in enumerate(str(row_sel["top_reasons"]).split(" / "), 1):
                st.write(f"{i}. {t}")

