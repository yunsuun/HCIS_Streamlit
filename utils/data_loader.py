import pandas as pd
import streamlit as st
from config import PD_COL_CANDIDATES, ID_COL, MODEL_DF_PARQUET

# def _try_read_parquet(path: str):
#     if not path:
#         return None
#     if os.path.exists(path):
#         return pd.read_parquet(path)
#     return None

@st.cache_data(show_spinner=False)
def load_base_df(data_version: int = 0):
    """운영용 베이스 데이터 로더.
    우선순위:
      1) config.DATA_CANDIDATES 중 존재하는 첫 파일
    """
    if MODEL_DF_PARQUET.exists():
        return pd.read_parquet(MODEL_DF_PARQUET), str(MODEL_DF_PARQUET)
    return None, None

# @st.cache_data(show_spinner=False)
# def load_oof_df():
#     """OOF(검증용) 예측 로더. 고객 화면에서는 사용하지 말 것."""
#     if os.path.exists(OOF_PRED_PATH):
#         return pd.read_parquet(OOF_PRED_PATH), OOF_PRED_PATH
#     return None, None

def ensure_id(df: pd.DataFrame):
    if df is None:
        return df
    if ID_COL not in df.columns:
        raise KeyError(f"ID 컬럼 '{ID_COL}' 이(가) 데이터에 없습니다.")
    return df

def pick_pd_column(df: pd.DataFrame):
    for c in PD_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None
