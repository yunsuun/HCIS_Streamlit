import pandas as pd
import streamlit as st
from config import PD_COL_CANDIDATES, ID_COL, MODEL_DF_PARQUET, DEFAULT_SAMPLE_PARQUET

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
        df = pd.read_parquet(MODEL_DF_PARQUET)
        return df, str(MODEL_DF_PARQUET)

    # 2) 없으면 기본 샘플 사용
    if DEFAULT_SAMPLE_PARQUET.exists():
        df = pd.read_parquet(DEFAULT_SAMPLE_PARQUET)
        return df, str(DEFAULT_SAMPLE_PARQUET)

    # 3) 둘 다 없으면 로드 실패
    return None, None

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
