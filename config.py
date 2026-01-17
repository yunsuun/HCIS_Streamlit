# ===========================================================
# HCIS Streamlit (Operational) - Config
# ===========================================================
# ✔ 본 파일은 HCIS 시스템의 "정책 단일 진실 원천(SSOT)"입니다.
# ✔ 모든 페이지(app/pages/utils)는 이 파일의 값만 참조합니다.
# ✔ 운영 기준은 절대값 기준을 사용하며, 분위수(quantile)는 사용하지 않습니다.
#
# Score 설계:
#   score = OFFSET + FACTOR * (1−PD)/PD
# ===========================================================
import numpy as np
from pathlib import Path
APP_TITLE = "HCIS 신용평가 시스템 (운영 로직)"

# ---------------- Data ----------------

PD_COL_CANDIDATES = ["pd_operational", "pd_hat", "pd", "PD", "prob_default"]

ID_COL = "sk_id_curr"
TARGET_COL = "target"

BASE_DIR = Path(__file__).resolve().parent
ST_DATA_DIR = BASE_DIR / "st_data"

MAPPING_PATH = ST_DATA_DIR / "reason_code_mapping.parquet"
MODEL_DF_PARQUET = ST_DATA_DIR / "model_df.parquet"
DEFAULT_SAMPLE_PARQUET = ST_DATA_DIR / "model_df_default.parquet"

# ---------------- Score policy ----------------

OFFSET = 600
PDO = 50
FACTOR = PDO / np.log(2)

SCORE_MIN = 0
SCORE_MAX = 1200

# 부도율 상,하한 적용
PD_FLOOR = 1e-6     # 최소값 설정, 0으로 나누기 방지
PD_CEIL = 0.999999  # 상한값

# ---------------- Decision policy (Dual cut-off) ----------------

T_LOW = 675     # low 컷
T_HIGH = 720    # high 컷
TOP_N = 10      # shap값 상위 몇개 사용?




# ---------------- Grade policy (ABSOLUTE CUTS) ----------------

PD_GRADE_CUTS = {
    "A": 0.02,
    "B": 0.05,
    "C": 0.10,
    "D": 0.20,
    "E": 1.00,
}

GRADE_ORDER = ["A", "B", "C", "D", "E"]

# ---------------- Monitoring (Admin only) ----------------

PSI_THRESHOLDS = {
    "안정": 0.10,
    "관찰 필요": 0.25,
}


# ---------------- Display ----------------

CURRENCY_COL_HINTS = ["amt_", "AMT_"]
