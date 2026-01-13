import joblib
from pathlib import Path

from modules.calibrators import (
    BaseCalibrator,
    NoneCalibrator,
    PlattCalibrator,
    IsotonicCalibrator,
)

OLD = Path("artifacts/model/v1.0.0_XGB_artifact.joblib")
NEW = Path("artifacts/model/v1.0.1_XGB_artifact.joblib")

artifact = joblib.load(OLD)
joblib.dump(artifact, NEW)

print("✅ 재저장 완료:", NEW.resolve())