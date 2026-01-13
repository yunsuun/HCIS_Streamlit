from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

# 기존 Streamlit 페이지에서 쓰던 동일 모듈들
from modules.model_loader import load_artifact
from modules.preprocess import preprocess_features_only
from modules.align import sanitize_and_align
from modules.inference import predict_pd_upload_with_shap
from utils.hcis_core import pd_to_hcis


def build_model_df(in_path: Path, out_path: Path) -> pd.DataFrame:
    """
    업로드 parquet(app_test_4.parquet 같은 app 데이터) -> 전처리/추론 -> model_df.parquet 저장용 DF 생성
    out DF 컬럼: sk_id_curr, pd_hat, score
    """
    if not in_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {in_path}")

    df_raw = pd.read_parquet(in_path)

    # 1) 모델 아티팩트 로드
    model, calibrator, model_type, feature_names = load_artifact()

    # 2) 전처리 (features only)
    X, ids = preprocess_features_only(df_raw)

    # 3) 학습 피처 정렬/정제
    X = sanitize_and_align(X, feature_names)

    # 4) 추론
    pd_hat = predict_pd_upload_with_shap(model, calibrator, model_type, X)

    # 5) PD -> HCIS(score)
    score = pd_to_hcis(pd_hat)

    # 6) 결과 병합(최소 컬럼)
    result_df = pd.DataFrame({
        "sk_id_curr": ids,
        "pd_hat": pd_hat,
        "score": score
    })

    # (선택) 원본에 sk_id_curr가 있으면 left merge해서 누락 확인 가능
    if "sk_id_curr" in df_raw.columns:
        base_ids = df_raw[["sk_id_curr"]].copy()
        result_df = base_ids.merge(result_df, on="sk_id_curr", how="left")

    # 7) 저장
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(out_path, index=False)

    return result_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in",
        dest="in_path",
        required=True,
        help="Input parquet path (e.g., st_data/app_test_4.parquet)"
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        required=True,
        help="Output parquet path (e.g., st_data/model_df.parquet)"
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    in_path = (project_root / args.in_path).resolve() if not Path(args.in_path).is_absolute() else Path(args.in_path)
    out_path = (project_root / args.out_path).resolve() if not Path(args.out_path).is_absolute() else Path(args.out_path)

    df_out = build_model_df(in_path=in_path, out_path=out_path)
    print("✅ Saved:", out_path)
    print("shape:", df_out.shape)
    print("columns:", df_out.columns.tolist())


if __name__ == "__main__":
    main()
