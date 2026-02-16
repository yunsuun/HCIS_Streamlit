# scripts/score_all.py
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from modules.inference import predict_pd_only
from utils.hcis_core import compute_hcis_columns

pd_hat = predict_pd_only(model, calibrator, model_type, df_feat)

# ✅ 너 프로젝트 모듈에 맞게 바꿔야 하는 import 3개
# 1) feature mart에서 feat_all 읽기
# 2) PD 모델 로드 + predict_proba
# 3) PD -> HCIS 점수 변환
#
# 예시(이름만 바꾸면 됨)
# from st_data.db import read_table  # sqlite read helper
# from utils.modeling import load_pd_model, predict_pd
# from utils.hcis import pd_to_hcis_score

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--feat-all-path", type=str, default="", help="parquet path if using file")
    p.add_argument("--db-path", type=str, default="", help="sqlite db path if using DB")
    p.add_argument("--feat-all-table", type=str, default="feat_all", help="sqlite table name")
    p.add_argument("--model-path", type=str, required=True, help="saved PD model path")
    p.add_argument("--ids-path", type=str, default="", help="parquet of sample ids (sk_id_curr)")
    p.add_argument("--limit", type=int, default=0, help="limit rows for quick test")
    p.add_argument("--chunk-size", type=int, default=50000, help="chunk size for full scoring")
    p.add_argument("--out-path", type=str, default="outputs/score_result.parquet")
    return p.parse_args()

def ensure_dir(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def load_ids(ids_path: str) -> set[int] | None:
    if not ids_path:
        return None
    df_ids = pd.read_parquet(ids_path)
    # ids 파일 컬럼명이 다르면 여기만 수정
    col = "sk_id_curr"
    return set(df_ids[col].astype(int).tolist())

def load_feat_all_source(args) -> tuple[str, str]:
    """
    return (mode, locator)
    mode: 'parquet' or 'sqlite'
    """
    if args.feat_all_path:
        return "parquet", args.feat_all_path
    if args.db_path:
        return "sqlite", args.db_path
    raise ValueError("Provide either --feat-all-path (parquet) or --db-path (sqlite).")

def iter_feat_all_parquet(path: str, chunksize: int):
    # parquet는 chunk read가 애매해서 일단 전체 read 후 chunk slicing (30만이면 보통 가능)
    df = pd.read_parquet(path)
    n = len(df)
    for start in range(0, n, chunksize):
        yield df.iloc[start:start+chunksize].copy()

def iter_feat_all_sqlite(db_path: str, table: str, chunksize: int):
    import sqlite3
    con = sqlite3.connect(db_path)
    # rowid 기반 chunk (feat_all에 rowid가 없으면 pk로 쪼개는 방식으로 변경)
    q_cnt = f"SELECT COUNT(*) as cnt FROM {table}"
    total = pd.read_sql(q_cnt, con)["cnt"].iloc[0]

    offset = 0
    while offset < total:
        q = f"SELECT * FROM {table} LIMIT {chunksize} OFFSET {offset}"
        df = pd.read_sql(q, con)
        yield df
        offset += chunksize
    con.close()

def main():
    args = parse_args()
    ensure_dir(args.out_path)

    # ✅ PD 모델 로드
    model = joblib.load(args.model_path)

    ids_set = load_ids(args.ids_path)

    mode, locator = load_feat_all_source(args)
    if mode == "parquet":
        it = iter_feat_all_parquet(locator, args.chunk_size)
    else:
        it = iter_feat_all_sqlite(locator, args.feat_all_table, args.chunk_size)

    out_chunks = []
    seen = 0

    for chunk in it:
        # ✅ 필수 컬럼명: pk
        if "sk_id_curr" not in chunk.columns:
            raise KeyError("feat_all must contain 'sk_id_curr'.")

        # 샘플 ids 필터
        if ids_set is not None:
            chunk = chunk[chunk["sk_id_curr"].astype(int).isin(ids_set)]
        if args.limit and seen >= args.limit:
            break
        if args.limit:
            chunk = chunk.iloc[: max(0, args.limit - seen)]

        if len(chunk) == 0:
            continue

        # ✅ (B) 모델 입력 X 구성: 너 프로젝트의 전처리/컬럼정렬 로직으로 교체
        # X = build_model_matrix(chunk)  # TODO
        X = None  # TODO

        # ✅ (C) PD 예측: predict_proba 사용 (양성 클래스 확률)
        # pd_hat = predict_pd(model, X)  # returns np.array shape (n,)
        pd_hat = np.full(len(chunk), 0.02)  # TODO (임시)

        # ✅ (D) HCIS 변환
        # hcis = pd_to_hcis_score(pd_hat)  # returns np.array int/float
        hcis = (600 + 20 * np.log2(0.02 / pd_hat)).astype(float)  # TODO (임시 예시)

        res = pd.DataFrame({
            "sk_id_curr": chunk["sk_id_curr"].astype(int).values,
            "pd": pd_hat.astype(float),
            "hcis_score": hcis.astype(float),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        })

        out_chunks.append(res)
        seen += len(chunk)

    if not out_chunks:
        raise RuntimeError("No rows were scored. Check ids filter / input source.")

    out = pd.concat(out_chunks, ignore_index=True)
    out.to_parquet(args.out_path, index=False)
    print(f"✅ saved: {args.out_path}  rows={len(out):,}")

if __name__ == "__main__":
    main()
