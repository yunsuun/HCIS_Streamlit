import os
import glob
import sqlite3
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARQUET_DIR = BASE_DIR   # parquet 파일이 st_data 바로 아래에 있다고 가정
DB_PATH = os.path.join(BASE_DIR, "db", "hcis.db")

def normalize_col(col: str) -> str:
    col = col.strip().lower()
    col = col.replace(" ", "_").replace("-", "_")
    return col

def load_one_parquet(conn: sqlite3.Connection, parquet_path: str):
    table = os.path.splitext(os.path.basename(parquet_path))[0].lower()
    print(f"\n=== {table} 적재 중: {parquet_path} ===")

    df = pd.read_parquet(parquet_path)
    df.columns = [normalize_col(c) for c in df.columns]

    # SQLite 안전을 위해 전부 TEXT로 (나중에 CAST해서 씀)
    df = df.astype("string")

    df.to_sql(table, conn, if_exists="replace", index=False)

    # 키 컬럼 있으면 인덱스 생성
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table});").fetchall()]
    for key in ("sk_id_curr", "sk_id_bureau"):
        if key in cols:
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_{key} ON {table}({key});")

    conn.commit()
    print(f"✅ 완료: {table} ({len(df):,} rows)")

def main():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    parquet_files = sorted(glob.glob(os.path.join(PARQUET_DIR, "*.parquet")))
    if not parquet_files:
        raise SystemExit(
            f"❌ parquet 파일을 못 찾았어.\n"
            f"경로: {PARQUET_DIR}\n"
            f"예: app_train.parquet, bureau.parquet"
        )

    print(f"✅ DB 경로: {DB_PATH}")
    print(f"✅ 찾은 parquet 개수: {len(parquet_files)}")

    conn = sqlite3.connect(DB_PATH)
    try:
        for path in parquet_files:
            load_one_parquet(conn, path)
    finally:
        conn.close()

    print("\n🎉 모든 parquet → SQLite 적재 완료!")

if __name__ == "__main__":
    main()
