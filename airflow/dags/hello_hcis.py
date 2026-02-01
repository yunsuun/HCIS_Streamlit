from __future__ import annotations

from airflow import DAG
from airflow.decorators import task
from airflow.utils.dates import days_ago

from pathlib import Path
from datetime import datetime


with DAG(
    dag_id="hello_hcis",
    start_date=days_ago(1),
    schedule="@daily",
    catchup=False,
    tags=["tutorial"],
) as dag:

    @task
    def write_hello():
        out_dir = Path("/opt/airflow/data/outputs")
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / "hello.txt"
        out_path.write_text(f"hello airflow! {datetime.now().isoformat()}\n", encoding="utf-8")
        return str(out_path)

    @task
    def confirm(path: str):
        p = Path(path)
        return f"created: {p.exists()} | {path}"

    confirm(write_hello())
