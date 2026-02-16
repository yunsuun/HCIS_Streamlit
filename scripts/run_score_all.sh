#!/bin/bash
python scripts/score_all.py \
  --db-path st_data/db/hcis.db \
  --model-path artifacts/model/v1.0.2_XGB_artifact.joblib \
  --chunk-size 50000 \
  --out-path outputs/score_result.parquet
