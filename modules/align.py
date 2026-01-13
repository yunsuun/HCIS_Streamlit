import pandas as pd
import re
from collections import Counter

def sanitize_and_align(X: pd.DataFrame, feature_names):
    X = pd.get_dummies(X, dummy_na=True)

    cleaned = [re.sub(r"[^0-9a-zA-Z_]", "_", c) for c in X.columns]
    counter = Counter()
    final_cols = []
    for c in cleaned:
        counter[c] += 1
        final_cols.append(c if counter[c] == 1 else f"{c}_dup{counter[c]-1}")
    X.columns = final_cols

    return X.reindex(columns=feature_names, fill_value=0)