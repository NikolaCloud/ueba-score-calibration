"""Audit the derivative feature matrix without using it as research input."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
frame = pd.read_parquet(ROOT / "data/external/feature_matrix_v1.parquet")
results = dict(rows=len(frame), columns=frame.columns.tolist(), users=int(frame.user.nunique()),
               no_target_column=not bool({"label", "target", "malicious", "insider", "attack"} & set(frame)),
               duplicates=int(frame[["user", "day"]].duplicated().sum()),
               missing=frame.isna().sum().to_dict(), full_history_z_equivalence={})
for base in ["total_logon_activity", "http_activity_count", "total_emails", "file_activity_count", "usb_activity_count"]:
    column = base+"_z_user"
    group = frame.groupby("user")[base]
    mean = group.transform("mean")
    variants = {}
    for ddof in [0, 1]:
        deviation = group.transform("std", ddof=ddof)
        z = ((frame[base]-mean)/deviation.replace(0, np.nan)).fillna(0)
        residual = np.abs(frame[column]-z)
        variants[f"ddof_{ddof}"] = dict(max_abs_error=float(residual.max()),
            matching_fraction=float(np.isclose(frame[column], z, atol=1e-9, rtol=1e-9).mean()))
    results["full_history_z_equivalence"][column] = variants
results["interpretation"] = "Exact equivalence to whole-user-history normalization is evidence against accepting these features as independently verified causal features; producer code remains unavailable."
path = ROOT / "data/metadata/zenodo_audit.json"
path.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(json.dumps({k:v for k,v in results.items() if k not in {"columns", "missing"}}, indent=2))
