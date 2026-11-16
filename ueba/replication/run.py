"""Execute the complete prespecified r5.2 replication without touching r4.2 results."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..calibration import GlobalScore,RoleCalibration,UserCalibration,EqualCalibration,HierarchicalCalibration
from ..experiment import run_protocol
from ..splits import calendar_split
from .statistics import bootstrap_period,correct_families

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/r5_2"
PROTOCOLS = {"primary":[.5,.2,.15,.15], "early":[.2,.1,.1,.2],
             "temporal_1":[.3,.15,.1,.15], "temporal_2":[.4,.15,.1,.15], "temporal_3":[.5,.15,.1,.15]}


def frozen_models():
    return dict(M0=GlobalScore(),M1=RoleCalibration(30),M2=UserCalibration(5),
                M3=EqualCalibration(),M4=HierarchicalCalibration(30,200))


def lock_methods():
    calibration = list((ROOT / "ueba/calibration").glob("*.py"))
    proofs = {}
    original_bundle = ROOT / "dist/ueba_review_2026-11-02.zip"
    if original_bundle.exists():
        with zipfile.ZipFile(original_bundle) as archive:
            for path in calibration:
                relative = path.relative_to(ROOT).as_posix()
                if archive.read("ueba_review/"+relative) != path.read_bytes():
                    raise ValueError(f"Calibration changed since the r4.2 review bundle: {relative}")
                proofs[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    else:
        proofs = {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in calibration}
    return dict(calibration_sha256=proofs,matched_original_r4_2_bundle=original_bundle.exists(),
                M4_k_user=30,M4_k_role=200,detector="IsolationForest",n_estimators=200,
                max_samples=2048,seeds=[42,43,44,45,46],model_selection="none")


def preflight(frame,audit):
    OUTPUT.mkdir(parents=True,exist_ok=True)
    locks = lock_methods()
    (OUTPUT/"locked_methods.json").write_text(json.dumps(locks,indent=2),encoding="utf-8")
    feature_hash = hashlib.sha256((ROOT/"data/processed/r5.2/user_days.parquet").read_bytes()).hexdigest()
    protocol_hash = hashlib.sha256((ROOT/"docs/protocol_r5_2.md").read_bytes()).hexdigest()
    plan = dict(protocols=PROTOCOLS,features=audit["features"],data_sha256=feature_hash,
                protocol_sha256=protocol_hash,locked_methods=locks,
                source_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (ROOT/"ueba").rglob("*.py")})
    plan_path = OUTPUT/"preflight_plan.json"
    if plan_path.exists():
        prior = json.loads(plan_path.read_text())
        for field in ["protocols","features","data_sha256","protocol_sha256","locked_methods"]:
            if prior[field] != plan[field]:
                raise ValueError(f"Prespecified plan changed after it was locked: {field}")
    else:
        plan_path.write_text(json.dumps(plan,indent=2),encoding="utf-8")
    support,scenarios = [],[]
    for period,fractions in PROTOCOLS.items():
        blocks,summary = calendar_split(frame,fractions)
        support.extend(dict(row,period=period) for row in summary)
        for block,part in blocks.items():
            for scenario in range(1,6):
                selected = part.loc[part[f"scenario_{scenario}"].eq(1)]
                incidents = set(value for ids in selected.incident_ids for value in ids.split(";") if value)
                # Impersonation is not inferred from dates: these are official observed account IDs.
                scenarios.append(dict(period=period,block=block,scenario=scenario,
                    positive_days=len(selected),positive_accounts=int(selected.user.nunique()),incidents=len(incidents),
                    earliest=str(selected.date.min().date()) if len(selected) else "",
                    latest=str(selected.date.max().date()) if len(selected) else "",
                    supported=bool(len(selected)>=20 and selected.user.nunique()>=5),
                    recall_day_resolution=1/len(selected) if len(selected) else np.nan))
    pd.DataFrame(support).to_csv(OUTPUT/"all_split_support.csv",index=False)
    pd.DataFrame(scenarios).to_csv(OUTPUT/"scenario_period_support.csv",index=False)
    environment = dict(python=platform.python_version(),platform=platform.platform(),
        packages={p:importlib.metadata.version(p) for p in ["numpy","pandas","scikit-learn","scipy",
                  "statsmodels","matplotlib","pyarrow","indexed-bzip2"]})
    (OUTPUT/"environment.json").write_text(json.dumps(environment,indent=2),encoding="utf-8")
    print(pd.DataFrame(scenarios).query("block=='test'").to_string(index=False),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage",choices=["preflight","experiments","statistics","all"],default="all")
    args = parser.parse_args()
    frame = pd.read_parquet(ROOT/"data/processed/r5.2/user_days.parquet")
    audit = json.loads((ROOT/"data/processed/r5.2/data_audit.json").read_text())
    preflight(frame,audit)
    if args.stage in {"experiments","all"}:
        for period,fractions in PROTOCOLS.items():
            run_protocol(frame,audit["features"],"r5_2/"+period,fractions,model_map=frozen_models(),
                         history_caps=[],scenario_ids=range(1,6))
    if args.stage in {"statistics","all"}:
        results = [bootstrap_period(OUTPUT/name) for name in PROTOCOLS]
        correct_families(results,OUTPUT)


if __name__ == "__main__":
    main()
