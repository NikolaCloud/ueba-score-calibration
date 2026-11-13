"""Global calendar splits with support audits before fitting."""
import numpy as np
import pandas as pd


def calendar_split(frame, fractions=(0.5, 0.2, 0.15, 0.15)):
    if len(fractions) != 4 or any(x <= 0 for x in fractions) or sum(fractions) > 1.00000001:
        raise ValueError("Four positive fractions with sum <=1 required")
    if frame.date.isna().any() or frame[["user", "date"]].duplicated().any():
        raise ValueError("Invalid user-day keys")
    start = frame.date.min().normalize()
    stop = frame.date.max().normalize() + pd.Timedelta(days=1)
    days = (stop-start).days
    offsets = np.floor(np.r_[0, np.cumsum(fractions)] * days + 1e-8).astype(int)
    bounds = [start + pd.Timedelta(days=int(x)) for x in offsets]
    result, support = {}, []
    for i, name in enumerate(["train", "calibration", "validation", "test"]):
        sub = frame.loc[(frame.date >= bounds[i]) & (frame.date < bounds[i+1])].copy()
        if sub.empty:
            raise ValueError(f"Empty {name}")
        result[name] = sub
        positives = sub.loc[sub.label.eq(1)]
        positive_users = positives.user.nunique()
        support.append(dict(block=name, start=str(bounds[i].date()), stop_exclusive=str(bounds[i+1].date()),
                            days=len(sub), users=int(sub.user.nunique()), positives=int(sub.label.sum()),
                            positive_users=int(positive_users), prevalence=float(sub.label.mean()),
                            reliable_support=bool(len(positives) >= 20 and positive_users >= 5),
                            recall_one_day_resolution=float(1/len(positives)) if len(positives) else None,
                            # This is only an optimistic independent-day approximation, not
                            # the power of the clustered paired experiment.
                            optimistic_recall_95_halfwidth=float(1.96*np.sqrt(.25/len(positives)))
                            if len(positives) else None))
    return result, support
