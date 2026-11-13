from .base import anomaly_score


class EqualCalibration:
    """M3: equal mean; Profiles substitutes global for each absent profile."""
    def transform(self, raw_scores, components):
        return anomaly_score(.5 * components["p_user"] + .5 * components["p_role"])
