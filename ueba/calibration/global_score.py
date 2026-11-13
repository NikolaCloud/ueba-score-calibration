import numpy as np


class GlobalScore:
    """M0: unchanged anomaly score; common validation threshold is applied downstream."""
    def transform(self, raw_scores, components=None):
        result = np.asarray(raw_scores, dtype=float)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite score")
        return result.copy()
