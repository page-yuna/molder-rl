"""
Wrapper for inference with the tuned XGBoost model.
Call predict_fail(x) → 0 (OK) or 1 (NG).
"""

import numpy as np
import xgboost as xgb
from pathlib import Path

# ── load model ──────────────────────────────────────────────────────────
_MODEL_PATH = Path("models/xgb_quality_smote.json")
_MODEL = xgb.XGBClassifier()
_MODEL.load_model(_MODEL_PATH)

# ── tuned threshold (set by threshold-search script) ────────────────────
_THRESH = 0.0025  

# ── helper functions ────────────────────────────────────────────────────
def prob_fail(x: np.ndarray) -> float:
    """Return P(NG) for one observation."""
    x2d = x.reshape(1, -1).astype(np.float32)
    return float(_MODEL.predict_proba(x2d)[0, 1])

def predict_fail(x: np.ndarray) -> int:
    """0 = pass, 1 = NG using the tuned threshold."""
    return int(prob_fail(x) >= _THRESH)
