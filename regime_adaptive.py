"""
regime_adaptive.py — v2 enhancement layer
Adds graduated regime response on top of the static WCTE engine (bot.py).

Does NOT modify bot.py. This is an additive overlay:
  - Classifies market state into 4 tiers instead of a binary chop gate
  - Reweights the 'trend' rule group by regime
  - Scales position risk by regime instead of all-or-nothing blocking
  - Raises the entry bar in the caution zone rather than trading it at
    full size, which is what the v1 static engine effectively does at
    the edge of its chop filter

This file only ADDS logic — it imports the same Rule/RuleSet/indicators
from bot.py so the underlying 9+9 rule definitions are unchanged.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple

from bot import Config, indicators, build_signals, RuleSet, sl_tp, lot_size


# ═══════════════════════════════════════════════════════════════════════════
# REGIME CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════

REGIME_PARAMS = {
    # name              trend_mult  threshold  risk_mult
    "STRONG_TREND":   {"trend_mult": 1.25, "threshold": 0.55, "risk_mult": 1.00},
    "MODERATE_TREND": {"trend_mult": 1.00, "threshold": 0.55, "risk_mult": 1.00},
    "CAUTION":        {"trend_mult": 1.00, "threshold": 0.65, "risk_mult": 0.50},
    "RANGING":        {"trend_mult": 1.00, "threshold": 1.01, "risk_mult": 0.00},  # never triggers
}

CI_STRONG  = 45.0   # below this: strong trend
CI_CAUTION = 58.0   # v1's static cutoff — now the start of a graduated caution zone
CI_BLOCK   = 65.0   # above this: no trading at all


def classify_regime(row: pd.Series) -> str:
    ci = row["chop"]
    if ci < CI_STRONG:
        return "STRONG_TREND"
    elif ci < CI_CAUTION:
        return "MODERATE_TREND"
    elif ci < CI_BLOCK:
        return "CAUTION"
    else:
        return "RANGING"


def adaptive_score(ruleset: RuleSet, row: pd.Series, trend_mult: float = 1.0) -> float:
    """Weighted score with a per-group multiplier (trend group only, here)."""
    total_w = 0.0
    hit_w   = 0.0
    for r in ruleset.rules:
        w = r.weight * (trend_mult if r.group == "trend" else 1.0)
        total_w += w
        if r.eval(row):
            hit_w += w
    return hit_w / total_w if total_w else 0.0


@dataclass
class AdaptiveDecision:
    regime:      str
    buy_score:   float
    sell_score:  float
    threshold:   float
    risk_mult:   float
    signal:      str    # BUY / SELL / FLAT


def adaptive_decide(buy_r: RuleSet, sell_r: RuleSet, row: pd.Series) -> AdaptiveDecision:
    regime = classify_regime(row)
    p = REGIME_PARAMS[regime]

    bs = adaptive_score(buy_r,  row, p["trend_mult"])
    ss = adaptive_score(sell_r, row, p["trend_mult"])

    sig = "FLAT"
    if bs >= p["threshold"] and ss < p["threshold"]:
        sig = "BUY"
    elif ss >= p["threshold"] and bs < p["threshold"]:
        sig = "SELL"

    return AdaptiveDecision(regime, bs, ss, p["threshold"], p["risk_mult"], sig)


# ═══════════════════════════════════════════════════════════════════════════
# BASELINE (naive single-rule EMA crossover) — for comparison
# ═══════════════════════════════════════════════════════════════════════════

def baseline_decide(row: pd.Series) -> str:
    """Trivial EMA8/EMA21 crossover — no filter, no weighting."""
    if row["ema8"] > row["ema21"]:
        return "BUY"
    elif row["ema8"] < row["ema21"]:
        return "SELL"
    return "FLAT"
