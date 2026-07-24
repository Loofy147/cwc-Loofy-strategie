"""
sim_data.py — v2 data generation
Two upgrades over benchmark.py's gen_ohlcv:

1. Fat-tailed returns (Student-t innovations, df=4) instead of pure Gaussian.
   Crypto and gold both show excess kurtosis vs normal; using Gaussian-only
   returns understates tail risk in the SL/TP simulation.

2. Markov regime-switching series: one continuous price path that transitions
   between bull/bear/chop via a transition matrix, instead of three separate
   clean blocks. This tests the engine through actual regime CHANGES, which
   is where static-threshold systems typically fail (they lag the transition).
"""

import numpy as np
import pandas as pd


def gen_ohlcv_t(n: int, regime: str, seed: int = 42, df_t: float = 4.0) -> pd.DataFrame:
    """Same interface as benchmark.gen_ohlcv but with Student-t return innovations."""
    rng  = np.random.default_rng(seed)
    base = 2000.0

    if regime == "bull":
        drift, vol = 0.0006, 0.0025
        vol_fac = lambda i: 1.0 + 0.5 * np.sin(i / 30)
    elif regime == "bear":
        drift, vol = -0.0005, 0.0028
        vol_fac = lambda i: 1.2 + 0.3 * np.sin(i / 25)
    else:  # chop
        drift, vol = 0.0001, 0.0015
        vol_fac = lambda i: 0.7 + 0.3 * np.abs(np.sin(i / 10))

    # Student-t scaled to match target vol (t-dist variance = df/(df-2))
    t_scale = np.sqrt((df_t - 2) / df_t)

    closes  = [base]
    volumes = []
    for i in range(1, n):
        shock = rng.standard_t(df_t) * t_scale
        ret   = drift + vol * vol_fac(i) * shock
        closes.append(closes[-1] * (1 + ret))
        volumes.append(int(rng.integers(500, 3000) * vol_fac(i)))
    volumes.append(int(rng.integers(500, 3000)))

    closes = np.array(closes)
    highs  = closes * (1 + rng.uniform(0, 0.004, n))
    lows   = closes * (1 - rng.uniform(0, 0.004, n))
    opens  = np.roll(closes, 1); opens[0] = closes[0]

    return pd.DataFrame({
        "time":        pd.date_range("2024-01-01", periods=n, freq="15min"),
        "open":        opens,
        "high":        np.maximum(highs, np.maximum(opens, closes)),
        "low":         np.minimum(lows,  np.minimum(opens, closes)),
        "close":       closes,
        "tick_volume": np.array(volumes),
    })


# Regime dynamics reused per-bar inside the Markov walk
_REGIME_DYNAMICS = {
    "bull": dict(drift=0.0006,  vol=0.0025, vol_fac=lambda i: 1.0 + 0.5 * np.sin(i / 30)),
    "bear": dict(drift=-0.0005, vol=0.0028, vol_fac=lambda i: 1.2 + 0.3 * np.sin(i / 25)),
    "chop": dict(drift=0.0001,  vol=0.0015, vol_fac=lambda i: 0.7 + 0.3 * np.abs(np.sin(i / 10))),
}

# Transition probabilities per bar (15m bars). Regimes are "sticky" —
# average persistence ~200-400 bars (2-4 days on M15) before switching.
_TRANSITION = {
    "bull": {"bull": 0.9970, "bear": 0.0010, "chop": 0.0020},
    "bear": {"bull": 0.0010, "bear": 0.9970, "chop": 0.0020},
    "chop": {"bull": 0.0015, "bear": 0.0015, "chop": 0.9970},
}


def gen_markov_series(n: int, seed: int = 42, df_t: float = 4.0,
                       start_regime: str = "chop") -> pd.DataFrame:
    """
    One continuous OHLCV series where the underlying regime switches via a
    Markov chain. Returns df with an extra 'true_regime' column (ground truth,
    for evaluating the classifier — not used by the trading engine itself).
    """
    rng = np.random.default_rng(seed)
    base = 2000.0
    t_scale = np.sqrt((df_t - 2) / df_t)

    regime = start_regime
    closes  = [base]
    volumes = []
    regimes = [regime]

    for i in range(1, n):
        # Possibly transition
        probs = _TRANSITION[regime]
        regime = rng.choice(list(probs.keys()), p=list(probs.values()))
        regimes.append(regime)

        dyn = _REGIME_DYNAMICS[regime]
        shock = rng.standard_t(df_t) * t_scale
        ret   = dyn["drift"] + dyn["vol"] * dyn["vol_fac"](i) * shock
        closes.append(closes[-1] * (1 + ret))
        volumes.append(int(rng.integers(500, 3000) * dyn["vol_fac"](i)))
    volumes.append(int(rng.integers(500, 3000)))

    closes = np.array(closes)
    highs  = closes * (1 + rng.uniform(0, 0.004, n))
    lows   = closes * (1 - rng.uniform(0, 0.004, n))
    opens  = np.roll(closes, 1); opens[0] = closes[0]

    return pd.DataFrame({
        "time":        pd.date_range("2024-01-01", periods=n, freq="15min"),
        "open":        opens,
        "high":        np.maximum(highs, np.maximum(opens, closes)),
        "low":         np.minimum(lows,  np.minimum(opens, closes)),
        "close":       closes,
        "tick_volume": np.array(volumes),
        "true_regime": regimes,
    })
