"""
benchmark_v2.py — Monte Carlo validation + walk-forward + baseline comparison

Three experiments, all reproducible from this single script:

  EXP 1 — Monte Carlo clean regimes (N=30 seeds/regime)
          Static v1 engine, mean ± std of win rate / EV / max drawdown

  EXP 2 — Walk-forward Markov regime-switching series (N=15 seeds)
          Static v1 vs Adaptive v2 vs naive EMA-crossover baseline
          Reports equity curve stats: total return (R), max drawdown (R),
          Sharpe-like ratio (mean/std of per-trade R), trade count

  EXP 3 — Regime classifier accuracy
          Compares the CI-based regime label against the ground-truth
          Markov regime used to generate the data
"""

import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from bot import Config, indicators, build_signals, sl_tp
from sim_data import gen_ohlcv_t, gen_markov_series
from regime_adaptive import (
    classify_regime, adaptive_score, adaptive_decide,
    baseline_decide, REGIME_PARAMS
)

CFG = Config(signal_threshold=0.55, filter_chop_max=58.0,
             atr_sl_mult=2.0, atr_tp_mult=4.0)
BUY_R, SELL_R, FILT_R = build_signals(CFG)


# ═══════════════════════════════════════════════════════════════════════════
# EXP 1 — MONTE CARLO ON CLEAN REGIMES (static v1 engine)
# ═══════════════════════════════════════════════════════════════════════════

def simulate_static_trades(df: pd.DataFrame, sl_mult: float, tp_mult: float) -> list:
    """Returns list of trade R-multiples (unit of risk) for the static v1 engine."""
    trades = []
    in_trade = False
    side = sl = tp = None

    for _, row in df.iterrows():
        if in_trade:
            if side == "BUY":
                if row["low"] <= sl:
                    trades.append(-1.0); in_trade = False
                elif row["high"] >= tp:
                    trades.append(tp_mult / sl_mult); in_trade = False
            else:
                if row["high"] >= sl:
                    trades.append(-1.0); in_trade = False
                elif row["low"] <= tp:
                    trades.append(tp_mult / sl_mult); in_trade = False
            continue

        if not FILT_R.triggered(row):
            continue
        atr = row["atr"]
        bt, st = BUY_R.triggered(row), SELL_R.triggered(row)
        if bt and not st:
            side = "BUY";  sl = row["close"] - atr * sl_mult; tp = row["close"] + atr * tp_mult
            in_trade = True
        elif st and not bt:
            side = "SELL"; sl = row["close"] + atr * sl_mult; tp = row["close"] - atr * tp_mult
            in_trade = True

    return trades


def max_drawdown_R(trades: list) -> float:
    if not trades:
        return 0.0
    equity = np.cumsum(trades)
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    return float(dd.max())


def exp1_monte_carlo(n_seeds: int = 30, n_bars: int = 700):
    print(f"\n[EXP 1] Monte Carlo — {n_seeds} seeds × 3 regimes × {n_bars} bars (fat-tailed, df=4)")
    results = {}
    for regime in ["bull", "bear", "chop"]:
        wrs, evs, dds, ns = [], [], [], []
        for seed in range(n_seeds):
            df = gen_ohlcv_t(n_bars, regime, seed=1000 + seed)
            df = indicators(df).dropna().reset_index(drop=True)
            trades = simulate_static_trades(df, CFG.atr_sl_mult, CFG.atr_tp_mult)
            if trades:
                arr = np.array(trades)
                wrs.append((arr > 0).mean())
                evs.append(arr.mean())
                dds.append(max_drawdown_R(trades))
                ns.append(len(trades))
        results[regime] = {
            "n_trades_mean": round(float(np.mean(ns)), 1),
            "win_rate_mean": round(float(np.mean(wrs)), 3),
            "win_rate_std":  round(float(np.std(wrs)), 3),
            "ev_mean":       round(float(np.mean(evs)), 4),
            "ev_std":        round(float(np.std(evs)), 4),
            "max_dd_mean":   round(float(np.mean(dds)), 2),
            "max_dd_std":    round(float(np.std(dds)), 2),
        }
        r = results[regime]
        print(f"  {regime:6}  n={r['n_trades_mean']:>5.1f} trades  "
              f"WR={r['win_rate_mean']:.1%}±{r['win_rate_std']:.1%}  "
              f"EV={r['ev_mean']:+.3f}±{r['ev_std']:.3f}  "
              f"MaxDD={r['max_dd_mean']:.1f}R±{r['max_dd_std']:.1f}R")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# EXP 2 — WALK-FORWARD MARKOV SERIES: static vs adaptive vs baseline
# ═══════════════════════════════════════════════════════════════════════════

def simulate_adaptive_trades(df: pd.DataFrame) -> list:
    """Regime-adaptive v2 engine — graduated risk_mult scales R directly."""
    trades = []
    in_trade = False
    side = sl = tp = None
    risk_mult = 1.0

    for _, row in df.iterrows():
        if in_trade:
            if side == "BUY":
                if row["low"] <= sl:
                    trades.append(-1.0 * risk_mult); in_trade = False
                elif row["high"] >= tp:
                    trades.append((CFG.atr_tp_mult / CFG.atr_sl_mult) * risk_mult); in_trade = False
            else:
                if row["high"] >= sl:
                    trades.append(-1.0 * risk_mult); in_trade = False
                elif row["low"] <= tp:
                    trades.append((CFG.atr_tp_mult / CFG.atr_sl_mult) * risk_mult); in_trade = False
            continue

        dec = adaptive_decide(BUY_R, SELL_R, row)
        if dec.signal == "FLAT" or dec.risk_mult == 0:
            continue
        atr = row["atr"]
        side = dec.signal
        risk_mult = dec.risk_mult
        if side == "BUY":
            sl = row["close"] - atr * CFG.atr_sl_mult
            tp = row["close"] + atr * CFG.atr_tp_mult
        else:
            sl = row["close"] + atr * CFG.atr_sl_mult
            tp = row["close"] - atr * CFG.atr_tp_mult
        in_trade = True

    return trades


def simulate_baseline_trades(df: pd.DataFrame, sl_mult: float = 1.5, tp_mult: float = 3.0) -> list:
    """Naive EMA8/EMA21 crossover, no filter, fixed 1.5/3.0 ATR SL/TP."""
    trades = []
    in_trade = False
    side = sl = tp = None

    for _, row in df.iterrows():
        if in_trade:
            if side == "BUY":
                if row["low"] <= sl:
                    trades.append(-1.0); in_trade = False
                elif row["high"] >= tp:
                    trades.append(tp_mult / sl_mult); in_trade = False
            else:
                if row["high"] >= sl:
                    trades.append(-1.0); in_trade = False
                elif row["low"] <= tp:
                    trades.append(tp_mult / sl_mult); in_trade = False
            continue

        sig = baseline_decide(row)
        if sig == "FLAT":
            continue
        atr = row["atr"]
        side = sig
        if side == "BUY":
            sl = row["close"] - atr * sl_mult; tp = row["close"] + atr * tp_mult
        else:
            sl = row["close"] + atr * sl_mult; tp = row["close"] - atr * tp_mult
        in_trade = True

    return trades


def sharpe_like(trades: list) -> float:
    if len(trades) < 2:
        return 0.0
    arr = np.array(trades)
    return float(arr.mean() / arr.std()) if arr.std() > 0 else 0.0


def exp2_walk_forward(n_seeds: int = 15, n_bars: int = 4000):
    print(f"\n[EXP 2] Walk-forward Markov regime-switching — {n_seeds} seeds × {n_bars} bars")
    print(f"         (one continuous series per seed, regime transitions mid-series)")

    engines = {"static": [], "adaptive": [], "baseline": []}

    for seed in range(n_seeds):
        df = gen_markov_series(n_bars, seed=2000 + seed)
        df = indicators(df).dropna().reset_index(drop=True)

        t_static   = simulate_static_trades(df, CFG.atr_sl_mult, CFG.atr_tp_mult)
        t_adaptive = simulate_adaptive_trades(df)
        t_baseline = simulate_baseline_trades(df)

        engines["static"].append(t_static)
        engines["adaptive"].append(t_adaptive)
        engines["baseline"].append(t_baseline)

    summary = {}
    for name, all_trades in engines.items():
        totals, dds, sharpes, ns = [], [], [], []
        for trades in all_trades:
            if not trades:
                continue
            totals.append(sum(trades))
            dds.append(max_drawdown_R(trades))
            sharpes.append(sharpe_like(trades))
            ns.append(len(trades))
        summary[name] = {
            "n_trades_mean":  round(float(np.mean(ns)), 1) if ns else 0,
            "total_R_mean":   round(float(np.mean(totals)), 2) if totals else 0,
            "total_R_std":    round(float(np.std(totals)), 2) if totals else 0,
            "max_dd_mean":    round(float(np.mean(dds)), 2) if dds else 0,
            "sharpe_mean":    round(float(np.mean(sharpes)), 3) if sharpes else 0,
        }
        s = summary[name]
        print(f"  {name:10}  n={s['n_trades_mean']:>6.1f} trades  "
              f"TotalR={s['total_R_mean']:>+7.2f}±{s['total_R_std']:.2f}  "
              f"MaxDD={s['max_dd_mean']:>5.1f}R  Sharpe={s['sharpe_mean']:+.3f}")

    return summary, engines


# ═══════════════════════════════════════════════════════════════════════════
# EXP 3 — REGIME CLASSIFIER ACCURACY (vs ground truth Markov label)
# ═══════════════════════════════════════════════════════════════════════════

def exp3_classifier_accuracy(n_bars: int = 4000, seed: int = 42):
    print(f"\n[EXP 3] Regime classifier accuracy vs ground-truth Markov regime")
    df = gen_markov_series(n_bars, seed=seed)
    df = indicators(df).dropna().reset_index(drop=True)

    df["detected"] = df.apply(classify_regime, axis=1)
    # Map detected 4-tier → 2-tier (tradeable vs not) for comparison against
    # ground truth's 3-tier (bull/bear/chop), since the classifier doesn't
    # know direction — it only knows trending vs ranging.
    df["detected_tradeable"] = df["detected"].isin(["STRONG_TREND", "MODERATE_TREND"])
    df["truth_tradeable"]    = df["true_regime"].isin(["bull", "bear"])

    agree = (df["detected_tradeable"] == df["truth_tradeable"]).mean()
    print(f"  Trending-vs-ranging agreement: {agree:.1%}  (n={len(df)} bars)")

    # Confusion breakdown
    tp = ((df["detected_tradeable"]) & (df["truth_tradeable"])).sum()
    fp = ((df["detected_tradeable"]) & (~df["truth_tradeable"])).sum()
    fn = ((~df["detected_tradeable"]) & (df["truth_tradeable"])).sum()
    tn = ((~df["detected_tradeable"]) & (~df["truth_tradeable"])).sum()
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    print(f"  Precision (traded & was real trend): {precision:.1%}")
    print(f"  Recall (of real trends, % detected): {recall:.1%}")

    return {"agreement": round(float(agree), 3), "precision": round(float(precision), 3),
            "recall": round(float(recall), 3), "tp": int(tp), "fp": int(fp),
            "fn": int(fn), "tn": int(tn)}


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t0 = time.perf_counter()
    print("=" * 78)
    print("  WCTE v2 — Monte Carlo + Walk-Forward Validation Suite")
    print("=" * 78)

    exp1 = exp1_monte_carlo(n_seeds=30, n_bars=700)
    exp2, engines = exp2_walk_forward(n_seeds=15, n_bars=4000)
    exp3 = exp3_classifier_accuracy(n_bars=4000, seed=42)

    elapsed = time.perf_counter() - t0
    print(f"\nTotal runtime: {elapsed:.1f}s")

    out = {"exp1_monte_carlo": exp1, "exp2_walk_forward": exp2, "exp3_classifier": exp3}
    Path("/home/claude/mt5_bot/benchmark_v2.json").write_text(json.dumps(out, indent=2))
    print("Saved: benchmark_v2.json")

    # Save one representative engine's trade list for chart generation
    np.save("/home/claude/mt5_bot/_static_trades_sample.npy", np.array(engines["static"][0]))
    np.save("/home/claude/mt5_bot/_adaptive_trades_sample.npy", np.array(engines["adaptive"][0]))
    np.save("/home/claude/mt5_bot/_baseline_trades_sample.npy", np.array(engines["baseline"][0]))
