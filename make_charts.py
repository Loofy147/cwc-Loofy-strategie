"""
make_charts.py — generates all PNG visual assets for the SKILL.md submission.
Every chart is built from data actually computed in benchmark_v2.py / this run.
No illustrative/fake data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from bot import Config, indicators, build_signals
from sim_data import gen_markov_series
import regime_adaptive as ra
from benchmark_v2 import (simulate_static_trades, simulate_adaptive_trades,
                            simulate_baseline_trades, max_drawdown_R, sharpe_like)

OUT = Path("/home/claude/mt5_bot/examples")
OUT.mkdir(exist_ok=True)

# Consistent dark-on-light finance-chart style, no seaborn dependency
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.grid": True,
    "grid.color": "#e0e0e0",
    "grid.linewidth": 0.6,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
}
)

COLOR_STATIC   = "#2563eb"   # blue
COLOR_ADAPTIVE = "#059669"   # green
COLOR_BASELINE = "#dc2626"   # red
COLOR_BULL     = "#d1fae5"
COLOR_BEAR     = "#fee2e2"
COLOR_CHOP     = "#f3f4f6"

CFG = Config(signal_threshold=0.55, filter_chop_max=58.0, atr_sl_mult=2.0, atr_tp_mult=4.0)
BUY_R, SELL_R, FILT_R = build_signals(CFG)

ra.CI_STRONG, ra.CI_CAUTION, ra.CI_BLOCK = 45.0, 58.0, 65.0
ra.REGIME_PARAMS = {
    "STRONG_TREND":   {"trend_mult": 1.25, "threshold": 0.55, "risk_mult": 1.00},
    "MODERATE_TREND": {"trend_mult": 1.00, "threshold": 0.55, "risk_mult": 1.00},
    "CAUTION":        {"trend_mult": 1.00, "threshold": 0.65, "risk_mult": 0.50},
    "RANGING":        {"trend_mult": 1.00, "threshold": 1.01, "risk_mult": 0.00},
}


# ═══════════════════════════════════════════════════════════════════════════
# CHART 1 — Equity curves: static vs adaptive vs baseline (median-perf seed)
# ═══════════════════════════════════════════════════════════════════════════

def chart_equity_curves():
    seed = 5000 + 14   # the median-performing static seed found in the 25-seed run
    df = gen_markov_series(4000, seed=seed)
    df = indicators(df).dropna().reset_index(drop=True)

    t_static   = simulate_static_trades(df, CFG.atr_sl_mult, CFG.atr_tp_mult)
    t_adaptive = simulate_adaptive_trades(df)
    t_baseline = simulate_baseline_trades(df)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for trades, label, color in [
        (t_static,   f"WCTE Static (n={len(t_static)}, final={sum(t_static):+.1f}R)",     COLOR_STATIC),
        (t_adaptive, f"WCTE Adaptive (n={len(t_adaptive)}, final={sum(t_adaptive):+.1f}R)", COLOR_ADAPTIVE),
        (t_baseline, f"Naive EMA Crossover (n={len(t_baseline)}, final={sum(t_baseline):+.1f}R)", COLOR_BASELINE),
    ]:
        equity = np.concatenate([[0], np.cumsum(trades)])
        ax.plot(equity, label=label, color=color, linewidth=1.8)

    ax.set_xlabel("Trade number")
    ax.set_ylabel("Cumulative R (risk-multiples)")
    ax.set_title("Equity Curves — Single Representative Walk-Forward Series\n"
                  "(Markov regime-switching, 4000 M15 bars, median-performing seed)")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.axhline(0, color="#888", linewidth=0.8, linestyle="--")
    fig.tight_layout()
    fig.savefig(OUT / "equity_curves.png", dpi=150)
    plt.close(fig)
    print("Saved equity_curves.png")


# ═══════════════════════════════════════════════════════════════════════════
# CHART 2 — Price + regime timeline with trade entry markers
# ═══════════════════════════════════════════════════════════════════════════

def chart_regime_timeline():
    seed = 5000 + 14
    df = gen_markov_series(4000, seed=seed)
    df = indicators(df).dropna().reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1]})

    # Background shading by true regime
    regime_colors = {"bull": COLOR_BULL, "bear": COLOR_BEAR, "chop": COLOR_CHOP}
    prev = df["true_regime"].iloc[0]
    start = 0
    for i in range(1, len(df)):
        if df["true_regime"].iloc[i] != prev or i == len(df) - 1:
            ax1.axvspan(start, i, color=regime_colors[prev], alpha=0.6, linewidth=0)
            start = i
            prev = df["true_regime"].iloc[i]

    ax1.plot(df.index, df["close"], color="#1f2937", linewidth=0.9)

    # Trade markers (static engine)
    in_trade, side, entry_i = False, None, None
    buy_x, buy_y, sell_x, sell_y = [], [], [], []
    for i, row in df.iterrows():
        if in_trade:
            atr = None
            continue
    # Simpler: re-derive entries directly
    in_trade = False
    side = sl = tp = None
    for i, row in df.iterrows():
        if in_trade:
            if side == "BUY":
                if row["low"] <= sl or row["high"] >= tp:
                    in_trade = False
            else:
                if row["high"] >= sl or row["low"] <= tp:
                    in_trade = False
            continue
        if not FILT_R.triggered(row):
            continue
        bt, st = BUY_R.triggered(row), SELL_R.triggered(row)
        atr = row["atr"]
        if bt and not st:
            buy_x.append(i); buy_y.append(row["close"])
            side = "BUY"; sl = row["close"] - atr * CFG.atr_sl_mult; tp = row["close"] + atr * CFG.atr_tp_mult
            in_trade = True
        elif st and not bt:
            sell_x.append(i); sell_y.append(row["close"])
            side = "SELL"; sl = row["close"] + atr * CFG.atr_sl_mult; tp = row["close"] - atr * CFG.atr_tp_mult
            in_trade = True

    ax1.scatter(buy_x, buy_y, marker="^", color=COLOR_ADAPTIVE, s=28, zorder=5, label=f"BUY entry (n={len(buy_x)})")
    ax1.scatter(sell_x, sell_y, marker="v", color=COLOR_BASELINE, s=28, zorder=5, label=f"SELL entry (n={len(sell_x)})")

    legend_elems = [Patch(facecolor=COLOR_BULL, label="True regime: bull"),
                    Patch(facecolor=COLOR_BEAR, label="True regime: bear"),
                    Patch(facecolor=COLOR_CHOP, label="True regime: chop")]
    ax1.legend(handles=legend_elems + ax1.get_legend_handles_labels()[0],
               loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
    ax1.set_ylabel("Price")
    ax1.set_title("WCTE Static Engine — Trade Entries Over a Regime-Switching Walk-Forward Series\n"
                   "(background shading = ground-truth Markov regime, not visible to the engine)")

    ax2.plot(df.index, df["chop"], color="#7c3aed", linewidth=0.9, label="Choppiness Index (14)")
    ax2.axhline(58.0, color=COLOR_BASELINE, linestyle="--", linewidth=1.0, label="Filter cutoff (58.0)")
    ax2.set_ylabel("CI(14)")
    ax2.set_xlabel("Bar index (M15)")
    ax2.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "regime_timeline.png", dpi=150)
    plt.close(fig)
    print("Saved regime_timeline.png")


# ═══════════════════════════════════════════════════════════════════════════
# CHART 3 — Rule importance (score sensitivity)
# ═══════════════════════════════════════════════════════════════════════════

def chart_rule_importance():
    from sim_data import gen_ohlcv_t
    df = gen_ohlcv_t(600, "bull", seed=7)
    df = indicators(df).dropna().reset_index(drop=True)

    base_scores = []
    delta_scores = {r.name: [] for r in BUY_R.rules}
    for _, row in df.iterrows():
        if not FILT_R.triggered(row):
            continue
        base = BUY_R.score(row)
        for target in BUY_R.rules:
            saved = target.weight
            target.weight = 0.0
            delta_scores[target.name].append(BUY_R.score(row) - base)
            target.weight = saved

    means = {k: np.mean(v) for k, v in delta_scores.items() if v}
    names = sorted(means, key=lambda k: means[k])
    vals  = [means[k] for k in names]
    colors = [COLOR_STATIC if v < 0 else COLOR_BASELINE for v in vals]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(names, vals, color=colors)
    ax.axvline(0, color="#333", linewidth=0.8)
    ax.set_xlabel("Mean score Δ when rule weight is zeroed (bull regime)")
    ax.set_title("Rule Importance — Score Sensitivity Analysis\n"
                  "(negative = rule drives score up; positive = rule is anti-correlated with trend)")
    fig.tight_layout()
    fig.savefig(OUT / "rule_importance.png", dpi=150)
    plt.close(fig)
    print("Saved rule_importance.png")


# ═══════════════════════════════════════════════════════════════════════════
# CHART 4 — Monte Carlo Sharpe / EV comparison (25 seeds) with error bars
# ═══════════════════════════════════════════════════════════════════════════

def chart_mc_comparison():
    N_SEEDS = 25
    engines = {"static": [], "adaptive": [], "baseline": []}
    for seed in range(N_SEEDS):
        df = gen_markov_series(4000, seed=5000 + seed)
        df = indicators(df).dropna().reset_index(drop=True)
        engines["static"].append(simulate_static_trades(df, CFG.atr_sl_mult, CFG.atr_tp_mult))
        engines["adaptive"].append(simulate_adaptive_trades(df))
        engines["baseline"].append(simulate_baseline_trades(df))

    labels = ["WCTE\nStatic", "WCTE\nAdaptive", "Naive\nEMA Crossover"]
    colors = [COLOR_STATIC, COLOR_ADAPTIVE, COLOR_BASELINE]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    for ax, metric_name, metric_fn, ylabel in [
        (axes[0], "Sharpe-like ratio", sharpe_like, "mean / std of per-trade R"),
        (axes[1], "Max Drawdown", max_drawdown_R, "R-multiples"),
        (axes[2], "EV per trade", lambda t: sum(t) / len(t) if t else 0, "R-multiples"),
    ]:
        means, stds = [], []
        for name in ["static", "adaptive", "baseline"]:
            vals = [metric_fn(t) for t in engines[name] if t]
            means.append(np.mean(vals))
            stds.append(np.std(vals) / np.sqrt(len(vals)))  # SEM
        bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors, alpha=0.85)
        ax.set_title(metric_name, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.axhline(0, color="#888", linewidth=0.6)

    fig.suptitle(f"Monte Carlo Comparison — {N_SEEDS} independent walk-forward series (error bars = SEM)",
                 fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "mc_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved mc_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════
# CHART 5 — Win rate distributions by regime (from clean-regime Monte Carlo)
# ═══════════════════════════════════════════════════════════════════════════

def chart_winrate_distribution():
    from sim_data import gen_ohlcv_t
    from benchmark_v2 import simulate_static_trades as sim

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"bull": COLOR_ADAPTIVE, "bear": COLOR_STATIC, "chop": "#f59e0b"}

    for regime in ["bull", "bear", "chop"]:
        wrs = []
        for seed in range(30):
            df = gen_ohlcv_t(700, regime, seed=1000 + seed)
            df = indicators(df).dropna().reset_index(drop=True)
            trades = sim(df, CFG.atr_sl_mult, CFG.atr_tp_mult)
            if trades:
                arr = np.array(trades)
                wrs.append((arr > 0).mean())
        ax.hist(wrs, bins=12, alpha=0.55, label=f"{regime} (μ={np.mean(wrs):.1%})", color=colors[regime])

    ax.axvline(0.333, color="#333", linestyle=":", linewidth=1.2,
               label="Breakeven WR at 2:1 RR (33.3%)")
    ax.set_xlabel("Win rate (per 700-bar seed)")
    ax.set_ylabel("Seed count")
    ax.set_title("Win Rate Distribution Across 30 Monte Carlo Seeds per Regime\n"
                  "(fat-tailed synthetic data, Student-t df=4)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "winrate_distribution.png", dpi=150)
    plt.close(fig)
    print("Saved winrate_distribution.png")


if __name__ == "__main__":
    chart_equity_curves()
    chart_regime_timeline()
    chart_rule_importance()
    chart_mc_comparison()
    chart_winrate_distribution()
    print("\nAll charts saved to", OUT)
