"""
config_optimized.py
Derived from benchmark + RR sweep.

Key findings:
  - sl_mult=2.0 / tp_mult=4.0 (RR 2:1) maximizes avg EV across all regimes (+0.6463)
  - threshold=0.55 balances signal frequency vs noise suppression
  - chop_max=58.0 drops chop filter pass to 32% without cutting real signals
  - ema_stack_up (weight=2.5) is the dominant rule — never reduce it
  - stoch_cross_up and bb_lower_half are anti-correlated with trend;
    they act as mean-reversion gates — acceptable at low weight
"""

from bot import Config

XAUUSD_M15 = Config(
    symbol           = "XAUUSD",
    timeframe        = "M15",
    bars             = 500,
    risk_per_trade   = 0.01,       # 1% balance risk per trade
    atr_sl_mult      = 2.0,        # ← optimized (was 1.5)
    atr_tp_mult      = 4.0,        # ← optimized (was 3.0) → RR 2:1
    max_spread_pts   = 30,
    max_positions    = 1,
    signal_threshold = 0.55,       # ← optimized (was 0.60)
    filter_chop_max  = 58.0,       # ← optimized (was 61.8)
    rsi_oversold     = 40.0,
    rsi_overbought   = 60.0,
    poll_interval    = 5.0,
    magic            = 20250101,
)

BTC_M5 = Config(
    symbol           = "BTCUSD",
    timeframe        = "M5",
    bars             = 500,
    risk_per_trade   = 0.005,      # 0.5% — higher vol
    atr_sl_mult      = 2.0,
    atr_tp_mult      = 4.0,
    max_spread_pts   = 50,
    max_positions    = 1,
    signal_threshold = 0.55,
    filter_chop_max  = 58.0,
    poll_interval    = 3.0,
    magic            = 20250102,
)

# Usage:
#   from config_optimized import XAUUSD_M15
#   bot = MT5Bot(XAUUSD_M15)
