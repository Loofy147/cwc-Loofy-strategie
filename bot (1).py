"""
MT5 Rules-Based Trading Bot
Pure signal-rule → risk → execution pipeline
Designed for XAUUSD / BTC / any symbol

Usage:
    python bot.py --symbol XAUUSD --tf M15 --risk 0.01

Architecture:
    Data → Indicators → RuleEngine → RiskEngine → MT5Executor → SQLiteJournal
"""

from __future__ import annotations
import argparse
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

# ── MT5 import is optional (falls back to sim mode) ──────────────────────────
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    # Market
    symbol:          str   = "XAUUSD"
    timeframe:       str   = "M15"          # M1 M5 M15 M30 H1 H4 D1
    bars:            int   = 500

    # Risk
    risk_per_trade:  float = 0.01           # fraction of balance
    atr_sl_mult:     float = 1.5
    atr_tp_mult:     float = 3.0            # RR = 2.0
    max_spread_pts:  int   = 30
    max_positions:   int   = 1

    # Rules
    signal_threshold: float = 0.60          # MAJORITY gate
    filter_chop_max:  float = 61.8          # Choppiness ceiling
    rsi_oversold:     float = 40.0
    rsi_overbought:   float = 60.0

    # Loop
    poll_interval:   float = 5.0            # seconds

    # Broker
    login:    int = 0
    password: str = ""
    server:   str = ""

    # Internal
    magic:    int = 20250101

    def tf_const(self):
        if not MT5_AVAILABLE:
            return None
        mapping = {
            "M1": mt5.TIMEFRAME_M1,  "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,"M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,  "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }
        return mapping.get(self.timeframe, mt5.TIMEFRAME_M15)


# ═══════════════════════════════════════════════════════════════════════════════
# INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════

def indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all indicators in-place. Returns df with new columns."""
    c, h, l, v = df["close"], df["high"], df["low"], df["tick_volume"]

    # ── EMAs ──────────────────────────────────────────────────────────────────
    for n in (8, 21, 50, 200):
        df[f"ema{n}"] = c.ewm(span=n, adjust=False).mean()
    df["sma20"] = c.rolling(20).mean()

    # ── RSI(14) ───────────────────────────────────────────────────────────────
    d = c.diff()
    df["rsi"] = 100 - 100 / (
        1 + d.clip(lower=0).rolling(14).mean()
          / (-d.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
    )

    # ── MACD(12,26,9) ─────────────────────────────────────────────────────────
    df["macd"]   = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    df["macd_s"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_h"] = df["macd"] - df["macd_s"]

    # ── Bollinger(20,2) ───────────────────────────────────────────────────────
    std = c.rolling(20).std()
    df["bb_u"] = df["sma20"] + 2 * std
    df["bb_l"] = df["sma20"] - 2 * std
    df["bb_pct"] = ((c - df["bb_l"]) / (df["bb_u"] - df["bb_l"])).clip(0, 1)

    # ── ATR(14) ───────────────────────────────────────────────────────────────
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["tr1"] = tr

    # ── Stochastic(14,3) ──────────────────────────────────────────────────────
    lo, hi = l.rolling(14).min(), h.rolling(14).max()
    df["stoch_k"] = (100 * (c - lo) / (hi - lo).replace(0, np.nan)).fillna(50)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # ── Choppiness Index(14) ──────────────────────────────────────────────────
    atr_sum = df["tr1"].rolling(14).sum()
    price_range = (h.rolling(14).max() - l.rolling(14).min()).replace(0, np.nan)
    df["chop"] = 100 * np.log10(atr_sum / price_range) / np.log10(14)

    # ── Volume ratio ──────────────────────────────────────────────────────────
    df["vol_ratio"] = (v / v.rolling(20).mean()).fillna(1.0)

    # ── Momentum ──────────────────────────────────────────────────────────────
    df["mom10"] = c.pct_change(10)

    return df.copy()


# ═══════════════════════════════════════════════════════════════════════════════
# RULE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Rule:
    name:   str
    fn:     Callable[[pd.Series], bool]
    weight: float = 1.0
    group:  str   = ""          # optional grouping for diagnostics

    def eval(self, row: pd.Series) -> bool:
        try:
            return bool(self.fn(row))
        except Exception:
            return False


@dataclass
class RuleSet:
    name:      str
    rules:     List[Rule]
    logic:     Literal["AND", "OR", "MAJORITY", "WEIGHTED"] = "MAJORITY"
    threshold: float = 0.60

    def score(self, row: pd.Series) -> float:
        tw = sum(r.weight for r in self.rules)
        hw = sum(r.weight for r in self.rules if r.eval(row))
        return hw / tw if tw else 0.0

    def triggered(self, row: pd.Series) -> bool:
        match self.logic:
            case "AND":      return all(r.eval(row) for r in self.rules)
            case "OR":       return any(r.eval(row) for r in self.rules)
            case "MAJORITY" | "WEIGHTED":
                             return self.score(row) >= self.threshold

    def breakdown(self, row: pd.Series) -> Dict[str, bool]:
        return {r.name: r.eval(row) for r in self.rules}


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL DEFINITIONS  ← edit here to tune the strategy
# ═══════════════════════════════════════════════════════════════════════════════

def build_signals(cfg: Config) -> Tuple[RuleSet, RuleSet, RuleSet]:

    # ── BUY rules ─────────────────────────────────────────────────────────────
    buy = RuleSet(
        name="BUY",
        logic="MAJORITY",
        threshold=cfg.signal_threshold,
        rules=[
            # Trend
            Rule("ema_stack_up",      lambda r: r.ema8 > r.ema21 > r.ema50,                       weight=2.5, group="trend"),
            Rule("price_gt_ema21",    lambda r: r.close > r.ema21,                                weight=1.5, group="trend"),
            Rule("above_ema200",      lambda r: r.close > r.ema200,                               weight=1.0, group="trend"),
            # Momentum
            Rule("rsi_bull_zone",     lambda r: cfg.rsi_oversold < r.rsi < 70,                    weight=1.5, group="momentum"),
            Rule("macd_bull",         lambda r: r.macd > r.macd_s and r.macd_h > 0,              weight=2.0, group="momentum"),
            Rule("stoch_cross_up",    lambda r: r.stoch_k > r.stoch_d and r.stoch_k < 80,        weight=1.0, group="momentum"),
            Rule("pos_momentum",      lambda r: r.mom10 > 0,                                      weight=1.0, group="momentum"),
            # Mean reversion bonus
            Rule("bb_lower_half",     lambda r: r.bb_pct < 0.50,                                  weight=0.5, group="structure"),
            # Volume
            Rule("vol_confirm",       lambda r: r.vol_ratio > 1.05,                               weight=1.0, group="volume"),
        ]
    )

    # ── SELL rules ────────────────────────────────────────────────────────────
    sell = RuleSet(
        name="SELL",
        logic="MAJORITY",
        threshold=cfg.signal_threshold,
        rules=[
            Rule("ema_stack_down",    lambda r: r.ema8 < r.ema21 < r.ema50,                       weight=2.5, group="trend"),
            Rule("price_lt_ema21",    lambda r: r.close < r.ema21,                                weight=1.5, group="trend"),
            Rule("below_ema200",      lambda r: r.close < r.ema200,                               weight=1.0, group="trend"),
            Rule("rsi_bear_zone",     lambda r: cfg.rsi_overbought > r.rsi > 30,                  weight=1.5, group="momentum"),
            Rule("macd_bear",         lambda r: r.macd < r.macd_s and r.macd_h < 0,              weight=2.0, group="momentum"),
            Rule("stoch_cross_down",  lambda r: r.stoch_k < r.stoch_d and r.stoch_k > 20,        weight=1.0, group="momentum"),
            Rule("neg_momentum",      lambda r: r.mom10 < 0,                                      weight=1.0, group="momentum"),
            Rule("bb_upper_half",     lambda r: r.bb_pct > 0.50,                                  weight=0.5, group="structure"),
            Rule("vol_confirm",       lambda r: r.vol_ratio > 1.05,                               weight=1.0, group="volume"),
        ]
    )

    # ── MARKET FILTER (AND gate — all must pass) ──────────────────────────────
    filt = RuleSet(
        name="FILTER",
        logic="AND",
        rules=[
            Rule("not_choppy",   lambda r: r.chop < cfg.filter_chop_max, weight=1.0),
            Rule("atr_positive", lambda r: r.atr > 0,                    weight=1.0),
            Rule("vol_active",   lambda r: r.vol_ratio > 0.25,           weight=1.0),
        ]
    )

    return buy, sell, filt


# ═══════════════════════════════════════════════════════════════════════════════
# RISK ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def sl_tp(row: pd.Series, side: str, sl_mult: float, tp_mult: float, digits: int) -> Tuple[float, float]:
    atr = row["atr"]
    p   = row["close"]
    if side == "BUY":
        return round(p - atr * sl_mult, digits), round(p + atr * tp_mult, digits)
    else:
        return round(p + atr * sl_mult, digits), round(p - atr * tp_mult, digits)


def lot_size(balance: float, risk_pct: float, sl_dist: float,
             tick_val: float, tick_sz: float,
             lot_min: float, lot_max: float, lot_step: float) -> float:
    if sl_dist <= 0 or tick_sz <= 0:
        return lot_min
    ticks   = sl_dist / tick_sz
    raw_lot = (balance * risk_pct) / (ticks * tick_val)
    stepped = round(raw_lot / lot_step) * lot_step
    return round(max(lot_min, min(lot_max, stepped)), 2)


# ═══════════════════════════════════════════════════════════════════════════════
# JOURNAL
# ═══════════════════════════════════════════════════════════════════════════════

class Journal:
    def __init__(self, path: str = "trades.db"):
        self.conn = sqlite3.connect(path)
        self._migrate()

    def _migrate(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS signals (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                ts         TEXT, symbol TEXT,
                buy_score  REAL, sell_score REAL,
                triggered  TEXT, bar_close REAL, atr REAL
            );
            CREATE TABLE IF NOT EXISTS trades (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                ts         TEXT, symbol TEXT, side TEXT,
                lot        REAL, entry REAL, sl REAL, tp REAL,
                exit_price REAL, pnl REAL,
                buy_score  REAL, sell_score REAL,
                ticket     INTEGER, status TEXT
            );
        """)
        self.conn.commit()

    def log_signal(self, ts, symbol, buy_score, sell_score, triggered, bar_close, atr):
        self.conn.execute(
            "INSERT INTO signals (ts,symbol,buy_score,sell_score,triggered,bar_close,atr) "
            "VALUES (?,?,?,?,?,?,?)",
            (str(ts), symbol, round(buy_score,4), round(sell_score,4),
             triggered, bar_close, atr)
        )
        self.conn.commit()

    def log_trade(self, **kw):
        cols = ",".join(kw.keys())
        vals = tuple(kw.values())
        q    = ",".join("?" * len(kw))
        self.conn.execute(f"INSERT INTO trades ({cols}) VALUES ({q})", vals)
        self.conn.commit()

    def summary(self) -> pd.DataFrame:
        return pd.read_sql("SELECT * FROM trades ORDER BY id DESC LIMIT 50", self.conn)


# ═══════════════════════════════════════════════════════════════════════════════
# MT5 BOT
# ═══════════════════════════════════════════════════════════════════════════════

class MT5Bot:
    def __init__(self, cfg: Config):
        self.cfg  = cfg
        self.j    = Journal()
        self._setup_log()
        self.buy_rules = self.sell_rules = self.filt_rules = None

    def _setup_log(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.FileHandler("bot.log"), logging.StreamHandler()],
        )
        self.log = logging.getLogger("MT5Bot")

    # ── MT5 helpers ───────────────────────────────────────────────────────────

    def connect(self) -> bool:
        if not MT5_AVAILABLE:
            self.log.error("MetaTrader5 package not installed")
            return False
        ok = mt5.initialize(
            login=self.cfg.login,
            password=self.cfg.password,
            server=self.cfg.server,
        )
        if not ok:
            self.log.error(f"MT5 init failed: {mt5.last_error()}")
            return False
        info = mt5.account_info()
        self.log.info(f"Connected — {self.cfg.server} | Balance: {info.balance:.2f} {info.currency}")
        return True

    def disconnect(self):
        if MT5_AVAILABLE:
            mt5.shutdown()
        self.log.info("Disconnected")

    def _data(self) -> Optional[pd.DataFrame]:
        rates = mt5.copy_rates_from_pos(self.cfg.symbol, self.cfg.tf_const(), 0, self.cfg.bars)
        if rates is None or len(rates) < 201:
            return None
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return indicators(df).dropna()

    def _positions(self) -> list:
        p = mt5.positions_get(symbol=self.cfg.symbol)
        return list(p) if p else []

    def _spread_ok(self) -> bool:
        tick  = mt5.symbol_info_tick(self.cfg.symbol)
        sinfo = mt5.symbol_info(self.cfg.symbol)
        if not tick or not sinfo:
            return False
        spread = int((tick.ask - tick.bid) / sinfo.point)
        return spread <= self.cfg.max_spread_pts

    def _send(self, side: str, volume: float, sl: float, tp: float) -> Optional[int]:
        tick  = mt5.symbol_info_tick(self.cfg.symbol)
        sinfo = mt5.symbol_info(self.cfg.symbol)
        price = tick.ask if side == "BUY" else tick.bid
        otype = mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL
        req = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       self.cfg.symbol,
            "volume":       volume,
            "type":         otype,
            "price":        price,
            "sl":           sl,
            "tp":           tp,
            "deviation":    20,
            "magic":        self.cfg.magic,
            "comment":      "RulesBot",
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        res = mt5.order_send(req)
        if res.retcode != mt5.TRADE_RETCODE_DONE:
            self.log.error(f"Order rejected: [{res.retcode}] {res.comment}")
            return None
        self.log.info(f"✓ {side} {volume} {self.cfg.symbol} @ {res.price} | SL={sl} TP={tp} | #{res.order}")
        return res.order

    def _close(self, pos) -> bool:
        tick      = mt5.symbol_info_tick(self.cfg.symbol)
        is_buy    = pos.type == mt5.ORDER_TYPE_BUY
        close_t   = mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY
        close_p   = tick.bid            if is_buy else tick.ask
        req = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       self.cfg.symbol,
            "volume":       pos.volume,
            "type":         close_t,
            "position":     pos.ticket,
            "price":        close_p,
            "deviation":    20,
            "magic":        self.cfg.magic,
            "comment":      "RulesBot close",
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        res = mt5.order_send(req)
        if res.retcode != mt5.TRADE_RETCODE_DONE:
            self.log.error(f"Close failed: {res.retcode}")
            return False
        self.log.info(f"✓ Closed #{pos.ticket}")
        return True

    # ── Core loop ─────────────────────────────────────────────────────────────

    def run(self):
        self.buy_rules, self.sell_rules, self.filt_rules = build_signals(self.cfg)
        self.log.info(f"Rules → BUY({len(self.buy_rules.rules)}) "
                      f"SELL({len(self.sell_rules.rules)}) FILTER({len(self.filt_rules.rules)})")

        last_bar = None

        while True:
            try:
                df = self._data()
                if df is None:
                    time.sleep(self.cfg.poll_interval)
                    continue

                bar = df.iloc[-2]           # last closed bar
                if bar["time"] == last_bar:
                    time.sleep(self.cfg.poll_interval)
                    continue
                last_bar = bar["time"]

                # ── Filter ────────────────────────────────────────────────────
                if not self.filt_rules.triggered(bar):
                    self.log.debug(f"Filter blocked | chop={bar.chop:.1f}")
                    time.sleep(self.cfg.poll_interval)
                    continue

                b_score = self.buy_rules.score(bar)
                s_score = self.sell_rules.score(bar)
                b_trig  = self.buy_rules.triggered(bar)
                s_trig  = self.sell_rules.triggered(bar)

                self.log.info(
                    f"Bar {bar.time} | close={bar.close:.3f} | "
                    f"BUY={b_score:.2f}{'✓' if b_trig else '✗'} "
                    f"SELL={s_score:.2f}{'✓' if s_trig else '✗'} | "
                    f"chop={bar.chop:.1f} rsi={bar.rsi:.1f} atr={bar.atr:.3f}"
                )

                self.j.log_signal(
                    bar.time, self.cfg.symbol, b_score, s_score,
                    "BUY" if b_trig else ("SELL" if s_trig else "NONE"),
                    bar.close, bar.atr
                )

                positions = self._positions()

                # ── Entry ─────────────────────────────────────────────────────
                if len(positions) < self.cfg.max_positions:
                    if b_trig and not s_trig:
                        self._enter("BUY",  bar, b_score, s_score)
                    elif s_trig and not b_trig:
                        self._enter("SELL", bar, b_score, s_score)

                # ── Exit on opposing signal ───────────────────────────────────
                for pos in positions:
                    pos_side = "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL"
                    if pos_side == "BUY"  and s_trig:
                        self.log.info(f"Flip signal → closing BUY #{pos.ticket}")
                        self._close(pos)
                    elif pos_side == "SELL" and b_trig:
                        self.log.info(f"Flip signal → closing SELL #{pos.ticket}")
                        self._close(pos)

            except KeyboardInterrupt:
                self.log.info("Interrupted — stopping")
                break
            except Exception as e:
                self.log.error(f"Loop error: {e}", exc_info=True)

            time.sleep(self.cfg.poll_interval)

    def _enter(self, side: str, bar: pd.Series, b_score: float, s_score: float):
        if not self._spread_ok():
            self.log.warning("Spread too wide — skipped")
            return

        sinfo   = mt5.symbol_info(self.cfg.symbol)
        acct    = mt5.account_info()
        sl, tp  = sl_tp(bar, side, self.cfg.atr_sl_mult, self.cfg.atr_tp_mult, sinfo.digits)
        sl_dist = abs(bar.close - sl)
        vol     = lot_size(
            acct.balance, self.cfg.risk_per_trade, sl_dist,
            sinfo.trade_tick_value, sinfo.trade_tick_size,
            sinfo.volume_min, sinfo.volume_max, sinfo.volume_step,
        )

        ticket = self._send(side, vol, sl, tp)
        if ticket:
            self.j.log_trade(
                ts=str(bar.time), symbol=self.cfg.symbol, side=side,
                lot=vol, entry=bar.close, sl=sl, tp=tp,
                exit_price=None, pnl=None,
                buy_score=round(b_score, 4), sell_score=round(s_score, 4),
                ticket=ticket, status="OPEN",
            )


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="MT5 Rules-Based Bot")
    p.add_argument("--symbol",   default="XAUUSD")
    p.add_argument("--tf",       default="M15")
    p.add_argument("--bars",     type=int,   default=500)
    p.add_argument("--risk",     type=float, default=0.01)
    p.add_argument("--sl-mult",  type=float, default=1.5)
    p.add_argument("--tp-mult",  type=float, default=3.0)
    p.add_argument("--threshold",type=float, default=0.60)
    p.add_argument("--login",    type=int,   default=0)
    p.add_argument("--password", default="")
    p.add_argument("--server",   default="")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = Config(
        symbol           = args.symbol,
        timeframe        = args.tf,
        bars             = args.bars,
        risk_per_trade   = args.risk,
        atr_sl_mult      = args.sl_mult,
        atr_tp_mult      = args.tp_mult,
        signal_threshold = args.threshold,
        login            = args.login,
        password         = args.password,
        server           = args.server,
    )
    bot = MT5Bot(cfg)
    if bot.connect():
        try:
            bot.run()
        finally:
            bot.disconnect()
