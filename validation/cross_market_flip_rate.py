"""
Cross-Market Flip Rate -- Test 2 of Case A Validation.

H1: filtered regime flip rate scales with market microstructure fragility,
    so VNINDEX (frontier, retail-dominated) should exhibit a higher flip
    rate than the S&P 500 (developed, institutional) under the IDENTICAL
    hysteresis configuration. Bitcoin (24/7, no circuit breakers) acts as
    a natural upper-bound reference.
H0: flip rate is a property of the GMM/hysteresis machinery alone, not of
    market microstructure -- under identical config, all three markets
    produce statistically indistinguishable flip rates.

Methodology
-----------
- IDENTICAL hysteresis params across all three markets (delta_hard=0.60,
  delta_soft=0.35, t_persist=8). No per-market tuning.
- IDENTICAL GMM hyperparameters (n_components=3, covariance_type='full',
  n_init=10, random_state=42). Each market gets its own fit since
  cluster geometry is market-specific, but the hyperparameter recipe is
  fixed.
- Three windows / normalizations reported:
    HEADLINE      : common window 2022-01-01 -> 2026-04-17, native calendars.
    ROBUSTNESS 1  : same common window, BTC scaled to 252-day "equity year".
    ROBUSTNESS 2  : each market's FULL labelable window, native calendars.
- The 504-day rolling SPE_Z window precludes labeling the first ~2 years
  of each dataset; the March 2020 COVID shock is therefore unlabelable
  by construction. The common window starts at 2022-01-01 -- the earliest
  date by which all three markets have valid features -- and covers
  Russia-Ukraine war onset, global credit tightening (SVB, Credit Suisse),
  and Vietnamese domestic credit events (Tan Hoang Minh, bond freeze) --
  a sufficient cross-section of macro regimes to test microstructure
  dependency.

Run:
    python validation/cross_market_flip_rate.py
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from validation._features import DEFAULT_HYSTERESIS, TRADING_DAYS
from scripts.extract_flip_dates import _cached_pipeline


# ==============================================================================
# CONFIG
# ==============================================================================
COMMON_START = "2022-01-01"
COMMON_END   = "2026-04-17"
DATA_START   = "2020-01-01"
DATA_END     = "2026-04-17"

# Native calendar bars per year. VN/SPX trade ~252 weekdays minus holidays;
# BTC trades 24/7 (~365). These constants are the multipliers used to
# convert per-bar flip propensity into a flip-rate-per-calendar-year.
NATIVE_BPY = {
    "VNINDEX": 252,
    "SP500":   252,
    "BTC":     365,
}
EQUITY_BPY = 252  # used for robustness panel 1

# Threshold for declaring H1 supported on the headline panel.
RATIO_THRESHOLD = 1.5    # VN flip rate must exceed SPX by at least this factor
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
RESULT_PATH = os.path.join(RESULTS_DIR, "cross_market_result.json")


COVID_DISCLOSURE = (
    "The 504-day rolling SPE_Z window precludes labeling for approximately "
    "the first two years of each dataset. The COVID-19 shock (March 2020) "
    "falls in the unlabelable region. Cross-market headline comparison covers "
    "2022-01-01 onwards, spanning Russia-Ukraine war onset, global credit "
    "tightening (SVB, Credit Suisse), and Vietnamese domestic credit events "
    "(Tan Hoang Minh, bond market freeze) -- a range of macro regimes "
    "sufficient to test cross-market microstructure dependency."
)


# ==============================================================================
# DATACLASSES
# ==============================================================================
@dataclass
class MarketRow:
    market:           str
    n_bars:           int
    n_flips:          int
    years_native:     float
    years_equity:     float
    flips_per_yr:     float    # native if use_native else equity-equivalent
    window_start:     str
    window_end:       str
    multiplier:       int      # 252 or 365 used for this row


@dataclass
class TableResult:
    name: str
    description: str
    rows: Dict[str, MarketRow]                # keyed by market
    vn_over_sp_ratio: float
    vn_over_btc_ratio: float
    ordering: str                              # e.g. "BTC > VN > SPX"
    h1_supported: bool                          # VN > SPX AND ratio > RATIO_THRESHOLD


# ==============================================================================
# PIPELINE FETCH (cached)
# ==============================================================================
def fetch_market(market: str) -> dict:
    """Returns the full _features.run_full_pipeline output for one market."""
    print(f"  Fetching {market} ...")
    out = _cached_pipeline(market, DATA_START, DATA_END)
    feat = out["features"]
    print(
        f"    {market}: {len(out['ohlcv'])} OHLCV bars, "
        f"{len(feat)} labelable bars "
        f"({feat.index[0].date()} -> {feat.index[-1].date()})"
    )
    return out


# ==============================================================================
# FLIP-RATE STATS PER MARKET
# ==============================================================================
def _flips_in_window(
    labels: pd.Series,
    window: Tuple[str, str] | None,
) -> tuple[pd.Series, int, int]:
    """Restrict label series to window (or use full labelable region)."""
    if window is not None:
        s, e = pd.Timestamp(window[0]), pd.Timestamp(window[1])
        sub = labels[(labels.index >= s) & (labels.index <= e)]
    else:
        sub = labels
    n_bars = len(sub)
    n_flips = (
        int((np.diff(sub.astype(int).values) != 0).sum())
        if n_bars >= 2 else 0
    )
    return sub, n_bars, n_flips


def market_row(
    market: str,
    pipeline: dict,
    window: Tuple[str, str] | None,
    use_native: bool,
) -> MarketRow:
    sub, n_bars, n_flips = _flips_in_window(pipeline["filtered_labels"], window)
    if n_bars < 2:
        raise RuntimeError(f"{market}: window too small ({n_bars} bars).")

    native_mult = NATIVE_BPY[market]
    multiplier = native_mult if use_native else EQUITY_BPY
    years_native = n_bars / native_mult
    years_equity = n_bars / EQUITY_BPY
    flips_per_yr = n_flips * multiplier / n_bars

    return MarketRow(
        market=market,
        n_bars=n_bars,
        n_flips=n_flips,
        years_native=years_native,
        years_equity=years_equity,
        flips_per_yr=flips_per_yr,
        window_start=str(sub.index[0].date()),
        window_end=str(sub.index[-1].date()),
        multiplier=multiplier,
    )


def build_table(
    name: str,
    description: str,
    pipelines: Dict[str, dict],
    window: Tuple[str, str] | None,
    use_native: bool,
) -> TableResult:
    rows = {
        m: market_row(m, p, window, use_native)
        for m, p in pipelines.items()
    }
    vn = rows["VNINDEX"].flips_per_yr
    sp = rows["SP500"].flips_per_yr
    bt = rows["BTC"].flips_per_yr
    vn_sp = vn / sp if sp > 0 else float("inf")
    vn_bt = vn / bt if bt > 0 else float("inf")

    ordered = sorted(rows.items(), key=lambda kv: -kv[1].flips_per_yr)
    ordering_str = " > ".join(f"{m}({r.flips_per_yr:.1f})" for m, r in ordered)
    h1_supported = (vn > sp) and (vn_sp >= RATIO_THRESHOLD)

    return TableResult(
        name=name, description=description, rows=rows,
        vn_over_sp_ratio=vn_sp,
        vn_over_btc_ratio=vn_bt,
        ordering=ordering_str,
        h1_supported=h1_supported,
    )


# ==============================================================================
# REPORTING
# ==============================================================================
def _format_table(t: TableResult) -> str:
    header = (
        f"{'':<22}{'VN':>11}{'SPX':>11}{'BTC':>11}\n"
        f"{'-' * 55}\n"
    )
    fmt = lambda key, fn: (
        f"{key:<22}"
        f"{fn(t.rows['VNINDEX']):>11}"
        f"{fn(t.rows['SP500']):>11}"
        f"{fn(t.rows['BTC']):>11}\n"
    )
    body = (
        fmt("Flips/yr", lambda r: f"{r.flips_per_yr:.2f}")
        + fmt("n_flips", lambda r: f"{r.n_flips}")
        + fmt("n_bars", lambda r: f"{r.n_bars}")
        + fmt("Years (native)", lambda r: f"{r.years_native:.2f}")
        + fmt("Multiplier", lambda r: f"{r.multiplier}")
        + fmt("Window start", lambda r: r.window_start)
        + fmt("Window end",   lambda r: r.window_end)
    )
    tail = (
        f"\n  VN / SPX ratio:   {t.vn_over_sp_ratio:.2f}x\n"
        f"  VN / BTC ratio:   {t.vn_over_btc_ratio:.2f}x\n"
        f"  Ordering:         {t.ordering}\n"
        f"  H1 supported:     {t.h1_supported}  "
        f"(VN > SPX AND ratio >= {RATIO_THRESHOLD})\n"
    )
    return header + body + tail


def format_full_report(
    headline: TableResult,
    robust1: TableResult,
    robust2: TableResult,
) -> str:
    lines = [
        "=" * 70,
        "CROSS-MARKET FLIP RATE -- T2 RESULTS",
        "=" * 70,
        "",
        "Hysteresis config (identical across markets):",
        f"  delta_hard = {DEFAULT_HYSTERESIS['delta_hard']}",
        f"  delta_soft = {DEFAULT_HYSTERESIS['delta_soft']}",
        f"  t_persist  = {DEFAULT_HYSTERESIS['t_persist']}",
        "",
        f"HEADLINE -- {headline.description}",
        "-" * 70,
        _format_table(headline),
        "",
        f"ROBUSTNESS 1 -- {robust1.description}",
        "-" * 70,
        _format_table(robust1),
        "",
        f"ROBUSTNESS 2 -- {robust2.description}",
        "-" * 70,
        _format_table(robust2),
        "",
        "VERDICT",
        "-" * 70,
        f"  H1 (VN flip rate > SPX flip rate) on HEADLINE: "
        f"{'SUPPORTED' if headline.h1_supported else 'INCONCLUSIVE'}",
        f"  Robust across all three tables:                "
        f"{'YES' if all(t.h1_supported for t in (headline, robust1, robust2)) else 'NO'}",
        "",
        "  Per-table H1 status:",
        f"    HEADLINE      (common window, native cal.) : {headline.h1_supported}",
        f"    ROBUSTNESS 1  (common window, equity year) : {robust1.h1_supported}",
        f"    ROBUSTNESS 2  (full window,   native cal.) : {robust2.h1_supported}",
        "",
        "DISCLOSURE",
        "-" * 70,
        f"  {COVID_DISCLOSURE}",
    ]
    return "\n".join(lines)


def save_results(
    headline: TableResult,
    robust1: TableResult,
    robust2: TableResult,
    path: str = RESULT_PATH,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def _table_to_dict(t: TableResult) -> dict:
        return {
            "name": t.name,
            "description": t.description,
            "rows": {m: asdict(r) for m, r in t.rows.items()},
            "vn_over_sp_ratio": t.vn_over_sp_ratio,
            "vn_over_btc_ratio": t.vn_over_btc_ratio,
            "ordering": t.ordering,
            "h1_supported": t.h1_supported,
        }

    payload = {
        "config": {
            "hysteresis": dict(DEFAULT_HYSTERESIS),
            "common_window": [COMMON_START, COMMON_END],
            "data_window":   [DATA_START,   DATA_END],
            "ratio_threshold": RATIO_THRESHOLD,
            "native_bpy": NATIVE_BPY,
            "equity_bpy": EQUITY_BPY,
        },
        "covid_disclosure": COVID_DISCLOSURE,
        "headline":     _table_to_dict(headline),
        "robustness_1": _table_to_dict(robust1),
        "robustness_2": _table_to_dict(robust2),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ==============================================================================
# ENTRY
# ==============================================================================
def main() -> int:
    print("=" * 70)
    print(f"FETCHING MARKETS  ({DATA_START} -> {DATA_END})")
    print("=" * 70)
    pipelines = {
        "VNINDEX": fetch_market("VNINDEX"),
        "SP500":   fetch_market("SP500"),
        "BTC":     fetch_market("BTC"),
    }

    headline = build_table(
        name="HEADLINE",
        description=(
            f"common window {COMMON_START} -> {COMMON_END}, "
            "native calendars (BTC=365, VN/SPX=252)"
        ),
        pipelines=pipelines,
        window=(COMMON_START, COMMON_END),
        use_native=True,
    )
    robust1 = build_table(
        name="ROBUSTNESS_1",
        description=(
            f"common window {COMMON_START} -> {COMMON_END}, "
            "equity-equivalent normalization (all multiplied by 252)"
        ),
        pipelines=pipelines,
        window=(COMMON_START, COMMON_END),
        use_native=False,
    )
    robust2 = build_table(
        name="ROBUSTNESS_2",
        description="full per-market labelable window, native calendars",
        pipelines=pipelines,
        window=None,
        use_native=True,
    )

    print()
    print(format_full_report(headline, robust1, robust2))
    save_results(headline, robust1, robust2)
    print(f"\n  Result saved to: {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
