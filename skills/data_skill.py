"""
Data Ingestion Layer -- Financial Entropy Agent
Thu thap du lieu thi truong VN-Index va ro VN30 tu vnstock / yfinance.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from functools import lru_cache


# ==============================================================================
# VN30 TICKER LIST (HOSE)
# ==============================================================================
VN30_TICKERS_YF: list[str] = [
    "ACB.VN", "BCM.VN", "BID.VN", "BVH.VN", "CTG.VN", "FPT.VN",
    "GAS.VN", "GVR.VN", "HDB.VN", "HPG.VN", "MBB.VN", "MSN.VN",
    "MWG.VN", "PLX.VN", "POW.VN", "SAB.VN", "SHB.VN", "STB.VN",
    "TCB.VN", "TPB.VN", "VCB.VN", "VHM.VN", "VIB.VN", "VIC.VN",
    "VJC.VN", "VNM.VN", "VPB.VN", "VRE.VN",
]

VN30_TICKERS_VCI: list[str] = [t.replace(".VN", "") for t in VN30_TICKERS_YF]


# ==============================================================================
# CORE: VN-INDEX OHLCV
# ==============================================================================
def fetch_vnindex(
    ticker: str = "VNINDEX",
    start_date: str = "2020-01-01",
    end_date: str | None = None,
    source: str = "VCI",
) -> pd.DataFrame:
    """
    Lay du lieu OHLCV cua VN-Index tu vnstock.
    Returns: DataFrame voi DatetimeIndex, columns = [Open, High, Low, Close, Volume].
    """
    from vnstock import Vnstock

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    stock = Vnstock().stock(symbol=ticker, source=source)
    df: pd.DataFrame = stock.quote.history(start=start_date, end=end_date)

    # Chuan hoa columns
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        },
        inplace=True,
    )

    # Forward fill missing values
    df.ffill(inplace=True)
    return df


# ==============================================================================
# CORE: VN30 COMPONENT RETURNS
# ==============================================================================
def fetch_vn30_returns(
    start_date: str = "2020-01-01",
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Lay bang returns (pct_change) cua ro VN30 tu yfinance.
    Returns: DataFrame NxM, moi cot la 1 ticker, gia tri la daily return.
    """
    import yfinance as yf

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    df: pd.DataFrame = yf.download(
        VN30_TICKERS_YF, start=start_date, end=end_date
    )["Close"]

    return df.ffill().pct_change().dropna(how="all")


# ==============================================================================
# FALLBACK: LOCAL FILE
# ==============================================================================
def load_local_file(path: str) -> pd.DataFrame:
    """
    Doc du lieu OHLCV tu file CSV hoac Excel (fallback khi API bi rate-limit).
    Tu dong detect cot ngay va chuan hoa column names.
    """
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Dinh dang file khong ho tro: {path}")

    # Detect date column
    date_candidates = [c for c in df.columns if str(c).lower().strip() in ("date", "time", "ngay", "ngày")]
    if date_candidates:
        df[date_candidates[0]] = pd.to_datetime(df[date_candidates[0]])
        df.set_index(date_candidates[0], inplace=True)
        df.sort_index(inplace=True)

    # Chuan hoa ten cot
    col_map: dict[str, str] = {}
    for c in df.columns:
        key = str(c).lower().strip()
        if key == "open":
            col_map[c] = "Open"
        elif key == "high":
            col_map[c] = "High"
        elif key == "low":
            col_map[c] = "Low"
        elif key == "close":
            col_map[c] = "Close"
        elif key == "volume":
            col_map[c] = "Volume"
    df.rename(columns=col_map, inplace=True)

    df.ffill(inplace=True)
    return df


# ==============================================================================
# CONVENIENCE: GET LATEST MARKET DATA
# ==============================================================================
def get_latest_market_data(
    ticker: str = "VNINDEX",
    start_date: str = "2020-01-01",
    end_date: str | None = None,
    fallback_path: str | None = None,
) -> pd.DataFrame:
    """
    Ham tien ich chinh: lay du lieu thi truong moi nhat.
    Uu tien API (vnstock), neu that bai thi fallback sang local file.
    Returns: DataFrame OHLCV voi DatetimeIndex, da forward-fill.
    """
    try:
        df = fetch_vnindex(ticker=ticker, start_date=start_date, end_date=end_date)
        if df.empty:
            raise RuntimeError("API tra ve DataFrame rong.")
        return df
    except Exception as e:
        if fallback_path:
            print(f"[WARN] API failed ({e}), chuyen sang local file: {fallback_path}")
            return load_local_file(fallback_path)
        raise RuntimeError(
            f"Khong the lay du lieu tu API va khong co fallback_path. Loi goc: {e}"
        ) from e


# ==============================================================================
# TESTING BLOCK
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TEST: Fetch real-time VNINDEX data")
    print("=" * 60)

    df = get_latest_market_data(
        ticker="VNINDEX",
        start_date="2024-01-01",
    )

    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Index range: {df.index.min()} -> {df.index.max()}")
    print(f"Missing values:\n{df[['Open','High','Low','Close','Volume']].isna().sum()}")
    print(f"\nLast 5 rows:")
    print(df[["Open", "High", "Low", "Close", "Volume"]].tail(5))
