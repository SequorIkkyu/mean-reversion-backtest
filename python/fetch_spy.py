print("FETCH_SPY: script started")

import yfinance as yf

def main():
    print("FETCH_SPY: downloading...")
    df = yf.download(
        "SPY",
        start="2025-07-01",
        end="2025-12-31",
        interval="1d",
        auto_adjust=False,
        progress=False
    )
    print("FETCH_SPY: downloaded, shape =", df.shape)

    if df.empty:
        raise RuntimeError("Downloaded DataFrame is empty")

    out = "data/raw/spy_2025_Jul_Dec.csv"
    df.to_csv(out)
    print(f"FETCH_SPY: saved -> {out}")

if __name__ == "__main__":
    main()
