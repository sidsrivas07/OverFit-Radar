"""Unit tests for data validation, caching, and corporate action adjustment checks."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from overfit_radar.data.validator import validate_ohlcv, check_split_adjustment
from overfit_radar.data.loader import save_to_cache, load_from_cache


def test_validate_ohlcv_valid(synthetic_gbm_prices):
    """Ensure standard valid OHLCV data passes validation cleanly."""
    df = synthetic_gbm_prices(n_days=100, seed=42)
    # Should not raise any error
    validate_ohlcv(df, symbol="TEST.NS")


def test_validate_ohlcv_missing_columns():
    """Ensure missing essential columns are caught immediately."""
    df = pd.DataFrame({"close": [100.0, 101.0]})
    with pytest.raises(ValueError, match="missing required columns"):
        validate_ohlcv(df, symbol="TEST.NS")


def test_validate_ohlcv_high_low_violation():
    """Ensure High < Low anomaly is flagged."""
    dates = pd.bdate_range("2023-01-01", periods=2)
    df = pd.DataFrame(
        {
            "open": [100.0, 100.0],
            "high": [95.0, 105.0],  # Invalid: high < low on day 0
            "low": [98.0, 99.0],
            "close": [99.0, 101.0],
            "adjusted_close": [99.0, 101.0],
            "volume": [1000.0, 1000.0],
        },
        index=dates,
    )
    with pytest.raises(ValueError, match="High price is lower than Low price"):
        validate_ohlcv(df, symbol="TEST.NS")


def test_cache_save_and_load(tmp_path: Path, synthetic_gbm_prices):
    """Test saving dataframe to Parquet and retrieving it from disk."""
    df = synthetic_gbm_prices(n_days=50, seed=42)
    symbol = "RELIANCE.NS"
    
    # Save to temporary cache
    cache_file = save_to_cache(df, symbol=symbol, cache_dir=tmp_path)
    assert cache_file.is_file()
    
    # Load back
    loaded_df = load_from_cache(symbol=symbol, cache_dir=tmp_path)
    assert loaded_df is not None
    pd.testing.assert_frame_equal(df, loaded_df, check_freq=False)


def test_check_split_adjustment_detects_unadjusted_plunge():
    """
    Test corporate action adjustment check:
    Simulates a 10:1 stock split where unadjusted close plunged by 90%,
    verifying our checker detects when split adjustment is missing vs properly applied.
    """
    dates = pd.date_range("2022-07-25", periods=5, freq="D")
    
    # Case A: Unadjusted plunge on split date (2022-07-28)
    unadjusted_df = pd.DataFrame(
        {
            "open": [1000.0, 1000.0, 1000.0, 100.0, 102.0],
            "high": [1010.0, 1010.0, 1010.0, 105.0, 103.0],
            "low": [990.0, 990.0, 990.0, 99.0, 100.0],
            "close": [1000.0, 1000.0, 1000.0, 101.0, 102.0],
            "adjusted_close": [1000.0, 1000.0, 1000.0, 101.0, 102.0], # Unadjusted!
            "volume": [50000.0] * 5,
        },
        index=dates,
    )
    # The return on 2022-07-28 is ~ -90%, which is an unadjusted anomaly
    assert not check_split_adjustment(unadjusted_df, split_date="2022-07-28", max_allowed_abs_return=0.20)
    
    # Case B: Properly adjusted prices (backward-adjusted prior to split)
    adjusted_df = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0, 102.0],
            "high": [101.0, 101.0, 101.0, 105.0, 103.0],
            "low": [99.0, 99.0, 99.0, 99.0, 100.0],
            "close": [100.0, 100.0, 100.0, 101.0, 102.0],
            "adjusted_close": [100.0, 100.0, 100.0, 101.0, 102.0],
            "volume": [50000.0] * 5,
        },
        index=dates,
    )
    assert check_split_adjustment(adjusted_df, split_date="2022-07-28", max_allowed_abs_return=0.20)


@pytest.mark.integration
def test_cached_benchmark_symbols_offline():
    """Verify that cached real benchmark symbols load offline and pass integrity checks."""
    from overfit_radar.config import get_project_root
    cache_dir = get_project_root() / "data" / "raw"
    parquet_files = list(cache_dir.glob("*.parquet"))
    if not parquet_files:
        pytest.skip('Market cache not present; run scripts/download_sample_universe.py for integration tests')
    assert len(parquet_files) >= 10, f"Expected at least 10 cached symbols, found {len(parquet_files)}"

    # Verify Tata Steel split adjustment on cached real data
    ts_df = load_from_cache("TATASTEEL.NS", cache_dir)
    assert ts_df is not None
    validate_ohlcv(ts_df, "TATASTEEL.NS")
    assert check_split_adjustment(ts_df, "2022-07-28", max_allowed_abs_return=0.20)
    assert ts_df.index[0].year <= 2010
    assert len(ts_df) > 3000

    # Verify SBI split adjustment on cached real data
    sbi_df = load_from_cache("SBIN.NS", cache_dir)
    assert sbi_df is not None
    validate_ohlcv(sbi_df, "SBIN.NS")
    assert check_split_adjustment(sbi_df, "2014-11-20", max_allowed_abs_return=0.20)
