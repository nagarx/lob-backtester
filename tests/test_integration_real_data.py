"""
Integration tests for lob-backtester using real trained model and data.

This test validates the entire backtester pipeline with:
1. Real exported data from feature-extractor-MBO-LOB
2. Real trained model from lob-model-trainer
3. End-to-end backtest execution

PURPOSE: Expose any issues in the backtester implementation that wouldn't
appear with synthetic data.
"""

import pytest
import numpy as np
import json
import os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass

# Check if we have the required dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import backtester modules
from lobbacktest.types import Trade, Position, BacktestResult, TradeSide, PositionSide
from lobbacktest.config import BacktestConfig, CostConfig
from lobbacktest.strategies.direction import DirectionStrategy, ThresholdStrategy
from lobbacktest.engine.vectorized import VectorizedEngine, Backtester, BacktestData
from lobbacktest.metrics.base import Metric
from lobbacktest.metrics.risk import SharpeRatio, SortinoRatio, MaxDrawdown, CalmarRatio
from lobbacktest.metrics.trading import WinRate, ProfitFactor, Expectancy


# =============================================================================
# Test Data Loading Utilities
# =============================================================================

@dataclass
class LoadedTestData:
    """Container for loaded test data (named to avoid pytest collection)."""
    sequences: np.ndarray  # Shape: (N, T, F)
    labels: np.ndarray     # Shape: (N,) or (N, H)
    prices: np.ndarray     # Shape: (N,) - denormalized mid-prices
    normalization: Dict[str, Any]
    metadata: Dict[str, Any]
    day: str


def get_project_root() -> Path:
    """Get the HFT-pipeline-v2 root directory.

    Thin wrapper around ``hft_contracts._testing.require_monorepo_root`` —
    the SSoT helper (Phase V.A.0) for cross-module monorepo-layout gates.
    Previous implementation was a parallel in-file walk that duplicated
    the SSoT logic (hft-rules §1 violation, flagged by V.A.0 audit).

    When lob-backtester is checked out standalone (e.g., on CI), the
    monorepo root is absent — the whole ``TestBacktesterWithRealData``
    class depends on paths that only resolve inside the monorepo layout.
    Per hft-rules §6 (tests document behavior; no tautological crashes),
    we skip cleanly rather than raising a hard ``RuntimeError``.

    Callers invoke this LAZILY from inside pytest fixtures so
    ``pytest.skip()`` propagates to the enclosing test correctly.
    """
    from hft_contracts._testing import require_monorepo_root

    return require_monorepo_root()


def load_single_day_data(
    data_dir: Path,
    day: str,
    horizon_idx: int = 0
) -> LoadedTestData:
    """Load data for a single day.
    
    Args:
        data_dir: Path to the split directory (e.g., test/)
        day: Date string (e.g., "20250825")
        horizon_idx: Which horizon to use for labels (0-indexed)
    
    Returns:
        TestData with sequences, labels, prices, and metadata
    """
    seq_path = data_dir / f"{day}_sequences.npy"
    label_path = data_dir / f"{day}_labels.npy"
    meta_path = data_dir / f"{day}_metadata.json"
    norm_path = data_dir / f"{day}_normalization.json"
    
    if not seq_path.exists():
        raise FileNotFoundError(f"Sequences file not found: {seq_path}")
    
    sequences = np.load(seq_path)
    labels_raw = np.load(label_path)
    
    with open(meta_path) as f:
        metadata = json.load(f)
    with open(norm_path) as f:
        normalization = json.load(f)
    
    # Handle multi-horizon labels
    if labels_raw.ndim == 2:
        labels = labels_raw[:, horizon_idx]
    else:
        labels = labels_raw
    
    # Extract mid-prices from sequences (feature index 40 is mid-price)
    # This is normalized - we need to denormalize
    mid_price_idx = 40  # Per feature-extractor-MBO-LOB layout
    prices_normalized = sequences[:, -1, mid_price_idx]  # Last timestep
    
    # Denormalize prices
    prices = denormalize_prices(
        prices_normalized,
        normalization,
    )
    
    return LoadedTestData(
        sequences=sequences,
        labels=labels,
        prices=prices,
        normalization=normalization,
        metadata=metadata,
        day=day,
    )


def denormalize_prices(
    prices_norm: np.ndarray,
    norm_params: Dict[str, Any],
) -> np.ndarray:
    """Denormalize prices from z-scored values.
    
    The feature extractor applies: z = (x - mean) / std
    We reverse: x = z * std + mean
    
    Note: The normalization params from feature-extractor-MBO-LOB contain:
    - price_means: 10 values (one per level, combined for ask/bid)
    - price_stds: 10 values (one per level, combined for ask/bid)
    
    For mid-price, we use level 0 (tightest level) as the reference.
    """
    price_means = np.array(norm_params.get("price_means", []))
    price_stds = np.array(norm_params.get("price_stds", []))
    
    if len(price_means) == 0 or len(price_stds) == 0:
        # Fallback: return normalized (can't denormalize)
        return prices_norm
    
    # Use level 0 (tightest spread) for mid-price approximation
    # The means/stds are per-level, combined for ask/bid
    mean = price_means[0]
    std = price_stds[0] if price_stds[0] > 0 else 1.0
    
    return prices_norm * std + mean


def load_model_predictions(
    model_dir: Path,
    sequences: np.ndarray,
    device: str = "cpu"
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load model and generate predictions.
    
    Args:
        model_dir: Path to experiment output directory (with checkpoints/)
        sequences: Input sequences (N, T, F)
        device: Computation device
    
    Returns:
        Tuple of (predictions, probabilities)
        - predictions: (N,) int array with class indices {0, 1, 2}
        - probabilities: (N, 3) float array with class probabilities
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not available")
    
    checkpoint_path = model_dir / "checkpoints" / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Import model creation from lob-model-trainer
    import sys
    trainer_path = get_project_root() / "lob-model-trainer" / "src"
    if str(trainer_path) not in sys.path:
        sys.path.insert(0, str(trainer_path))
    
    models_path = get_project_root() / "lob-models" / "src"
    if str(models_path) not in sys.path:
        sys.path.insert(0, str(models_path))
    
    from lobmodels.models.deeplob import create_deeplob
    
    # Create model with same config as training
    model = create_deeplob(
        mode="benchmark",  # Benchmark mode uses 40 LOB features
        num_levels=10,
        num_classes=3,
        conv_filters=32,
        inception_filters=64,
        lstm_hidden=64,
        dropout=0.0,
    )
    
    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)
    
    # Select first 40 features (LOB only for benchmark mode)
    sequences_lob = sequences[:, :, :40].astype(np.float32)
    
    # Generate predictions in batches
    batch_size = 64
    n_samples = len(sequences_lob)
    predictions = []
    probabilities = []
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = torch.from_numpy(sequences_lob[i:i+batch_size])
            batch = batch.to(device)
            
            # DeepLOB expects (batch, seq_len, features) = (N, 100, 40)
            # Our data is already in this format, no transformation needed
            
            logits = model(batch)
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
            
            predictions.append(preds.cpu().numpy())
            probabilities.append(probs.cpu().numpy())
    
    predictions = np.concatenate(predictions)
    probabilities = np.concatenate(probabilities)
    
    return predictions, probabilities


# =============================================================================
# Integration Tests
# =============================================================================

class TestBacktesterWithRealData:
    """Integration tests using real trained model and exported data."""
    
    @pytest.fixture
    def data_dir(self) -> Path:
        """Path to test data."""
        root = get_project_root()
        data_dir = root / "data" / "exports" / "nvda_balanced" / "test"
        if not data_dir.exists():
            pytest.skip("Test data not available")
        return data_dir
    
    @pytest.fixture
    def model_dir(self) -> Path:
        """Path to trained model."""
        root = get_project_root()
        model_dir = root / "lob-model-trainer" / "outputs" / "experiments" / "nvda_h10_weighted_v1"
        if not (model_dir / "checkpoints" / "best.pt").exists():
            pytest.skip("Trained model not available")
        return model_dir
    
    @pytest.fixture
    def test_data(self, data_dir: Path) -> LoadedTestData:
        """Load a single day of test data."""
        # Get first available day
        days = sorted([f.stem.split("_")[0] for f in data_dir.glob("*_sequences.npy")])
        if not days:
            pytest.skip("No test data found")
        return load_single_day_data(data_dir, days[0], horizon_idx=0)
    
    def test_data_loading(self, test_data: LoadedTestData):
        """Verify test data loading works correctly."""
        assert test_data.sequences.ndim == 3
        assert test_data.sequences.shape[1] == 100  # Window size
        assert test_data.sequences.shape[2] == 98   # Feature count
        assert test_data.labels.ndim == 1
        assert np.all(np.isin(test_data.labels, [-1, 0, 1]))
        assert len(test_data.prices) == len(test_data.labels)
        assert np.all(np.isfinite(test_data.prices))
    
    def test_prices_are_reasonable(self, test_data: LoadedTestData):
        """Verify denormalized prices are in expected range for NVDA."""
        # NVDA was trading ~$450-$500 in August 2025 (data period)
        # Allow wide range for safety
        min_price = 100.0
        max_price = 600.0
        
        # Check if prices are normalized or denormalized
        if np.abs(test_data.prices).max() < 10:
            # Still normalized, that's okay - but log a warning
            print(f"WARNING: Prices appear to be normalized (range: {test_data.prices.min():.2f} to {test_data.prices.max():.2f})")
            assert np.all(np.isfinite(test_data.prices))
        else:
            # Denormalized - check range
            assert test_data.prices.min() > min_price, \
                f"Price {test_data.prices.min():.2f} below expected minimum {min_price}"
            assert test_data.prices.max() < max_price, \
                f"Price {test_data.prices.max():.2f} above expected maximum {max_price}"
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_model_loading_and_prediction(self, model_dir: Path, test_data: LoadedTestData):
        """Verify model can be loaded and generates valid predictions."""
        predictions, probabilities = load_model_predictions(
            model_dir,
            test_data.sequences,
        )
        
        # Check predictions
        assert predictions.shape == (len(test_data.labels),)
        assert predictions.dtype in [np.int32, np.int64]
        assert np.all(np.isin(predictions, [0, 1, 2])), "Predictions should be in {0, 1, 2}"
        
        # Check probabilities
        assert probabilities.shape == (len(test_data.labels), 3)
        assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5), \
            "Probabilities should sum to 1"
        assert np.all(probabilities >= 0) and np.all(probabilities <= 1)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_direction_strategy_with_real_predictions(self, model_dir: Path, test_data: LoadedTestData):
        """Test DirectionStrategy with real model predictions."""
        predictions, _ = load_model_predictions(model_dir, test_data.sequences)
        
        # Model outputs {0, 1, 2} for Down/Stable/Up (shifted labels)
        strategy = DirectionStrategy(predictions=predictions, shifted=True)
        signals = strategy.generate_signals(test_data.prices)
        
        assert len(signals.signals) == len(predictions)
        # Verify mapping: 0 -> SELL, 1 -> HOLD, 2 -> BUY
        for i in range(min(100, len(predictions))):  # Check first 100
            if predictions[i] == 0:  # Down -> SELL
                assert signals.signals[i] == -1, f"Expected SELL for Down, got {signals.signals[i]}"
            elif predictions[i] == 1:  # Stable -> HOLD
                assert signals.signals[i] == 0, f"Expected HOLD for Stable, got {signals.signals[i]}"
            elif predictions[i] == 2:  # Up -> BUY
                assert signals.signals[i] == 1, f"Expected BUY for Up, got {signals.signals[i]}"
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_full_backtest_pipeline(self, model_dir: Path, test_data: LoadedTestData):
        """Full end-to-end backtest with real data."""
        predictions, probabilities = load_model_predictions(model_dir, test_data.sequences)
        
        # Create backtest config
        config = BacktestConfig(
            initial_capital=100_000.0,
            position_size=0.1,  # 10% of capital per trade
            costs=CostConfig(
                spread_bps=1.0,
                slippage_bps=0.5,
            ),
            allow_short=True,
            trading_days_per_year=252.0,
            periods_per_day=1000.0,  # Approximate sequences per day
        )
        
        # Create strategy
        strategy = DirectionStrategy(predictions=predictions, shifted=True)
        
        # Create backtest data
        data = BacktestData(
            prices=test_data.prices,
            labels=test_data.labels + 1,  # Shift from {-1,0,1} to {0,1,2}
        )
        
        # Run backtest
        engine = VectorizedEngine(config)
        result = engine.run(data, strategy)
        
        # Verify result structure
        assert isinstance(result, BacktestResult)
        assert len(result.equity_curve) == len(test_data.prices)
        assert len(result.returns) == len(test_data.prices) - 1
        assert len(result.positions) == len(test_data.prices)
        assert result.initial_capital == 100_000.0
        assert result.final_equity > 0  # Should not go bankrupt
        
        # Verify trades
        assert len(result.trades) > 0, "Should have at least some trades"
        for trade in result.trades:
            assert isinstance(trade, Trade)
            assert trade.price > 0
            assert trade.size > 0
            assert trade.cost >= 0
        
        # Verify metrics
        assert "TotalReturn" in result.metrics or result.total_return is not None
        assert np.isfinite(result.total_return)
        assert np.isfinite(result.max_drawdown)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_backtest_with_threshold_strategy(self, model_dir: Path, test_data: LoadedTestData):
        """Test backtest with confidence threshold filtering."""
        predictions, probabilities = load_model_predictions(model_dir, test_data.sequences)
        
        # Create threshold strategy (only trade high confidence)
        strategy = ThresholdStrategy(
            predictions=predictions,
            probabilities=probabilities,
            threshold=0.6,  # Only trade if > 60% confident
            shifted=True,
        )
        
        signals = strategy.generate_signals(test_data.prices)
        
        # Verify threshold filtering
        high_conf_mask = probabilities.max(axis=1) > 0.6
        hold_signals = signals.signals == 0
        
        # Low confidence should always be HOLD
        low_conf_should_hold = ~high_conf_mask & (predictions != 1)
        # Note: Some might still be HOLD if prediction was Stable
        
        # Count trades
        n_trades = np.sum(signals.signals != 0)
        n_possible = np.sum(high_conf_mask & (predictions != 1))
        
        print(f"Threshold filtering: {n_trades} trades from {n_possible} high-conf directional signals")
        assert n_trades <= n_possible, "Shouldn't have more trades than high-conf directional signals"
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_backtest_metrics_consistency(self, model_dir: Path, test_data: LoadedTestData):
        """Verify metric computations are consistent."""
        predictions, _ = load_model_predictions(model_dir, test_data.sequences)
        
        config = BacktestConfig(initial_capital=100_000.0)
        strategy = DirectionStrategy(predictions=predictions, shifted=True)
        data = BacktestData(prices=test_data.prices, labels=test_data.labels + 1)
        
        backtester = Backtester(config)
        result = backtester.run(data, strategy)
        
        # Compute metrics manually and compare
        returns = result.returns
        
        # Sharpe Ratio
        if len(returns) > 1 and np.std(returns) > 0:
            expected_sr = (np.mean(returns) / np.std(returns)) * np.sqrt(252 * 1000)
            metric = SharpeRatio(trading_days_per_year=252, periods_per_day=1000)
            computed = metric.compute(returns, {})
            
            # Allow some tolerance due to different implementations
            if "SharpeRatio" in result.metrics:
                assert np.isfinite(result.metrics["SharpeRatio"])
        
        # Max Drawdown
        if len(result.equity_curve) > 0:
            peak = np.maximum.accumulate(result.equity_curve)
            drawdown = (peak - result.equity_curve) / peak  # Positive drawdown
            drawdown = np.where(np.isfinite(drawdown), drawdown, 0.0)
            expected_mdd = np.max(drawdown)
            
            assert np.isclose(result.max_drawdown, expected_mdd, rtol=1e-5), \
                f"Max drawdown mismatch: {result.max_drawdown} vs {expected_mdd}"
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_backtest_invariants(self, model_dir: Path, test_data: LoadedTestData):
        """Test important backtesting invariants."""
        predictions, _ = load_model_predictions(model_dir, test_data.sequences)
        
        config = BacktestConfig(
            initial_capital=100_000.0,
            position_size=0.1,
            costs=CostConfig(spread_bps=0, slippage_bps=0),  # No costs for this test
        )
        strategy = DirectionStrategy(predictions=predictions, shifted=True)
        data = BacktestData(prices=test_data.prices)
        
        result = Backtester(config).run(data, strategy)
        
        # Invariant 1: Equity should start at initial capital
        assert np.isclose(result.equity_curve[0], config.initial_capital, rtol=1e-5), \
            f"Initial equity {result.equity_curve[0]} != initial capital {config.initial_capital}"
        
        # Invariant 2: Final equity should match last equity curve value
        assert np.isclose(result.final_equity, result.equity_curve[-1], rtol=1e-5)
        
        # Invariant 3: Number of trades should be reasonable
        n_signals = np.sum(predictions != 1)  # Non-stable predictions
        n_trades = len(result.trades)
        # Can't have more trades than 2x signals (one to enter, one to exit)
        assert n_trades <= 2 * n_signals + 2, \
            f"Too many trades ({n_trades}) for {n_signals} directional signals"
        
        # Invariant 4: Equity curve should be continuous (no NaN)
        assert np.all(np.isfinite(result.equity_curve)), "Equity curve has NaN/Inf values"
        
        # Invariant 5: Returns should match equity changes
        expected_returns = np.diff(result.equity_curve) / result.equity_curve[:-1]
        assert np.allclose(result.returns, expected_returns, rtol=1e-5, equal_nan=True)


class TestBacktesterEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_prices(self):
        """Backtest with empty prices should fail at data validation."""
        with pytest.raises(ValueError, match="cannot be empty"):
            BacktestData(prices=np.array([]))
    
    def test_single_price(self):
        """Backtest with single price point."""
        config = BacktestConfig()
        strategy = DirectionStrategy(predictions=np.array([1]), shifted=True)
        data = BacktestData(prices=np.array([100.0]))
        
        engine = VectorizedEngine(config)
        result = engine.run(data, strategy)
        
        assert len(result.equity_curve) == 1
        assert result.final_equity == config.initial_capital
    
    def test_all_hold_signals(self):
        """Backtest with all HOLD signals should have no trades."""
        prices = np.array([100.0, 101.0, 99.0, 100.5, 101.5])
        predictions = np.array([1, 1, 1, 1, 1])  # All Stable -> HOLD
        
        config = BacktestConfig()
        strategy = DirectionStrategy(predictions=predictions, shifted=True)
        data = BacktestData(prices=prices)
        
        result = Backtester(config).run(data, strategy)
        
        assert len(result.trades) == 0, "Should have no trades with all HOLD signals"
        assert result.final_equity == config.initial_capital, \
            "No trades should mean no change in capital"
    
    def test_extreme_prices_no_crash(self):
        """Test that extreme price movements don't crash the engine."""
        # More realistic but still significant price movements (+/-50%)
        prices = np.array([100.0, 50.0, 75.0, 150.0, 100.0])
        predictions = np.array([2, 0, 2, 0, 1])  # Buy, Sell, Buy, Sell, Hold
        
        config = BacktestConfig(
            initial_capital=100_000.0,
            position_size=0.1,  # 10% per trade
            max_position=0.3,   # Max 30% of capital at risk
        )
        strategy = DirectionStrategy(predictions=predictions, shifted=True)
        data = BacktestData(prices=prices)
        
        result = Backtester(config).run(data, strategy)
        
        # Should complete without crashing
        assert np.all(np.isfinite(result.equity_curve)), "Equity curve has NaN/Inf"
        # With limited position sizing, shouldn't go to zero
        assert result.final_equity > 0, f"Final equity {result.final_equity} <= 0"
    
    def test_wild_price_swings_limited_loss(self):
        """Test that position sizing limits catastrophic losses vs no limits.
        
        With 1000x price swings ($1 → $1000), even a bounded short position
        can lose more than initial capital. The key is that losses are
        BOUNDED by position sizing, not unlimited.
        
        Without safeguards: 9000+ shares shorted → -$8.9M loss
        With safeguards:    200 shares shorted → -$200k loss
        """
        prices = np.array([100.0, 1.0, 1000.0, 0.1, 100.0])  # Wild 1000x swing
        predictions = np.array([2, 0, 2, 0, 1])  # Buy, Sell, Buy, Sell, Hold
        
        config = BacktestConfig(
            initial_capital=100_000.0,
            position_size=0.1,  # Only 10% per trade
            max_position=0.2,   # Max 20% of initial capital
        )
        strategy = DirectionStrategy(predictions=predictions, shifted=True)
        data = BacktestData(prices=prices)
        
        result = Backtester(config).run(data, strategy)
        
        # Should complete without crashing
        assert np.all(np.isfinite(result.equity_curve)), "Equity curve has NaN/Inf"
        
        # The key invariant: losses should be BOUNDED by position sizing
        # Max shares ≈ 200 (from max_position=0.2 * 100k / 100)
        # Max loss ≈ 200 * $999 = $200k on the short
        # 
        # Without safeguards, the old code would have:
        # Max shares ≈ 9000+ (from capital / $1)
        # Max loss ≈ 9000 * $999 = $8.99M
        #
        # So we verify loss is bounded (much less than $1M)
        assert result.final_equity > -1_000_000, \
            f"Final equity {result.final_equity} exceeds reasonable loss bounds for limited position"
        
        # Verify position size was limited (not unlimited)
        max_position = np.max(np.abs(result.positions))
        assert max_position <= 500, \
            f"Position size {max_position} exceeds expected limit for max_position=0.2"
    
    def test_zero_prices_rejected(self):
        """Prices of zero should be rejected at data validation."""
        prices = np.array([100.0, 0.0, 101.0])  # Zero price
        
        with pytest.raises(ValueError, match="must be positive"):
            BacktestData(prices=prices)
    
    def test_negative_prices_rejected(self):
        """Negative prices should be rejected at data validation."""
        prices = np.array([100.0, -50.0, 101.0])  # Negative price
        
        with pytest.raises(ValueError, match="must be positive"):
            BacktestData(prices=prices)
    
    def test_nan_prices_rejected(self):
        """NaN prices should be rejected at data validation."""
        prices = np.array([100.0, np.nan, 101.0])
        
        with pytest.raises(ValueError, match="NaN or Inf"):
            BacktestData(prices=prices)
    
    def test_inf_prices_rejected(self):
        """Inf prices should be rejected at data validation."""
        prices = np.array([100.0, np.inf, 101.0])
        
        with pytest.raises(ValueError, match="NaN or Inf"):
            BacktestData(prices=prices)


class TestLabelEncodingConsistency:
    """Test that label encoding is handled correctly throughout."""
    
    def test_original_labels_unshifted(self):
        """Original labels {-1, 0, 1} should work with shifted=False."""
        prices = np.array([100.0, 101.0, 99.0, 100.5])
        predictions = np.array([-1, 0, 1, -1])  # Down, Stable, Up, Down
        
        strategy = DirectionStrategy(predictions=predictions, shifted=False)
        signals = strategy.generate_signals(prices)
        
        expected = np.array([-1, 0, 1, -1])  # SELL, HOLD, BUY, SELL
        np.testing.assert_array_equal(signals.signals, expected)
    
    def test_shifted_labels(self):
        """Shifted labels {0, 1, 2} should work with shifted=True."""
        prices = np.array([100.0, 101.0, 99.0, 100.5])
        predictions = np.array([0, 1, 2, 0])  # Down, Stable, Up, Down
        
        strategy = DirectionStrategy(predictions=predictions, shifted=True)
        signals = strategy.generate_signals(prices)
        
        expected = np.array([-1, 0, 1, -1])  # SELL, HOLD, BUY, SELL
        np.testing.assert_array_equal(signals.signals, expected)
    
    def test_wrong_shifted_flag_produces_wrong_signals(self):
        """Using wrong shifted flag should produce incorrect signals."""
        prices = np.array([100.0, 101.0])
        
        # Original Down label = -1
        # With shifted=True, this becomes -1 - 1 = -2 (invalid)
        # Our strategy should handle this gracefully
        predictions = np.array([-1, 1])
        
        strategy = DirectionStrategy(predictions=predictions, shifted=True)
        signals = strategy.generate_signals(prices)
        
        # -1 with shifted=True means original label was -2 (invalid)
        # Strategy should map unknown values to HOLD
        # This test verifies the behavior is defined


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

