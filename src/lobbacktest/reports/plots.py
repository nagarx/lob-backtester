"""
Visualization utilities for backtest results.

Provides functions for generating standard backtest plots:
- Equity curves
- Returns distributions
- Drawdown charts
- Multi-model comparisons
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from lobbacktest.types import BacktestResult


def _import_matplotlib():
    """Import matplotlib with error handling."""
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        raise ImportError(
            "matplotlib required for plotting. Install with: pip install matplotlib"
        )


def plot_equity_curve(
    result: BacktestResult,
    figsize: Tuple[int, int] = (12, 6),
    title: str = "Equity Curve",
    show_initial: bool = True,
) -> "plt.Figure":
    """
    Plot equity curve from backtest results.

    Args:
        result: BacktestResult from backtest
        figsize: Figure size (width, height)
        title: Plot title
        show_initial: If True, show horizontal line at initial capital

    Returns:
        matplotlib Figure
    """
    plt = _import_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    # Plot equity
    ax.plot(result.equity_curve, label="Equity", color="blue", linewidth=1.5)

    # Initial capital line
    if show_initial:
        ax.axhline(
            y=result.initial_capital,
            color="gray",
            linestyle="--",
            linewidth=1,
            label="Initial Capital",
        )

    # Formatting
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Equity ($)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

    plt.tight_layout()
    return fig


def plot_returns_distribution(
    result: BacktestResult,
    figsize: Tuple[int, int] = (10, 6),
    bins: int = 50,
    title: str = "Returns Distribution",
) -> "plt.Figure":
    """
    Plot histogram of period returns.

    Args:
        result: BacktestResult from backtest
        figsize: Figure size
        bins: Number of histogram bins
        title: Plot title

    Returns:
        matplotlib Figure
    """
    plt = _import_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    # Convert to percentage
    returns_pct = result.returns * 100

    # Plot histogram
    ax.hist(returns_pct, bins=bins, color="blue", alpha=0.7, edgecolor="black")

    # Zero line
    ax.axvline(x=0, color="red", linestyle="-", linewidth=1)

    # Mean line
    mean_return = np.mean(returns_pct)
    ax.axvline(
        x=mean_return,
        color="green",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {mean_return:.4f}%",
    )

    # Formatting
    ax.set_xlabel("Return (%)")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_drawdown(
    result: BacktestResult,
    figsize: Tuple[int, int] = (12, 6),
    title: str = "Drawdown",
) -> "plt.Figure":
    """
    Plot drawdown over time.

    Args:
        result: BacktestResult from backtest
        figsize: Figure size
        title: Plot title

    Returns:
        matplotlib Figure
    """
    plt = _import_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    # Compute drawdown
    equity = result.equity_curve
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak * 100  # As percentage

    # Plot
    ax.fill_between(
        range(len(drawdown)),
        drawdown,
        0,
        color="red",
        alpha=0.5,
    )
    ax.plot(drawdown, color="red", linewidth=0.5)

    # Max drawdown line
    max_dd = np.max(drawdown)
    max_dd_idx = np.argmax(drawdown)
    ax.axhline(
        y=max_dd,
        color="darkred",
        linestyle="--",
        label=f"Max DD: {max_dd:.2f}%",
    )

    # Formatting
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Drawdown (%)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()  # Drawdown is negative

    plt.tight_layout()
    return fig


def plot_comparison(
    results: Dict[str, BacktestResult],
    figsize: Tuple[int, int] = (14, 10),
    normalize: bool = True,
) -> "plt.Figure":
    """
    Plot comparison of multiple backtest results.

    Args:
        results: Dict mapping model name to BacktestResult
        figsize: Figure size
        normalize: If True, normalize equity to start at 1.0

    Returns:
        matplotlib Figure with equity curves and metrics
    """
    plt = _import_matplotlib()

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    # 1. Equity curves
    ax1 = axes[0, 0]
    for (name, result), color in zip(results.items(), colors):
        equity = result.equity_curve
        if normalize:
            equity = equity / equity[0]
        ax1.plot(equity, label=name, color=color, linewidth=1.5)

    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Normalized Equity" if normalize else "Equity ($)")
    ax1.set_title("Equity Curves")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # 2. Drawdown comparison
    ax2 = axes[0, 1]
    for (name, result), color in zip(results.items(), colors):
        equity = result.equity_curve
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak * 100
        ax2.plot(drawdown, label=name, color=color, linewidth=1)

    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_title("Drawdown Comparison")
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()

    # 3. Bar chart of Sharpe ratios
    ax3 = axes[1, 0]
    names = list(results.keys())
    sharpes = [results[n].metrics.get("SharpeRatio", 0) for n in names]
    ax3.bar(names, sharpes, color=colors)
    ax3.set_ylabel("Sharpe Ratio")
    ax3.set_title("Risk-Adjusted Returns")
    ax3.grid(True, alpha=0.3, axis="y")
    ax3.axhline(y=0, color="black", linewidth=0.5)

    # Rotate labels if many models
    if len(names) > 4:
        ax3.tick_params(axis="x", rotation=45)

    # 4. Bar chart of total returns
    ax4 = axes[1, 1]
    returns = [results[n].total_return * 100 for n in names]
    ax4.bar(names, returns, color=colors)
    ax4.set_ylabel("Total Return (%)")
    ax4.set_title("Total Returns")
    ax4.grid(True, alpha=0.3, axis="y")
    ax4.axhline(y=0, color="black", linewidth=0.5)

    if len(names) > 4:
        ax4.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    return fig


def plot_positions(
    result: BacktestResult,
    figsize: Tuple[int, int] = (12, 8),
) -> "plt.Figure":
    """
    Plot position changes over time alongside price.

    Args:
        result: BacktestResult from backtest
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    plt = _import_matplotlib()

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # 1. Price
    ax1 = axes[0]
    ax1.plot(result.prices, color="blue", linewidth=0.5)
    ax1.set_ylabel("Price ($)")
    ax1.set_title("Price and Positions")
    ax1.grid(True, alpha=0.3)

    # 2. Positions
    ax2 = axes[1]
    positions = result.positions

    # Fill areas
    ax2.fill_between(
        range(len(positions)),
        positions,
        0,
        where=positions > 0,
        color="green",
        alpha=0.5,
        label="Long",
    )
    ax2.fill_between(
        range(len(positions)),
        positions,
        0,
        where=positions < 0,
        color="red",
        alpha=0.5,
        label="Short",
    )

    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Position")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

