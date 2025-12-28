"""
Text summary reports for backtest results.

Provides utilities for generating formatted text reports
and comparison tables.
"""

from typing import Dict, List

from lobbacktest.types import BacktestResult


def generate_report(
    result: BacktestResult,
    title: str = "Backtest Report",
) -> str:
    """
    Generate a comprehensive text report from backtest results.

    Args:
        result: BacktestResult from a backtest run
        title: Report title

    Returns:
        Formatted multi-line string report
    """
    lines = [
        "=" * 70,
        title.center(70),
        "=" * 70,
        "",
        "CAPITAL",
        "-" * 40,
        f"  Initial Capital:     ${result.initial_capital:>15,.2f}",
        f"  Final Equity:        ${result.final_equity:>15,.2f}",
        f"  Total P&L:           ${result.total_pnl:>+15,.2f}",
        f"  Total Return:        {result.total_return * 100:>+15.2f}%",
        "",
        "TRADING ACTIVITY",
        "-" * 40,
        f"  Total Trades:        {result.total_trades:>15,}",
        f"  Data Points:         {len(result.equity_curve):>15,}",
        "",
    ]

    # Metrics section
    if result.metrics:
        lines.append("PERFORMANCE METRICS")
        lines.append("-" * 40)

        # Group metrics
        returns_metrics = {k: v for k, v in result.metrics.items() if "Return" in k}
        risk_metrics = {
            k: v
            for k, v in result.metrics.items()
            if k in ["SharpeRatio", "SortinoRatio", "MaxDrawdown", "CalmarRatio"]
        }
        trading_metrics = {
            k: v
            for k, v in result.metrics.items()
            if k
            in [
                "WinRate",
                "ProfitFactor",
                "AverageWin",
                "AverageLoss",
                "PayoffRatio",
                "Expectancy",
            ]
        }
        prediction_metrics = {
            k: v
            for k, v in result.metrics.items()
            if k in ["DirectionalAccuracy", "SignalRate", "UpPrecision", "DownPrecision"]
        }

        if returns_metrics:
            lines.append("  Returns:")
            for name, value in returns_metrics.items():
                if isinstance(value, float):
                    lines.append(f"    {name:25s} {value * 100:>+10.2f}%")

        if risk_metrics:
            lines.append("  Risk:")
            for name, value in risk_metrics.items():
                if isinstance(value, float):
                    if "Drawdown" in name:
                        lines.append(f"    {name:25s} {value * 100:>10.2f}%")
                    else:
                        lines.append(f"    {name:25s} {value:>+10.4f}")

        if trading_metrics:
            lines.append("  Trading:")
            for name, value in trading_metrics.items():
                if isinstance(value, float):
                    if name == "WinRate":
                        lines.append(f"    {name:25s} {value * 100:>10.2f}%")
                    elif name in ["AverageWin", "AverageLoss", "Expectancy"]:
                        lines.append(f"    {name:25s} ${value:>+10.2f}")
                    else:
                        lines.append(f"    {name:25s} {value:>10.4f}")

        if prediction_metrics:
            lines.append("  Prediction:")
            for name, value in prediction_metrics.items():
                if isinstance(value, float):
                    lines.append(f"    {name:25s} {value * 100:>10.2f}%")

    lines.extend(["", "=" * 70])

    return "\n".join(lines)


def comparison_table(
    results: Dict[str, BacktestResult],
    metrics: List[str] = None,
) -> str:
    """
    Generate a comparison table for multiple backtest results.

    Args:
        results: Dict mapping model name to BacktestResult
        metrics: Optional list of metrics to include

    Returns:
        Formatted comparison table
    """
    if not results:
        return "No results to compare"

    # Default metrics
    if metrics is None:
        metrics = [
            "TotalReturn",
            "SharpeRatio",
            "MaxDrawdown",
            "WinRate",
            "ProfitFactor",
            "DirectionalAccuracy",
        ]

    # Header
    model_names = list(results.keys())
    col_width = max(15, max(len(name) for name in model_names) + 2)

    lines = [
        "=" * (25 + col_width * len(model_names)),
        "MODEL COMPARISON",
        "=" * (25 + col_width * len(model_names)),
        "",
    ]

    # Column headers
    header = f"{'Metric':<25s}"
    for name in model_names:
        header += f"{name:>{col_width}s}"
    lines.append(header)
    lines.append("-" * (25 + col_width * len(model_names)))

    # Metric rows
    for metric in metrics:
        row = f"{metric:<25s}"
        for name in model_names:
            value = results[name].metrics.get(metric, float("nan"))
            if isinstance(value, float):
                if "Return" in metric or "Rate" in metric or "Accuracy" in metric:
                    row += f"{value * 100:>{col_width - 1}.2f}%"
                elif "Drawdown" in metric:
                    row += f"{value * 100:>{col_width - 1}.2f}%"
                else:
                    row += f"{value:>{col_width}.4f}"
            else:
                row += f"{str(value):>{col_width}s}"
        lines.append(row)

    # Summary row
    lines.append("-" * (25 + col_width * len(model_names)))

    # Final equity
    row = f"{'Final Equity':<25s}"
    for name in model_names:
        row += f"${results[name].final_equity:>{col_width - 1},.0f}"
    lines.append(row)

    # Total trades
    row = f"{'Total Trades':<25s}"
    for name in model_names:
        row += f"{results[name].total_trades:>{col_width},}"
    lines.append(row)

    lines.append("=" * (25 + col_width * len(model_names)))

    return "\n".join(lines)

