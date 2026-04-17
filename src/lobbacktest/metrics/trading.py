"""
Trading-specific metrics.

Metrics:
- WinRate: Proportion of winning trades
- ProfitFactor: Gross profit / gross loss
- AverageWin: Mean profit on winning trades
- AverageLoss: Mean loss on losing trades
- PayoffRatio: Average win / average loss
"""

from typing import Any, Dict, List, Mapping

import numpy as np

from lobbacktest.metrics.base import Metric


class WinRate(Metric):
    """
    Win Rate (fraction of profitable trades).

    Formula:
        WinRate = num_winning_trades / total_trades

    Where:
        winning_trade = trade with pnl > 0

    Reference:
        Standard trading metric

    Notes:
        - Returns value between 0 and 1
        - Requires "trade_pnls" in context
    """

    def __init__(self, *, name: str = None):
        """
        Initialize WinRate metric.

        Args:
            name: Optional custom name (default: "WinRate")
        """
        self._name = name or "WinRate"

    @property
    def name(self) -> str:
        return self._name

    def compute(
        self,
        returns: np.ndarray,
        context: Dict[str, Any],
    ) -> Mapping[str, float]:
        """
        Compute win rate from trade P&Ls.

        Args:
            returns: Not used directly (uses trade_pnls from context)
            context: Must contain:
                - "trade_pnls": Array of P&L per trade

        Returns:
            {"WinRate": win_rate} where win_rate in [0, 1]

        Edge cases:
            - No trades: 0.0
            - All winners: 1.0
            - All losers: 0.0
        """
        trade_pnls = context.get("trade_pnls", np.array([]))

        if len(trade_pnls) == 0:
            return {self.name: 0.0}

        trade_pnls = np.asarray(trade_pnls)
        n_winning = np.sum(trade_pnls > 0)
        n_total = len(trade_pnls)

        win_rate = n_winning / n_total

        return {self.name: float(win_rate)}


class ProfitFactor(Metric):
    """
    Profit Factor (gross profit / gross loss).

    Formula:
        ProfitFactor = sum(winning_pnls) / abs(sum(losing_pnls))

    Reference:
        Standard trading metric

    Notes:
        - > 1.0 means profitable overall
        - Undefined if no losses (returns 0)
        - Requires "trade_pnls" in context
    """

    def __init__(self, *, name: str = None):
        """
        Initialize ProfitFactor metric.

        Args:
            name: Optional custom name (default: "ProfitFactor")
        """
        self._name = name or "ProfitFactor"

    @property
    def name(self) -> str:
        return self._name

    def compute(
        self,
        returns: np.ndarray,
        context: Dict[str, Any],
    ) -> Mapping[str, float]:
        """
        Compute profit factor from trade P&Ls.

        Args:
            returns: Not used directly (uses trade_pnls from context)
            context: Must contain:
                - "trade_pnls": Array of P&L per trade

        Returns:
            {"ProfitFactor": profit_factor}

        Edge cases:
            - No trades: 0.0
            - No losses: 100.0 (capped)
            - No wins: 0.0
        """
        trade_pnls = context.get("trade_pnls", np.array([]))

        if len(trade_pnls) == 0:
            return {self.name: 0.0}

        trade_pnls = np.asarray(trade_pnls)

        gross_profit = np.sum(trade_pnls[trade_pnls > 0])
        gross_loss = np.abs(np.sum(trade_pnls[trade_pnls < 0]))

        if gross_loss == 0:
            # No losses - cap at reasonable value
            if gross_profit > 0:
                return {self.name: 100.0}
            else:
                return {self.name: 0.0}

        profit_factor = gross_profit / gross_loss

        return {self.name: float(profit_factor)}


class AverageWin(Metric):
    """
    Average winning trade P&L.

    Formula:
        AverageWin = mean(pnl for pnl > 0)

    Reference:
        Standard trading metric
    """

    def __init__(self, *, name: str = None):
        """
        Initialize AverageWin metric.

        Args:
            name: Optional custom name (default: "AverageWin")
        """
        self._name = name or "AverageWin"

    @property
    def name(self) -> str:
        return self._name

    def compute(
        self,
        returns: np.ndarray,
        context: Dict[str, Any],
    ) -> Mapping[str, float]:
        """
        Compute average winning trade.

        Args:
            returns: Not used directly
            context: Must contain "trade_pnls"

        Returns:
            {"AverageWin": average_win}

        Edge cases:
            - No winning trades: 0.0
        """
        trade_pnls = context.get("trade_pnls", np.array([]))

        if len(trade_pnls) == 0:
            return {self.name: 0.0}

        trade_pnls = np.asarray(trade_pnls)
        winning = trade_pnls[trade_pnls > 0]

        if len(winning) == 0:
            return {self.name: 0.0}

        avg_win = np.mean(winning)

        return {self.name: float(avg_win)}


class AverageLoss(Metric):
    """
    Average losing trade P&L (as positive value).

    Formula:
        AverageLoss = abs(mean(pnl for pnl < 0))

    Reference:
        Standard trading metric

    Notes:
        - Returns positive value (magnitude of average loss)
    """

    def __init__(self, *, name: str = None):
        """
        Initialize AverageLoss metric.

        Args:
            name: Optional custom name (default: "AverageLoss")
        """
        self._name = name or "AverageLoss"

    @property
    def name(self) -> str:
        return self._name

    def compute(
        self,
        returns: np.ndarray,
        context: Dict[str, Any],
    ) -> Mapping[str, float]:
        """
        Compute average losing trade.

        Args:
            returns: Not used directly
            context: Must contain "trade_pnls"

        Returns:
            {"AverageLoss": average_loss} (positive value)

        Edge cases:
            - No losing trades: 0.0
        """
        trade_pnls = context.get("trade_pnls", np.array([]))

        if len(trade_pnls) == 0:
            return {self.name: 0.0}

        trade_pnls = np.asarray(trade_pnls)
        losing = trade_pnls[trade_pnls < 0]

        if len(losing) == 0:
            return {self.name: 0.0}

        avg_loss = np.abs(np.mean(losing))

        return {self.name: float(avg_loss)}


class PayoffRatio(Metric):
    """
    Payoff Ratio (average win / average loss).

    Formula:
        PayoffRatio = AverageWin / AverageLoss

    Also known as:
        - Reward-to-risk ratio
        - Risk/reward ratio (inverted)

    Reference:
        Standard trading metric

    Notes:
        - > 1.0 means average win exceeds average loss
        - Combined with WinRate for expectancy
    """

    def __init__(self, *, name: str = None):
        """
        Initialize PayoffRatio metric.

        Args:
            name: Optional custom name (default: "PayoffRatio")
        """
        self._name = name or "PayoffRatio"

    @property
    def name(self) -> str:
        return self._name

    def compute(
        self,
        returns: np.ndarray,
        context: Dict[str, Any],
    ) -> Mapping[str, float]:
        """
        Compute payoff ratio.

        Args:
            returns: Not used directly
            context: May contain:
                - "AverageWin": Pre-computed
                - "AverageLoss": Pre-computed
                - "trade_pnls": If above not present

        Returns:
            {"PayoffRatio": payoff_ratio}

        Edge cases:
            - No losing trades: 100.0 (capped)
            - No winning trades: 0.0
        """
        # Get or compute average win/loss
        if "AverageWin" in context and "AverageLoss" in context:
            avg_win = context["AverageWin"]
            avg_loss = context["AverageLoss"]
        else:
            trade_pnls = context.get("trade_pnls", np.array([]))
            if len(trade_pnls) == 0:
                return {self.name: 0.0}

            trade_pnls = np.asarray(trade_pnls)
            winning = trade_pnls[trade_pnls > 0]
            losing = trade_pnls[trade_pnls < 0]

            avg_win = np.mean(winning) if len(winning) > 0 else 0.0
            avg_loss = np.abs(np.mean(losing)) if len(losing) > 0 else 0.0

        if avg_loss == 0:
            if avg_win > 0:
                return {self.name: 100.0}
            else:
                return {self.name: 0.0}

        payoff = avg_win / avg_loss

        return {self.name: float(payoff)}


class Expectancy(Metric):
    """
    Trade Expectancy (expected value per trade).

    Formula:
        Expectancy = WinRate * AverageWin - (1 - WinRate) * AverageLoss

    Also known as:
        - Mathematical expectation
        - Edge

    Reference:
        Van Tharp (1998). "Trade Your Way to Financial Freedom"

    Notes:
        - Positive expectancy = profitable system
        - Measured in same units as P&L (USD)
    """

    def __init__(self, *, name: str = None):
        """
        Initialize Expectancy metric.

        Args:
            name: Optional custom name (default: "Expectancy")
        """
        self._name = name or "Expectancy"

    @property
    def name(self) -> str:
        return self._name

    def compute(
        self,
        returns: np.ndarray,
        context: Dict[str, Any],
    ) -> Mapping[str, float]:
        """
        Compute trade expectancy.

        Args:
            returns: Not used directly
            context: May contain pre-computed WinRate, AverageWin, AverageLoss

        Returns:
            {"Expectancy": expectancy}
        """
        # Get or compute components
        trade_pnls = context.get("trade_pnls", np.array([]))
        if len(trade_pnls) == 0:
            return {self.name: 0.0}

        trade_pnls = np.asarray(trade_pnls)

        # Win rate
        if "WinRate" in context:
            win_rate = context["WinRate"]
        else:
            win_rate = np.sum(trade_pnls > 0) / len(trade_pnls)

        # Average win
        if "AverageWin" in context:
            avg_win = context["AverageWin"]
        else:
            winning = trade_pnls[trade_pnls > 0]
            avg_win = np.mean(winning) if len(winning) > 0 else 0.0

        # Average loss
        if "AverageLoss" in context:
            avg_loss = context["AverageLoss"]
        else:
            losing = trade_pnls[trade_pnls < 0]
            avg_loss = np.abs(np.mean(losing)) if len(losing) > 0 else 0.0

        expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss

        return {self.name: float(expectancy)}

