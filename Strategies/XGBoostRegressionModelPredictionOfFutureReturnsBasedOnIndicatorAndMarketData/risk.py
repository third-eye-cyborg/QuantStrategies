#region imports
from AlgorithmImports import *
#endregion
from QuantConnect.Algorithm.Framework.Risk import RiskManagementModel

# bracket risk model class
class BracketRiskModel(RiskManagementModel):
    '''Creates a trailing stop loss for the maximumDrawdownPercent value and a profit taker for the maximumUnrealizedProfitPercent value'''
    def __init__(self, maximumDrawdownPercent = 0.05, maximumUnrealizedProfitPercent = 0.05):
        self.maximumDrawdownPercent = -abs(maximumDrawdownPercent)
        self.trailingHighs = dict()
        self.maximumUnrealizedProfitPercent = abs(maximumUnrealizedProfitPercent)

    def ManageRisk(self, algorithm, targets):
        riskAdjustedTargets = list()
        for kvp in algorithm.Securities:
            symbol = kvp.Key
            security = kvp.Value

            # Remove if not invested
            if not security.Invested:
                self.trailingHighs.pop(symbol, None)
                continue
            pnl = security.Holdings.UnrealizedProfitPercent
            
            if pnl > self.maximumUnrealizedProfitPercent:
                # liquidate
                algorithm.Debug(f"Profit Taken: {security.Symbol}")
                algorithm.Log(f"Profit Taken: {security.Symbol}")
                riskAdjustedTargets.append(PortfolioTarget(security.Symbol, 0))
                return riskAdjustedTargets
                
            # Add newly invested securities
            if symbol not in self.trailingHighs:
                self.trailingHighs[symbol] = security.Holdings.AveragePrice   # Set to average holding cost
                continue

            # Check for new highs and update - set to tradebar high
            if self.trailingHighs[symbol] < security.High:
                self.trailingHighs[symbol] = security.High
                continue

            # Check for securities past the drawdown limit
            securityHigh = self.trailingHighs[symbol]
            drawdown = (security.Low / securityHigh) - 1

                
            if drawdown < self.maximumDrawdownPercent:
                # liquidate
                algorithm.Debug(f"Losses Taken: {security.Symbol}")
                algorithm.Log(f"Losses Taken: {security.Symbol}")
                riskAdjustedTargets.append(PortfolioTarget(symbol, 0))
                
        return riskAdjustedTargets
