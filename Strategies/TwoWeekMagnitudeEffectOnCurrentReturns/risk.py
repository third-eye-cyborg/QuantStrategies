class BracketRiskModel(RiskManagementModel):
    '''Provides an implementation of IRiskManagementModel that limits the maximum possible loss
    measured from the highest unrealized profit'''
    def __init__(self, maximumDrawdownPercent = 0.05, maximumUnrealizedProfitPercent = 0.05):
        '''Initializes a new instance of the TrailingStopRiskManagementModel class
        Args:
            maximumDrawdownPercent: The maximum percentage drawdown allowed for algorithm portfolio compared with the highest unrealized profit, defaults to 5% drawdown'''
        self.maximumDrawdownPercent = -abs(maximumDrawdownPercent)
        self.trailingHighs = dict()
        self.maximumUnrealizedProfitPercent = abs(maximumUnrealizedProfitPercent)
        
    def ManageRisk(self, algorithm, targets):
        '''Manages the algorithm's risk at each time step
        Args:
            algorithm: The algorithm instance
            targets: The current portfolio targets to be assessed for risk'''
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
                algorithm.Debug("Profit Taken")
                algorithm.Log("Profit Taken")
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
                algorithm.Debug("Losses Taken")
                algorithm.Log("Losses Taken")
                riskAdjustedTargets.append(PortfolioTarget(symbol, 0))
                
        return riskAdjustedTargets
