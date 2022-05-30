from Execution.ImmediateExecutionModel import ImmediateExecutionModel
from Portfolio.EqualWeightingPortfolioConstructionModel import EqualWeightingPortfolioConstructionModel

from AlgorithmImports import *

from risk import BracketRiskModel
from alpha import MyHistoricalReturnsAlphaModel

class TwoWeekMagnitudeEffectOnCurrentReturns(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2015, 5, 8)  # Set Start Date
        self.SetCash(1000000)  # Set Strategy Cash
        
        self.AddAlpha(MyHistoricalReturnsAlphaModel(14, Resolution.Daily))
        
        self.SetExecution(ImmediateExecutionModel())
        
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel())
        
        # uses universe selection to build a universe with 150 high volume stocks that have earnings within 3 months and low ATR(Average True Range) which signals low volatility
        self.SetUniverseSelection(FineFundamentalUniverseSelectionModel(self.CoarseSelectionFunction, self.FineSelectionFunction)) 
        self.UniverseSettings.Resolution = Resolution.Daily # makes sure the universe is on a daily rebalance period
        self.period = 14
        
        self.up_value = 0.10
        self.down_value = 0.05
        self.SetRiskManagement(BracketRiskModel(self.down_value, self.up_value))
        
        self.SetBrokerageModel(AlphaStreamsBrokerageModel())
        
    def CoarseSelectionFunction(self, coarse):
        # sort the universe by high dollar volume
        sortedByDollarVolume = sorted(coarse, key=lambda x: x.DollarVolume, reverse=True)
        # filter any assets that are above $5.00 and have a trading volume of over 10,000,000. then return the top 275
        filtered = [ x.Symbol for x in sortedByDollarVolume if x.HasFundamentalData and x.Price >= 5 and x.DollarVolume > 10000000 ][:275]
        return [ x for x in filtered ] # passes to FineSelectionFunction()
        
    def FineSelectionFunction(self, fine):
        # filters assets to return the top 150 that have earnings dates within 3 months
        filteredByDates = [ x for x in fine if x.EarningReports.FileDate.year == self.Time.year and x.EarningReports.FileDate.month - self.Time.month <= 3 ][:150]
        
        # gives us our atr indicator values
        for symbol in filteredByDates:
            self.AddSecurity(symbol.Symbol, Resolution.Daily)
            history = self.History([symbol.Symbol], self.period, Resolution.Daily)
            atr = self.ATR(symbol.Symbol, self.period, Resolution.Daily)
            for bar in history.itertuples():
                tradebar = TradeBar(bar.Index[1], symbol, bar.open, bar.high, bar.low, bar.close, bar.volume)
                atr.Update(tradebar)
            symbol.atr = atr.Current.Value
        
        # use this to sort by low to high ATR values 
        sortedByVolatility = sorted(filteredByDates, key=lambda x: abs(x.atr), reverse=False)
        
        return [ x.Symbol for x in sortedByVolatility ] # gives us our universe
