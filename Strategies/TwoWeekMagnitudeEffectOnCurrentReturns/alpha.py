class MyHistoricalReturnsAlphaModel(AlphaModel):
    '''Uses Historical returns to create insights.'''
        
    def __init__(self, *args, **kwargs):
        '''Initializes a new default instance of the HistoricalReturnsAlphaModel class.
        Args:
            lookback(int): Historical return lookback period
            resolution: The resolution of historical data'''
        self.lookback = kwargs['lookback'] if 'lookback' in kwargs else 1
        self.resolution = kwargs['resolution'] if 'resolution' in kwargs else Resolution.Daily
        self.predictionInterval = Time.Multiply(Extensions.ToTimeSpan(self.resolution), self.lookback)
        self.symbolDataBySymbol = {}

    def Update(self, algorithm, data):
        '''Updates this alpha model with the latest data from the algorithm.
        This is called each time the algorithm receives data for subscribed securities
        Args:
            algorithm: The algorithm instance
            data: The new data available
        Returns:
            The new insights generated'''
        insights = []

        for symbol, symbolData in self.symbolDataBySymbol.items():
            if symbolData.CanEmit:
                
                direction = InsightDirection.Flat
                magnitude = symbolData.Return
                if magnitude > .10: direction = InsightDirection.Down
                if magnitude < -.10: direction = InsightDirection.Up
                
                insights.append(Insight.Price(symbol, self.predictionInterval, direction, magnitude, None))
                
        return insights
            
    def OnSecuritiesChanged(self, algorithm, changes):
        '''Event fired each time the we add/remove securities from the data feed
        Args:
            algorithm: The algorithm instance that experienced the change in securities
            changes: The security additions and removals from the algorithm'''
            
        # clean up data for removed securities
        for removed in changes.RemovedSecurities:
            symbolData = self.symbolDataBySymbol.pop(removed.Symbol, None)
            if symbolData is not None:
                symbolData.RemoveConsolidators(algorithm)
                
        # initialize data for added securities
        symbols = [ x.Symbol for x in changes.AddedSecurities ]
        history = algorithm.History(symbols, self.lookback, self.resolution)
        if history.empty: return
            
        tickers = history.index.levels[0]
        for ticker in tickers:
            symbol = SymbolCache.GetSymbol(ticker)
                
            if symbol not in self.symbolDataBySymbol:
                symbolData = SymbolData(symbol, self.lookback)
                self.symbolDataBySymbol[symbol] = symbolData
                symbolData.RegisterIndicators(algorithm, self.resolution)
                symbolData.WarmUpIndicators(history.loc[ticker])
                
                
class SymbolData:
    '''Contains data specific to a symbol required by this model'''
    def __init__(self, symbol, lookback):
        self.Symbol = symbol
        self.ROCP = RateOfChange('{}.ROCP({})'.format(symbol, lookback), lookback)
        
        self.Consolidator = None
        self.previous = 0
            
    def RegisterIndicators(self, algorithm, resolution):
        self.Consolidator = algorithm.ResolveConsolidator(self.Symbol, resolution)
        algorithm.RegisterIndicator(self.Symbol, self.ROCP, self.Consolidator)
            
    def RemoveConsolidators(self, algorithm):
        if self.Consolidator is not None:
            algorithm.SubscriptionManager.RemoveConsolidator(self.Symbol, self.Consolidator)
            
    def WarmUpIndicators(self, history):
        for tuple in history.itertuples():
            self.ROCP.Update(tuple.Index, tuple.close)
            
    @property
    def Return(self):
        return float(self.ROCP.Current.Value)
            
    @property
    def CanEmit(self):
        if self.previous == self.ROCP.Samples:
            return False
            
        self.previous = self.ROCP.Samples
        return self.ROCP.IsReady
        
    def __str__(self, **kwargs):
        return '{}: {:.2%}'.format(self.ROC.Name, (1 + self.Return)**252 - 1)
