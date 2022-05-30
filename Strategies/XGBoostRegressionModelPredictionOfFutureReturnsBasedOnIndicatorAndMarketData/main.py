from AlgorithmImports import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error as MSE

import sklearn as sk
import xgboost as xgb
import pandas as pd
import numpy as np

from QuantConnect.Algorithm.Framework.Portfolio import PortfolioTarget
from Execution.ImmediateExecutionModel import ImmediateExecutionModel

from QuantConnect import *
from risk import BracketRiskModel

class ModulatedNadionReplicator(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2015, 1, 1)  # Set Start Date
        self.SetCash(100000)  # Set Strategy Cash
        
        # the coarse universe selection model that takes top volume stocks over 5,000,000 in volume and between $20 - $200
        # then in the fine universe selection it gets the top stocks sorted by the Earnings Filing Dates fundemental metric
        # set the resolution to be hour
        self.__numberOfSymbols = 30
        self.__numberOfSymbolsFine = 10
        self.SetUniverseSelection(FineFundamentalUniverseSelectionModel(self.CoarseSelectionFunction, self.FineSelectionFunction))
        self.UniverseSettings.Resolution = Resolution.Hour
        
        # standard immidiate execution model
        self.SetExecution(ImmediateExecutionModel())
        
        # insight weighted portfolio model with 5% of the portfolio liquid
        self.Settings.FreePortfolioValuePercentage = 0.05
        self.SetPortfolioConstruction(InsightWeightingPortfolioConstructionModel())
        
        # custom bracket risk model, trailing stop with profit taker
        self.up_value = 0.15
        self.down_value = 0.05
        self.SetRiskManagement(BracketRiskModel(self.down_value, self.up_value))
        
        # standard alpha streams brokerage model
        self.SetBrokerageModel(AlphaStreamsBrokerageModel())
        
        # get ready the symbol data dictionary to hold our data passed from the SymbolData class
        self.symbolDataBySymbol = {}
        self.isTrained = False
        self.symbolFlag = {}
        
        # global dictionary to store our models
        self.models = {}
        
        # the market symbol in this example "SPY"
        self.AddEquity("SPY")
        
        # training and scheduling methods
        self.Train(self.TrainingMethod)
        self.Train(self.DateRules.MonthEnd(), self.TimeRules.At(8,0), self.TrainingMethod)
        self.Schedule.On(self.DateRules.EveryDay('SPY'), self.TimeRules.AfterMarketOpen('SPY', 30), self.Predict)
        
        # coarse universe
    def CoarseSelectionFunction(self, coarse):
        # rebalance logic for a one month rebalance at midnight
        if not self.Time.day % 30 == 1 and not self.Time.hour % 24 == 1: return self.Universe.Unchanged
        # sort descending by daily dollar volume
        sortedByDollarVolume = sorted(coarse, key=lambda x: x.DollarVolume, reverse=True)
        filtered = [ x.Symbol for x in sortedByDollarVolume if x.HasFundamentalData and x.Price >= 20 and x.Price <= 200 and x.DollarVolume > 5000000 ]
        return [ x for x in filtered[:self.__numberOfSymbols] ]
        # fine universe
    def FineSelectionFunction(self, fine):
        # sort by most recent earnings dates
        sortedByEarnings = sorted(fine, key=lambda x: x.EarningReports.FileDate, reverse=False)[:self.__numberOfSymbolsFine] # sort by most recent earnings dates
        return [ x.Symbol for x in sortedByEarnings ]
        
        
    def TrainingMethod(self):
        # lets the predict function know when the model has been trained
        self.isTrained = False
        scores = []
        # go through the dictionary of symbol data we recieved from our SymbolData class
        for symbol, symbolData in self.symbolDataBySymbol.items():
            try:
                # don't run the model on the "SPY" symbol due to this being our market indicator symbol
                if str(symbol) == 'SPY': break
                self.symbolFlag[symbol] = False
                
                # wait until the indicator data is ready
                if not (symbolData.atr.IsReady and symbolData.rsi.IsReady and symbolData.market_rocp.IsReady and symbolData.rocp.IsReady): continue
                # load up the indicator values into numpy arrays, as this is the format we will need for the xgboost regressor
                market_rocp = np.array([x.Value for x in symbolData.market_rocpWin]).reshape(-1,1)
                rocp = np.array([x.Value for x in symbolData.rocpWin]).reshape(-1,1)
                rsi = np.array([x.Value for x in symbolData.rsiWin]).reshape(-1,1)
                atr = np.array([x.Value for x in symbolData.atrWin]).reshape(-1,1)
                # scale the data down for better feature quality
                scaler = MinMaxScaler(feature_range=(-1, 1))
                scaler.fit(rocp)
                scaled_rocp = scaler.transform(rocp)
                
                scaler.fit(atr)
                scaled_atr = scaler.transform(atr)
                scaler.fit(market_rocp)
                scaled_market_rocp = scaler.transform(market_rocp)
                scaler.fit(rsi)
                scaled_rsi = scaler.transform(rsi)
                
                # create our indicator dataframe to feed into our train_test_split funtion
                indicator_df = pd.DataFrame(np.hstack((scaled_market_rocp, scaled_rsi, scaled_atr)))
                
                # split our data into training and testing groups for our model
                x_train, x_valid, y_train, y_valid = train_test_split(indicator_df, scaled_rocp, test_size=0.35, random_state=42)
                
                # fit the model to the data with the eval set with the parameters below passed to the random search optimizer with 5 iterations and a cv of 4
                parameters = {
                        'n_estimators': [100, 200, 300, 400],
                        'learning_rate': [0.001, 0.005, 0.01, 0.05],
                        'max_depth': [8, 10, 12, 15],
                        'gamma': [0.001, 0.005, 0.01, 0.02],
                        'random_state': [42]
                     }
                eval_set = [(x_train, y_train), (x_valid, y_valid)]
                self.models[symbol] = xgb.XGBRegressor(eval_set=eval_set, objective='reg:squarederror', verbose=False)
                
                self.models[symbol] = RandomizedSearchCV(estimator=self.models[symbol], param_distributions=parameters, n_iter=5, scoring='neg_mean_squared_error', cv=4, verbose=1)
                
                self.models[symbol].fit(x_train, y_train, eval_set=eval_set, verbose=False)

                y_pred = self.models[symbol].predict(x_valid)
                
                score = np.sqrt(MSE(y_valid, y_pred))
                
                
                # you can always experiment with the more ways of setting your trained flag like below
                # scores.append(score)
                # if len(scores) > 2:
                #     scores_avg = sum(scores) / len(scores)
                #     if score < scores_avg:
                #         self.symbolFlag[symbol] = True
                #         self.Debug(f'Scores Average: {scores_avg}')
                        
                self.symbolFlag[symbol] = True
                
                self.Debug(f'Score: {score}')
                
                self.Debug(f'Trained Model: {symbol}')
                
                self.Debug("Models Trained")
            except:
                self.Debug(f"error_training {symbol}")
        self.isTrained = True
        return 
            
    def Predict(self):
        # only predict if the model is trained
        if self.isTrained == True:
            # go through our symbol data from the SymbolData class
            for symbol, symbolData in self.symbolDataBySymbol.items():
                try:
                    # don't run the model on the "SPY" symbol due to this being our market indicator symbol
                    if str(symbol) == 'SPY': break
                    if symbol in self.symbolFlag and self.symbolFlag[symbol] == True:
                        
                        # wait until the indicator data is ready
                        if not (symbolData.atr.IsReady and symbolData.rsi.IsReady and symbolData.market_rocp.IsReady and symbolData.rocp.IsReady): continue
                        # load up the indicator values into numpy arrays, as this is the format we will need for the xgboost regressor
                        market_rocp = np.array([x.Value for x in symbolData.market_rocpWin]).reshape(-1,1)
                        rocp = np.array([x.Value for x in symbolData.rocpWin]).reshape(-1,1)
                        rsi = np.array([x.Value for x in symbolData.rsiWin]).reshape(-1,1)
                        atr = np.array([x.Value for x in symbolData.atrWin]).reshape(-1,1)
                        # scale the data down for better feature quality
                        scaler = MinMaxScaler(feature_range=(-1, 1))
                        scaler.fit(rocp)
                        scaled_rocp = scaler.transform(rocp)
                        
                        scaler.fit(atr)
                        scaled_atr = scaler.transform(atr)
                        scaler.fit(market_rocp)
                        scaled_market_rocp = scaler.transform(market_rocp)
                        scaler.fit(rsi)
                        scaled_rsi = scaler.transform(rsi)
                        
                        # create our indicator dataframe to feed into our train_test_split funtion
                        indicator_df = pd.DataFrame(np.hstack((scaled_market_rocp, scaled_rsi, scaled_atr)))
                        
                        # make sure our model exists
                        if symbol in self.models:
                            # load our model that we know is ready due to our flag for the model being tained is true     
                            model = self.models[symbol]
                            
                            # predict on new data with our model
                            model_predict = model.predict(indicator_df)
                            # get the most current prediction we will use for our insight logic
                            magnitude = round(float(model_predict[-1]), 2)
                            
                            # get some important metrics we can use for our insights
                            market_avg = (sum(scaled_market_rocp) / len(scaled_market_rocp))
                            rocp_avg = (sum(scaled_rocp) / len(scaled_rocp))
                            mag = abs(magnitude)
                            
                            # direction starts flat
                            direction = InsightDirection.Flat
                            
                            self.Debug(f"Predicted Model: {symbol}")
                            self.Debug(f'prediction: {magnitude}')
                            self.Debug(f'rocp_avg: {rocp_avg}')
                            self.Debug(f'market_avg: {market_avg}')

                            # you could change your risk values to be more dynamic values
                            
                            self.down_value = abs(market_avg) * 0.2
                            self.up_value = mag * 0.8

                            # if prediction is above 0.5%, go long
                            if magnitude > 0.5:
                                direction = InsightDirection.Up
                                
                            # if the prediction is below -0.5%, go short
                            if magnitude < -0.5:
                                direction = InsightDirection.Down
                                
                            self.Debug(f'Direction: {str(direction)}')
                            
                            # emit our signal using the rocp_avg as the expected move, the market_avg as the confidence, 
                            # and the absolute magnitude multiplied by .3 as our weight 
                            self.EmitInsights(Insight.Price(symbol, timedelta(hours = 24), direction, rocp_avg, market_avg, None, (.3 * mag)))
                except:
                    self.Debug(f"issue trading {symbol}")
        return
        
    # on symbol change we need to remove the symbols that are dropping off our universe from our subscription manager
    # alternatively we call our history data, instantiate our SymbolData class, and update our indcator data
    def OnSecuritiesChanged(self, changes):
        symbols = [ x.Symbol for x in changes.RemovedSecurities ]
        if len(symbols) > 0:
            for subscription in self.SubscriptionManager.Subscriptions:
                if subscription.Symbol in symbols:
                    self.symbolDataBySymbol.pop(subscription.Symbol, None)
                if subscription.Symbol in self.models:
                    self.models.pop(subscription.Symbol, None)
                if subscription.Symbol in self.symbolFlag:
                    self.symbolFlag.pop(subscription.Symbol, None)
                subscription.Consolidators.Clear()
                    
        addedSymbols = [ x.Symbol for x in changes.AddedSecurities if x.Symbol not in self.symbolDataBySymbol]
        
        if len(addedSymbols) == 0: return
        history = self.History(addedSymbols, 24, Resolution.Hour)
        
        for symbol in addedSymbols:
            symbolData = SymbolData(symbol, self)
            self.symbolDataBySymbol[symbol] = symbolData
            if not history.empty:
                ticker = SymbolCache.GetTicker(symbol)
                if ticker not in history.index.levels[0]:
                    continue
                    
                for tuple in history.loc[ticker].itertuples():
                    symbolData.market_rocp.Update(tuple.Index, tuple.close)
                    symbolData.rocp.Update(tuple.Index, tuple.close)
                    symbolData.rsi.Update(tuple.Index, tuple.close)
        for bar in history.itertuples():
            tradebar = TradeBar(bar.Index[1], symbol, bar.open, bar.high, bar.low, bar.close, bar.volume)
            symbolData.atr.Update(tradebar)
                    
# SymbolData class that is used to get us our indicator values
class SymbolData:
    '''Contains data 
    specific to a symbol required by this model'''
    def __init__(self, symbol, algorithm):
        self.Insight = None
        self.Symbol = symbol
        self.market_rocp = algorithm.ROCP('SPY', 24)
        self.rocp = algorithm.ROCP(symbol, 24)
        self.rsi = algorithm.RSI(symbol,12, MovingAverageType.Simple)
        self.atr = algorithm.ATR(symbol, 24)
        
        self.window_length = 24
        
        self.market_rocp.Updated += self.market_RocpUpdated
        self.market_rocpWin = RollingWindow[IndicatorDataPoint](self.window_length) 
        
        self.rocp.Updated += self.RocpUpdated
        self.rocpWin = RollingWindow[IndicatorDataPoint](self.window_length)
        
        self.rsi.Updated += self.RsiUpdated
        self.rsiWin = RollingWindow[IndicatorDataPoint](self.window_length)
        
        self.atr.Updated += self.AtrUpdated
        self.atrWin = RollingWindow[IndicatorDataPoint](self.window_length)
        
    def market_RocpUpdated(self, sender, updated):
        self.market_rocpWin.Add(updated)
        
    def RocpUpdated(self, sender, updated):
        self.rocpWin.Add(updated)
        
    def RsiUpdated(self, sender, updated):
        self.rsiWin.Add(updated)
        
    def AtrUpdated(self, sender, updated):
        self.atrWin.Add(updated)
