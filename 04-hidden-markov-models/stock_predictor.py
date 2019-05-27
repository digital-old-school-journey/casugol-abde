import pandas as pd
import logging
 
class StockPredictor(object):
    def __init__(self, company, n_latency_days=10):
        self._init_logger()
 
        self.company = company
        self.n_latency_days = n_latency_days
        self.data = pd.read_csv(
            'data/company_data/{company}.csv'.format(company=self.company))
 
 
    def _init_logger(self):
        self._logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.DEBUG)
 
 
    @staticmethod
    def _extract_features(data):
        open_price = np.array(data['open'])
        close_price = np.array(data['close'])
        high_price = np.array(data['high'])
        low_price = np.array(data['low'])
 
        # Compute the fraction change in close, high and low prices
        # which would be used a feature
        frac_change = (close_price - open_price) / open_price
        frac_high = (high_price - open_price) / open_price
        frac_low = (open_price - low_price) / open_price
 
        return np.column_stack((frac_change, frac_high, frac_low))
 
 
# Predictor for GOOGL stocks
stock_predictor = StockPredictor(company='GOOGL')