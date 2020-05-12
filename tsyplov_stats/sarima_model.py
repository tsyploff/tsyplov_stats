
import numpy as np
from tsyplov_stats.wolfram_functions import *
from tsyplov_stats.arima_model import diff, accumulate
from tsyplov_stats.sarma_model import SARMA 
from tsyplov_stats.autoregression_model import AutoRegression


def seasonal_diff(ts, D, s):
    start, end, dts = list(), list(), ts.copy()
    
    for _ in range(D):
        start.append(dts[:s])
        end.append(dts[-s:])
        dts = dts[s:] - dts[:-s]
        
    return dts, start[::-1], end[::-1]

def seasonal_cumsum(dts, ts0, s):
    '''
    >>> len(start)
    s
    '''
    k  = s - len(dts) % s
    ts = np.hstack((ts0, dts, np.zeros(k)))
    ts = np.cumsum(partition(ts, s, s), axis=0).flatten()
    return ts[:-k]

def seasonal_accumulate(dts, start, s):
    integrate = lambda dy, y0: seasonal_cumsum(dy, y0, s)
    return fold(integrate, dts, start)


class SARIMA(AutoRegression):

    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        '''
        order = (p, d, q)
        seasonal_order = (P, D, Q, seasonal lag)
        '''
        self.p, self.d, self.q = order
        self.P, self.D, self.Q, self.s = seasonal_order

        self.true_values   = np.zeros(2)
        self.fitted_values = np.zeros(2)
        self.residuals     = np.zeros(2)
        self.coef          = np.zeros(self.p + self.q + self.P + self.Q + 1)

        self.reg = SARMA(order=(self.p, self.q), seasonal_order=(self.P, self.Q, self.s))

        self.series = np.zeros(2)
        
        self.dts   = np.zeros(2)
        self.start = np.zeros(2)
        self.end   = np.zeros(2)
        
        self.seasonal_dts   = np.zeros(2)
        self.seasonal_start = np.zeros(2)
        self.seasonal_end   = np.zeros(2)

    def fit(self, ts):
        self.reset_to_default() #model clearing
        self.series = ts.copy()
        self.dts, self.start, self.end = diff(ts, self.d)

        self.seasonal_dts, self.seasonal_start, self.seasonal_end = seasonal_diff(self.dts, self.D, self.s)

        self.reg.fit(self.seasonal_dts)
        self.coef = self.reg.coef

        self.fitted_values = accumulate(seasonal_accumulate(self.reg.fitted_values, self.seasonal_start, self.s), self.start)
        self.true_values   = ts[-len(self.fitted_values):]
        self.residuals     = self.true_values - self.fitted_values

        return self

    def predict(self, h=1):
        return accumulate(seasonal_accumulate(self.reg.predict(h), self.seasonal_end, self.s), self.end)[-h:]

    def reset_to_default(self):
        self.true_values   = np.zeros(2)
        self.fitted_values = np.zeros(2)
        self.residuals     = np.zeros(2)
        self.coef          = np.zeros(self.p + self.q + self.P + self.Q + 1)

        self.reg = SARMA(order=(self.p, self.q), seasonal_order=(self.P, self.Q, self.s))

        self.series = np.zeros(2)
        
        self.dts   = np.zeros(2)
        self.start = np.zeros(2)
        self.end   = np.zeros(2)
        
        self.seasonal_dts   = np.zeros(2)
        self.seasonal_start = np.zeros(2)
        self.seasonal_end   = np.zeros(2)
        return SARIMA(order=(self.p, self.d, self.q), seasonal_order=(self.P, self.D, self.Q, self.s))
