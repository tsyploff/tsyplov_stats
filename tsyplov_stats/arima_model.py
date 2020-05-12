
import numpy as np
from tsyplov_stats.wolfram_functions import *
from tsyplov_stats.autoregression_model import *
from tsyplov_stats.arma_model import *

def diff(ts, d):
    '''Gives (1 - B)^d y, the list of 
    first values each of y, (1 - B) y, (1 - B)^2 y, ...
    and the list of last values each of y, (1 - B) y, (1 - B)^2 y
        '''
    start, end, dts = list(), list(), ts.copy()
        
    for _ in range(d):
        start.append(dts[0])
        end.append(dts[-1])
        dts = np.diff(dts)

    return dts, start[::-1], end[::-1]
        
def accumulate(dts, start):
    '''Returns back y using (1 - B)^d y and [y_0, (1 - B) y_1, ...]
    dts – (1 - B)^d y

    '''
    integrate = lambda dy, y0: np.cumsum(np.insert(dy, 0, y0))
    return fold(integrate, dts, start)


class ARIMA(AutoRegression):
    
    def __init__(self, p=1, d=1, q=1):
        self.p = p
        self.d = d
        self.q = q

        self.true_values   = np.zeros(2)
        self.fitted_values = np.zeros(2)
        self.residuals     = np.zeros(2)
        
        self.coef = np.zeros(p + q + 1)
        self.reg  = ARMA(p, q)

        self.series = np.zeros(2)
        self.dts    = np.zeros(2)
        self.start  = np.zeros(2)
        self.end    = np.zeros(2)

    def fit(self, ts):
        self.reset_to_default() #model clear
        self.series = ts.copy()
        self.dts, self.start, self.end = diff(ts, self.d)
        
        self.reg.fit(self.dts)
        self.coef = self.reg.coef

        self.true_values   = ts[self.p + self.q:]
        self.fitted_values = accumulate(self.reg.fitted_values, self.start)
        self.residuals     = self.true_values - self.fitted_values 

        return self

    def predict(self, h=1):
        return accumulate(self.reg.predict(h), self.end)[-h:]

    def reset_to_default(self):
        self.true_values   = np.zeros(2)
        self.fitted_values = np.zeros(2)
        self.residuals     = np.zeros(2)
        
        self.coef = np.zeros(self.p + self.q + 1)
        self.reg  = ARMA(self.p, self.q)

        self.series = np.zeros(2)
        self.dts    = np.zeros(2)
        self.start  = np.zeros(2)
        self.end    = np.zeros(2)
        return ARIMA(self.p, self.d, self.q)
