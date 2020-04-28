
import numpy as np
from tsyplov_stats.wolfram_functions import *
from tsyplov_stats.linearregression_model import *

class AutoRegression():
    '''Ordinary AutoRegression.
    
    Attributes
    
    coef: estimated coefficients for the linear regression problem.

    true_values: array of y_train

    fitted_values: estimated values for x_train

    residuals: array of differences between true_values and fitted_values

    series: the original time series

    p: the quantity of using coefficients

    Example

    >>> import numpy as np
    >>> import tsyplov_stats
    >>> ts = np.array([25, 21, 21, 41, 12, 33])
    >>> reg = tsyplov_stats.autoregression_model(p=2).fit(ts)
    >>> reg.coef
    array([71.27997359, -0.65118864, -1.1346476 ])
    >>> reg.mae()
    5.086329081303894
    >>> reg.rmse()
    6.336921026562879
    
    '''

    def __init__(self, p=2):
        '''Initializes model'''
        self.true_values   = np.zeros(2)
        self.fitted_values = np.zeros(2)
        self.residuals     = np.zeros(2)
        self.series = np.zeros(2)
        self.coef   = np.zeros(p + 1)
        self.p      = p

    def fit(self, ts):
        '''Fits model'''
        self.series      = ts.copy()
        self.true_values = ts[self.p:]

        x    = partition(ts[:-1], self.p, 1)
        x    = np.hstack((np.ones(len(x)).reshape(len(x), 1), x))
        x_tr = x.transpose()

        self.coef          = np.linalg.inv(x_tr.dot(x)).dot(x_tr).dot(self.true_values)
        self.fitted_values = x.dot(self.coef)
        self.residuals     = self.true_values - self.fitted_values

        return self
    
    def predict(self, h=1):
        '''Gives forecast for x using model self'''
        if self.p != 0:
            for _ in range(h):
                fc = self.coef[0] + self.coef[1:].dot(self.series[-self.p:]) #forecast
                fc = np.array([fc])
                self.series = np.hstack((self.series, fc))
            self.series, fc = self.series[:-h], self.series[-h:]
            return fc
        else:
            return np.zeros(h) + self.coef[0]

    def mae(self):
        '''Gives the mean absolute error'''
        return np.mean(np.abs(self.residuals))
    
    def mse(self):
        '''Gives the mean square error'''
        return np.mean((self.residuals)**2)
    
    def rmse(self):
        '''Gives the root from mean square error'''
        return np.sqrt(self.mse())