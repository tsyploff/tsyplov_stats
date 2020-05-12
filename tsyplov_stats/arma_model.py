
import numpy as np
from tsyplov_stats.wolfram_functions import *
from tsyplov_stats.autoregression_model import *


class ARMA(AutoRegression):
    '''Ordinary ARMA.
    
    Attributes
    
    coef: estimated coefficients for the linear regression problem.

    true_values: array of y_train

    fitted_values: estimated values for x_train

    residuals: array of differences between true_values and fitted_values

    series: the original time series

    p: the quantity of using autoregrression coefficients

    q: the quantity of using moving average coefficients

    reg_p: autoregrression model
    
    reg_q: moving average model

    Example

    >>> import numpy as np
    >>> import tsyplov_stats
    >>> ts = np.array([25, 21, 21, 41, 12, 33])
    >>> reg = tsyplov_stats.arma_model(p=2, q=2).fit(ts)
    >>> reg.coef
    array([ 7.33908002e+01, -6.51188635e-01, -1.13464760e+00,  4.69453047e-03, -1.12732754e-01])
    >>> reg.mae()
    0.17014982892848085
    >>> reg.rmse()
    0.23573711065376157
    
    '''

    def __init__(self, p=2, q=2):
        '''Initializes model'''
        super().__init__(p)
        self.reg_p = AutoRegression(p)
        self.reg_q = AutoRegression(q)
        self.coef  = np.zeros(p + q + 1)
        self.q = q

    def fit(self, ts):
        '''Fits model'''
        self = self.reset_to_default() #model clear
        self.series      = ts.copy()
        self.true_values = ts[self.p + self.q]
        self.reg_p.fit(ts)
        self.reg_q.fit(self.reg_p.residuals)

        bias = self.reg_p.coef[0] + self.reg_q.coef[0]
        bias = np.array([bias])
        self.coef = np.hstack((bias, self.reg_p.coef[1:], self.reg_q.coef[1:]))

        self.fitted_values = self.reg_p.fitted_values[self.q:] + self.reg_q.fitted_values
        self.residuals     = self.true_values - self.fitted_values

        return self

    def predict(self, h=1):
        '''Gives forecast for x using model self'''
        residuals_fc = self.reg_q.predict(h)
        if self.p != 0:
            for t in range(h):
                fc = self.coef[0] + self.reg_p.coef[1:].dot(self.series[-self.p:]) + residuals_fc[t] #forecast
                fc = np.array([fc])
                self.series = np.hstack((self.series, fc))
                self.series, fc = self.series[:-h], self.series[-h:]
                return fc
        else:
            return self.coef[0] + residuals_fc

    def aic(self):
        '''The Akaike information criterion (AIC) is 
        an estimator of out-of-sample prediction error 
        and thereby relative quality of statistical 
        models for a given set of data.[1][2] Given a 
        collection of models for the data, AIC estimates 
        the quality of each model, relative to each of 
        the other models. Thus, AIC provides a means for 
        model selection.'''
        k = self.p + self.q + 1
        n = len(self.true_values)
        return 2*k + n*(np.log(2*np.pi*self.mse()) + 1)

    def reset_to_default(self):
        self.reg_p = AutoRegression(self.p)
        self.reg_q = AutoRegression(self.q)
        self.true_values   = np.zeros(2)
        self.fitted_values = np.zeros(2)
        self.residuals     = np.zeros(2)
        self.series = np.zeros(2)
        self.coef   = np.zeros(self.p + self.q + 1)
        return ARMA(self.p, self.q)
