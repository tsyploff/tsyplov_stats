
import numpy as np
from tsyplov_stats.wolfram_functions import *
from tsyplov_stats.linearregression_model import *
from tsyplov_stats.autoregression_model import *


class SARMA(AutoRegression):
    
    def __init__(self, order=(1, 1), seasonal_order=(1, 1, 12)):
        '''
        order = (p, q)
        seasonal_order = (P, Q, seasonal lag)
        '''
        self.p, self.q = order
        self.P, self.Q, self.s = seasonal_order

        self.true_values   = np.zeros(2)
        self.fitted_values = np.zeros(2)
        self.residuals     = np.zeros(2)
        self.coef          = np.zeros(self.p + self.q + self.P + self.Q + 1)

        self.sar = LinearRegression()
        self.sma = LinearRegression()

    def fit(self, ts):
        self.reset_to_default() #model clearing

        span = len(ts) - max(self.P*self.s, self.p)

        tmp1 = partition(ts, 1 + self.p, 1)
        tmp1 = tmp1[-span:]
        tmp2 = [ts[:-(i * self.s)][-span:] for i in range(1, 1 + self.P)][::-1] #seasonal lags
        tmp2 = np.array(tmp2).transpose()
        tmp3 = np.ones(span).reshape(-1, 1)

        if self.P == 0:
        	sarX = np.hstack((tmp3, tmp1[:, :-1]))
        else:
        	sarX = np.hstack((tmp3, tmp2, tmp1[:, :-1]))
        
        sarY = tmp1[:, -1]

        self.sar.fit(sarX, sarY)

        span -= max(self.Q*self.s, self.q)

        tmp1 = partition(self.sar.residuals, 1 + self.q, 1)
        tmp1 = tmp1[-span:]
        tmp2 = [self.sar.residuals[:-(i * self.s)][-span:] for i in range(1, 1 + self.Q)][::-1] #seasonal lags
        tmp2 = np.array(tmp2).transpose()
        tmp3 = np.ones(span).reshape(-1, 1)

        if self.Q == 0:
        	smaX = np.hstack((tmp3, tmp1[:, :-1]))
        else:
        	smaX = np.hstack((tmp3, tmp2, tmp1[:, :-1]))
        
        smaY = tmp1[:, -1]
        
        self.sma.fit(smaX, smaY)

        self.true_values   = ts[-span:]
        self.fitted_values = self.sar.fitted_values[-span:] + self.sma.fitted_values
        self.residuals     = self.true_values - self.fitted_values
        self.coef          = np.insert(np.hstack((self.sar.coef[1:], self.sma.coef[1:])), 0, self.sar.coef[0] + self.sma.coef[0])

        return self

    def predict(self, h=1):
        
        for _ in range(h):
            tmp1 = self.residuals[-self.q:]
            tmp2 = [self.residuals[-(i * self.s)] for i in range(1, 1 + self.Q)][::-1]
            tmp2 = np.array(tmp2)
            tmp3 = np.ones(1)
            vect = np.hstack((tmp3, tmp2, tmp1))
            self.residuals = np.hstack((self.residuals, vect.dot(self.sma.coef)))

        self.residuals, residuals_fc = take_drop(self.residuals, -h)

        for i in range(h):
            tmp1 = self.true_values[-self.p:]
            tmp2 = [self.true_values[-(i * self.s)] for i in range(1, 1 + self.P)][::-1]
            tmp2 = np.array(tmp2)
            tmp3 = np.ones(1)
            vect = np.hstack((tmp3, tmp2, tmp1))
            self.true_values = np.hstack((self.true_values, vect.dot(self.sar.coef) + residuals_fc[i]))

        self.true_values, fc = take_drop(self.true_values, -h)	

        return fc
        
    def reset_to_default(self):
        self.true_values   = np.zeros(2)
        self.fitted_values = np.zeros(2)
        self.residuals     = np.zeros(2)
        self.coef          = np.zeros(self.p + self.q + self.P + self.Q + 1)

        self.sar = LinearRegression()
        self.sma = LinearRegression()

        order = self.p, self.q
        seasonal_order = self.P, self.Q, self.s
        
        return SARMA(order, seasonal_order)
