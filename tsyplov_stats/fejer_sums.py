
import numpy as np
from tsyplov_stats.wolfram_functions import *

def seasonality_mean(s, ts):
    '''Gives seasonality mean'''
    length = len(ts)//s
    y = ts[-(length*s):]
    return np.mean(y.reshape(length, s), axis=0)

class FejerSums():
    
    def __init__(self, n=10, s=7):
        '''Initializes model'''
        self.n = n #the order of the amount
        self.s = s #seasonality
        self.bar_y = np.zeros(2)
        self.coef  = np.zeros(n)
        self.true_values   = np.zeros(2)
        self.fitted_values = np.zeros(2)
        self.residuals     = np.zeros(2)

    def fourier_coefficients(self, k):
        '''Gives the couple of coefficients (a_k, b_k) of Fourier series in model'''
        a = 2*self.bar_y.dot(np.cos(k*np.linspace(0, 2*np.pi, self.s, endpoint=False)))/self.s
        b = 2*self.bar_y.dot(np.sin(k*np.linspace(0, 2*np.pi, self.s, endpoint=False)))/self.s
        return np.array([a, b])
        
    def fit(self, ts):
        '''Fits model'''
        self.reset_to_default() #model clear
        self.series      = ts.copy()
        self.bar_y = seasonality_mean(self.s, ts)
        self.coef  = np.hstack(tuple(self.fourier_coefficients(k) for k in range(1, self.n)))
        self.true_values = ts

        self.fitted_values = []

        cos = lambda x: np.cos(2*np.pi*x/self.s)
        sin = lambda x: np.sin(2*np.pi*x/self.s)

        for t in range(len(ts)):
            funs = np.hstack(tuple((self.n - k)*np.array([cos(t), sin(t)])/self.n for k in range(1, self.n)))
            self.fitted_values.append(np.mean(self.bar_y) + self.coef.dot(funs))
        
        self.fitted_values = np.array(self.fitted_values)
        self.residuals = self.true_values - self.fitted_values

        return self

    def predict(self, h=1):
        '''Gives forecast for x using model self'''
        result = []

        cos = lambda x: np.cos(2*np.pi*x/self.s)
        sin = lambda x: np.sin(2*np.pi*x/self.s)

        for t in range(len(self.true_values), len(self.true_values) + h):
            funs = np.hstack(tuple((self.n - k)*np.array([cos(t), sin(t)])/self.n for k in range(1, self.n)))
            result.append(np.mean(self.bar_y) + self.coef.dot(funs))
            
        return np.array(result)

    def reset_to_default(self):
        self.bar_y = np.zeros(2)
        self.coef  = np.zeros(self.n)
        self.true_values   = np.zeros(2)
        self.fitted_values = np.zeros(2)
        self.residuals     = np.zeros(2)
        return FejerSums(self.n, self.s)


class RecursiveFejerSums():
    '''Fejer sums for case of several seasonalities'''
    
    def __init__(self, n=10, s=[7]):
        '''Initializes model'''
        self.n = n #the order of the amount
        self.s = s #seasonalities
        self.sigmas = []
        self.true_values   = np.zeros(2)
        self.fitted_values = np.zeros(2)
        self.residuals     = np.zeros(2)
    
    def fit(self, ts):
        '''fits all of models'''
        self.reset_to_default() #model clear
        self.true_values   = ts.copy()
        self.fitted_values = np.zeros(len(ts))
        self.residuals     = ts.copy()

        for i in self.s:
            model = FejerSums(self.n, i).fit(self.residuals)
            self.fitted_values += model.fitted_values
            self.residuals = model.residuals
            self.sigmas.append(model)

        return self

    def predict(self, h=1):
        forecast = np.zeros(h)
        for model in self.sigmas:
            forecast += model.predict(h)
        return forecast

    def reset_to_default(self):
        self.sigmas = []
        self.true_values   = np.zeros(2)
        self.fitted_values = np.zeros(2)
        self.residuals     = np.zeros(2)
        return RecursiveFejerSums(self.n, self.s)
