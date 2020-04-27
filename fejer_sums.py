
import numpy as np
from wolfram_functions import *

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

    def seasonality_mean(self, ts):
        '''Gives seasonality mean'''
        length = len(y)//self.s
        y = ts[-(length*self.s):]
        return np.mean(y.reshape(length, s), axis=0)

    def fourier_coefficients(self, k):
        '''Gives the couple of coefficients (a_k, b_k) of Fourier series in model'''
        a = 2*self.bar_y.dot(np.cos(k*np.linspace(0, 2*np.pi, self.s, endpoint=False)))/self.s
        b = 2*self.bar_y.dot(np.sin(k*np.linspace(0, 2*np.pi, self.s, endpoint=False)))/self.s
        return np.array([a, b])
        
    def fit(self, ts):
        '''Fits model'''
        self.bar_y = self.seasonality_mean(ts)
        self.coef  = np.hstack(tuple(self.fourier_coefficients(k) for k in range(1, self.n)))
        self.true_values = ts

        self.fitted_values = []

        for t in range(len(ts)):
            cos = lambda x: np.cos(2*np.pi*x/self.s)
            sin = lambda x: np.sin(2*np.pi*x/self.s)
            funs = np.hstack(tuple((self.n - k)*np.array([cos(t), sin(t)])/self.n for k in range(1, self.n)))
            self.fitted_values.append(np.mean(self.bar_y) + self.coef.dot(funs))
        
        self.fitted_values = np.array(self.fitted_values)
        self.residuals = self.true_values - self.fitted_values

        return self

    def predict(self, h=1):
        '''Gives forecast for x using model self'''
        result = []
        for t in range(len(ts)):
            funs = np.hstack(tuple((self.n - k)*np.array([np.cos(2*np.pi*t/self.s), np.sin(2*np.pi*t/self.s)])/self.n for k in range(1, self.n)))
            result.append(np.mean(self.bar_y) + self.coef.dot(funs))
        return np.array(result)
