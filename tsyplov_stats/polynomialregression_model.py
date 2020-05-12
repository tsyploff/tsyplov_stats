
import numpy as np
from tsyplov_stats.wolfram_functions import *
from tsyplov_stats.linearregression_model import *

class PolynomialRegression(LinearRegression):
    '''Ordinary least squares Linear Regression. Builds a polynomial regression model for times series (!).
    
    Attributes
    
    coef: estimated coefficients for the linear regression problem.

    d: max degree polynomial
    
    '''
    
    def __init__(self, d=1):
        '''Initializes model'''
        super().__init__()
        self.d = d 
        
    def fit(self, ts):
        '''Fits model'''
        self.reset_to_default() #model clearing
        x = np.power(np.arange(len(ts)), np.arange(self.d + 1).reshape(-1, 1)).transpose()
        reg = LinearRegression().fit(x, ts)
        self.coef = reg.coef
        self.true_values   = reg.true_values
        self.fitted_values = reg.fitted_values
        self.residuals     = reg.residuals
        return self

    def predict(self, h=1):
        '''Gives forecast for h times'''
        x = np.power(np.arange(len(self.true_values), len(self.true_values) + h), np.arange(self.d + 1).reshape(-1, 1)).transpose()
        return x.dot(self.coef)

    def reset_to_default(self):
        self.coef = np.zeros(2)
        self.true_values   = np.zeros(2)
        self.fitted_values = np.zeros(2)
        self.residuals     = np.zeros(2)
        return PolynomialRegression(self.d)
