
import numpy as np
from tsyplov_stats.wolfram_functions import *

class LinearRegression():
    '''Ordinary least squares Linear Regression.
    
    Attributes
    
    coef: estimated coefficients for the linear regression problem.

    Example

    >>> import numpy as np
    >>> import tsyplov_stats
    >>> x = np.array([[1, 0], [1, 1], [1, 3], [1, 5]])
    >>> y = np.array([1, 0, 2, 4])
    >>> reg = tsyplov_stats.linearregression_model().fit(x, y)
    >>> reg.coef
    array([0.18644068, 0.69491525])
    >>> reg.score()
    0.814043583535109
    >>> reg.residuals
    array([ 0.81355932, -0.88135593, -0.27118644,  0.33898305])
    
    '''
    
    def __init__(self):
        '''Initializes model'''
        self.coef = np.zeros(2)
        self.true_values   = np.zeros(2)
        self.fitted_values = np.zeros(2)
        self.residuals     = np.zeros(2)
        
    def fit(self, x, y):
        '''Fits model'''
        self.reset_to_default() #model clear
        x_tr = x.transpose()
        self.coef = np.linalg.inv(x_tr.dot(x)).dot(x_tr).dot(y)
        self.true_values   = y.copy()
        self.fitted_values = self.predict(x)
        self.residuals     = self.true_values - self.fitted_values
        return self

    def predict(self, x):
        '''Gives forecast for x using model self'''
        return x.dot(self.coef)
        
    def score(self, x, y):
        '''Gives the coefficien of determination'''
        return 1 - np.var(self.residuals)/np.var(y)
    
    def mae(self, x, y):
        '''Gives the mean absolute error'''
        return np.mean(np.abs(self.residuals))
    
    def mse(self, x, y):
        '''Gives the mean square error'''
        return np.mean((self.residuals)**2)
    
    def rmse(self, x, y):
        '''Gives the root from mean square error'''
        return np.sqrt(self.mse(x, y))

    def reset_to_default(self):
        self.coef = np.zeros(2)
        self.true_values   = np.zeros(2)
        self.fitted_values = np.zeros(2)
        self.residuals     = np.zeros(2)
        return LinearRegression()
