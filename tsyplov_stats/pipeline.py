
import numpy as np

class Pipeline():
    
    def __init__(self, models):
        self.models = models
        self.true_values   = np.zeros(2)
        self.fitted_values = np.zeros(2)
        self.residuals     = np.zeros(2)

    def fit(self, ts):
        self.reset_to_default() #model clear
        self.true_values = ts.copy()
        self.residuals   = ts.copy()
        
        for model in self.models:
            model.fit(self.residuals)
            self.residuals = model.residuals

        self.true_values   = self.true_values[-len(self.residuals):]
        self.fitted_values = self.true_values - self.residuals
        
        return self

    def predict(self, h=1):
        fc = np.zeros(h)
        for model in self.models:
            fc += model.predict(h)
        return fc

    def reset_to_default(self):
        self.true_values   = np.zeros(2)
        self.fitted_values = np.zeros(2)
        self.residuals     = np.zeros(2)
        return Pipeline(self.models)
