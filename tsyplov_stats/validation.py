
import numpy as np
from tsyplov_stats.wolfram_functions import *


def mean_square_error(x, y):
	return np.mean((x - y)**2)

def root_mean_square_error(x, y):
	return np.sqrt(np.mean((x - y)**2))

def mean_absolute_error(x, y):
	return np.mean(np.abs(x - y))

def time_series_split(ts, n, k):
    '''Gives the list of k pairs (train, test), where 
    len(test) == n; len(train) >= len(test)'''
    return [take_drop(ts[:n + i], i) for i in range(len(ts) - n, len(ts) - (k + 1)*n, -n)][::-1]
     
def cross_val_score(model, metric, ts_split):
    '''Gives the self.metric score on each test in time seiries split
    using self.model fitted on train
    '''
    h = len(ts_split[0][1])
    result = list()
    model_ = model.reset_to_default()

    for train, test in ts_split:
        forecast = model_.fit(train).predict(h)
        result.append(metric(forecast, test))

    return np.array(result)
