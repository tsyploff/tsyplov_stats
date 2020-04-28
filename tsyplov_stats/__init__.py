
__all__ = ['arma_model', 'autoregression_model', 'fejer_sums', 'linearregression_model', 'polynomialregression_model', 'wolfram_functions']

from tsyplov_stats.wolfram_functions import *

def invboxcox(ts, lmbda):
    if lmbda != 0:
        return np.exp(np.log(lmbda*ts + 1) / lmbda)
    else:
        return np.exp(ts)

def cross_validation_split(ts, n):
	'''Gives the list of tuples (train, test), where len(test) is always equal n'''
	result = []
	for i in range(len(ts), len(ts) - (n - 1)*(len(ts)//n), -n):
		result.append(take_drop(ts[:i], n))
	return result[::-1]