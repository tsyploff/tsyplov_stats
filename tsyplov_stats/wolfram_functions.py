
import numpy as np
import matplotlib.pyplot as plt

def partition(lst, n, d):
    '''Partitions list into sublists of length n with offset d.

    >>> partition(np.arange(6), 3, 1)
    array([[0, 1, 2],
           [1, 2, 3],
           [2, 3, 4], 
           [3, 4, 5]])

    '''
    indexes_1 = np.arange(n)
    indexes_2 = np.arange(0, len(lst) - n + 1, d)
    return lst[indexes_1[np.newaxis] + indexes_2[:, np.newaxis]]

def fold(f, x, lst):
    '''Fold just like fold in Wolfram

    >>> fold(lambda x, y: 'f[{}, {}]'.format(x, y), 0, np.arange(1, 5))
    'f[f[f[f[0, 1], 2], 3], 4]'

    '''
    if len(lst) != 0:
        return fold(f, f(x, lst[0]), lst[1:])
    else:
        return x

def fold_list(f, x, lst):
    '''FoldList just like FoldList in Wolfram

    >>> fold_list(lambda x, y: 'f[{}, {}]'.format(x, y), 0, np.arange(1, 3))
    [0, 'f[0, 1]', 'f[f[0, 1], 2]']

    '''
    if len(lst) != 0:
        return [x] + fold_list(f, f(x, lst[0]), lst[1:])
    else:
        return [x]

def nest(f, x, k):
    '''Nest just like nest in Wolfram

    >>> import tsyplov_optimize
    >>> tsyplov_optimize.nest(lambda x: 'f[{}]'.format(x), 'x', 5)
    'f[f[f[f[f[x]]]]]'

    '''
    if k != 0:
        return nest(f, f(x), k - 1)
    else:
        return x

def nest_list(f, x, k):
    '''NestList just like NestList in Wolfram

    >>> import tsyplov_optimize
    >>> tsyplov_optimize.nest_list(lambda x: 'f[{}]'.format(x), 'x', 5)
    ['x', 'f[x]', 'f[f[x]]', 'f[f[f[x]]]', 'f[f[f[f[x]]]]', 'f[f[f[f[f[x]]]]]']

    '''
    if k != 0:
        return [x] + nest_list(f, f(x), k - 1)
    else:
        return [x]

def nest_while(f, x, cond, max_iter=1000):
    '''NestWhile just like NestWhile in Wolfram

    >>> import tsyplov_optimize
    >>> tsyplov_optimize/nest_while(lambda x: x//2, 123456, lambda x: x%2 == 0)
    1929

    '''
    if cond(x) and max_iter > 0:
        return nest_while(f, f(x), cond, max_iter - 1)
    else:
        return x

def nest_while_list(f, expr, test, m=1, max_iter=1024):
    '''NestWhileList just like NestWhileList in Wolfram

    - generates a list of the results of applying f repeatedly, 
    starting with expr, and continuing until applying test to the result no longer yields True. 

    - supplies the most recent m results as arguments for test at each step. 

    - applies f at most max_iter times and at least one time.

    >>> import tsyplov_optimize
    >>> tsyplov_optimize.nest_while_list(lambda x: x//2, 123456, lambda x: x[0]%2 == 0)
    array([123456,  61728,  30864,  15432,   7716,   3858,   1929])

    '''
    lst = [expr, f(expr)]
    itr = max_iter
    while test(lst[-m:]) and itr > 0:
        lst.append(f(lst[-1]))
        itr -= 1
    return np.array(lst)

def take_drop(a, n):
    '''gives the tuple of a[:n] and a[n:]'''
    return a[:n].copy(), a[n:].copy()

def list_line_plot(data, plot_legends=0, image_size=(10, 6)):
    '''Build plot as ListLinePlot in Wolfram'''
    fig, ax = plt.subplots(figsize=image_size)

    if plot_legends == 0:
        plot_legends = [str(i) for i in range(len(data))]
    
    i = 0
    for x, y in data:
        ax.plot(x, y, label=plot_legends[i])
        i += 1

    plt.legend()
    plt.show()