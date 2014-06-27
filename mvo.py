"""NOTE: This code is largely a replication of Ondrej Martinsky's fantastic Python implementation of
mean-variance optimization techniques. You can find the source of this code at:

https://code.google.com/p/quantandfinancial/source/browse/trunk/example_black_litterman.py

I would also recommend his blog, Quant and Financial, which contains easy and intuitive explanations
of various financial modeling processes (although unfortunately there are only a few entries):

http://www.quantandfinancial.com/

I have asked for permission to use this code but have not heard back yet. I am not trying to
pass off Ondrej's work as my own.
"""

import numpy as np
import scipy.optimize as spo


def port_mean(W, R):
    """Calculates portfolio mean"""
    return sum(W*R)


def port_var(W, C):
    """Calculates portfolio variance"""
    return np.dot(np.dot(W, C), W)


def port_mean_var(W, R, C):
    """Single function to return portfolio mean and variance"""
    return port_mean(W, R), port_var(W, C)


def get_W_R_C(hist_prices, mkt_caps): #Think of a better name

    prices = np.matrix(hist_prices)

    #Create r x 1 matrix of asset weights
    W = np.array(mkt_caps)/sum(mkt_caps)

    #Find returns over historical period
    rows, cols = prices.shape
    returns = np.empty([rows, cols-1])
    for r in xrange(rows):
        for c in xrange(cols-1):
            returns[r, c] = (prices[r, c + 1]/prices[r, c]) - 1

    #Create r x 1 expected returns as the average of historical returns
    R = np.array([0]*rows, dtype=float)
    for r in xrange(rows):
        R[r] = np.mean(returns[r])

    #Create r x r covariance matrix
    C = np.cov(returns)

    #Annualize returns and covariances
    R = (1 + R)**250 - 1
    C *= 250
    return W, R, C


def fitness(W, R, C, r, fit_func):
    """Calculates the fitness of the given portfolio, based on fit_func.

fit_func should be defined as:

fit_func=lambda mean, var, r: (function of mean, var, r)

where mean and var are calculated in the function. Two examples:

fit_func=lambda mean, var, r: var + (100*abs(mean-r)) - returns the variance of the portfolio plus a penalty
fit_func=lambda mean, var, r: (mean - r)/(var**.5) - returns the sharpe ratio of the portfolio
"""
    mean, var = port_mean_var(W, R, C)
    return fit_func(mean, var, r)


def solve_tangency(R, C, rf, fit_func=lambda mean, var, rf: mean-rf/(var**.5)):
    """Calculates the tangency portfolio given a set of expected returns (R), covariances (C),
risk-free rate (rf), and a fitness function (fit_func) which defaults to the Sharpe ratio.

Returns the weights of the tangency portfolio.
"""
    n = len(R)
    #Begin with equal weights
    W = np.ones([n])/n

    # Replace expected returns that are less than rf with rf + a super small value
    # since if it's less than rf, the sharpe ratio is negative, which ruins minimization
    np.place(R, R<rf, rf+0.00001)

    # Set boundaries on weights - no shorting or leverage allowed. Can probably incorporate
    # this functionality easily, though.
    bounds = [(0., 1.) for i in xrange(n)]

    # Set constraints as defined in SciPy's documentation for minimize. 'fun' forces the weights to sum to 1
    constraints = ({'type':'eq',
                    'fun': lambda W: sum(W)-1.})

    # Minimize fitness by changing W with (R, C, rf, fit_func) as given using the SLQSP method
    # based on the above defined constraints and bounds
    tangency = spo.minimize(fun=fitness,
                            x0=W,
                            args=(R, C, rf, fit_func),
                            method='SLSQP',
                            constraints=constraints,
                            bounds=bounds)
    if not tangency.success:
        raise BaseException(tangency.message)
    return tangency.x


def solve_frontier(R, C, num_points=20, fit_func=lambda mean, var, r: var + (50*abs(mean-r))):
    """Calculates the frontier of efficient portfolios given a set of expected returns (R),
covariances (C), number of desired points(num_points), and a fitness function (fit_func)
which defaults to (var + (50*abs(mean-r)))

Returns the frontier means, frontier variances, and frontier weights.
"""
    n = len(R)
    front_mean, front_var, front_weights = [], [], []

    # Loop through range of target returns
    for r in np.linspace(min(R), max(R), num=num_points):
        #Begin with equal weights
        W = np.ones([n])/n

        # Set boundaries on weights - no shorting or leverage allowed. Can probably incorporate
        # this functionality easily, though.
        bounds = [(0., 1.) for i in xrange(n)]

        # Set constraints as defined in SciPy's documentation for minimize. 'fun' forces the weights to sum to 1
        constraints = ({'type':'eq',
                        'fun': lambda W: sum(W)-1.})

        # Minimize fitness by changing W with (R, C, r, fit_func) as given using the SLQSP method
        # based on the above defined constraints and bounds
        efficient = spo.minimize(fun=fitness,
                                 x0=W,
                                 args=(R, C, r, fit_func),
                                 method='SLSQP',
                                 constraints=constraints,
                                 bounds=bounds)
        if not efficient.success:
            raise BaseException(efficient.message)

        # Add data points to the frontier lists
        front_mean.append(r)
        front_var.append(port_var(efficient.x, C))
        front_weights.append(efficient.x)
    return front_mean, front_var, front_weights


if __name__ == '__main__':
    import yahoo_data as yh
    from datetime import date
    symbols = ['VBR','AGG','VCLT','VEU']
    prices = []
    caps = [1.0,1.0,1.0,1.0] # Not used in optimization
    for symbol in symbols:
        print 'Getting returns for %s...' % symbol
        hist_prices = yh.get_historical_data(symbol, date(2011, 1, 1), date(2014, 6, 1), 'd')
        prices.append(hist_prices['Close'])
        if len(hist_prices['Close']) != len(prices[0]):
            raise Exception('Missing data values for at least one symbol')
    test_W, test_R, test_C = get_W_R_C(prices, caps)
    print test_R
    print test_C
    print 'Optimizing...'
    test_means, test_vars, test_weights = solve_frontier(test_R, test_C)
    print 'Finished!'
    for i in xrange(len(test_means)):
        print test_means[i], test_vars[i]**.5, [round(num, 2) for num in test_weights[i]]