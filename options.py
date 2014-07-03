__author__ = 'aphillips'

from pyfi import *
import copy
import random
import math

#Various functions used to assist in options pricing


def _check_option_error(S, sigma, K, T, n=1, r=0.0, q=0.0, ex=None, call=True, geom=True):
    if not isinstance(S, float) and not isinstance(S, int):
        raise Exception('Invalid type for S. Must be a float or int')
    if not isinstance(sigma, float):
        raise Exception('Invalid type for sigma. Must be a float')
    if not isinstance(K, int) and not isinstance(K, float):
        raise Exception('Invalid type for K. Must be an integer or a float')
    if K <= 0:
        raise Exception('Invalid value for K. Must be greater than 0')
    if not isinstance(T, int) and not isinstance(T, float):
        raise Exception('Invalid type for T. Must be an integer or a float')
    if T <= 0:
        raise Exception('Invalid value for T. Must be greater than 0')
    if not isinstance(n, int):
        raise Exception('Invalid type for n. Must be an integer')
    if n <= 0:
        raise Exception('Invalid value for n. Must be greater than 0')
    if not isinstance(r, float):
        raise Exception('Invalid type for r. Must be a float')
    if not isinstance(q, float):
        raise Exception('Invalid type for q. Must be a float')
    if not isinstance(ex, list) and ex is not None:
        raise Exception('Invalid type for ex. Must be a list')
    if not isinstance(call, bool):
        raise Exception('Invalid type for call. Must be a boolean')
    if not isinstance(geom, bool):
        raise Exception('Invalid type for geom. Must be a boolean')
    if ((r-q)*float(T)/n) >= sigma*((float(T)/n)**.5):
        raise Exception('Invalid risk-free rate, need ((r-q)*(T/n)) < sigma*((T/n)**.5)')


def calc_bin_greeks(stock_tree, price_tree, dt):
    delta = (price_tree[1][1] - price_tree[1][0])/(stock_tree[1][1] - stock_tree[1][0])
    if len(stock_tree) >= 2 and len(price_tree) >= 2:
        n1 = (price_tree[2][2] - price_tree[2][1])/(stock_tree[2][2] - stock_tree[2][1])
        n2 = (price_tree[2][1] - price_tree[2][0])/(stock_tree[2][1] - stock_tree[2][0])
        gamma = (n1 - n2)/(stock_tree[2][2] - stock_tree[2][0])
        theta = (price_tree[0][0] - price_tree[2][1])/(2*dt)
    else:
        gamma = 0
        theta = 0
    return (delta, gamma, theta)


def calc_bs_greeks():
    pass


def phi(x):
    """phi(x)

Finds the CDF of a Normal(0, 1) distribution evaluated at x.
"""
    return (1.0 + math.erf(x/math.sqrt(2.0)))/2.0


#Vanilla Options - Binomial Method---------------------------------------------------

###CRR-------------------------------------------------------------------------------


class EuropeanCRR(object):
    """EuropeanCRR(S, sigma, K, T, n, r, q=0, call=True)

Creates a European option. European options give the option buyer the right
to buy (call) or sell (put) a stock at a future date for a specified strike
price K. The payout is max(ST-K, 0) for a call and max(K-ST, 0) for a put,
where ST is the final sock price.

This class takes the following arguments:

S: underlying stock price
sigma: underlying stock annual volatility
K: strike price of the option
T: length of the life of the option (in annual terms)
n: number of periods in the tree (time step between periods = T/n)
r: annual risk-free rate, compounded continuously
q: annual continuous dividend rate, defaults to 0
call: boolean indicating if the option is a call or put option, defaults to True

If any of the parameters change, run [object name].calc_price() to generate the new price.
"""
    def __init__(self, S, sigma, K, T, n, r, q=0.0, call=True):
        _check_option_error(S, sigma, K, T, n, r, q=q, call=call)
        self.S = S
        self.sigma = sigma
        self.K = K
        self.T = T
        self.n = n
        self.dt = float(T)/n
        self.r = r
        self.q = q
        self.call = call
        self.delta = 0
        self.gamma = 0
        self.theta = 0
        self.price_tree = []
        self.stock_tree = []
        self.price = self.calc_price()

    def make_tree(self):
        stock_tree = []
        u = math.exp(self.sigma*(self.dt**.5))
        d = 1/u
        for t in xrange(self.n+1):
            prices = []
            for num_u in xrange(t+1):
                prices.append(self.S*(u**num_u)*(d**(t-num_u)))
            stock_tree.append(prices)
        self.stock_tree = stock_tree
        return stock_tree

    def calc_price(self):
        self.stock_tree = self.make_tree()
        price_tree = copy.deepcopy(self.stock_tree)
        u = math.exp(self.sigma*(self.dt**.5))
        d = 1/u
        pu = (math.exp((self.r - self.q)*self.dt) - d)/(u - d)
        pd = 1 - pu
        for i, price in enumerate(price_tree[-1]):
            if self.call:
                price_tree[-1][i] = max(price - self.K, 0)
            else:
                price_tree[-1][i] = max(self.K - price, 0)
        for t in xrange(self.n-1, -1, -1):
            for i in xrange(t+1):
                price_tree[t][i] = (pd*price_tree[t+1][i])+(pu*price_tree[t+1][i+1])
                price_tree[t][i] *= math.exp((self.r-self.q)*self.dt*-1)
        self.price = price_tree[0][0]
        self.price_tree = price_tree
        self.delta, self.gamma, self.theta = calc_bin_greeks(self.stock_tree, self.price_tree, self.dt)
        return self.price

    def print_tree(self, stock=False):
        if stock:
            for line in self.stock_tree:
                print '\t'.join([str(round(price, 2)) for price in line])
        else:
            for line in self.price_tree:
                print '\t'.join([str(round(price, 2)) for price in line])


class AmericanCRR(object):
    """AmericanCRR(S, sigma, K, T, n, r, q=0, call=True)

Creates an American option. An American option gives the option buyer
the right to buy (call) or sell (put) a stock at a specified strike price.
The buyer may exercise this option before expiry at any time. Whenever the
option is exercised, the payout is max(St - K, 0) for calls and
max(K - St, 0) for puts, where St is the stock price at time t.

This class takes the following arguments:

S: underlying stock price
sigma: underlying stock annual volatility
K: strike price of the option
T: length of the life of the option (in annual terms)
n: number of periods in the tree (time step between periods = T/n)
r: annual risk-free rate, compounded continuously
q: annual continuous dividend rate, defaults to 0
call: boolean indicating if the option is a call or put option, defaults to True

If any of the parameters change, run [object name].calc_price() to generate the new price.
"""
    def __init__(self, S, sigma, K, T, n, r, q=0.0, call=True):
        _check_option_error(S, sigma, K, T, n, r, q=q, call=call)
        self.S = S
        self.sigma = sigma
        self.K = K
        self.T = T
        self.n = n
        self.dt = float(T)/n
        self.r = r
        self.q = q
        self.call = call
        self.delta = 0
        self.gamma = 0
        self.theta = 0
        self.price_tree = []
        self.stock_tree = []
        self.price = self.calc_price()

    def make_tree(self):
        stock_tree = []
        u = math.exp(self.sigma*(self.dt**.5))
        d = 1/u
        for t in xrange(self.n+1):
            prices = []
            for num_u in xrange(t+1):
                prices.append(self.S*(u**num_u)*(d**(t-num_u)))
            stock_tree.append(prices)
        self.stock_tree = stock_tree
        return stock_tree

    def calc_price(self):
        self.stock_tree = self.make_tree()
        price_tree = copy.deepcopy(self.stock_tree)
        u = math.exp(self.sigma*(self.dt**.5))
        d = 1/u
        pu = (math.exp((self.r - self.q)*self.dt) - d)/(u - d)
        pd = 1 - pu
        for i, price in enumerate(price_tree[-1]):
            if self.call:
                price_tree[-1][i] = max(price - self.K, 0)
            else:
                price_tree[-1][i] = max(self.K - price, 0)
        for t in xrange(self.n-1, -1, -1):
            for i in xrange(t+1):
                option_value = (pd*price_tree[t+1][i])+(pu*price_tree[t+1][i+1])
                if self.call:
                    exercise_value = max(price_tree[t][i] - self.K, 0)
                else:
                    exercise_value = max(self.K - price_tree[t][i], 0)
                price_tree[t][i] = max(option_value, exercise_value)
                price_tree[t][i] *= math.exp((self.r-self.q)*self.dt*-1)
        self.price = price_tree[0][0]
        self.price_tree = price_tree
        self.delta, self.gamma, self.theta = calc_bin_greeks(self.stock_tree, self.price_tree, self.dt)
        return self.price

    def print_tree(self, stock=False):
        if stock:
            for line in self.stock_tree:
                print '\t'.join([str(round(price, 2)) for price in line])
        else:
            for line in self.price_tree:
                print '\t'.join([str(round(price, 2)) for price in line])


class BermudanCRR(object):
    """BermudanCRR(S, sigma, K, T, n, r, q=0, ex=[], call=True)

Creates an Bermudan option. A Bermudan option gives the option buyer
the right to buy (call) or sell (put) a stock at a specified strike price.
The buyer may exercise this option at certain points in time. Whenever the
option is exercised, the payout is max(St - K, 0) for calls and
max(K - St, 0) for puts, where St is the stock price at time t.

This class takes the following arguments:

S: underlying stock price
sigma: underlying stock annual volatility
K: strike price of the option
T: length of the life of the option (in annual terms)
n: number of periods in the tree (time step between periods = T/n)
r: annual risk-free rate, compounded continuously
q: annual continuous dividend rate, defaults to 0
ex: a list containing the periods at which exercising is allowed.
call: boolean indicating if the option is a call or put option, defaults to True

If any of the parameters change, run [object name].calc_price() to generate the new price.
"""
    def __init__(self, S, sigma, K, T, n, r, q=0.0, ex=None, call=True):
        _check_option_error(S, sigma, K, T, n, r, q=q, ex=ex, call=call)
        self.S = S
        self.sigma = sigma
        self.K = K
        self.T = T
        self.n = n
        self.dt = float(T)/n
        self.r = r
        self.q = q
        self.ex = ex
        self.call = call
        self.delta = 0
        self.gamma = 0
        self.theta = 0
        self.price_tree = []
        self.stock_tree = []
        self.price = self.calc_price()

    def make_tree(self):
        stock_tree = []
        u = math.exp(self.sigma*(self.dt**.5))
        d = 1/u
        for t in xrange(self.n+1):
            prices = []
            for num_u in xrange(t+1):
                prices.append(self.S*(u**num_u)*(d**(t-num_u)))
            stock_tree.append(prices)
        self.stock_tree = stock_tree
        return stock_tree

    def calc_price(self):
        self.stock_tree = self.make_tree()
        price_tree = copy.deepcopy(self.stock_tree)
        u = math.exp(self.sigma*(self.dt**.5))
        d = 1/u
        pu = (math.exp((self.r - self.q)*self.dt) - d)/(u - d)
        pd = 1 - pu
        #Find exercise value at final nodes
        for i, price in enumerate(price_tree[-1]):
            if self.call:
                price_tree[-1][i] = max(price - self.K, 0)
            else:
                price_tree[-1][i] = max(self.K - price, 0)
        #Move backwards through the tree to determine option values
        for t in xrange(self.n-1, -1, -1):
            for i in xrange(t+1):
                option_value = (pd*price_tree[t+1][i])+(pu*price_tree[t+1][i+1])
                #Use appropriate payout function
                if self.call:
                    exercise_value = max(price_tree[t][i] - self.K, 0)
                else:
                    exercise_value = max(self.K - price_tree[t][i], 0)
                #Check if holder can exercise
                if t in self.ex:
                    price_tree[t][i] = max(option_value, exercise_value)
                else:
                    price_tree[t][i] = option_value
                #Discount price
                price_tree[t][i] *= math.exp((self.r-self.q)*self.dt*-1)
        self.price = price_tree[0][0]
        self.price_tree = price_tree
        self.delta, self.gamma, self.theta = calc_bin_greeks(self.stock_tree, self.price_tree, self.dt)
        return self.price

    def print_tree(self, stock=False):
        if stock:
            for line in self.stock_tree:
                print '\t'.join([str(round(price, 2)) for price in line])
        else:
            for line in self.price_tree:
                print '\t'.join([str(round(price, 2)) for price in line])


###Jarrow-Rudd-----------------------------------------------------------------------

class EuropeanJR(object):
    """EuropeanJR(S, sigma, K, T, n, r, q=0, call=True)

Creates a European option. European options give the option buyer the right
to buy (call) or sell (put) a stock at a future date for a specified strike
price K. The payout is max(ST-K, 0) for a call and max(K-ST, 0) for a put,
where ST is the final sock price.

This class takes the following arguments:

S: underlying stock price
sigma: underlying stock annual volatility
K: strike price of the option
T: length of the life of the option (in annual terms)
n: number of periods in the tree (time step between periods = T/n)
r: annual risk-free rate, compounded continuously
q: annual continuous dividend rate, defaults to 0
call: boolean indicating if the option is a call or put option, defaults to True

If any of the parameters change, run [object name].calc_price() to generate the new price.
"""
    def __init__(self, S, sigma, K, T, n, r, q=0.0, call=True):
        _check_option_error(S, sigma, K, T, n, r, q=q, call=call)
        self.S = S
        self.sigma = sigma
        self.K = K
        self.T = T
        self.n = n
        self.dt = float(T)/n
        self.r = r
        self.q = q
        self.call = call
        self.delta = 0
        self.gamma = 0
        self.theta = 0
        self.price_tree = []
        self.stock_tree = []
        self.price = self.calc_price()

    def make_tree(self):
        #Source for u and d: http://www.goddardconsulting.ca/option-pricing-binomial-alts.html
        stock_tree = []
        u = math.exp((self.r - self.q - (.5*(self.sigma**2)))*self.dt + self.sigma*(self.dt**.5))
        d = math.exp((self.r - self.q - (.5*(self.sigma**2)))*self.dt - self.sigma*(self.dt**.5))
        for t in xrange(self.n+1):
            prices = []
            for num_u in xrange(t+1):
                prices.append(self.S*(u**num_u)*(d**(t-num_u)))
            stock_tree.append(prices)
        self.stock_tree = stock_tree
        return stock_tree

    def calc_price(self):
        self.stock_tree = self.make_tree()
        price_tree = copy.deepcopy(self.stock_tree)
        pu = .5
        pd = .5
        #Find the exercise value at final nodes
        for i, price in enumerate(price_tree[-1]):
            if self.call:
                price_tree[-1][i] = max(price - self.K, 0)
            else:
                price_tree[-1][i] = max(self.K - price, 0)
        #Move backwards through the tree to determine option values
        for t in xrange(self.n-1, -1, -1):
            for i in xrange(t+1):
                price_tree[t][i] = (pd*price_tree[t+1][i])+(pu*price_tree[t+1][i+1])
                #Discount the option price
                price_tree[t][i] *= math.exp((self.r-self.q)*self.dt*-1)
        self.price = price_tree[0][0]
        self.price_tree = price_tree
        self.delta, self.gamma, self.theta = calc_bin_greeks(self.stock_tree, self.price_tree, self.dt)
        return self.price

    def print_tree(self, stock=False):
        if stock:
            for line in self.stock_tree:
                print '\t'.join([str(round(price, 2)) for price in line])
        else:
            for line in self.price_tree:
                print '\t'.join([str(round(price, 2)) for price in line])


class AmericanJR(object):
    """AmericanJR(S, sigma, K, T, n, r, q=0, call=True)

Creates an American option. An American option gives the option buyer
the right to buy (call) or sell (put) a stock at a specified strike price.
The buyer may exercise this option before expiry at any time. Whenever the
option is exercised, the payout is max(St - K, 0) for calls and
max(K - St, 0) for puts, where St is the stock price at time t.

This class takes the following arguments:

S: underlying stock price
sigma: underlying stock annual volatility
K: strike price of the option
T: length of the life of the option (in annual terms)
n: number of periods in the tree (time step between periods = T/n)
r: annual risk-free rate, compounded continuously
q: annual continuous dividend rate, defaults to 0
call: boolean indicating if the option is a call or put option, defaults to True

If any of the parameters change, run [object name].calc_price() to generate the new price.
"""
    def __init__(self, S, sigma, K, T, n, r, q=0.0, call=True):
        _check_option_error(S, sigma, K, T, n, r, q=q, call=call)
        self.S = S
        self.sigma = sigma
        self.K = K
        self.T = T
        self.n = n
        self.dt = float(T)/n
        self.r = r
        self.q = q
        self.call = call
        self.delta = 0
        self.gamma = 0
        self.theta = 0
        self.price_tree = []
        self.stock_tree = []
        self.price = self.calc_price()

    def make_tree(self):
        #Source for u and d: http://www.goddardconsulting.ca/option-pricing-binomial-alts.html
        stock_tree = []
        u = math.exp((self.r - self.q - (.5*(self.sigma**2)))*self.dt + self.sigma*(self.dt**.5))
        d = math.exp((self.r - self.q - (.5*(self.sigma**2)))*self.dt - self.sigma*(self.dt**.5))
        for t in xrange(self.n+1):
            prices = []
            for num_u in xrange(t+1):
                prices.append(self.S*(u**num_u)*(d**(t-num_u)))
            stock_tree.append(prices)
        self.stock_tree = stock_tree
        return stock_tree

    def calc_price(self):
        self.stock_tree = self.make_tree()
        price_tree = copy.deepcopy(self.stock_tree)
        pu = .5
        pd = .5
        #Find exercise value at final nodes
        for i, price in enumerate(price_tree[-1]):
            if self.call:
                price_tree[-1][i] = max(price - self.K, 0)
            else:
                price_tree[-1][i] = max(self.K - price, 0)
        #Move backwards through the tree to determine option values
        for t in xrange(self.n-1, -1, -1):
            for i in xrange(t+1):
                option_value = (pd*price_tree[t+1][i])+(pu*price_tree[t+1][i+1])
                #Use appropriate payout function
                if self.call:
                    exercise_value = max(price_tree[t][i] - self.K, 0)
                else:
                    exercise_value = max(self.K - price_tree[t][i], 0)
                price_tree[t][i] = max(option_value, exercise_value)
                #Discount the option price
                price_tree[t][i] *= math.exp((self.r-self.q)*self.dt*-1)
        self.price = price_tree[0][0]
        self.price_tree = price_tree
        self.delta, self.gamma, self.theta = calc_bin_greeks(self.stock_tree, self.price_tree, self.dt)
        return self.price

    def print_tree(self, stock=False):
        if stock:
            for line in self.stock_tree:
                print '\t'.join([str(round(price, 2)) for price in line])
        else:
            for line in self.price_tree:
                print '\t'.join([str(round(price, 2)) for price in line])


class BermudanJR(object):
    """BermudanJR(S, sigma, K, T, n, r, q=0, ex=[], call=True)

Creates an Bermudan option. A Bermudan option gives the option buyer
the right to buy (call) or sell (put) a stock at a specified strike price.
The buyer may exercise this option at certain points in time. Whenever the
option is exercised, the payout is max(St - K, 0) for calls and
max(K - St, 0) for puts, where St is the stock price at time t.

This class takes the following arguments:

S: underlying stock price
sigma: underlying stock annual volatility
K: strike price of the option
T: length of the life of the option (in annual terms)
n: number of periods in the tree (time step between periods = T/n)
r: annual risk-free rate, compounded continuously
q: annual continuous dividend rate, defaults to 0
ex: a list containing the periods at which exercising is allowed.
call: boolean indicating if the option is a call or put option, defaults to True

If any of the parameters change, run [object name].calc_price() to generate the new price.
"""
    def __init__(self, S, sigma, K, T, n, r, q=0.0, ex=None, call=True):
        _check_option_error(S, sigma, K, T, n, r, q=q, ex=ex, call=call)
        self.S = S
        self.sigma = sigma
        self.K = K
        self.T = T
        self.n = n
        self.dt = float(T)/n
        self.r = r
        self.q = q
        self.ex = ex
        self.call = call
        self.delta = 0
        self.gamma = 0
        self.theta = 0
        self.price_tree = []
        self.stock_tree = []
        self.price = self.calc_price()

    def make_tree(self):
        #Source for u and d: http://www.goddardconsulting.ca/option-pricing-binomial-alts.html
        stock_tree = []
        u = math.exp((self.r - self.q - (.5*(self.sigma**2)))*self.dt + self.sigma*(self.dt**.5))
        d = math.exp((self.r - self.q - (.5*(self.sigma**2)))*self.dt - self.sigma*(self.dt**.5))
        for t in xrange(self.n+1):
            prices = []
            for num_u in xrange(t+1):
                prices.append(self.S*(u**num_u)*(d**(t-num_u)))
            stock_tree.append(prices)
        self.stock_tree = stock_tree
        return stock_tree

    def calc_price(self):
        self.stock_tree = self.make_tree()
        price_tree = copy.deepcopy(self.stock_tree)
        pu = .5
        pd = .5
        #Find exercise value at final nodes
        for i, price in enumerate(price_tree[-1]):
            if self.call:
                price_tree[-1][i] = max(price - self.K, 0)
            else:
                price_tree[-1][i] = max(self.K - price, 0)
        #Move backwards through the tree to determine option values
        for t in xrange(self.n-1, -1, -1):
            for i in xrange(t+1):
                option_value = (pd*price_tree[t+1][i])+(pu*price_tree[t+1][i+1])
                #Use appropriate payout function
                if self.call:
                    exercise_value = max(price_tree[t][i] - self.K, 0)
                else:
                    exercise_value = max(self.K - price_tree[t][i], 0)
                #Check if holder can exercise
                if t in self.ex:
                    price_tree[t][i] = max(option_value, exercise_value)
                else:
                    price_tree[t][i] = option_value
                #Discount price
                price_tree[t][i] *= math.exp((self.r-self.q)*self.dt*-1)
        self.price = price_tree[0][0]
        self.price_tree = price_tree
        self.delta, self.gamma, self.theta = calc_bin_greeks(self.stock_tree, self.price_tree, self.dt)
        return self.price

    def print_tree(self, stock=False):
        if stock:
            for line in self.stock_tree:
                print '\t'.join([str(round(price, 2)) for price in line])
        else:
            for line in self.price_tree:
                print '\t'.join([str(round(price, 2)) for price in line])


###Tian------------------------------------------------------------------------------

class EuropeanTian(object):
    """EuropeanTian(S, sigma, K, T, n, r, q=0, call=True)

Creates a European option. European options give the option buyer the right
to buy (call) or sell (put) a stock at a future date for a specified strike
price K. The payout is max(ST-K, 0) for a call and max(K-ST, 0) for a put,
where ST is the final sock price.

This class takes the following arguments:

S: underlying stock price
sigma: underlying stock annual volatility
K: strike price of the option
T: length of the life of the option (in annual terms)
n: number of periods in the tree (time step between periods = T/n)
r: annual risk-free rate, compounded continuously
q: annual continuous dividend rate, defaults to 0
call: boolean indicating if the option is a call or put option, defaults to True

If any of the parameters change, run [object name].calc_price() to generate the new price.
"""
    def __init__(self, S, sigma, K, T, n, r, q=0.0, call=True):
        _check_option_error(S, sigma, K, T, n, r, q=q, call=call)
        self.S = S
        self.sigma = sigma
        self.K = K
        self.T = T
        self.n = n
        self.dt = float(T)/n
        self.r = r
        self.q = q
        self.call = call
        self.delta = 0
        self.gamma = 0
        self.theta = 0
        self.price_tree = []
        self.stock_tree = []
        self.price = self.calc_price()

    def make_tree(self):
        #Source for u and d: http://www.goddardconsulting.ca/option-pricing-binomial-alts.html
        stock_tree = []
        V = math.exp((self.sigma**2)*self.dt)
        u = .5 * math.exp((self.r - self.q) * self.dt) * V * (V + 1 + ((V**2) + (2*V) - 3)**.5)
        d = .5 * math.exp((self.r - self.q) * self.dt) * V * (V + 1 - ((V**2) + (2*V) - 3)**.5)
        for t in xrange(self.n+1):
            prices = []
            for num_u in xrange(t+1):
                prices.append(self.S*(u**num_u)*(d**(t-num_u)))
            stock_tree.append(prices)
        self.stock_tree = stock_tree
        return stock_tree

    def calc_price(self):
        self.stock_tree = self.make_tree()
        price_tree = copy.deepcopy(self.stock_tree)
        V = math.exp((self.sigma**2)*self.dt)
        u = .5 * math.exp((self.r - self.q) * self.dt) * V * (V + 1 + ((V**2) + (2*V) - 3)**.5)
        d = .5 * math.exp((self.r - self.q) * self.dt) * V * (V + 1 - ((V**2) + (2*V) - 3)**.5)
        pu = (math.exp((self.r - self.q)*self.dt) - d)/(u - d)
        pd = 1 - pu
        #Find the exercise value at final nodes
        for i, price in enumerate(price_tree[-1]):
            if self.call:
                price_tree[-1][i] = max(price - self.K, 0)
            else:
                price_tree[-1][i] = max(self.K - price, 0)
        #Move backwards through the tree to determine option values
        for t in xrange(self.n-1, -1, -1):
            for i in xrange(t+1):
                price_tree[t][i] = (pd*price_tree[t+1][i])+(pu*price_tree[t+1][i+1])
                #Discount the option price
                price_tree[t][i] *= math.exp((self.r-self.q)*self.dt*-1)
        self.price = price_tree[0][0]
        self.price_tree = price_tree
        self.delta, self.gamma, self.theta = calc_bin_greeks(self.stock_tree, self.price_tree, self.dt)
        return self.price

    def print_tree(self, stock=False):
        if stock:
            for line in self.stock_tree:
                print '\t'.join([str(round(price, 2)) for price in line])
        else:
            for line in self.price_tree:
                print '\t'.join([str(round(price, 2)) for price in line])


class AmericanTian(object):
    """AmericanTian(S, sigma, K, T, n, r, q=0, call=True)

Creates an American option. An American option gives the option buyer
the right to buy (call) or sell (put) a stock at a specified strike price.
The buyer may exercise this option before expiry at any time. Whenever the
option is exercised, the payout is max(St - K, 0) for calls and
max(K - St, 0) for puts, where St is the stock price at time t.

This class takes the following arguments:

S: underlying stock price
sigma: underlying stock annual volatility
K: strike price of the option
T: length of the life of the option (in annual terms)
n: number of periods in the tree (time step between periods = T/n)
r: annual risk-free rate, compounded continuously
q: annual continuous dividend rate, defaults to 0
call: boolean indicating if the option is a call or put option, defaults to True

If any of the parameters change, run [object name].calc_price() to generate the new price.
"""
    def __init__(self, S, sigma, K, T, n, r, q=0.0, call=True):
        _check_option_error(S, sigma, K, T, n, r, q=q, call=call)
        self.S = S
        self.sigma = sigma
        self.K = K
        self.T = T
        self.n = n
        self.dt = float(T)/n
        self.r = r
        self.q = q
        self.call = call
        self.delta = 0
        self.gamma = 0
        self.theta = 0
        self.price_tree = []
        self.stock_tree = []
        self.price = self.calc_price()

    def make_tree(self):
        #Source for u and d: http://www.goddardconsulting.ca/option-pricing-binomial-alts.html
        stock_tree = []
        V = math.exp((self.sigma**2)*self.dt)
        u = .5 * math.exp((self.r - self.q) * self.dt) * V * (V + 1 + ((V**2) + (2*V) - 3)**.5)
        d = .5 * math.exp((self.r - self.q) * self.dt) * V * (V + 1 - ((V**2) + (2*V) - 3)**.5)
        for t in xrange(self.n+1):
            prices = []
            for num_u in xrange(t+1):
                prices.append(self.S*(u**num_u)*(d**(t-num_u)))
            stock_tree.append(prices)
        self.stock_tree = stock_tree
        return stock_tree

    def calc_price(self):
        self.stock_tree = self.make_tree()
        price_tree = copy.deepcopy(self.stock_tree)
        V = math.exp((self.sigma**2)*self.dt)
        u = .5 * math.exp((self.r - self.q) * self.dt) * V * (V + 1 + ((V**2) + (2*V) - 3)**.5)
        d = .5 * math.exp((self.r - self.q) * self.dt) * V * (V + 1 - ((V**2) + (2*V) - 3)**.5)
        pu = (math.exp((self.r - self.q)*self.dt) - d)/(u - d)
        pd = 1 - pu
        #Find exercise value at final nodes
        for i, price in enumerate(price_tree[-1]):
            if self.call:
                price_tree[-1][i] = max(price - self.K, 0)
            else:
                price_tree[-1][i] = max(self.K - price, 0)
        #Move backwards through the tree to determine option values
        for t in xrange(self.n-1, -1, -1):
            for i in xrange(t+1):
                option_value = (pd*price_tree[t+1][i])+(pu*price_tree[t+1][i+1])
                #Use appropriate payout function
                if self.call:
                    exercise_value = max(price_tree[t][i] - self.K, 0)
                else:
                    exercise_value = max(self.K - price_tree[t][i], 0)
                price_tree[t][i] = max(option_value, exercise_value)
                #Discount the option price
                price_tree[t][i] *= math.exp((self.r-self.q)*self.dt*-1)
        self.price = price_tree[0][0]
        self.price_tree = price_tree
        self.delta, self.gamma, self.theta = calc_bin_greeks(self.stock_tree, self.price_tree, self.dt)
        return self.price

    def print_tree(self, stock=False):
        if stock:
            for line in self.stock_tree:
                print '\t'.join([str(round(price, 2)) for price in line])
        else:
            for line in self.price_tree:
                print '\t'.join([str(round(price, 2)) for price in line])


class BermudanTian(object):
    """BermudanTian(S, sigma, K, T, n, r, q=0, ex=[], call=True)

Creates an Bermudan option. A Bermudan option gives the option buyer
the right to buy (call) or sell (put) a stock at a specified strike price.
The buyer may exercise this option at certain points in time. Whenever the
option is exercised, the payout is max(St - K, 0) for calls and
max(K - St, 0) for puts, where St is the stock price at time t.

This class takes the following arguments:

S: underlying stock price
sigma: underlying stock annual volatility
K: strike price of the option
T: length of the life of the option (in annual terms)
n: number of periods in the tree (time step between periods = T/n)
r: annual risk-free rate, compounded continuously
q: annual continuous dividend rate, defaults to 0
ex: a list containing the periods at which exercising is allowed.
call: boolean indicating if the option is a call or put option, defaults to True

If any of the parameters change, run [object name].calc_price() to generate the new price.
"""
    def __init__(self, S, sigma, K, T, n, r, q=0.0, ex=None, call=True):
        _check_option_error(S, sigma, K, T, n, r, q=q, ex=ex, call=call)
        self.S = S
        self.sigma = sigma
        self.K = K
        self.T = T
        self.n = n
        self.dt = float(T)/n
        self.r = r
        self.q = q
        self.ex = ex
        self.call = call
        self.delta = 0
        self.gamma = 0
        self.theta = 0
        self.price_tree = []
        self.stock_tree = []
        self.price = self.calc_price()

    def make_tree(self):
        #Source for u and d: http://www.goddardconsulting.ca/option-pricing-binomial-alts.html
        stock_tree = []
        V = math.exp((self.sigma**2)*self.dt)
        u = .5 * math.exp((self.r - self.q) * self.dt) * V * (V + 1 + ((V**2) + (2*V) - 3)**.5)
        d = .5 * math.exp((self.r - self.q) * self.dt) * V * (V + 1 - ((V**2) + (2*V) - 3)**.5)
        for t in xrange(self.n+1):
            prices = []
            for num_u in xrange(t+1):
                prices.append(self.S*(u**num_u)*(d**(t-num_u)))
            stock_tree.append(prices)
        self.stock_tree = stock_tree
        return stock_tree

    def calc_price(self):
        self.stock_tree = self.make_tree()
        price_tree = copy.deepcopy(self.stock_tree)
        V = math.exp((self.sigma**2)*self.dt)
        u = .5 * math.exp((self.r - self.q) * self.dt) * V * (V + 1 + ((V**2) + (2*V) - 3)**.5)
        d = .5 * math.exp((self.r - self.q) * self.dt) * V * (V + 1 - ((V**2) + (2*V) - 3)**.5)
        pu = (math.exp((self.r - self.q)*self.dt) - d)/(u - d)
        pd = 1 - pu
        #Find exercise value at final nodes
        for i, price in enumerate(price_tree[-1]):
            if self.call:
                price_tree[-1][i] = max(price - self.K, 0)
            else:
                price_tree[-1][i] = max(self.K - price, 0)
        #Move backwards through the tree to determine option values
        for t in xrange(self.n-1, -1, -1):
            for i in xrange(t+1):
                option_value = (pd*price_tree[t+1][i])+(pu*price_tree[t+1][i+1])
                #Use appropriate payout function
                if self.call:
                    exercise_value = max(price_tree[t][i] - self.K, 0)
                else:
                    exercise_value = max(self.K - price_tree[t][i], 0)
                #Check if holder can exercise
                if t in self.ex:
                    price_tree[t][i] = max(option_value, exercise_value)
                else:
                    price_tree[t][i] = option_value
                #Discount price
                price_tree[t][i] *= math.exp((self.r-self.q)*self.dt*-1)
        self.price = price_tree[0][0]
        self.price_tree = price_tree
        self.delta, self.gamma, self.theta = calc_bin_greeks(self.stock_tree, self.price_tree, self.dt)
        return self.price

    def print_tree(self, stock=False):
        if stock:
            for line in self.stock_tree:
                print '\t'.join([str(round(price, 2)) for price in line])
        else:
            for line in self.price_tree:
                print '\t'.join([str(round(price, 2)) for price in line])


#Binary Options - Binomial Method----------------------------------------------------

###CRR-------------------------------------------------------------------------------


class EuropeanBinaryCRR(object):
    """EuropeanBinaryCRR(S, sigma, K, T, n, r, q=0, call=True, cash=True)

Creates a European option. European options give the option buyer the right
to buy (call) or sell (put) a stock at a future date for a specified strike
price K. The payout is max(ST-K, 0) for a call and max(K-ST, 0) for a put,
where ST is the final sock price.

This class takes the following arguments:

S: underlying stock price
sigma: underlying stock annual volatility
K: strike price of the option
T: length of the life of the option (in annual terms)
n: number of periods in the tree (time step between periods = T/n)
r: annual risk-free rate, compounded continuously
q: annual continuous dividend rate, defaults to 0
call: boolean indicating if the option is a call or put option, defaults to True
cash: indicates if the option is cash-or-nothing or asset-or-nothing, defaults to True

If any of the parameters change, run [object name].calc_price() to generate the new price.
"""
    def __init__(self, S, sigma, K, T, n, r, q=0.0, call=True, cash=True):
        _check_option_error(S, sigma, K, T, n, r, q=q, call=call)
        self.S = S
        self.sigma = sigma
        self.K = K
        self.T = T
        self.n = n
        self.dt = float(T)/n
        self.r = r
        self.q = q
        self.call = call
        self.cash = cash
        self.delta = 0
        self.gamma = 0
        self.theta = 0
        self.price_tree = []
        self.stock_tree = []
        self.price = self.calc_price()

    def make_tree(self):
        stock_tree = []
        u = math.exp(self.sigma*(self.dt**.5))
        d = 1/u
        for t in xrange(self.n+1):
            prices = []
            for num_u in xrange(t+1):
                prices.append(self.S*(u**num_u)*(d**(t-num_u)))
            stock_tree.append(prices)
        self.stock_tree = stock_tree
        return stock_tree

    def calc_price(self):
        self.stock_tree = self.make_tree()
        price_tree = copy.deepcopy(self.stock_tree)
        u = math.exp(self.sigma*(self.dt**.5))
        d = 1/u
        pu = (math.exp((self.r - self.q)*self.dt) - d)/(u - d)
        pd = 1 - pu
        #Find exercise value at final nodes
        for i, price in enumerate(price_tree[-1]):
            if self.call and price - self.K > 0:
                #Evaluates to 1 if cash, asset price if not cash
                price_tree[-1][i] = (1*self.cash)+(self.stock_tree[-1][i]*(not self.cash))
            elif not self.call and self.K - price > 0:
                price_tree[-1][i] = (1*self.cash)+(self.stock_tree[-1][i]*(not self.cash))
            else:
                price_tree[-1][i] = 0
        #Move backwards through the tree to determine option values
        for t in xrange(self.n-1, -1, -1):
            for i in xrange(t+1):
                price_tree[t][i] = (pd*price_tree[t+1][i])+(pu*price_tree[t+1][i+1])
                price_tree[t][i] *= math.exp((self.r-self.q)*self.dt*-1)
        self.price = price_tree[0][0]
        self.price_tree = price_tree
        self.delta, self.gamma, self.theta = calc_bin_greeks(self.stock_tree, self.price_tree, self.dt)
        return self.price

    def print_tree(self, stock=False):
        if stock:
            for line in self.stock_tree:
                print '\t'.join([str(round(price, 2)) for price in line])
        else:
            for line in self.price_tree:
                print '\t'.join([str(round(price, 2)) for price in line])


class AmericanBinaryCRR(object):
    """AmericanBinaryCRR(S, sigma, K, T, n, r, q=0, call=True, cash=True)

Creates an American option. An American option gives the option buyer
the right to buy (call) or sell (put) a stock at a specified strike price.
The buyer may exercise this option before expiry at any time. Whenever the
option is exercised, the payout is max(St - K, 0) for calls and
max(K - St, 0) for puts, where St is the stock price at time t.

This class takes the following arguments:

S: underlying stock price
sigma: underlying stock annual volatility
K: strike price of the option
T: length of the life of the option (in annual terms)
n: number of periods in the tree (time step between periods = T/n)
r: annual risk-free rate, compounded continuously
q: annual continuous dividend rate, defaults to 0
call: boolean indicating if the option is a call or put option, defaults to True
cash: indicates if the option is cash-or-nothing or asset-or-nothing, defaults to True

If any of the parameters change, run [object name].calc_price() to generate the new price.
"""
    def __init__(self, S, sigma, K, T, n, r, q=0.0, call=True, cash=True):
        _check_option_error(S, sigma, K, T, n, r, q=q, call=call)
        self.S = S
        self.sigma = sigma
        self.K = K
        self.T = T
        self.n = n
        self.dt = float(T)/n
        self.r = r
        self.q = q
        self.call = call
        self.cash = cash
        self.delta = 0
        self.gamma = 0
        self.theta = 0
        self.price_tree = []
        self.stock_tree = []
        self.price = self.calc_price()

    def make_tree(self):
        stock_tree = []
        u = math.exp(self.sigma*(self.dt**.5))
        d = 1/u
        for t in xrange(self.n+1):
            prices = []
            for num_u in xrange(t+1):
                prices.append(self.S*(u**num_u)*(d**(t-num_u)))
            stock_tree.append(prices)
        self.stock_tree = stock_tree
        return stock_tree

    def calc_price(self):
        self.stock_tree = self.make_tree()
        price_tree = copy.deepcopy(self.stock_tree)
        u = math.exp(self.sigma*(self.dt**.5))
        d = 1/u
        pu = (math.exp((self.r - self.q)*self.dt) - d)/(u - d)
        pd = 1 - pu
        #Find exercise value at final nodes
        for i, price in enumerate(price_tree[-1]):
            if self.call and price - self.K > 0:
                price_tree[-1][i] = (1*self.cash)+(self.stock_tree[-1][i]*(not self.cash))
            elif not self.call and self.K - price > 0:
                price_tree[-1][i] = (1*self.cash)+(self.stock_tree[-1][i]*(not self.cash))
            else:
                price_tree[-1][i] = 0
        #Move backwards through the tree to determine option values
        for t in xrange(self.n-1, -1, -1):
            for i in xrange(t+1):
                option_value = (pd*price_tree[t+1][i])+(pu*price_tree[t+1][i+1])
                if self.call and price_tree[t][i] - self.K > 0:
                    exercise_value = (1*self.cash)+(self.stock_tree[t][i]*(not self.cash))
                elif not self.call and self.K - price_tree[t][i] > 0:
                    exercise_value = (1*self.cash)+(self.stock_tree[t][i]*(not self.cash))
                else:
                    exercise_value = 0
                price_tree[t][i] = max(option_value, exercise_value)
                price_tree[t][i] *= math.exp((self.r-self.q)*self.dt*-1)
        self.price = price_tree[0][0]
        self.price_tree = price_tree
        self.delta, self.gamma, self.theta = calc_bin_greeks(self.stock_tree, self.price_tree, self.dt)
        return self.price

    def print_tree(self, stock=False):
        if stock:
            for line in self.stock_tree:
                print '\t'.join([str(round(price, 2)) for price in line])
        else:
            for line in self.price_tree:
                print '\t'.join([str(round(price, 2)) for price in line])


class BermudanBinaryCRR(object):
    """BermudanBinaryCRR(S, sigma, K, T, n, r, q=0, ex=[], call=True, cash=True)

Creates an Bermudan option. A Bermudan option gives the option buyer
the right to buy (call) or sell (put) a stock at a specified strike price.
The buyer may exercise this option at certain points in time. Whenever the
option is exercised, the payout is max(St - K, 0) for calls and
max(K - St, 0) for puts, where St is the stock price at time t.

This class takes the following arguments:

S: underlying stock price
sigma: underlying stock annual volatility
K: strike price of the option
T: length of the life of the option (in annual terms)
n: number of periods in the tree (time step between periods = T/n)
r: annual risk-free rate, compounded continuously
q: annual continuous dividend rate, defaults to 0
ex: a list containing the periods at which exercising is allowed.
call: boolean indicating if the option is a call or put option, defaults to True
cash: indicates if the option is cash-or-nothing or asset-or-nothing, defaults to True

If any of the parameters change, run [object name].calc_price() to generate the new price.
"""
    def __init__(self, S, sigma, K, T, n, r, q=0.0, ex=None, call=True, cash=True):
        _check_option_error(S, sigma, K, T, n, r, q=q, ex=ex, call=call)
        self.S = S
        self.sigma = sigma
        self.K = K
        self.T = T
        self.n = n
        self.dt = float(T)/n
        self.r = r
        self.q = q
        self.ex = ex
        self.call = call
        self.cash = cash
        self.delta = 0
        self.gamma = 0
        self.theta = 0
        self.price_tree = []
        self.stock_tree = []
        self.price = self.calc_price()

    def make_tree(self):
        stock_tree = []
        u = math.exp(self.sigma*(self.dt**.5))
        d = 1/u
        for t in xrange(self.n+1):
            prices = []
            for num_u in xrange(t+1):
                prices.append(self.S*(u**num_u)*(d**(t-num_u)))
            stock_tree.append(prices)
        self.stock_tree = stock_tree
        return stock_tree

    def calc_price(self):
        self.stock_tree = self.make_tree()
        price_tree = copy.deepcopy(self.stock_tree)
        u = math.exp(self.sigma*(self.dt**.5))
        d = 1/u
        pu = (math.exp((self.r - self.q)*self.dt) - d)/(u - d)
        pd = 1 - pu
        #Find exercise value at final nodes
        for i, price in enumerate(price_tree[-1]):
            if self.call and price - self.K > 0:
                price_tree[-1][i] = (1*self.cash)+(self.stock_tree[-1][i]*(not self.cash))
            elif not self.call and self.K - price > 0:
                price_tree[-1][i] = (1*self.cash)+(self.stock_tree[-1][i]*(not self.cash))
            else:
                price_tree[-1][i] = 0
        #Move backwards through the tree to determine option values
        for t in xrange(self.n-1, -1, -1):
            for i in xrange(t+1):
                option_value = (pd*price_tree[t+1][i])+(pu*price_tree[t+1][i+1])
                #Use appropriate payout function
                if self.call and price_tree[t][i] - self.K > 0:
                    exercise_value = (1*self.cash)+(self.stock_tree[t][i]*(not self.cash))
                elif not self.call and self.K - price_tree[t][i] > 0:
                    exercise_value = (1*self.cash)+(self.stock_tree[t][i]*(not self.cash))
                else:
                    exercise_value = 0
                #Check if holder can exercise
                if t in self.ex:
                    price_tree[t][i] = max(option_value, exercise_value)
                else:
                    price_tree[t][i] = option_value
                #Discount price
                price_tree[t][i] *= math.exp((self.r-self.q)*self.dt*-1)
        self.price = price_tree[0][0]
        self.price_tree = price_tree
        self.delta, self.gamma, self.theta = calc_bin_greeks(self.stock_tree, self.price_tree, self.dt)
        return self.price

    def print_tree(self, stock=False):
        if stock:
            for line in self.stock_tree:
                print '\t'.join([str(round(price, 2)) for price in line])
        else:
            for line in self.price_tree:
                print '\t'.join([str(round(price, 2)) for price in line])


###Jarrow-Rudd-----------------------------------------------------------------------

class EuropeanBinaryJR(object):
    """EuropeanBinaryJR(S, sigma, K, T, n, r, q=0, call=True, cash=True)

Creates a European option. European options give the option buyer the right
to buy (call) or sell (put) a stock at a future date for a specified strike
price K. The payout is max(ST-K, 0) for a call and max(K-ST, 0) for a put,
where ST is the final sock price.

This class takes the following arguments:

S: underlying stock price
sigma: underlying stock annual volatility
K: strike price of the option
T: length of the life of the option (in annual terms)
n: number of periods in the tree (time step between periods = T/n)
r: annual risk-free rate, compounded continuously
q: annual continuous dividend rate, defaults to 0
call: boolean indicating if the option is a call or put option, defaults to True
cash: indicates if the option is cash-or-nothing or asset-or-nothing, defaults to True

If any of the parameters change, run [object name].calc_price() to generate the new price.
"""
    def __init__(self, S, sigma, K, T, n, r, q=0.0, call=True, cash=True):
        _check_option_error(S, sigma, K, T, n, r, q=q, call=call)
        self.S = S
        self.sigma = sigma
        self.K = K
        self.T = T
        self.n = n
        self.dt = float(T)/n
        self.r = r
        self.q = q
        self.call = call
        self.cash=cash
        self.delta = 0
        self.gamma = 0
        self.theta = 0
        self.price_tree = []
        self.stock_tree = []
        self.price = self.calc_price()

    def make_tree(self):
        #Source for u and d: http://www.goddardconsulting.ca/option-pricing-binomial-alts.html
        stock_tree = []
        u = math.exp((self.r - self.q - (.5*(self.sigma**2)))*self.dt + self.sigma*(self.dt**.5))
        d = math.exp((self.r - self.q - (.5*(self.sigma**2)))*self.dt - self.sigma*(self.dt**.5))
        for t in xrange(self.n+1):
            prices = []
            for num_u in xrange(t+1):
                prices.append(self.S*(u**num_u)*(d**(t-num_u)))
            stock_tree.append(prices)
        self.stock_tree = stock_tree
        return stock_tree

    def calc_price(self):
        self.stock_tree = self.make_tree()
        price_tree = copy.deepcopy(self.stock_tree)
        pu = .5
        pd = .5
        #Find the exercise value at final nodes
        for i, price in enumerate(price_tree[-1]):
            if self.call and price - self.K > 0:
                price_tree[-1][i] = (1*self.cash)+(self.stock_tree[-1][i]*(not self.cash))
            elif not self.call and self.K - price > 0:
                price_tree[-1][i] = (1*self.cash)+(self.stock_tree[-1][i]*(not self.cash))
            else:
                price_tree[-1][i] = 0
        #Move backwards through the tree to determine option values
        for t in xrange(self.n-1, -1, -1):
            for i in xrange(t+1):
                price_tree[t][i] = (pd*price_tree[t+1][i])+(pu*price_tree[t+1][i+1])
                #Discount the option price
                price_tree[t][i] *= math.exp((self.r-self.q)*self.dt*-1)
        self.price = price_tree[0][0]
        self.price_tree = price_tree
        self.delta, self.gamma, self.theta = calc_bin_greeks(self.stock_tree, self.price_tree, self.dt)
        return self.price

    def print_tree(self, stock=False):
        if stock:
            for line in self.stock_tree:
                print '\t'.join([str(round(price, 2)) for price in line])
        else:
            for line in self.price_tree:
                print '\t'.join([str(round(price, 2)) for price in line])


class AmericanBinaryJR(object):
    """AmericanBinaryJR(S, sigma, K, T, n, r, q=0, call=True, cash=True)

Creates an American option. An American option gives the option buyer
the right to buy (call) or sell (put) a stock at a specified strike price.
The buyer may exercise this option before expiry at any time. Whenever the
option is exercised, the payout is max(St - K, 0) for calls and
max(K - St, 0) for puts, where St is the stock price at time t.

This class takes the following arguments:

S: underlying stock price
sigma: underlying stock annual volatility
K: strike price of the option
T: length of the life of the option (in annual terms)
n: number of periods in the tree (time step between periods = T/n)
r: annual risk-free rate, compounded continuously
q: annual continuous dividend rate, defaults to 0
call: boolean indicating if the option is a call or put option, defaults to True
cash: indicates if the option is cash-or-nothing or asset-or-nothing, defaults to True

If any of the parameters change, run [object name].calc_price() to generate the new price.
"""
    def __init__(self, S, sigma, K, T, n, r, q=0.0, call=True, cash=True):
        _check_option_error(S, sigma, K, T, n, r, q=q, call=call)
        self.S = S
        self.sigma = sigma
        self.K = K
        self.T = T
        self.n = n
        self.dt = float(T)/n
        self.r = r
        self.q = q
        self.call = call
        self.cash = cash
        self.delta = 0
        self.gamma = 0
        self.theta = 0
        self.price_tree = []
        self.stock_tree = []
        self.price = self.calc_price()

    def make_tree(self):
        #Source for u and d: http://www.goddardconsulting.ca/option-pricing-binomial-alts.html
        stock_tree = []
        u = math.exp((self.r - self.q - (.5*(self.sigma**2)))*self.dt + self.sigma*(self.dt**.5))
        d = math.exp((self.r - self.q - (.5*(self.sigma**2)))*self.dt - self.sigma*(self.dt**.5))
        for t in xrange(self.n+1):
            prices = []
            for num_u in xrange(t+1):
                prices.append(self.S*(u**num_u)*(d**(t-num_u)))
            stock_tree.append(prices)
        self.stock_tree = stock_tree
        return stock_tree

    def calc_price(self):
        self.stock_tree = self.make_tree()
        price_tree = copy.deepcopy(self.stock_tree)
        pu = .5
        pd = .5
        #Find exercise value at final nodes
        for i, price in enumerate(price_tree[-1]):
            if self.call and price - self.K > 0:
                price_tree[-1][i] = (1*self.cash)+(self.stock_tree[-1][i]*(not self.cash))
            elif not self.call and self.K - price > 0:
                price_tree[-1][i] = (1*self.cash)+(self.stock_tree[-1][i]*(not self.cash))
            else:
                price_tree[-1][i] = 0
        #Move backwards through the tree to determine option values
        for t in xrange(self.n-1, -1, -1):
            for i in xrange(t+1):
                option_value = (pd*price_tree[t+1][i])+(pu*price_tree[t+1][i+1])
                #Use appropriate payout function
                if self.call and price_tree[t][i] - self.K > 0:
                    exercise_value = (1*self.cash)+(self.stock_tree[t][i]*(not self.cash))
                elif not self.call and self.K - price_tree[t][i] > 0:
                    exercise_value = (1*self.cash)+(self.stock_tree[t][i]*(not self.cash))
                else:
                    exercise_value = 0
                price_tree[t][i] = max(option_value, exercise_value)
                #Discount the option price
                price_tree[t][i] *= math.exp((self.r-self.q)*self.dt*-1)
        self.price = price_tree[0][0]
        self.price_tree = price_tree
        self.delta, self.gamma, self.theta = calc_bin_greeks(self.stock_tree, self.price_tree, self.dt)
        return self.price

    def print_tree(self, stock=False):
        if stock:
            for line in self.stock_tree:
                print '\t'.join([str(round(price, 2)) for price in line])
        else:
            for line in self.price_tree:
                print '\t'.join([str(round(price, 2)) for price in line])


class BermudanBinaryJR(object):
    """BermudanBinaryJR(S, sigma, K, T, n, r, q=0, ex=[], call=True, cash=True)

Creates an Bermudan option. A Bermudan option gives the option buyer
the right to buy (call) or sell (put) a stock at a specified strike price.
The buyer may exercise this option at certain points in time. Whenever the
option is exercised, the payout is max(St - K, 0) for calls and
max(K - St, 0) for puts, where St is the stock price at time t.

This class takes the following arguments:

S: underlying stock price
sigma: underlying stock annual volatility
K: strike price of the option
T: length of the life of the option (in annual terms)
n: number of periods in the tree (time step between periods = T/n)
r: annual risk-free rate, compounded continuously
q: annual continuous dividend rate, defaults to 0
ex: a list containing the periods at which exercising is allowed.
call: boolean indicating if the option is a call or put option, defaults to True
cash: indicates if the option is cash-or-nothing or asset-or-nothing, defaults to True

If any of the parameters change, run [object name].calc_price() to generate the new price.
"""
    def __init__(self, S, sigma, K, T, n, r, q=0.0, ex=None, call=True, cash=True):
        _check_option_error(S, sigma, K, T, n, r, q=q, ex=ex, call=call)
        self.S = S
        self.sigma = sigma
        self.K = K
        self.T = T
        self.n = n
        self.dt = float(T)/n
        self.r = r
        self.q = q
        self.ex = ex
        self.call = call
        self.cash = cash
        self.delta = 0
        self.gamma = 0
        self.theta = 0
        self.price_tree = []
        self.stock_tree = []
        self.price = self.calc_price()

    def make_tree(self):
        #Source for u and d: http://www.goddardconsulting.ca/option-pricing-binomial-alts.html
        stock_tree = []
        u = math.exp((self.r - self.q - (.5*(self.sigma**2)))*self.dt + self.sigma*(self.dt**.5))
        d = math.exp((self.r - self.q - (.5*(self.sigma**2)))*self.dt - self.sigma*(self.dt**.5))
        for t in xrange(self.n+1):
            prices = []
            for num_u in xrange(t+1):
                prices.append(self.S*(u**num_u)*(d**(t-num_u)))
            stock_tree.append(prices)
        self.stock_tree = stock_tree
        return stock_tree

    def calc_price(self):
        self.stock_tree = self.make_tree()
        price_tree = copy.deepcopy(self.stock_tree)
        pu = .5
        pd = .5
        #Find exercise value at final nodes
        for i, price in enumerate(price_tree[-1]):
            if self.call and price - self.K > 0:
                price_tree[-1][i] = (1*self.cash)+(self.stock_tree[-1][i]*(not self.cash))
            elif not self.call and self.K - price > 0:
                price_tree[-1][i] = (1*self.cash)+(self.stock_tree[-1][i]*(not self.cash))
            else:
                price_tree[-1][i] = 0
        #Move backwards through the tree to determine option values
        for t in xrange(self.n-1, -1, -1):
            for i in xrange(t+1):
                option_value = (pd*price_tree[t+1][i])+(pu*price_tree[t+1][i+1])
                #Use appropriate payout function
                if self.call and price_tree[t][i] - self.K > 0:
                    exercise_value = (1*self.cash)+(self.stock_tree[t][i]*(not self.cash))
                elif not self.call and self.K - price_tree[t][i] > 0:
                    exercise_value = (1*self.cash)+(self.stock_tree[t][i]*(not self.cash))
                else:
                    exercise_value = 0
                #Check if holder can exercise
                if t in self.ex:
                    price_tree[t][i] = max(option_value, exercise_value)
                else:
                    price_tree[t][i] = option_value
                #Discount price
                price_tree[t][i] *= math.exp((self.r-self.q)*self.dt*-1)
        self.price = price_tree[0][0]
        self.price_tree = price_tree
        self.delta, self.gamma, self.theta = calc_bin_greeks(self.stock_tree, self.price_tree, self.dt)
        return self.price

    def print_tree(self, stock=False):
        if stock:
            for line in self.stock_tree:
                print '\t'.join([str(round(price, 2)) for price in line])
        else:
            for line in self.price_tree:
                print '\t'.join([str(round(price, 2)) for price in line])


###Tian------------------------------------------------------------------------------

class EuropeanBinaryTian(object):
    """EuropeanBinaryTian(S, sigma, K, T, n, r, q=0, call=True, cash=True)

Creates a European option. European options give the option buyer the right
to buy (call) or sell (put) a stock at a future date for a specified strike
price K. The payout is max(ST-K, 0) for a call and max(K-ST, 0) for a put,
where ST is the final sock price.

This class takes the following arguments:

S: underlying stock price
sigma: underlying stock annual volatility
K: strike price of the option
T: length of the life of the option (in annual terms)
n: number of periods in the tree (time step between periods = T/n)
r: annual risk-free rate, compounded continuously
q: annual continuous dividend rate, defaults to 0
call: boolean indicating if the option is a call or put option, defaults to True
cash: indicates if the option is cash-or-nothing or asset-or-nothing, defaults to True

If any of the parameters change, run [object name].calc_price() to generate the new price.
"""
    def __init__(self, S, sigma, K, T, n, r, q=0.0, call=True, cash=True):
        _check_option_error(S, sigma, K, T, n, r, q=q, call=call)
        self.S = S
        self.sigma = sigma
        self.K = K
        self.T = T
        self.n = n
        self.dt = float(T)/n
        self.r = r
        self.q = q
        self.call = call
        self.cash = cash
        self.delta = 0
        self.gamma = 0
        self.theta = 0
        self.price_tree = []
        self.stock_tree = []
        self.price = self.calc_price()

    def make_tree(self):
        #Source for u and d: http://www.goddardconsulting.ca/option-pricing-binomial-alts.html
        stock_tree = []
        V = math.exp((self.sigma**2)*self.dt)
        u = .5 * math.exp((self.r - self.q) * self.dt) * V * (V + 1 + ((V**2) + (2*V) - 3)**.5)
        d = .5 * math.exp((self.r - self.q) * self.dt) * V * (V + 1 - ((V**2) + (2*V) - 3)**.5)
        for t in xrange(self.n+1):
            prices = []
            for num_u in xrange(t+1):
                prices.append(self.S*(u**num_u)*(d**(t-num_u)))
            stock_tree.append(prices)
        self.stock_tree = stock_tree
        return stock_tree

    def calc_price(self):
        self.stock_tree = self.make_tree()
        price_tree = copy.deepcopy(self.stock_tree)
        V = math.exp((self.sigma**2)*self.dt)
        u = .5 * math.exp((self.r - self.q) * self.dt) * V * (V + 1 + ((V**2) + (2*V) - 3)**.5)
        d = .5 * math.exp((self.r - self.q) * self.dt) * V * (V + 1 - ((V**2) + (2*V) - 3)**.5)
        pu = (math.exp((self.r - self.q)*self.dt) - d)/(u - d)
        pd = 1 - pu
        #Find the exercise value at final nodes
        for i, price in enumerate(price_tree[-1]):
            if self.call and price - self.K > 0:
                price_tree[-1][i] = (1*self.cash)+(self.stock_tree[-1][i]*(not self.cash))
            elif not self.call and self.K - price > 0:
                price_tree[-1][i] = (1*self.cash)+(self.stock_tree[-1][i]*(not self.cash))
            else:
                price_tree[-1][i] = 0
        #Move backwards through the tree to determine option values
        for t in xrange(self.n-1, -1, -1):
            for i in xrange(t+1):
                price_tree[t][i] = (pd*price_tree[t+1][i])+(pu*price_tree[t+1][i+1])
                #Discount the option price
                price_tree[t][i] *= math.exp((self.r-self.q)*self.dt*-1)
        self.price = price_tree[0][0]
        self.price_tree = price_tree
        self.delta, self.gamma, self.theta = calc_bin_greeks(self.stock_tree, self.price_tree, self.dt)
        return self.price

    def print_tree(self, stock=False):
        if stock:
            for line in self.stock_tree:
                print '\t'.join([str(round(price, 2)) for price in line])
        else:
            for line in self.price_tree:
                print '\t'.join([str(round(price, 2)) for price in line])


class AmericanBinaryTian(object):
    """AmericanBinaryTian(S, sigma, K, T, n, r, q=0, call=True, cash=True)

Creates an American option. An American option gives the option buyer
the right to buy (call) or sell (put) a stock at a specified strike price.
The buyer may exercise this option before expiry at any time. Whenever the
option is exercised, the payout is max(St - K, 0) for calls and
max(K - St, 0) for puts, where St is the stock price at time t.

This class takes the following arguments:

S: underlying stock price
sigma: underlying stock annual volatility
K: strike price of the option
T: length of the life of the option (in annual terms)
n: number of periods in the tree (time step between periods = T/n)
r: annual risk-free rate, compounded continuously
q: annual continuous dividend rate, defaults to 0
call: boolean indicating if the option is a call or put option, defaults to True
cash: indicates if the option is cash-or-nothing or asset-or-nothing, defaults to True

If any of the parameters change, run [object name].calc_price() to generate the new price.
"""
    def __init__(self, S, sigma, K, T, n, r, q=0.0, call=True, cash=True):
        _check_option_error(S, sigma, K, T, n, r, q=q, call=call)
        self.S = S
        self.sigma = sigma
        self.K = K
        self.T = T
        self.n = n
        self.dt = float(T)/n
        self.r = r
        self.q = q
        self.call = call
        self.cash = cash
        self.delta = 0
        self.gamma = 0
        self.theta = 0
        self.price_tree = []
        self.stock_tree = []
        self.price = self.calc_price()

    def make_tree(self):
        #Source for u and d: http://www.goddardconsulting.ca/option-pricing-binomial-alts.html
        stock_tree = []
        V = math.exp((self.sigma**2)*self.dt)
        u = .5 * math.exp((self.r - self.q) * self.dt) * V * (V + 1 + ((V**2) + (2*V) - 3)**.5)
        d = .5 * math.exp((self.r - self.q) * self.dt) * V * (V + 1 - ((V**2) + (2*V) - 3)**.5)
        for t in xrange(self.n+1):
            prices = []
            for num_u in xrange(t+1):
                prices.append(self.S*(u**num_u)*(d**(t-num_u)))
            stock_tree.append(prices)
        self.stock_tree = stock_tree
        return stock_tree

    def calc_price(self):
        self.stock_tree = self.make_tree()
        price_tree = copy.deepcopy(self.stock_tree)
        V = math.exp((self.sigma**2)*self.dt)
        u = .5 * math.exp((self.r - self.q) * self.dt) * V * (V + 1 + ((V**2) + (2*V) - 3)**.5)
        d = .5 * math.exp((self.r - self.q) * self.dt) * V * (V + 1 - ((V**2) + (2*V) - 3)**.5)
        pu = (math.exp((self.r - self.q)*self.dt) - d)/(u - d)
        pd = 1 - pu
        #Find exercise value at final nodes
        for i, price in enumerate(price_tree[-1]):
            if self.call and price - self.K > 0:
                price_tree[-1][i] = (1*self.cash)+(self.stock_tree[-1][i]*(not self.cash))
            elif not self.call and self.K - price > 0:
                price_tree[-1][i] = (1*self.cash)+(self.stock_tree[-1][i]*(not self.cash))
            else:
                price_tree[-1][i] = 0
        #Move backwards through the tree to determine option values
        for t in xrange(self.n-1, -1, -1):
            for i in xrange(t+1):
                option_value = (pd*price_tree[t+1][i])+(pu*price_tree[t+1][i+1])
                #Use appropriate payout function
                if self.call and price_tree[t][i] - self.K > 0:
                    exercise_value = (1*self.cash)+(self.stock_tree[t][i]*(not self.cash))
                elif not self.call and self.K - price_tree[t][i] > 0:
                    exercise_value = (1*self.cash)+(self.stock_tree[t][i]*(not self.cash))
                else:
                    exercise_value = 0
                price_tree[t][i] = max(option_value, exercise_value)
                #Discount the option price
                price_tree[t][i] *= math.exp((self.r-self.q)*self.dt*-1)
        self.price = price_tree[0][0]
        self.price_tree = price_tree
        self.delta, self.gamma, self.theta = calc_bin_greeks(self.stock_tree, self.price_tree, self.dt)
        return self.price

    def print_tree(self, stock=False):
        if stock:
            for line in self.stock_tree:
                print '\t'.join([str(round(price, 2)) for price in line])
        else:
            for line in self.price_tree:
                print '\t'.join([str(round(price, 2)) for price in line])


class BermudanBinaryTian(object):
    """BermudanBinaryTian(S, sigma, K, T, n, r, q=0, ex=[], call=True, cash=True)

Creates an Bermudan option. A Bermudan option gives the option buyer
the right to buy (call) or sell (put) a stock at a specified strike price.
The buyer may exercise this option at certain points in time. Whenever the
option is exercised, the payout is max(St - K, 0) for calls and
max(K - St, 0) for puts, where St is the stock price at time t.

This class takes the following arguments:

S: underlying stock price
sigma: underlying stock annual volatility
K: strike price of the option
T: length of the life of the option (in annual terms)
n: number of periods in the tree (time step between periods = T/n)
r: annual risk-free rate, compounded continuously
q: annual continuous dividend rate, defaults to 0
ex: a list containing the periods at which exercising is allowed.
call: boolean indicating if the option is a call or put option, defaults to True
cash: indicates if the option is cash-or-nothing or asset-or-nothing, defaults to True

If any of the parameters change, run [object name].calc_price() to generate the new price.
"""
    def __init__(self, S, sigma, K, T, n, r, q=0.0, ex=None, call=True, cash=True):
        _check_option_error(S, sigma, K, T, n, r, q=q, ex=ex, call=call)
        self.S = S
        self.sigma = sigma
        self.K = K
        self.T = T
        self.n = n
        self.dt = float(T)/n
        self.r = r
        self.q = q
        self.ex = ex
        self.call = call
        self.cash = cash
        self.delta = 0
        self.gamma = 0
        self.theta = 0
        self.price_tree = []
        self.stock_tree = []
        self.price = self.calc_price()

    def make_tree(self):
        #Source for u and d: http://www.goddardconsulting.ca/option-pricing-binomial-alts.html
        stock_tree = []
        V = math.exp((self.sigma**2)*self.dt)
        u = .5 * math.exp((self.r - self.q) * self.dt) * V * (V + 1 + ((V**2) + (2*V) - 3)**.5)
        d = .5 * math.exp((self.r - self.q) * self.dt) * V * (V + 1 - ((V**2) + (2*V) - 3)**.5)
        for t in xrange(self.n+1):
            prices = []
            for num_u in xrange(t+1):
                prices.append(self.S*(u**num_u)*(d**(t-num_u)))
            stock_tree.append(prices)
        self.stock_tree = stock_tree
        return stock_tree

    def calc_price(self):
        self.stock_tree = self.make_tree()
        price_tree = copy.deepcopy(self.stock_tree)
        V = math.exp((self.sigma**2)*self.dt)
        u = .5 * math.exp((self.r - self.q) * self.dt) * V * (V + 1 + ((V**2) + (2*V) - 3)**.5)
        d = .5 * math.exp((self.r - self.q) * self.dt) * V * (V + 1 - ((V**2) + (2*V) - 3)**.5)
        pu = (math.exp((self.r - self.q)*self.dt) - d)/(u - d)
        pd = 1 - pu
        #Find exercise value at final nodes
        for i, price in enumerate(price_tree[-1]):
            if self.call and price - self.K > 0:
                price_tree[-1][i] = (1*self.cash)+(self.stock_tree[-1][i]*(not self.cash))
            elif not self.call and self.K - price > 0:
                price_tree[-1][i] = (1*self.cash)+(self.stock_tree[-1][i]*(not self.cash))
            else:
                price_tree[-1][i] = 0
        #Move backwards through the tree to determine option values
        for t in xrange(self.n-1, -1, -1):
            for i in xrange(t+1):
                option_value = (pd*price_tree[t+1][i])+(pu*price_tree[t+1][i+1])
                #Use appropriate payout function
                if self.call and price_tree[t][i] - self.K > 0:
                    exercise_value = (1*self.cash)+(self.stock_tree[t][i]*(not self.cash))
                elif not self.call and self.K - price_tree[t][i] > 0:
                    exercise_value = (1*self.cash)+(self.stock_tree[t][i]*(not self.cash))
                else:
                    exercise_value = 0
                #Check if holder can exercise
                if t in self.ex:
                    price_tree[t][i] = max(option_value, exercise_value)
                else:
                    price_tree[t][i] = option_value
                #Discount price
                price_tree[t][i] *= math.exp((self.r-self.q)*self.dt*-1)
        self.price = price_tree[0][0]
        self.price_tree = price_tree
        self.delta, self.gamma, self.theta = calc_bin_greeks(self.stock_tree, self.price_tree, self.dt)
        return self.price

    def print_tree(self, stock=False):
        if stock:
            for line in self.stock_tree:
                print '\t'.join([str(round(price, 2)) for price in line])
        else:
            for line in self.price_tree:
                print '\t'.join([str(round(price, 2)) for price in line])


#Vanilla Options - Black-Scholes-----------------------------------------------------

class EuropeanBS(object):

    def __init__(self, S, sigma, K, T, r, q, call=True):
        _check_option_error(S=S, sigma=sigma, K=K, T=T, r=r, q=q, call=call)
        self.S = float(S)
        self.sigma = sigma
        self.K = float(K)
        self.T = T
        self.r = r
        self.q = q
        self.call = call
        self.price = 0
        self.calc_price()

    def calc_price(self):
        d1 = (math.log(self.S/self.K) + (self.r - self.q + (.5*(self.sigma**2)))*self.T)/(self.sigma*(self.T**.5))
        d2 = d1 - (self.sigma*(self.T**.5))
        if self.call:
            Nd1, Nd2 = phi(d1), phi(d2)
            self.price = (math.exp(-self.q*self.T)*self.S*Nd1) - (math.exp(-self.r*self.T)*self.K*Nd2)
        else:
            Nd1, Nd2 = phi(d1*-1), phi(d2*-1)
            self.price = (math.exp(-self.r*self.T)*self.K*Nd2) - (math.exp(-self.q*self.T)*self.S*Nd1)
        return self.price

    def implied_vol(self, mkt_price, precision=10):
        # This process is identical for all BS objects - this can probably be refactored more optimally.
        # Preserving true value
        actual_sigma = self.sigma
        # Same search process as irr function
        if mkt_price == self.price:
            return self.sigma
        elif mkt_price > self.price:
            delta = 0.1
        else:
            delta = -0.1
        while abs(delta) >= (10**(precision*-1)):
            self.sigma += delta
            self.calc_price()
            if self.price == mkt_price:
                imp_vol = self.sigma
                # Restore initial values
                self.sigma = actual_sigma
                self.calc_price()
                return imp_vol
            else:
                if self.price < mkt_price and delta < 0:
                    delta /= -10
                elif self.price > mkt_price and delta > 0:
                    delta /= -10
        imp_vol = self.sigma
        self.sigma = actual_sigma
        self.calc_price()
        return imp_vol


class EuropeanBinaryBS(object):

    def __init__(self, S, sigma, K, T, r, q, call=True, cash=True):
        _check_option_error(S=S, sigma=sigma, K=K, T=T, r=r, q=q, call=call)
        self.S = float(S)
        self.sigma = sigma
        self.K = float(K)
        self.T = T
        self.r = r
        self.q = q
        self.call = call
        self.cash = cash
        self.price = 0
        self.calc_price()

    def calc_price(self):
        #Source: http://en.wikipedia.org/wiki/Binary_option#Black.E2.80.93Scholes_valuation
        d1 = (math.log(self.S/self.K) + (self.r - self.q + (.5*(self.sigma**2)))*self.T)/(self.sigma*(self.T**.5))
        d2 = d1 - (self.sigma*(self.T**.5))
        if self.call and self.cash:
            self.price = math.exp(-(self.r*self.T))*phi(d2)
        elif not self.call and self.cash:
            self.price = math.exp(-(self.r*self.T))*phi(d2*-1)
        elif self.call and not self.cash:
            self.price = self.S*math.exp(-(self.q*self.T))*phi(d1)
        else:
            self.price = self.S*math.exp(-(self.q*self.T))*phi(d1*-1)
        return self.price

    #TODO: Derive formulas for volatility for d1 and d2 to get implied vol.


#Options - Monte Carlo---------------------------------------------------------------

class EuropeanMCLogNormConstVol(object):
    # http://finance.bi.no/~bernt/gcc_prog/recipes/recipes/node12.html
    def __init__(self, S, sigma, K, T, r, q, call=True):
        self.S = S
        self.sigma = sigma
        self.K = K
        self.T = T
        self.r = r
        self.q = q
        self.call = call
        self.price = 0

    def calc_price(self, num_obs, num_sims):
        # Uses antithetic path variance reduction
        dt = self.T/float(num_obs)
        payoff = 0
        if self.call:
            payoff_func = lambda a, b: max(a - b, 0)
        else:
            payoff_func = lambda a, b: max(b - a, 0)
        for n in xrange(num_sims/2):
            Si1, Si2 = self.S, self.S
            for i in xrange(num_obs):
                dWt1 = random.normalvariate(0, 1)
                dWt2 = dWt1 * -1
                Si1 *= math.exp((((self.r-self.q) - (.5*(self.sigma**2))) * dt) + (self.sigma*dWt1*(dt**.5)))
                Si2 *= math.exp((((self.r-self.q) - (.5*(self.sigma**2))) * dt) + (self.sigma*dWt2*(dt**.5)))
            payoff += payoff_func(Si1, self.K) + payoff_func(Si2, self.K)
        self.price = math.exp(-(self.r - self.q)*self.T)*(payoff/num_sims)
        return self.price


class AsianMCLogNormConstVol(object):
    # http://finance.bi.no/~bernt/gcc_prog/recipes/recipes/node12.html
    def __init__(self, S, sigma, K, T, r, q, call=True, geometric=True):
        self.S = S
        self.sigma = sigma
        self.K = K
        self.T = T
        self.r = r
        self.q = q
        self.call = call
        self.geometric = geometric
        self.price = 0

    def calc_price(self, num_obs, num_sims):
        # Uses antithetic path variance reduction
        dt = self.T/float(num_obs)
        payoff = 0
        if self.geometric:
            base_avg = 1
            inc_avg = lambda a, b, c: (a**(1.0/b)) * c
        else:
            base_avg = 0
            inc_avg = lambda a, b, c: (a/b) + c
        if self.call:
            payoff_func = lambda a, b: max(a - b, 0)
        else:
            payoff_func = lambda a, b: max(b - a, 0)
        for n in xrange(num_sims/2):
            avg1 = base_avg
            avg2 = base_avg
            Si1 = self.S
            Si2 = self.S
            for i in xrange(num_obs):
                dWt1 = random.normalvariate(0, 1)
                dWt2 = dWt1 * -1
                Si1 *= math.exp((((self.r-self.q) - (.5*(self.sigma**2))) * dt) + (self.sigma*dWt1*(dt**.5)))
                Si2 *= math.exp((((self.r-self.q) - (.5*(self.sigma**2))) * dt) + (self.sigma*dWt2*(dt**.5)))
                avg1 = inc_avg(Si1, num_obs, avg1)
                avg2 = inc_avg(Si2, num_obs, avg2)
            payoff += payoff_func(avg1, self.K) + payoff_func(avg2, self.K)
        self.price = math.exp(-(self.r - self.q)*self.T)*(payoff/num_sims)
        return self.price


class LookbackFixedMCLogNormConstVol(object):
    # http://finance.bi.no/~bernt/gcc_prog/recipes/recipes/node12.html
    def __init__(self, S, K, sigma, T, r, q, call=True):
        self.S = S
        self.sigma = sigma
        self.K = K
        self.T = T
        self.r = r
        self.q = q
        self.call = call
        self.price = 0

    def calc_price(self, num_obs, num_sims):
        # Uses antithetic path variance reduction
        dt = self.T/float(num_obs)
        payoff = 0
        if self.call:
            get_X = lambda a, b: max(a, b)
            payoff_func = lambda a, b: max(a - b, 0)
        else:
            get_X = lambda a, b: min(a, b)
            payoff_func = lambda a, b: max(b - a, 0)
        for n in xrange(num_sims/2):
            X1 = self.S
            X2 = self.S
            Si1 = self.S
            Si2 = self.S
            for i in xrange(num_obs):
                dWt1 = random.normalvariate(0, 1)
                dWt2 = dWt1 * -1
                Si1 *= math.exp((((self.r-self.q) - (.5*(self.sigma**2))) * dt) + (self.sigma*dWt1*(dt**.5)))
                Si2 *= math.exp((((self.r-self.q) - (.5*(self.sigma**2))) * dt) + (self.sigma*dWt2*(dt**.5)))
                X1 = get_X(X1, Si1)
                X2 = get_X(X2, Si2)
            payoff += payoff_func(X1, self.K) + payoff_func(X2, self.K)
        self.price = math.exp(-(self.r - self.q)*self.T)*(payoff/num_sims)
        return self.price


class LookbackFloatingMCLogNormConstVol(object):
    # http://finance.bi.no/~bernt/gcc_prog/recipes/recipes/node12.html
    def __init__(self, S, sigma, T, r, q, call=True):
        self.S = S
        self.sigma = sigma
        self.T = T
        self.r = r
        self.q = q
        self.call = call
        self.price = 0

    def calc_price(self, num_obs, num_sims):
        # Uses antithetic path variance reduction
        dt = self.T/float(num_obs)
        payoff = 0
        if self.call:
            get_X = lambda a, b: min(a, b)
            payoff_func = lambda a, b: a - b
        else:
            get_X = lambda a, b: max(a, b)
            payoff_func = lambda a, b: b - a
        for n in xrange(num_sims/2):
            X1 = self.S
            X2 = self.S
            Si1 = self.S
            Si2 = self.S
            for i in xrange(num_obs):
                dWt1 = random.normalvariate(0, 1)
                dWt2 = dWt1 * -1
                Si1 *= math.exp((((self.r-self.q) - (.5*(self.sigma**2))) * dt) + (self.sigma*dWt1*(dt**.5)))
                Si2 *= math.exp((((self.r-self.q) - (.5*(self.sigma**2))) * dt) + (self.sigma*dWt2*(dt**.5)))
                X1 = get_X(X1, Si1)
                X2 = get_X(X2, Si2)
            payoff += payoff_func(Si1, X1) + payoff_func(Si2, X2)
        self.price = math.exp(-(self.r - self.q)*self.T)*(payoff/num_sims)
        return self.price


if __name__ == '__main__':
    S = 100
    sigma = .25
    K = 100
    T = 1
    r = .1
    q = 0.0
    call = True
    geometric = True
    base_option = EuropeanBS(S=S, sigma=sigma, K=K, T=T, r=r, q=q, call=call)
    print base_option.price
    my_option = EuropeanMCLogNormConstVol(S=S, sigma=sigma, K=K, T=T, r=r, q=q, call=call)
    print my_option.calc_price(num_obs=1, num_sims=500)
    my_option = AsianMCLogNormConstVol(S=S, sigma=sigma, K=K, T=T, r=r, q=q, call=call, geometric=geometric)
    print my_option.calc_price(num_obs=10, num_sims=1000)
    my_option = LookbackFixedMCLogNormConstVol(S=S, sigma=sigma, K=K, T=T, r=r, q=q, call=call)
    print my_option.calc_price(num_obs=1, num_sims=50)
    my_option = LookbackFloatingMCLogNormConstVol(S=S, sigma=sigma, T=T, r=r, q=q, call=call)
    print my_option.calc_price(num_obs=1, num_sims=50)