"""This is the primary module of the PyFi package. This module includes basic and advanced financial
functions, as well as classes to model certain financial instruments such as bonds, amortizing loans,
and equity options.

Current Functions:
pv - returns the present value of a series of cash flows
fv - returns the future value of a series of cash flows
irr - returns the Internal Rate of Return (IRR) of a series of cash flows
mirr - returns the Modified IRR of a series of cash flows
macD - returns the Macaulay duration of a series fo cash flows
modD - returns the Modified duration of a series of cash flows
convexity - returns the convexity of a series of cash flows
phi - returns the CDF of a N(0,1) distribution evaluated at a particular point

Current Classes:
Amortize - amortizes a beginning principal based on the arguments given. All attributes, such as interest paid, principal
           remaining, etc. are stored as lists over all payment periods for easy access.
Bond - creates a bond object. Price and YTM can be changed or calculated. Duration and convexity are also calculated
       automatically.
Stock - holds current value and volatility of a stock.
Options - various options are treated as classes in PyFi:
    - Binomial. Note Binary options can be either cash-or-nothing or asset-or-nothing. Delta, Gamma, and Theta are
      calculated automatically.
        - EuropeanCRR / EuropeanBinaryCRR
        - AmericanCRR / AmericanBinaryCRR
        - BermudanCRR / BermudanBinaryCRR
        - EuropeanJR / EuropeanBinaryJR
        - AmericanJR / AmericanBinaryJR
        - BermudanJR / BermudanBinaryJR
        - EuropeanTian / EuropeanBinaryTian
        - AmericanTian / AmericanBinaryTian
        - BermudanTian / BermudanBinaryTian
    - Closed-Form.
        - EuropeanBS / EuropeanBinaryBS
"""

import math
import copy
import random

#-------------------------------------------------------------------------------------
#Private Functions (input error checking)
#-------------------------------------------------------------------------------------


def _check_cf_error(cash_flows, apr, dt):
    """This function is used to check the inputs for numerous functions
used in this module. Without a dedicated function, there would be a lot
of repetition in each of the actual functions."""
    if not isinstance(cash_flows, list):
        raise Exception('Invalid type for cash flows. Must be a list.')
    if len(cash_flows) < 1:
        raise Exception('Invalid size of cash flow array - must contain at least 1 element.')
    if not isinstance(apr, float):
        raise Exception('Invalid type for apr. Must be a float.')
    if not isinstance(dt, int) and not isinstance(dt, float):
        raise Exception('Invalid type for dt. Must be an int or float.')
    if dt <= 0:
        raise Exception('Invalid value for dt. Must be greater than 0')


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


#-------------------------------------------------------------------------------------
#Functions (pv, fv, irr, macD, modD, convexity)
#-------------------------------------------------------------------------------------

#General financial functions----------------------------------------------------------

def pv(cash_flows, apr, dt=1):
    """This function calculates the present value of a series of
cash flows. Discounts to time 0, and assumes first cash flow
happens at first future time step.

cash_flows: an array of cash flows, with each cash flow being
            the cash flow at a specific point in time.
apr: the annual interest rate used, which should not be confused
     with the EAR or the discount rate
dt: size of the time step of the cash flows in years
           (i.e. 1 month = 1/12)
"""

    _check_cf_error(cash_flows, apr, dt)

    dis_rate = 1.0/(1+(apr*dt))
    total = 0
    #Loop through each cash flow, bring it back to time 0, then add it to the total
    for t, pmt in enumerate(cash_flows):
        total += pmt*(dis_rate**(t+1))
    return total


def fv(cash_flows, apr, dt=1):
    """This function calculates the future value of a series of
cash flows. Determines the future value as of the last cash
flow given.

cash_flows: an array of cash flows, with each cash flow being
            the cash flow at a specific point in time.
apr: the annual interest rate used, which should not be confused
     with the EAR or the discount rate
dt: size of the time step of the cash flows in years
           (i.e. 1 month = 1/12)
"""

    cf_pv = pv(cash_flows, apr, dt)
    if not cf_pv:
        return None
    #Future value = present value * ((1 + interest rate)**n)
    return cf_pv*((1+(apr*dt))**len(cash_flows))


def irr(cash_flows, dt=1):
    """Calculates the Internal Rate of Return for a series of
cash flows, which is the rate at which their present value is $0.

cash_flows: an array of cash flows, with each cash flow being
            the cash flow at a specific point in time.
dt: size of the time step of the cash flows in years 
           (i.e. 1 month = 1/12)
"""

    _check_cf_error(cash_flows, 0.0, dt)
    #Additional checking - irr will be infinite if all cash flows have same sign
    count_pos = 0
    for i, pmt in enumerate(cash_flows):
        if pmt > 0:
            count_pos += 1
        if (i+1) > count_pos and count_pos > 0: #some are positive, some negative
            break
    else:
        raise Exception('Invalid cash flows - must contain both positive and negative elements.')

    apr = 0.0
    #delta is used below as the step by which the guessed apr changes
    delta = 0.1
    #closest_pv is the best-guess pv which is checked against check_pv using the new apr guess
    closest_pv = pv(cash_flows=cash_flows, apr=apr, dt=dt)
    #Finding irr to 10 digits if not found exactly - not sure if this is accurate enough
    while abs(delta) >= (10**(-10)):
        if closest_pv == 0:
            return apr
        apr += delta
        check_pv = round(pv(cash_flows=cash_flows, apr=apr, dt=dt), 10)
        if abs(closest_pv-check_pv) > abs(closest_pv):
            #Indicates that check_pv and closest_pv have different signs
            #Move guess steps to lower order, switch direction of guesses if check_pv is closer to 0
            #otherwise, back up and don't change the closest_pv
            delta *= 0.1
            if abs(check_pv) < abs(closest_pv):
                closest_pv = check_pv
                delta *= -1
            else:
                apr -= (delta*10)
        elif abs(check_pv) > abs(closest_pv):
            #Indicates that guesses are moving in wrong direction
            #Move apr back one step, switch guess direction of guess movement
            apr -= delta
            delta *= (-1)
        else:
            #Guess is closer to 0 than previously and has not gone too far, so continue
            closest_pv = check_pv
    return apr


def mirr(inflows, outflows, reinv_rate, borrow_rate, dt=1):
    """Returns the Modified Internal Rate of Return on a series of cash inflows and outflows.

inflows: list of cash inflows, all numbers must be positive
outflows: list of cash outflows, all numbers must be negative. outflows[0] is assumed to happen at t0.
reinv_rate: the apr at which inflows may be reinvested
borrow_rate: the apr at which outflows must be borrowed at
dt: the time step between cash flows"""

    if not isinstance(inflows, list) or not isinstance(outflows, list):
        raise Exception('Invalid type for cash flow lists. Must be a list')
    if not all(inflows) >= 0 and all(outflows) <= 0:
        raise Exception('Invalid cash flow values. All inflows must be >= 0 and all outflows must be <= 0')
    if not isinstance(reinv_rate, float) or not isinstance(borrow_rate, float):
        raise Exception('Invalid type for rates. Must be floats.')

    n = max(len(inflows), len(outflows))-1
    #need to time-shift outflows, since first payment is assumed to be at t0
    if len(outflows) == 1:
        pv_outflows = outflows[0]*-1
    else:
        pv_outflows = (outflows[0] + pv(outflows[1:], borrow_rate, dt))*-1
    fv_inflows = fv(inflows, reinv_rate, dt)
    return ((fv_inflows/pv_outflows)**(1.0/n))-1


def macD(cash_flows, apr, dt):
    """Determines the Macaulay duration of a series of cash flows. This value
represents the weighted average maturity of the cash flows.

cash_flows: an array of cash flows, with each cash flow being
            the cash flow at a specific point in time.
apr: the annual interest rate used, which should not be confused
     with the EAR or the discount rate
dt: size of the time step of the cash flows in years
           (i.e. 1 month = 1/12)
"""

    _check_cf_error(cash_flows, apr, dt)

    dis_rate = 1.0/(1+(apr*dt))
    num = 0
    den = 0
    for t, pmt in enumerate(cash_flows):
        pv_pmt = pmt*(dis_rate**(t+1))
        num += (t+1)*dt*pv_pmt
        den += pv_pmt
    return num/den


def modD(cash_flows, apr, dt):
    """Determines the Modified duration of a series of cash flows. This value
represents the percentage change in the PV of the cash flows for a 1 percentage
point increase in the apr.

cash_flows: an array of cash flows, with each cash flow being
            the cash flow at a specific point in time.
apr: the annual interest rate used, which should not be confused
     with the EAR or the discount rate
dt: size of the time step of the cash flows in years
           (i.e. 1 month = 1/12)
"""

    return macD(cash_flows, apr, dt)/(1+(apr*dt))


def convexity(cash_flows, apr, dt):

    _check_cf_error(cash_flows, apr, dt)

    #Formula source: http://faculty.darden.virginia.edu/conroyb/Valuation/Val2002/F-1238.pdf

    dis_rate = 1.0/(1+(apr*dt))
    P = pv(cash_flows, apr, dt)
    total = 0
    for t, pmt in enumerate(cash_flows):
        weight = pmt*(dis_rate**(t+1))/P
        total += ((t+1)*(t+2)*(dt**2)) * weight * (dis_rate**2)
    return total


#Special functions - used in classes, or for particular cases ------------------------


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


#------------------------------------------------------------------------------------
#Classes (Assets, Options)
#------------------------------------------------------------------------------------

class Amortize(object):
    def __init__(self, principal=None, int_rate=None, num_payments=None, payment=None, future_value=None):
        """Rate must be a per-period interest rate"""

        #Check errors
        if not isinstance(principal, int) and not isinstance(principal, float) and principal is not None:
            raise Exception('Invalid type for principal. Must be int or float')
        if not isinstance(int_rate, float) and int_rate is not None:
            raise Exception('Invalid type for apr. Must be float')
        if not isinstance(num_payments, int) and num_payments is not None:
            raise Exception('Invalid type for num_payments. Must be int')
        if num_payments <= 0 and num_payments is not None:
            raise Exception('Invalid value for num_payments. Must be greater than 0')
        if not isinstance(payment, int) and not isinstance(payment, float) and payment is not None:
            raise Exception('Invalid type for payment. Must be int or float')
        if not isinstance(future_value, int) and not isinstance(future_value, float) and future_value is not None:
            raise Exception('Invalid type for future_value. Must be int or float')
        if [principal, int_rate, num_payments, payment, future_value].count(None) != 1:
            raise Exception('Exactly one argument must be omitted when initializing an Amortize object')

        self.principal = principal
        self.int_rate = int_rate
        self.num_payments = num_payments
        self.payment = payment
        self.future_value = future_value
        self.principal_remaining = []
        self.interest_paid = []
        self.principal_paid = []
        self.payments = []

        #Calculate the missing value
        if self.principal is None:
            self.calc_principal()
        elif self.future_value is None:
            self.calc_future_value()
        elif self.payment is None:
            self.calc_payment()
        elif self.num_payments is None:
            self.calc_num_payments()
        elif self.int_rate is None:
            self.calc_int_rate()

    def calc_principal(self):
        _pay_portion = self.payment*(1-(1/((1+self.int_rate)**self.num_payments)))/self.int_rate
        _fv_portion = self.future_value/((1+self.int_rate)**self.num_payments)
        self.principal = _pay_portion + _fv_portion
        self.update_table()
        return self.principal

    def calc_future_value(self):
        _pay_portion = self.payment*(1-(1/((1+self.int_rate)**self.num_payments)))/self.int_rate
        self.future_value = (self.principal-_pay_portion)*((1+self.int_rate)**self.num_payments)
        self.update_table()
        return self.future_value

    def calc_payment(self):
        _num = self.principal - (self.future_value/((1+self.int_rate)**self.num_payments))
        _den = (1-(1/((1+self.int_rate)**self.num_payments)))/self.int_rate
        self.payment = round(_num/_den, 2)
        self.update_table()
        return self.payment

    def calc_num_payments(self):
        _num = math.log(((self.int_rate*self.principal)-self.payment)/((self.int_rate*self.future_value)-self.payment))
        _den = math.log(1/(1+self.int_rate))
        self.num_payments = int(math.ceil(_num/_den))
        self.update_table()
        return self.num_payments

    def calc_int_rate(self):
        #Need to generate the total cash flows to find int rate (irr)
        _cash_flows = [self.principal*-1] + [self.payment]*self.num_payments
        _cash_flows[-1] += self.future_value
        self.int_rate = irr(_cash_flows)
        self.update_table()
        return self.int_rate

    #Use this formula to update the amortization table lists
    def update_table(self):
        self.principal_paid = [0]*self.num_payments
        self.interest_paid = [0]*self.num_payments
        self.principal_remaining = [0]*self.num_payments
        self.payments = [0]*self.num_payments
        rem_prin = self.principal
        for i in xrange(self.num_payments):
            end_prin = rem_prin * (1 + self.int_rate)
            self.interest_paid[i] = (end_prin - rem_prin)
            if end_prin <= self.payment:
                self.principal_paid[i] = rem_prin
                self.payments[i] = end_prin
            else:
                self.principal_paid[i] = (self.payment - self.interest_paid[i])
                self.payments[i] = self.payment
            rem_prin = max(0, end_prin - self.payment)
            self.principal_remaining[i] = rem_prin

    def print_table(self):
        #Kinda from http://stackoverflow.com/questions/13873927/printing-evenly-spaced-table-from-a-list-with-a-for-loop
        #TODO: figure out how to center AND format as 2 decimals AND use commas
        headers = ['Time', 'Payment', 'Principal Paid','Interest Paid','Principal Remaining']
        max_lens = [4+len(str(max(i, key=lambda x: len(str(x))))) for i in ([self.num_payments, headers[0]],
                                                                            self.payments + [headers[1]],
                                                                            self.principal_paid + [headers[2]],
                                                                            self.interest_paid + [headers[3]],
                                                                            self.principal_remaining + [headers[4]])]
        for i in xrange(len(self.principal_paid)):
            if i == 0:
                print '|'.join('{0:^{width}}'.format(x, width=y) for x, y in zip(headers, max_lens))
                print '-'*(sum(max_lens)+4)
            row = [i,
                   round(self.payments[i], 2),
                   round(self.principal_paid[i], 2),
                   round(self.interest_paid[i], 2),
                   round(self.principal_remaining[i], 2)]
            print '|'.join('{0:^{width}}'.format(x, width=y) for x, y in zip(row, max_lens))


class Bond(object):
    """Bond(length, par_value, coupon_rate,
    num_annual_coupons=2, price=None, ytm=None)

Creates a Bond object. A bond pays the bondholder periodic coupons equal to
a specific percentage of the bond's par value, and when the bond expires it
pays the bondholder the final coupon payment plus the par value. In practice,
coupon payments are typically twice per year. Create a bond and view its
attribute cash_flows to see the cash flow pattern.

This class takes the following arguments:

length: length of the bond, in years
par_value: par_value of the bond (value of the final payment, excluding coupon)
coupon_rate: percent of par value paid in coupons, expressed annually
num_annual_coupons: how many coupons are paid annually, defaults to 2
***The following two arguments are optional, and only one may be specified initially:
price: current market price of the bond
ytm: current yied to maturity of the bond, expressed annually

The attribute price may be accessed with name.price() and changed to new_price with name.price(new_price)
The attribute ytm may be accessed with name.ytm() and changed to new_ytm name.ytm(new_ytm)
Changing either of these variables will calculate the other, as well as update the durations.

Upon initialization, the following attributes are generated:
cash_flows - a list of the cash flows generated by this bond (time between payments = 1/num_annual_coupons)
macD - Macaulay duration of the bond (price and ytm must be specified)
modD - Modified duration of the bond (price and ytm must be specified)

Use the function info() to quickly view the bond's attributes.
"""

    def __init__(self, length, par_value, coupon_rate, num_annual_coupons=2, price=None, ytm=None):
        if price and ytm:
            raise Exception('Only one of "price" and "ytm" may be defined')
        #Assign variables
        self.length = length
        self.par_value = par_value
        self.coupon_rate = coupon_rate
        self.num_annual_coupons = num_annual_coupons
        self.macD = None
        self.modD = None
        self.convexity = None

        #Generate cash flows
        self.cash_flows = [par_value*coupon_rate/num_annual_coupons]*(length*num_annual_coupons)
        self.cash_flows[-1] += par_value

        #Assign either price of ytm. Durations will also be calculated.
        if price != None:
            self.price(price)
        elif ytm != None:
            self.ytm(ytm)

    def ytm(self, new_ytm=None):
        if not new_ytm:
            #User just wants to see or use the current ytm
            return self._ytm
        self._ytm = new_ytm

        #Need to get new price
        dt = 1.0/self.num_annual_coupons
        self._price = pv(self.cash_flows, self._ytm, dt)

        #Need to get new durations/convexity
        self.macD = macD(self.cash_flows, self._ytm, 1.0/self.num_annual_coupons)
        self.modD = modD(self.cash_flows, self._ytm, 1.0/self.num_annual_coupons)
        self.convexity = convexity(self.cash_flows, self._ytm, 1.0/self.num_annual_coupons)

    def price(self, new_price=None):
        if not new_price:
            #User just want to see or use current price
            return self._price
        self._price = new_price

        #Need to get new ytm
        pmts = [self._price*-1]+self.cash_flows
        dt = 1.0/self.num_annual_coupons
        self._ytm = irr(pmts, dt)

        #Get new durations
        self.macD = macD(self.cash_flows, self._ytm, 1.0/self.num_annual_coupons)
        self.modD = modD(self.cash_flows, self._ytm, 1.0/self.num_annual_coupons)
        self.convexity = convexity(self.cash_flows, self._ytm, 1.0/self.num_annual_coupons)

    def info(self):
        print 'Time to Expiry: %s' % self.length
        print 'Par Value:      $%s' % self.par_value
        print 'Coupon Rate:    %s percent' % (self.coupon_rate*100)
        print 'Annual Coupons: %s' % self.num_annual_coupons
        print 'Price:          $%s' % (round(self._price, 2))
        print 'YTM:            %s percent' % (round(self._ytm*100, 2))
        print 'macD:           %s' % self.macD
        print 'modD:           %s' % self.modD
        print 'convexity:      %s' % self.convexity


class Stock(object):
    """Stock(S0, sigma)

Creates a stock object. Takes the following arguments:

S0: Current stock price. Must be an integer or float.
sigma: Volatility of the stock. Must be a float."""
    def __init__(self, S0, sigma):
        if not isinstance(S0, int) and not isinstance(S0, float):
            raise Exception('S0 must be either an integer or a float.')
        if not isinstance(sigma, float):
            raise Exception('sigma must be a float.')
        self.S0 = S0
        self.sigma = sigma

    def info(self):
        print 'Price:      $%s' % self.S0
        print 'Volatility: %s percent' % round(self.sigma*100, 2)


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


#------------------------------------------------------------------------------------
#End
#------------------------------------------------------------------------------------

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
    print my_option.calc_price(num_obs=1, num_sims=50)
    my_option = AsianMCLogNormConstVol(S=S, sigma=sigma, K=K, T=T, r=r, q=q, call=call, geometric=geometric)
    print my_option.calc_price(num_obs=10, num_sims=1000)
    my_option = LookbackFixedMCLogNormConstVol(S=S, sigma=sigma, K=K, T=T, r=r, q=q, call=call)
    print my_option.calc_price(num_obs=1, num_sims=50)
    my_option = LookbackFloatingMCLogNormConstVol(S=S, sigma=sigma, T=T, r=r, q=q, call=call)
    print my_option.calc_price(num_obs=1, num_sims=50)