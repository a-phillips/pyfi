import math
import copy

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


def _check_option_error(S, K, T, n, r, q=0, ex=None, call=True, geom=True):
    if not isinstance(S, Stock):
        raise Exception('Invalid type for S. Must be a Stock object')
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
    if ((r-q)*float(T)/n) >= S.sigma*((float(T)/n)**.5):
        raise Exception('Invalid risk-free rate, need ((r-q)*(T/n)) < S.sigma*((T/n)**.5)')


#-------------------------------------------------------------------------------------
#Functions (pv, fv, irr, macD, modD, convexity)
#-------------------------------------------------------------------------------------

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

    #All formulas use the formula
    #principal = (payment)*(1-(1/((1+intrate)^n))/intrate) + FV/((1+intrate)^n)
    #Hopefully I didn't mess up my algebra
    #Int_rate found numerically

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

#Binomial Options - Regular----------------------------------------------------------

###CRR-------------------------------------------------------------------------------


class EuropeanCRR(object):
    """EuropeanCRR(S, K, T, n, r, q=0, call=True)

Creates a European option. European options give the option buyer the right
to buy (call) or sell (put) a stock at a future date for a specified strike
price K. The payout is max(ST-K, 0) for a call and max(K-ST, 0) for a put,
where ST is the final sock price.

This class takes the following arguments:

S: underlying stock, must be a Stock object
K: strike price of the option
T: length of the life of the option (in annual terms)
n: number of periods in the tree (time step between periods = T/n)
r: annual risk-free rate, compounded continuously
q: annual continuous dividend rate, defaults to 0
call: boolean indicating if the option is a call or put option, defaults to True

If any of the parameters change, run [object name].calc_price() to generate the new price.
"""
    def __init__(self, S, K, T, n, r, q=0.0, call=True):
        _check_option_error(S, K, T, n, r, q=q, call=call)
        self.S = S
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
        u = math.exp(self.S.sigma*(self.dt**.5))
        d = 1/u
        for t in xrange(self.n+1):
            prices = []
            for num_u in xrange(t+1):
                prices.append(self.S.S0*(u**num_u)*(d**(t-num_u)))
            stock_tree.append(prices)
        self.stock_tree = stock_tree
        return stock_tree

    def calc_price(self):
        self.stock_tree = self.make_tree()
        price_tree = copy.deepcopy(self.stock_tree)
        u = math.exp(self.S.sigma*(self.dt**.5))
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
    """AmericanCRR(S, K, T, n, r, q=0, call=True)

Creates an American option. An American option gives the option buyer
the right to buy (call) or sell (put) a stock at a specified strike price.
The buyer may exercise this option before expiry at any time. Whenever the
option is exercised, the payout is max(St - K, 0) for calls and
max(K - St, 0) for puts, where St is the stock price at time t.

This class takes the following arguments:

S: underlying stock, must be a Stock object
K: strike price of the option
T: length of the life of the option (in annual terms)
n: number of periods in the tree (time step between periods = T/n)
r: annual risk-free rate, compounded continuously
q: annual continuous dividend rate, defaults to 0
call: boolean indicating if the option is a call or put option, defaults to True

If any of the parameters change, run [object name].calc_price() to generate the new price.
"""
    def __init__(self, S, K, T, n, r, q=0.0, call=True):
        _check_option_error(S, K, T, n, r, q=q, call=call)
        self.S = S
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
        u = math.exp(self.S.sigma*(self.dt**.5))
        d = 1/u
        for t in xrange(self.n+1):
            prices = []
            for num_u in xrange(t+1):
                prices.append(self.S.S0*(u**num_u)*(d**(t-num_u)))
            stock_tree.append(prices)
        self.stock_tree = stock_tree
        return stock_tree

    def calc_price(self):
        self.stock_tree = self.make_tree()
        price_tree = copy.deepcopy(self.stock_tree)
        u = math.exp(self.S.sigma*(self.dt**.5))
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
    """BermudanCRR(S, K, T, n, r, q=0, ex=[], call=True)

Creates an Bermudan option. A Bermudan option gives the option buyer
the right to buy (call) or sell (put) a stock at a specified strike price.
The buyer may exercise this option at certain points in time. Whenever the
option is exercised, the payout is max(St - K, 0) for calls and
max(K - St, 0) for puts, where St is the stock price at time t.

This class takes the following arguments:

S: underlying stock, must be a Stock object
K: strike price of the option
T: length of the life of the option (in annual terms)
n: number of periods in the tree (time step between periods = T/n)
r: annual risk-free rate, compounded continuously
q: annual continuous dividend rate, defaults to 0
ex: a list containing the periods at which exercising is allowed.
call: boolean indicating if the option is a call or put option, defaults to True

If any of the parameters change, run [object name].calc_price() to generate the new price.
"""
    def __init__(self, S, K, T, n, r, q=0.0, ex=None, call=True):
        _check_option_error(S, K, T, n, r, q=q, ex=ex, call=call)
        self.S = S
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
        u = math.exp(self.S.sigma*(self.dt**.5))
        d = 1/u
        for t in xrange(self.n+1):
            prices = []
            for num_u in xrange(t+1):
                prices.append(self.S.S0*(u**num_u)*(d**(t-num_u)))
            stock_tree.append(prices)
        self.stock_tree = stock_tree
        return stock_tree

    def calc_price(self):
        self.stock_tree = self.make_tree()
        price_tree = copy.deepcopy(self.stock_tree)
        u = math.exp(self.S.sigma*(self.dt**.5))
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
    """EuropeanJR(S, K, T, n, r, q=0, call=True)

Creates a European option. European options give the option buyer the right
to buy (call) or sell (put) a stock at a future date for a specified strike
price K. The payout is max(ST-K, 0) for a call and max(K-ST, 0) for a put,
where ST is the final sock price.

This class takes the following arguments:

S: underlying stock, must be a Stock object
K: strike price of the option
T: length of the life of the option (in annual terms)
n: number of periods in the tree (time step between periods = T/n)
r: annual risk-free rate, compounded continuously
q: annual continuous dividend rate, defaults to 0
call: boolean indicating if the option is a call or put option, defaults to True

If any of the parameters change, run [object name].calc_price() to generate the new price.
"""
    def __init__(self, S, K, T, n, r, q=0.0, call=True):
        _check_option_error(S, K, T, n, r, q=q, call=call)
        self.S = S
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
        u = math.exp((self.r - self.q - (.5*(self.S.sigma**2)))*self.dt + self.S.sigma*(self.dt**.5))
        d = math.exp((self.r - self.q - (.5*(self.S.sigma**2)))*self.dt - self.S.sigma*(self.dt**.5))
        for t in xrange(self.n+1):
            prices = []
            for num_u in xrange(t+1):
                prices.append(self.S.S0*(u**num_u)*(d**(t-num_u)))
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
    """AmericanJR(S, K, T, n, r, q=0, call=True)

Creates an American option. An American option gives the option buyer
the right to buy (call) or sell (put) a stock at a specified strike price.
The buyer may exercise this option before expiry at any time. Whenever the
option is exercised, the payout is max(St - K, 0) for calls and
max(K - St, 0) for puts, where St is the stock price at time t.

This class takes the following arguments:

S: underlying stock, must be a Stock object
K: strike price of the option
T: length of the life of the option (in annual terms)
n: number of periods in the tree (time step between periods = T/n)
r: annual risk-free rate, compounded continuously
q: annual continuous dividend rate, defaults to 0
call: boolean indicating if the option is a call or put option, defaults to True

If any of the parameters change, run [object name].calc_price() to generate the new price.
"""
    def __init__(self, S, K, T, n, r, q=0.0, call=True):
        _check_option_error(S, K, T, n, r, q=q, call=call)
        self.S = S
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
        u = math.exp((self.r - self.q - (.5*(self.S.sigma**2)))*self.dt + self.S.sigma*(self.dt**.5))
        d = math.exp((self.r - self.q - (.5*(self.S.sigma**2)))*self.dt - self.S.sigma*(self.dt**.5))
        for t in xrange(self.n+1):
            prices = []
            for num_u in xrange(t+1):
                prices.append(self.S.S0*(u**num_u)*(d**(t-num_u)))
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
    """BermudanJR(S, K, T, n, r, q=0, ex=[], call=True)

Creates an Bermudan option. A Bermudan option gives the option buyer
the right to buy (call) or sell (put) a stock at a specified strike price.
The buyer may exercise this option at certain points in time. Whenever the
option is exercised, the payout is max(St - K, 0) for calls and
max(K - St, 0) for puts, where St is the stock price at time t.

This class takes the following arguments:

S: underlying stock, must be a Stock object
K: strike price of the option
T: length of the life of the option (in annual terms)
n: number of periods in the tree (time step between periods = T/n)
r: annual risk-free rate, compounded continuously
q: annual continuous dividend rate, defaults to 0
ex: a list containing the periods at which exercising is allowed.
call: boolean indicating if the option is a call or put option, defaults to True

If any of the parameters change, run [object name].calc_price() to generate the new price.
"""
    def __init__(self, S, K, T, n, r, q=0.0, ex=None, call=True):
        _check_option_error(S, K, T, n, r, q=q, ex=ex, call=call)
        self.S = S
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
        u = math.exp((self.r - self.q - (.5*(self.S.sigma**2)))*self.dt + self.S.sigma*(self.dt**.5))
        d = math.exp((self.r - self.q - (.5*(self.S.sigma**2)))*self.dt - self.S.sigma*(self.dt**.5))
        for t in xrange(self.n+1):
            prices = []
            for num_u in xrange(t+1):
                prices.append(self.S.S0*(u**num_u)*(d**(t-num_u)))
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
    """EuropeanTian(S, K, T, n, r, q=0, call=True)

Creates a European option. European options give the option buyer the right
to buy (call) or sell (put) a stock at a future date for a specified strike
price K. The payout is max(ST-K, 0) for a call and max(K-ST, 0) for a put,
where ST is the final sock price.

This class takes the following arguments:

S: underlying stock, must be a Stock object
K: strike price of the option
T: length of the life of the option (in annual terms)
n: number of periods in the tree (time step between periods = T/n)
r: annual risk-free rate, compounded continuously
q: annual continuous dividend rate, defaults to 0
call: boolean indicating if the option is a call or put option, defaults to True

If any of the parameters change, run [object name].calc_price() to generate the new price.
"""
    def __init__(self, S, K, T, n, r, q=0.0, call=True):
        _check_option_error(S, K, T, n, r, q=q, call=call)
        self.S = S
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
        V = math.exp((self.S.sigma**2)*self.dt)
        u = .5 * math.exp((self.r - self.q) * self.dt) * V * (V + 1 + ((V**2) + (2*V) - 3)**.5)
        d = .5 * math.exp((self.r - self.q) * self.dt) * V * (V + 1 - ((V**2) + (2*V) - 3)**.5)
        for t in xrange(self.n+1):
            prices = []
            for num_u in xrange(t+1):
                prices.append(self.S.S0*(u**num_u)*(d**(t-num_u)))
            stock_tree.append(prices)
        self.stock_tree = stock_tree
        return stock_tree

    def calc_price(self):
        self.stock_tree = self.make_tree()
        price_tree = copy.deepcopy(self.stock_tree)
        V = math.exp((self.S.sigma**2)*self.dt)
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
    """AmericanTian(S, K, T, n, r, q=0, call=True)

Creates an American option. An American option gives the option buyer
the right to buy (call) or sell (put) a stock at a specified strike price.
The buyer may exercise this option before expiry at any time. Whenever the
option is exercised, the payout is max(St - K, 0) for calls and
max(K - St, 0) for puts, where St is the stock price at time t.

This class takes the following arguments:

S: underlying stock, must be a Stock object
K: strike price of the option
T: length of the life of the option (in annual terms)
n: number of periods in the tree (time step between periods = T/n)
r: annual risk-free rate, compounded continuously
q: annual continuous dividend rate, defaults to 0
call: boolean indicating if the option is a call or put option, defaults to True

If any of the parameters change, run [object name].calc_price() to generate the new price.
"""
    def __init__(self, S, K, T, n, r, q=0.0, call=True):
        _check_option_error(S, K, T, n, r, q=q, call=call)
        self.S = S
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
        V = math.exp((self.S.sigma**2)*self.dt)
        u = .5 * math.exp((self.r - self.q) * self.dt) * V * (V + 1 + ((V**2) + (2*V) - 3)**.5)
        d = .5 * math.exp((self.r - self.q) * self.dt) * V * (V + 1 - ((V**2) + (2*V) - 3)**.5)
        for t in xrange(self.n+1):
            prices = []
            for num_u in xrange(t+1):
                prices.append(self.S.S0*(u**num_u)*(d**(t-num_u)))
            stock_tree.append(prices)
        self.stock_tree = stock_tree
        return stock_tree

    def calc_price(self):
        self.stock_tree = self.make_tree()
        price_tree = copy.deepcopy(self.stock_tree)
        V = math.exp((self.S.sigma**2)*self.dt)
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
    """BermudanTian(S, K, T, n, r, q=0, ex=[], call=True)

Creates an Bermudan option. A Bermudan option gives the option buyer
the right to buy (call) or sell (put) a stock at a specified strike price.
The buyer may exercise this option at certain points in time. Whenever the
option is exercised, the payout is max(St - K, 0) for calls and
max(K - St, 0) for puts, where St is the stock price at time t.

This class takes the following arguments:

S: underlying stock, must be a Stock object
K: strike price of the option
T: length of the life of the option (in annual terms)
n: number of periods in the tree (time step between periods = T/n)
r: annual risk-free rate, compounded continuously
q: annual continuous dividend rate, defaults to 0
ex: a list containing the periods at which exercising is allowed.
call: boolean indicating if the option is a call or put option, defaults to True

If any of the parameters change, run [object name].calc_price() to generate the new price.
"""
    def __init__(self, S, K, T, n, r, q=0.0, ex=None, call=True):
        _check_option_error(S, K, T, n, r, q=q, ex=ex, call=call)
        self.S = S
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
        V = math.exp((self.S.sigma**2)*self.dt)
        u = .5 * math.exp((self.r - self.q) * self.dt) * V * (V + 1 + ((V**2) + (2*V) - 3)**.5)
        d = .5 * math.exp((self.r - self.q) * self.dt) * V * (V + 1 - ((V**2) + (2*V) - 3)**.5)
        for t in xrange(self.n+1):
            prices = []
            for num_u in xrange(t+1):
                prices.append(self.S.S0*(u**num_u)*(d**(t-num_u)))
            stock_tree.append(prices)
        self.stock_tree = stock_tree
        return stock_tree

    def calc_price(self):
        self.stock_tree = self.make_tree()
        price_tree = copy.deepcopy(self.stock_tree)
        V = math.exp((self.S.sigma**2)*self.dt)
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


#Binomial Options - Binary----------------------------------------------------------

###CRR-------------------------------------------------------------------------------


class EuropeanBinaryCRR(object):
    """EuropeanCRR(S, K, T, n, r, q=0, call=True)

Creates a European option. European options give the option buyer the right
to buy (call) or sell (put) a stock at a future date for a specified strike
price K. The payout is max(ST-K, 0) for a call and max(K-ST, 0) for a put,
where ST is the final sock price.

This class takes the following arguments:

S: underlying stock, must be a Stock object
K: strike price of the option
T: length of the life of the option (in annual terms)
n: number of periods in the tree (time step between periods = T/n)
r: annual risk-free rate, compounded continuously
q: annual continuous dividend rate, defaults to 0
call: boolean indicating if the option is a call or put option, defaults to True

If any of the parameters change, run [object name].calc_price() to generate the new price.
"""
    def __init__(self, S, K, T, n, r, q=0.0, call=True):
        _check_option_error(S, K, T, n, r, q=q, call=call)
        self.S = S
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
        u = math.exp(self.S.sigma*(self.dt**.5))
        d = 1/u
        for t in xrange(self.n+1):
            prices = []
            for num_u in xrange(t+1):
                prices.append(self.S.S0*(u**num_u)*(d**(t-num_u)))
            stock_tree.append(prices)
        self.stock_tree = stock_tree
        return stock_tree

    def calc_price(self):
        self.stock_tree = self.make_tree()
        price_tree = copy.deepcopy(self.stock_tree)
        u = math.exp(self.S.sigma*(self.dt**.5))
        d = 1/u
        pu = (math.exp((self.r - self.q)*self.dt) - d)/(u - d)
        pd = 1 - pu
        for i, price in enumerate(price_tree[-1]):
            if self.call and price - self.K > 0:
                price_tree[-1][i] = 1
            elif not self.call and self.K - price > 0:
                price_tree[-1][i] = 1
            else:
                price_tree[-1][i] = 0
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
    """AmericanCRR(S, K, T, n, r, q=0, call=True)

Creates an American option. An American option gives the option buyer
the right to buy (call) or sell (put) a stock at a specified strike price.
The buyer may exercise this option before expiry at any time. Whenever the
option is exercised, the payout is max(St - K, 0) for calls and
max(K - St, 0) for puts, where St is the stock price at time t.

This class takes the following arguments:

S: underlying stock, must be a Stock object
K: strike price of the option
T: length of the life of the option (in annual terms)
n: number of periods in the tree (time step between periods = T/n)
r: annual risk-free rate, compounded continuously
q: annual continuous dividend rate, defaults to 0
call: boolean indicating if the option is a call or put option, defaults to True

If any of the parameters change, run [object name].calc_price() to generate the new price.
"""
    def __init__(self, S, K, T, n, r, q=0.0, call=True):
        _check_option_error(S, K, T, n, r, q=q, call=call)
        self.S = S
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
        u = math.exp(self.S.sigma*(self.dt**.5))
        d = 1/u
        for t in xrange(self.n+1):
            prices = []
            for num_u in xrange(t+1):
                prices.append(self.S.S0*(u**num_u)*(d**(t-num_u)))
            stock_tree.append(prices)
        self.stock_tree = stock_tree
        return stock_tree

    def calc_price(self):
        self.stock_tree = self.make_tree()
        price_tree = copy.deepcopy(self.stock_tree)
        u = math.exp(self.S.sigma*(self.dt**.5))
        d = 1/u
        pu = (math.exp((self.r - self.q)*self.dt) - d)/(u - d)
        pd = 1 - pu
        for i, price in enumerate(price_tree[-1]):
            if self.call and price - self.K > 0:
                price_tree[-1][i] = 1
            elif not self.call and self.K - price > 0:
                price_tree[-1][i] = 1
            else:
                price_tree[-1][i] = 0
        for t in xrange(self.n-1, -1, -1):
            for i in xrange(t+1):
                option_value = (pd*price_tree[t+1][i])+(pu*price_tree[t+1][i+1])
                if self.call and price_tree[t][i] - self.K > 0:
                    exercise_value = 1
                elif not self.call and self.K - price_tree[t][i] > 0:
                    exercise_value = 1
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
    """BermudanCRR(S, K, T, n, r, q=0, ex=[], call=True)

Creates an Bermudan option. A Bermudan option gives the option buyer
the right to buy (call) or sell (put) a stock at a specified strike price.
The buyer may exercise this option at certain points in time. Whenever the
option is exercised, the payout is max(St - K, 0) for calls and
max(K - St, 0) for puts, where St is the stock price at time t.

This class takes the following arguments:

S: underlying stock, must be a Stock object
K: strike price of the option
T: length of the life of the option (in annual terms)
n: number of periods in the tree (time step between periods = T/n)
r: annual risk-free rate, compounded continuously
q: annual continuous dividend rate, defaults to 0
ex: a list containing the periods at which exercising is allowed.
call: boolean indicating if the option is a call or put option, defaults to True

If any of the parameters change, run [object name].calc_price() to generate the new price.
"""
    def __init__(self, S, K, T, n, r, q=0.0, ex=None, call=True):
        _check_option_error(S, K, T, n, r, q=q, ex=ex, call=call)
        self.S = S
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
        u = math.exp(self.S.sigma*(self.dt**.5))
        d = 1/u
        for t in xrange(self.n+1):
            prices = []
            for num_u in xrange(t+1):
                prices.append(self.S.S0*(u**num_u)*(d**(t-num_u)))
            stock_tree.append(prices)
        self.stock_tree = stock_tree
        return stock_tree

    def calc_price(self):
        self.stock_tree = self.make_tree()
        price_tree = copy.deepcopy(self.stock_tree)
        u = math.exp(self.S.sigma*(self.dt**.5))
        d = 1/u
        pu = (math.exp((self.r - self.q)*self.dt) - d)/(u - d)
        pd = 1 - pu
        #Find exercise value at final nodes
        for i, price in enumerate(price_tree[-1]):
            if self.call and price - self.K > 0:
                price_tree[-1][i] = 1
            elif not self.call and self.K - price > 0:
                price_tree[-1][i] = 1
            else:
                price_tree[-1][i] = 0
        #Move backwards through the tree to determine option values
        for t in xrange(self.n-1, -1, -1):
            for i in xrange(t+1):
                option_value = (pd*price_tree[t+1][i])+(pu*price_tree[t+1][i+1])
                #Use appropriate payout function
                if self.call and price_tree[t][i] - self.K > 0:
                    exercise_value = 1
                elif not self.call and self.K - price_tree[t][i] > 0:
                    exercise_value = 1
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
    """EuropeanJR(S, K, T, n, r, q=0, call=True)

Creates a European option. European options give the option buyer the right
to buy (call) or sell (put) a stock at a future date for a specified strike
price K. The payout is max(ST-K, 0) for a call and max(K-ST, 0) for a put,
where ST is the final sock price.

This class takes the following arguments:

S: underlying stock, must be a Stock object
K: strike price of the option
T: length of the life of the option (in annual terms)
n: number of periods in the tree (time step between periods = T/n)
r: annual risk-free rate, compounded continuously
q: annual continuous dividend rate, defaults to 0
call: boolean indicating if the option is a call or put option, defaults to True

If any of the parameters change, run [object name].calc_price() to generate the new price.
"""
    def __init__(self, S, K, T, n, r, q=0.0, call=True):
        _check_option_error(S, K, T, n, r, q=q, call=call)
        self.S = S
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
        u = math.exp((self.r - self.q - (.5*(self.S.sigma**2)))*self.dt + self.S.sigma*(self.dt**.5))
        d = math.exp((self.r - self.q - (.5*(self.S.sigma**2)))*self.dt - self.S.sigma*(self.dt**.5))
        for t in xrange(self.n+1):
            prices = []
            for num_u in xrange(t+1):
                prices.append(self.S.S0*(u**num_u)*(d**(t-num_u)))
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
                price_tree[-1][i] = 1
            elif not self.call and self.K - price > 0:
                price_tree[-1][i] = 1
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
    """AmericanJR(S, K, T, n, r, q=0, call=True)

Creates an American option. An American option gives the option buyer
the right to buy (call) or sell (put) a stock at a specified strike price.
The buyer may exercise this option before expiry at any time. Whenever the
option is exercised, the payout is max(St - K, 0) for calls and
max(K - St, 0) for puts, where St is the stock price at time t.

This class takes the following arguments:

S: underlying stock, must be a Stock object
K: strike price of the option
T: length of the life of the option (in annual terms)
n: number of periods in the tree (time step between periods = T/n)
r: annual risk-free rate, compounded continuously
q: annual continuous dividend rate, defaults to 0
call: boolean indicating if the option is a call or put option, defaults to True

If any of the parameters change, run [object name].calc_price() to generate the new price.
"""
    def __init__(self, S, K, T, n, r, q=0.0, call=True):
        _check_option_error(S, K, T, n, r, q=q, call=call)
        self.S = S
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
        u = math.exp((self.r - self.q - (.5*(self.S.sigma**2)))*self.dt + self.S.sigma*(self.dt**.5))
        d = math.exp((self.r - self.q - (.5*(self.S.sigma**2)))*self.dt - self.S.sigma*(self.dt**.5))
        for t in xrange(self.n+1):
            prices = []
            for num_u in xrange(t+1):
                prices.append(self.S.S0*(u**num_u)*(d**(t-num_u)))
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
                price_tree[-1][i] = 1
            elif not self.call and self.K - price > 0:
                price_tree[-1][i] = 1
            else:
                price_tree[-1][i] = 0
        #Move backwards through the tree to determine option values
        for t in xrange(self.n-1, -1, -1):
            for i in xrange(t+1):
                option_value = (pd*price_tree[t+1][i])+(pu*price_tree[t+1][i+1])
                #Use appropriate payout function
                if self.call and price_tree[t][i] - self.K > 0:
                    exercise_value = 1
                elif not self.call and self.K - price_tree[t][i] > 0:
                    exercise_value = 1
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
    """BermudanJR(S, K, T, n, r, q=0, ex=[], call=True)

Creates an Bermudan option. A Bermudan option gives the option buyer
the right to buy (call) or sell (put) a stock at a specified strike price.
The buyer may exercise this option at certain points in time. Whenever the
option is exercised, the payout is max(St - K, 0) for calls and
max(K - St, 0) for puts, where St is the stock price at time t.

This class takes the following arguments:

S: underlying stock, must be a Stock object
K: strike price of the option
T: length of the life of the option (in annual terms)
n: number of periods in the tree (time step between periods = T/n)
r: annual risk-free rate, compounded continuously
q: annual continuous dividend rate, defaults to 0
ex: a list containing the periods at which exercising is allowed.
call: boolean indicating if the option is a call or put option, defaults to True

If any of the parameters change, run [object name].calc_price() to generate the new price.
"""
    def __init__(self, S, K, T, n, r, q=0.0, ex=None, call=True):
        _check_option_error(S, K, T, n, r, q=q, ex=ex, call=call)
        self.S = S
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
        u = math.exp((self.r - self.q - (.5*(self.S.sigma**2)))*self.dt + self.S.sigma*(self.dt**.5))
        d = math.exp((self.r - self.q - (.5*(self.S.sigma**2)))*self.dt - self.S.sigma*(self.dt**.5))
        for t in xrange(self.n+1):
            prices = []
            for num_u in xrange(t+1):
                prices.append(self.S.S0*(u**num_u)*(d**(t-num_u)))
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
                price_tree[-1][i] = 1
            elif not self.call and self.K - price > 0:
                price_tree[-1][i] = 1
            else:
                price_tree[-1][i] = 0
        #Move backwards through the tree to determine option values
        for t in xrange(self.n-1, -1, -1):
            for i in xrange(t+1):
                option_value = (pd*price_tree[t+1][i])+(pu*price_tree[t+1][i+1])
                #Use appropriate payout function
                if self.call and price_tree[t][i] - self.K > 0:
                    exercise_value = 1
                elif not self.call and self.K - price_tree[t][i] > 0:
                    exercise_value = 1
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
    """EuropeanTian(S, K, T, n, r, q=0, call=True)

Creates a European option. European options give the option buyer the right
to buy (call) or sell (put) a stock at a future date for a specified strike
price K. The payout is max(ST-K, 0) for a call and max(K-ST, 0) for a put,
where ST is the final sock price.

This class takes the following arguments:

S: underlying stock, must be a Stock object
K: strike price of the option
T: length of the life of the option (in annual terms)
n: number of periods in the tree (time step between periods = T/n)
r: annual risk-free rate, compounded continuously
q: annual continuous dividend rate, defaults to 0
call: boolean indicating if the option is a call or put option, defaults to True

If any of the parameters change, run [object name].calc_price() to generate the new price.
"""
    def __init__(self, S, K, T, n, r, q=0.0, call=True):
        _check_option_error(S, K, T, n, r, q=q, call=call)
        self.S = S
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
        V = math.exp((self.S.sigma**2)*self.dt)
        u = .5 * math.exp((self.r - self.q) * self.dt) * V * (V + 1 + ((V**2) + (2*V) - 3)**.5)
        d = .5 * math.exp((self.r - self.q) * self.dt) * V * (V + 1 - ((V**2) + (2*V) - 3)**.5)
        for t in xrange(self.n+1):
            prices = []
            for num_u in xrange(t+1):
                prices.append(self.S.S0*(u**num_u)*(d**(t-num_u)))
            stock_tree.append(prices)
        self.stock_tree = stock_tree
        return stock_tree

    def calc_price(self):
        self.stock_tree = self.make_tree()
        price_tree = copy.deepcopy(self.stock_tree)
        V = math.exp((self.S.sigma**2)*self.dt)
        u = .5 * math.exp((self.r - self.q) * self.dt) * V * (V + 1 + ((V**2) + (2*V) - 3)**.5)
        d = .5 * math.exp((self.r - self.q) * self.dt) * V * (V + 1 - ((V**2) + (2*V) - 3)**.5)
        pu = (math.exp((self.r - self.q)*self.dt) - d)/(u - d)
        pd = 1 - pu
        #Find the exercise value at final nodes
        for i, price in enumerate(price_tree[-1]):
            if self.call and price - self.K > 0:
                price_tree[-1][i] = 1
            elif not self.call and self.K - price > 0:
                price_tree[-1][i] = 1
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
    """AmericanTian(S, K, T, n, r, q=0, call=True)

Creates an American option. An American option gives the option buyer
the right to buy (call) or sell (put) a stock at a specified strike price.
The buyer may exercise this option before expiry at any time. Whenever the
option is exercised, the payout is max(St - K, 0) for calls and
max(K - St, 0) for puts, where St is the stock price at time t.

This class takes the following arguments:

S: underlying stock, must be a Stock object
K: strike price of the option
T: length of the life of the option (in annual terms)
n: number of periods in the tree (time step between periods = T/n)
r: annual risk-free rate, compounded continuously
q: annual continuous dividend rate, defaults to 0
call: boolean indicating if the option is a call or put option, defaults to True

If any of the parameters change, run [object name].calc_price() to generate the new price.
"""
    def __init__(self, S, K, T, n, r, q=0.0, call=True):
        _check_option_error(S, K, T, n, r, q=q, call=call)
        self.S = S
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
        V = math.exp((self.S.sigma**2)*self.dt)
        u = .5 * math.exp((self.r - self.q) * self.dt) * V * (V + 1 + ((V**2) + (2*V) - 3)**.5)
        d = .5 * math.exp((self.r - self.q) * self.dt) * V * (V + 1 - ((V**2) + (2*V) - 3)**.5)
        for t in xrange(self.n+1):
            prices = []
            for num_u in xrange(t+1):
                prices.append(self.S.S0*(u**num_u)*(d**(t-num_u)))
            stock_tree.append(prices)
        self.stock_tree = stock_tree
        return stock_tree

    def calc_price(self):
        self.stock_tree = self.make_tree()
        price_tree = copy.deepcopy(self.stock_tree)
        V = math.exp((self.S.sigma**2)*self.dt)
        u = .5 * math.exp((self.r - self.q) * self.dt) * V * (V + 1 + ((V**2) + (2*V) - 3)**.5)
        d = .5 * math.exp((self.r - self.q) * self.dt) * V * (V + 1 - ((V**2) + (2*V) - 3)**.5)
        pu = (math.exp((self.r - self.q)*self.dt) - d)/(u - d)
        pd = 1 - pu
        #Find exercise value at final nodes
        for i, price in enumerate(price_tree[-1]):
            if self.call and price - self.K > 0:
                price_tree[-1][i] = 1
            elif not self.call and self.K - price > 0:
                price_tree[-1][i] = 1
            else:
                price_tree[-1][i] = 0
        #Move backwards through the tree to determine option values
        for t in xrange(self.n-1, -1, -1):
            for i in xrange(t+1):
                option_value = (pd*price_tree[t+1][i])+(pu*price_tree[t+1][i+1])
                #Use appropriate payout function
                if self.call and price_tree[t][i] - self.K > 0:
                    exercise_value = 1
                elif not self.call and self.K - price_tree[t][i] > 0:
                    exercise_value = 1
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
    """BermudanTian(S, K, T, n, r, q=0, ex=[], call=True)

Creates an Bermudan option. A Bermudan option gives the option buyer
the right to buy (call) or sell (put) a stock at a specified strike price.
The buyer may exercise this option at certain points in time. Whenever the
option is exercised, the payout is max(St - K, 0) for calls and
max(K - St, 0) for puts, where St is the stock price at time t.

This class takes the following arguments:

S: underlying stock, must be a Stock object
K: strike price of the option
T: length of the life of the option (in annual terms)
n: number of periods in the tree (time step between periods = T/n)
r: annual risk-free rate, compounded continuously
q: annual continuous dividend rate, defaults to 0
ex: a list containing the periods at which exercising is allowed.
call: boolean indicating if the option is a call or put option, defaults to True

If any of the parameters change, run [object name].calc_price() to generate the new price.
"""
    def __init__(self, S, K, T, n, r, q=0.0, ex=None, call=True):
        _check_option_error(S, K, T, n, r, q=q, ex=ex, call=call)
        self.S = S
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
        V = math.exp((self.S.sigma**2)*self.dt)
        u = .5 * math.exp((self.r - self.q) * self.dt) * V * (V + 1 + ((V**2) + (2*V) - 3)**.5)
        d = .5 * math.exp((self.r - self.q) * self.dt) * V * (V + 1 - ((V**2) + (2*V) - 3)**.5)
        for t in xrange(self.n+1):
            prices = []
            for num_u in xrange(t+1):
                prices.append(self.S.S0*(u**num_u)*(d**(t-num_u)))
            stock_tree.append(prices)
        self.stock_tree = stock_tree
        return stock_tree

    def calc_price(self):
        self.stock_tree = self.make_tree()
        price_tree = copy.deepcopy(self.stock_tree)
        V = math.exp((self.S.sigma**2)*self.dt)
        u = .5 * math.exp((self.r - self.q) * self.dt) * V * (V + 1 + ((V**2) + (2*V) - 3)**.5)
        d = .5 * math.exp((self.r - self.q) * self.dt) * V * (V + 1 - ((V**2) + (2*V) - 3)**.5)
        pu = (math.exp((self.r - self.q)*self.dt) - d)/(u - d)
        pd = 1 - pu
        #Find exercise value at final nodes
        for i, price in enumerate(price_tree[-1]):
            if self.call and price - self.K > 0:
                price_tree[-1][i] = 1
            elif not self.call and self.K - price > 0:
                price_tree[-1][i] = 1
            else:
                price_tree[-1][i] = 0
        #Move backwards through the tree to determine option values
        for t in xrange(self.n-1, -1, -1):
            for i in xrange(t+1):
                option_value = (pd*price_tree[t+1][i])+(pu*price_tree[t+1][i+1])
                #Use appropriate payout function
                if self.call and price_tree[t][i] - self.K > 0:
                    exercise_value = 1
                elif not self.call and self.K - price_tree[t][i] > 0:
                    exercise_value = 1
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




#------------------------------------------------------------------------------------
#End
#------------------------------------------------------------------------------------

if __name__ == '__main__':
    print 'European CRR \t', EuropeanCRR(S=Stock(50, .4),K=50,T=2,n=10,r=.03,q=0.0,call=True).price
    print 'European JR \t', EuropeanJR(S=Stock(50, .4),K=50,T=2,n=10,r=.03,q=0.0,call=True).price
    print 'European Tian \t', EuropeanTian(S=Stock(50, .4),K=50,T=2,n=10,r=.03,q=0.0,call=True).price
    print 'European Binary CRR \t', EuropeanBinaryCRR(S=Stock(50, .4),K=50,T=2,n=10,r=.03,q=0.0,call=True).price
    print 'European Binary JR \t', EuropeanBinaryJR(S=Stock(50, .4),K=50,T=2,n=10,r=.03,q=0.0,call=True).price
    print 'European Binary Tian \t', EuropeanBinaryTian(S=Stock(50, .4),K=50,T=2,n=10,r=.03,q=0.0,call=True).price
    print 'American CRR \t', AmericanCRR(S=Stock(50, .4),K=50,T=2,n=10,r=.03,q=0.0,call=True).price
    print 'American JR \t', AmericanJR(S=Stock(50, .4),K=50,T=2,n=10,r=.03,q=0.0,call=True).price
    print 'American Tian \t', AmericanTian(S=Stock(50, .4),K=50,T=2,n=10,r=.03,q=0.0,call=True).price
    print 'American Binary CRR \t', AmericanBinaryCRR(S=Stock(50, .4),K=50,T=2,n=10,r=.03,q=0.0,call=True).price
    print 'American Binary JR \t', AmericanBinaryJR(S=Stock(50, .4),K=50,T=2,n=10,r=.03,q=0.0,call=True).price
    print 'American Binary Tian \t', AmericanBinaryTian(S=Stock(50, .4),K=50,T=2,n=10,r=.03,q=0.0,call=True).price
    print 'Bermudan CRR \t', BermudanCRR(S=Stock(50, .4),K=50,T=2,n=10,r=.03,q=0.0,call=True,ex=[1,3]).price
    print 'Bermudan JR \t', BermudanJR(S=Stock(50, .4),K=50,T=2,n=10,r=.03,q=0.0,call=True,ex=[1,3]).price
    print 'Bermudan Tian \t', BermudanTian(S=Stock(50, .4),K=50,T=2,n=10,r=.03,q=0.0,call=True,ex=[1,3]).price
    print 'Bermudan Binary CRR \t', BermudanBinaryCRR(S=Stock(50, .4),K=50,T=2,n=10,r=.03,q=0.0,call=True,ex=[1,3]).price
    print 'Bermudan Binary JR \t', BermudanBinaryJR(S=Stock(50, .4),K=50,T=2,n=10,r=.03,q=0.0,call=True,ex=[1,3]).price
    print 'Bermudan Binary Tian \t', BermudanBinaryTian(S=Stock(50, .4),K=50,T=2,n=10,r=.03,q=0.0,call=True,ex=[1,3]).price

#------------------------------------------------------------------------------------
#Probably not using this code
#------------------------------------------------------------------------------------

#Asian options typically aren't priced using binomial trees
"""
class AsianFixed(object):


    def __init__(self, S, K, T, n, r, q=0, call=True, geom=False):
        _check_option_error(S, K, T, n, r, q=q, call=call, geom=geom)
        self.S = S
        self.K = K
        self.T = T
        self.n = n
        self.dt = float(T)/n
        self.r = r
        self.q = q
        self.call = call
        self.avg = ('Geometric'*geom)+('Arithmetic'*(not geom))
        self.price_tree = []
        self.stock_tree = self.make_tree()
        self.price = self.calc_price()

    def make_tree(self):
        stock_tree = []
        u = math.exp(self.S.sigma*(self.dt**.5))
        d = 1/u
        for t in xrange(self.n+1):
            prices = []
            for num_u in xrange(t+1):
                prices.append(round(self.S.S0*(u**num_u)*(d**(t-num_u)),2))
            stock_tree.append(prices)
        return stock_tree

    def get_paths(self):
        paths = []
        curr_path = []
        paths = self._iter_thru_tree(self.stock_tree, paths, curr_path, 0, 0)
        return paths

    def _iter_thru_tree(self, tree, paths, curr_path, t, i):
        if t == len(tree)-1:
            paths.append(curr_path + [tree[t][i+1]])
            paths.append(curr_path + [tree[t][i]])
            return paths
        else:
            curr_path += [tree[t][i]]
            paths = self._iter_thru_tree(tree, paths, curr_path, t+1, i)
            if t > 0:
                curr_path = curr_path[:t] + [tree[t][i+1]]
                paths = self._iter_thru_tree(tree, paths, curr_path, t+1, i+1)
        return paths

    def calc_price(self):
        self.stock_tree = self.make_tree()
        price_tree = copy.deepcopy(self.stock_tree)
        u = math.exp(self.S.sigma*(self.dt**.5))
        d = 1/u
        #pu and pd represent risk-neutral probabilities of up and down movement, respectively
        pu = (math.exp((self.r - self.q)*self.dt) - d)/(u - d)
        pd = 1 - pu
        #All paths arriving at the same end result occur with the same probability, so the
        #averages are arrived at by finding the average for each path according to the
        #specified method, adding all averages for each end price, then dividing that sum
        #by the total number of ways that path occurs. I realize that those counts will be
        #nCk, but I figured it was easier to just keep count while iterating through
        #the paths than calculating factorials.
        paths = self.get_paths()
        avgs = [0]*(self.n+1)
        counts = [0]*(self.n+1)
        for i, path in enumerate(paths):
            if self.avg == 'Arithmetic':
                final_avg = sum(path)/(self.n + 1)
            else:
                final_avg = 1
                for price in path:
                    final_avg *= price
                final_avg = (final_avg**(1.0/(self.n+1)))
            index = self.stock_tree[-1].index(path[-1])
            avgs[index] += final_avg
            counts[index] += 1
        #Find the final period payouts as difference between avg stock price and strike
        for i, num in enumerate(avgs):
            avgs[i] = float(num)/counts[i]
            if self.call:
                price_tree[-1][i] = max(avgs[i] - self.K, 0)
            else:
                price_tree[-1][i] = max(self.K - avgs[i], 0)
        #Find price at t = 0 by starting at second-to-last time period, finding the value at
        #each node in that period as the pv of the risk-neutral expected value, then repeating
        #for prior time periods
        for t in xrange(self.n-1, -1, -1):
            for i in xrange(t+1):
                price_tree[t][i] = (pd*price_tree[t+1][i])+(pu*price_tree[t+1][i+1])
                price_tree[t][i] = round(price_tree[t][i]*math.exp((self.r-self.q)*self.dt*-1), 2) #discount
        self.price = price_tree[0][0]
        self.price_tree = price_tree
        return self.price

    def print_tree(self, stock=False):
        if stock:
            for line in self.stock_tree:
                print '\t'.join([str(price) for price in line])
        else:
            for line in self.price_tree:
                print '\t'.join([str(price) for price in line])


class AsianFloating(object):


    def __init__(self, S, T, n, r, q=0, call=True, geom=False):
        _check_option_error(S, K, T, n, r, q=q, call=call, geom=geom)
        self.S = S
        self.T = T
        self.n = n
        self.dt = float(T)/n
        self.r = r
        self.q = q
        self.call = call
        self.avg = ('Geometric'*geom)+('Arithmetic'*(not geom))
        self.price_tree = []
        self.stock_tree = self.make_tree()
        self.price = self.calc_price()

    def make_tree(self):
        stock_tree = []
        u = math.exp(self.S.sigma*(self.dt**.5))
        d = 1/u
        for t in xrange(self.n+1):
            prices = []
            for num_u in xrange(t+1):
                prices.append(round(self.S.S0*(u**num_u)*(d**(t-num_u)),2))
            stock_tree.append(prices)
        return stock_tree

    def get_paths(self):
        paths = []
        curr_path = []
        paths = self._iter_thru_tree(self.stock_tree, paths, curr_path, 0, 0)
        return paths

    def _iter_thru_tree(self, tree, paths, curr_path, t, i):
        if t == len(tree)-1:
            paths.append(curr_path + [tree[t][i+1]])
            paths.append(curr_path + [tree[t][i]])
            return paths
        else:
            curr_path += [tree[t][i]]
            paths = self._iter_thru_tree(tree, paths, curr_path, t+1, i)
            if t > 0:
                curr_path = curr_path[:t] + [tree[t][i+1]]
                paths = self._iter_thru_tree(tree, paths, curr_path, t+1, i+1)
        return paths

    def calc_price(self):
        self.stock_tree = self.make_tree()
        price_tree = copy.deepcopy(self.stock_tree)
        u = math.exp(self.S.sigma*(self.dt**.5))
        d = 1/u
        #pu and pd represent risk-neutral probabilities of up and down movement, respectively
        pu = (math.exp((self.r - self.q)*self.dt) - d)/(u - d)
        pd = 1 - pu
        #All paths arriving at the same end result occur with the same probability, so the
        #averages are arrived at by finding the average for each path according to the
        #specified method, adding all averages for each end price, then dividing that sum
        #by the total number of ways that path occurs. I realize that those counts will be
        #nCk, but I figured it was easier to just keep count while iterating through
        #the paths than calculating factorials.
        paths = self.get_paths()
        avgs = [0]*(self.n+1)
        counts = [0]*(self.n+1)
        for i, path in enumerate(paths):
            if self.avg == 'Arithmetic':
                final_avg = sum(path)/(self.n + 1)
            else:
                final_avg = 1
                for price in path:
                    final_avg *= price
                final_avg = (final_avg**(1.0/(self.n+1)))
            index = self.stock_tree[-1].index(path[-1])
            avgs[index] += final_avg
            counts[index] += 1
        #Find the final period payouts as difference between final stock price and
        #calculated strike price
        for i, num in enumerate(avgs):
            avgs[i] = float(num)/counts[i]
            if self.call:
                price_tree[-1][i] = max(price_tree[-1][i] - avgs[i], 0)
            else:
                price_tree[-1][i] = max(avgs[i] - price_tree[-1][i], 0)
        #Find price at t = 0 by starting at second-to-last time period, finding the value at
        #each node in that period as the pv of the risk-neutral expected value, then repeating
        #for prior time periods
        for t in xrange(self.n-1, -1, -1):
            for i in xrange(t+1):
                price_tree[t][i] = (pd*price_tree[t+1][i])+(pu*price_tree[t+1][i+1])
                price_tree[t][i] = round(price_tree[t][i]*math.exp((self.r-self.q)*self.dt*-1), 2) #discount
        self.price = price_tree[0][0]
        self.price_tree = price_tree
        return self.price

    def print_tree(self, stock=False):
        if stock:
            for line in self.stock_tree:
                print '\t'.join([str(price) for price in line])
        else:
            for line in self.price_tree:
                print '\t'.join([str(price) for price in line])

"""
