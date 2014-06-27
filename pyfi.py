"""This is the primary module of the PyFi package. This module includes basic and advanced financial
functions, as well as classes to model certain financial instruments such as bonds, amortizing loans,
and equity options.

Current Functions:
npv - returns the net present value of a series of cash flows
nfv - returns the net future value of a series of cash flows
pv - returns the pv of an investment with periodic payments
fv - returns the fv of an investment with periodic payments
pmt - returns the periodic payment of an investment with periodic payments
nper - returns the number of periods of an investment with periodic payments
rate - returns the interest rate of an investment with periodic payments
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


#-------------------------------------------------------------------------------------
#Functions (npv, nfv, irr, macD, modD, convexity)
#-------------------------------------------------------------------------------------

#General financial functions----------------------------------------------------------

def npv(cash_flows, apr, dt=1):
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


def nfv(cash_flows, apr, dt=1):
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

    cf_pv = npv(cash_flows, apr, dt)
    if not cf_pv:
        return None
    #Future value = present value * ((1 + interest rate)**n)
    return cf_pv*((1+(apr*dt))**len(cash_flows))


def pv(payment, int_rate, num_payments, future_value):
    pay_portion = payment*(1-(1/((1+int_rate)**num_payments)))/int_rate
    fv_portion = future_value/((1+int_rate)**num_payments)
    present_value = pay_portion + fv_portion
    return present_value


def fv(present_value, payment, int_rate, num_payments):
    pay_portion = payment*(1-(1/((1+int_rate)**num_payments)))/int_rate
    future_value = (present_value-pay_portion)*((1+int_rate)**num_payments)
    return future_value


def pmt(present_value, int_rate, num_payments, future_value):
    num = present_value - (future_value/((1+int_rate)**num_payments))
    den = (1-(1/((1+int_rate)**num_payments)))/int_rate
    payment = num/den
    return payment


def nper(present_value, payment, int_rate, future_value):
    num = math.log(((int_rate*present_value)-payment)/((int_rate*future_value)-payment))
    den = math.log(1/(1+int_rate))
    num_payments = int(math.ceil(num/den))
    return num_payments


def rate(present_value, payment, num_payments, future_value):
    cash_flows = [present_value*-1] + [payment]*num_payments
    cash_flows[-1] += future_value
    int_rate = irr(cash_flows)
    return int_rate


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
    closest_pv = npv(cash_flows=cash_flows, apr=apr, dt=dt)
    #Finding irr to 10 digits if not found exactly - not sure if this is accurate enough
    while abs(delta) >= (10**(-10)):
        if closest_pv == 0:
            return apr
        apr += delta
        check_pv = round(npv(cash_flows=cash_flows, apr=apr, dt=dt), 10)
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
        pv_outflows = (outflows[0] + npv(outflows[1:], borrow_rate, dt))*-1
    fv_inflows = nfv(inflows, reinv_rate, dt)
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
    P = npv(cash_flows, apr, dt)
    total = 0
    for t, pmt in enumerate(cash_flows):
        weight = pmt*(dis_rate**(t+1))/P
        total += ((t+1)*(t+2)*(dt**2)) * weight * (dis_rate**2)
    return total


#Special functions - used in classes, or for particular cases ------------------------



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
        self.principal = pv(self.payment, self.int_rate, self.num_payments, self.future_value)
        self.update_table()
        return self.principal

    def calc_future_value(self):
        self.future_value = fv(self.principal, self.payment, self.int_rate, self.num_payments)
        self.update_table()
        return self.future_value

    def calc_payment(self):
        self.payment = pmt(self.principal, self.int_rate, self.num_payments, self.future_value)
        self.update_table()
        return self.payment

    def calc_num_payments(self):
        self.num_payments = nper(self.principal, self.payment, self.int_rate, self.future_value)
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
        self._price = npv(self.cash_flows, self._ytm, dt)

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


#------------------------------------------------------------------------------------
#End
#------------------------------------------------------------------------------------

if __name__ == '__main__':
    pass