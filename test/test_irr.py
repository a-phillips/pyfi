import time
import scipy.optimize as spo
import numpy as np

"""New Results: trying to merge the two methods. It appears that adding the naive component makes the combined
algorithm faster than the bisection method, but not faster than the regular. Over 1000 iterations of finding the irr
for cash flows evenly spread over a wide range, we get:

Min irr: -0.1367968778
Max irr: 1.3981031821
Both: 0.24269080162 seconds
Bi: 0.247754812241 seconds
Reg: 0.163642883301 seconds

It looks like the naive method is considerably faster overall. I'm not sure why, since the search method seems pretty
basic, but it probably has to do with the fact that the IRR will generally fall in a reasonable range. I'm okay with
the algorithm taking longer for fringe cases while being much faster for more realistic cases.

--------Old Results---------------------------------------------------------

Results: prints cash flow, then each algo's result, # of iterations, and runtime

Seems as though the naive algorithm is more efficient for more reasonable results, while the bisection method is
much faster for much higher IRRs and extreme cases. Since I would expect actual use cases to be reasonable, the
more naive method may be better, since if the IRR < 1.0, it will experience a maximum of 100 iterations, and
most likely much less.

[-1000, 100, 10, 5, 3, 20, 15, 10, 10, 10, 100]
reg:	-0.1673520941 	32 	0.000318050384521
bis:	-0.1673520941 	46 	0.000497102737427
[-1000, 1000, 10, 5, 3, 20, 15, 10, 10, 10, 100]
reg:	0.0976350244 	32 	0.000298976898193
bis:	0.0976350244 	39 	0.000375032424927
[-1000, 10000, 10, 5, 3, 20, 15, 10, 10, 10, 100]
reg:	9.0010550433 	119 	0.00105094909668
bis:	9.0010550433 	41 	0.000557899475098
[-1000, 100000, 10, 5, 3, 20, 15, 10, 10, 10, 100]
reg:	99.000100503 	1007 	0.00595903396606
bis:	99.0001005029 	36 	0.000294923782349
[-100, 107]
reg:	0.07 	5 	1.90734863281e-05
bis:	0.07 	36 	0.000114917755127
[-100, 1070]
reg:	9.7 	98 	0.000266075134277
bis:	9.7 	34 	0.000142812728882
[-100, 10700]
reg:	106.0 	1061 	0.0027379989624
bis:	105.999999999 	34 	0.000170946121216
[-100, 50, -250, 1000, 300, 500, 200, -1]
reg:	1.1298029776 	35 	0.000144958496094
bis:	1.1298029776 	41 	0.000221014022827
[-950, 20, 20, 20, 20, 20, 20, 20, 1020]
reg:	0.0270339744 	31 	0.000134944915771
bis:	0.0270339744 	43 	0.000205993652344
[-950, 50, 50, 50, 50, 50, 50, 50, 10000]
reg:	0.3652357528 	39 	0.000166893005371
bis:	0.3652357528 	45 	0.000229835510254

"""



def pv(apr, cash_flows, dt=1):
    if apr*dt == -1:
        return 0
    dis_rate = 1.0/(1+(apr*dt))
    total = 0
    #Loop through each cash flow, bring it back to time 0, then add it to the total
    for t, pmt in enumerate(cash_flows):
        total += pmt*(dis_rate**(t+1))
    return total



def irrReg(cash_flows, dt=1):
    t0 = time.time()
    apr = 0.0
    #delta is used below as the step by which the guessed apr changes
    delta = 0.1
    #closest_pv is the best-guess pv which is checked against check_pv using the new apr guess
    closest_pv = pv(cash_flows=cash_flows, apr=apr, dt=dt)
    #Finding irr to 10 digits if not found exactly - not sure if this is accurate enough
    it = 1
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
        it += 1
    else:
        return apr


def new_irr(cash_flows, dt=1):
    min_func = lambda a, b, c: abs(pv(a, b, c))
    found = spo.fmin(func=min_func,
                     x0=np.array([.01]),
                     args=(cash_flows, dt),
                     xtol=10**-10,
                     full_output=False,
                     disp=False)
    return found

if __name__ == '__main__':
    t0 = time.time()
    for n in xrange(100):
        cfs = [-500, 100, 100, 100, 100, 125*n]
        new_irr(cfs)
    print time.time()-t0
    t0 = time.time()
    for n in xrange(100):
        cfs = [-500, 100, 100, 100, 100, 125*n]
        irrReg(cfs)
    print time.time()-t0







