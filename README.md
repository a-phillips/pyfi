#Welcome to PyFi

PyFi is a financial modeling module written in Python. It's meant to be an all-encompassing module for both basic
and advanced financial applications.

Thanks Christophe Rougeaux for helping correct the binomial model errors!

###Current Functionality

####Functions

* **pv/fv** - returns the present value/future value of a series of cash flows
```python
>>> print pv(cash_flows=[100, 200, 300], apr=.05, dt=.5)
566.503678124
>>> print fv(cash_flows=[100, 200, 300], apr=.05, dt=.5)
610.0625
```

* **irr** - returns the internal rate of return on a series of cash flows
```python
>>> print irr(cash_flows=[-566.503678, 100, 200, 300], dt=.5)
0.0500000002
```

* **macD/modD** - returns the Macaulay duration/Modified duration of a series of cash flows
```python
>>> print macD(cash_flows=[100, 200, 300], apr=.05, dt=.5)
1.15976846635
>>> print modD(cash_flows=[100, 200, 300], apr=.05, dt=.5)
1.13148143058
```

* **convexity** - returns the convexity of a series of cash flows
```python
>>> print convexity(cash_flows=[100, 200, 300], apr=.05, dt=.5)
1.96589018896
```

####Classes

* **Amortize** - creates an amortization table object.
  * Automatically calculates missing parameter upon initialization:
```python
>>> my_loan = Amortize(principal=10000, int_rate=.05/12, num_payments=60, future_value=0)
>>> print my_loan.payment
188.71233644
>>> my_loan = Amortize(principal=10000, int_rate=.05/12, payment=250, future_value=0)
>>> print my_loan.num_payments
44
my_loan = Amortize(int_rate=.05/12, num_payments=36, payment=250, future_value=0)
>>> print my_loan.principal
8341.42532094
```
  * Display the amortization table using `print_table()`, and access the principal, interest, etc. in each period
```python
>>> my_loan = Amortize(principal=10000, int_rate=.05, num_payments=5, future_value=0)
>>> my_loan.print_table()
  Time  |  Payment  |  Principal Paid  |  Interest Paid  |  Principal Remaining
---------------------------------------------------------------------------------
   0    |  2309.75  |     1809.75      |      500.0      |        8190.25
   1    |  2309.75  |     1900.24      |     409.51      |        6290.01
   2    |  2309.75  |     1995.25      |      314.5      |        4294.76
   3    |  2309.75  |     2095.01      |     214.74      |        2199.75
   4    |  2309.74  |     2199.75      |     109.99      |           0
>>> print my_loan.principal_paid[1]
1900.24
>>> my_loan.interest_paid[2]
314.5
>>> my_loan.principal_remaining[3]
2199.75
```
  * Change attributes of the table, then calculate a parameter to update the table
```python
>>> my_loan = Amortize(principal=10000, int_rate=.05, num_payments=5, future_value=0)
>>> my_loan.print_table()
  Time  |  Payment  |  Principal Paid  |  Interest Paid  |  Principal Remaining
---------------------------------------------------------------------------------
   0    |  2309.75  |     1809.75      |      500.0      |        8190.25
   1    |  2309.75  |     1900.24      |     409.51      |        6290.01
   2    |  2309.75  |     1995.25      |      314.5      |        4294.76
   3    |  2309.75  |     2095.01      |     214.74      |        2199.75
   4    |  2309.74  |     2199.75      |     109.99      |          0.0
>>> my_loan.payment = 2000
>>> my_loan.calc_num_payments()
6
>>> my_loan.print_table()
  Time  |  Payment  |  Principal Paid  |  Interest Paid  |  Principal Remaining
---------------------------------------------------------------------------------
   0    |  2000.0   |      1500.0      |      500.0      |        8500.0
   1    |  2000.0   |      1575.0      |      425.0      |        6925.0
   2    |  2000.0   |     1653.75      |     346.25      |        5271.25
   3    |  2000.0   |     1736.44      |     263.56      |        3534.81
   4    |  2000.0   |     1823.26      |     176.74      |        1711.55
   5    |  1797.13  |     1711.55      |      85.58      |          0.0
```
* **Bond** - creates a bond object.
  * Specify the parameters of the bond then, either `price` or `ytm` to initialize the bond. Whichever isn't specified
  can be seen using `ytm()` or `price()`
```python
>>> my_bond = Bond(length=5, par_value=1000, coupon_rate=.05, num_annual_coupons=2, ytm=.06)
>>> my_bond.price()
957.348985816121
```
  * Price and ytm can be updated with `price(new_price)` and `ytm(new_ytm)`
```python
>>> my_bond.price(1050)
>>> my_bond.ytm()
0.0388993771
```
  * A summary can be seen with `info()`. The attributes `macD`, `modD`, and `convexity` are calculated upon
  initialization and any change to price or ytm.
```python
>>> my_bond.info()
Time to Expiry: 5               #my_bond.length
Par Value:      $1000           #my_bond.par_value
Coupon Rate:    5.0 percent     #my_bond.coupon_rate
Annual Coupons: 2               #my_bond.num_annual_coupons
Price:          $957.35         #my_bond.price()
YTM:            6.0 percent     #my_bond.ytm()
macD:           4.47167860758   #my_bond.macD
modD:           4.34143554134   #my_bond.modD
Convexity:      22.3047280394   #my_bond.convexity
```

* **Stock** - creates a stock object. Primarily used for option pricing. Use `info()` to view the attributes.
```python
>>> my_stock = Stock(S0=50, sigma=.1)
>>> my_stock.info()
Price:      $50
Volatility: 10.0 percent
```

* **Binomial Options** - creates an option. The option is priced using a binomial tree. See the docstring
for each class for its specific use.
  * Cox-Ross-Rubinstein - original binomial model -  EuropeanCRR/AmericanCRR/BermudanCRR
  * Jarrow-Rudd - equal probability binomial model - EuropeanJR/AmericanJR/BermudanJR
  * Tian - the "moment matching" binomial model - EuropeanTian/AmericanTian/BermudanTian
```python
>>> my_option = AmericanCRR(S=Stock(50, .3), K=50, T=1, n=4, r=.03, call=True)
>>> my_option.print_tree() #Print option value tree and stock tree
6.29
1.98    10.91
0.0     4.09    18.24
0.0     0.0     8.46    28.79
0       0       0.0     17.49   41.11
>>> my_option.print_tree(stock=True)
50.0
43.04   58.09
37.04   50.0    67.49
31.88   43.04   58.09   78.42
27.44   37.04   50.0    67.49   91.11
>>> my_option.price
6.29
>>> my_option.K = 60 #Change attributes
>>> my_option.call=False
>>> my_option.print_tree()  #Doesn't change anything - need to calc_price
6.29
1.98    10.91
0.0     4.09    18.24
0.0     0.0     8.46    28.79
0       0       0.0     17.49   41.11
>>> my_option.calc_price()
11.93
>>> my_option.print_tree()
11.93
16.92   6.86
22.79   11.02   2.59
27.91   16.83   5.09    0.0
32.56   22.96   10.0    0       0
>>> print my_option.delta, my_option.gamma, my_option.theta  #Get greeks
-0.668438538206 0.0139963646614 1.82
>>> my_option = AmericanJR(S=Stock(50, .3), K=50, T=1, n=4, r=.03, call=True)
>>> my_option.price
6.41
```

###Future Functionality:

* **Portfolio Analysis** - real-time quotes, market information, allocation optimization
* **Interest Rate Models** - HJM, BGM, CIR, etc.
* **Monte Carlo Simulation** - framework for option pricing
* **Finite Differences*** - framework for option pricing




