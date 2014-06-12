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

* **irr/mirr** - returns the internal rate of return or modified irr on a series of cash flows
```python
>>> print irr(cash_flows=[-566.503678, 100, 200, 300], dt=.5)
0.0500000002
>>> print mirr(inflows=[0, 0, 5000, 2000], outflows=[-1000, -4000], reinv_rate=.12, borrow_rate=.1)
0.179085686035
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
  * Automatically calculates missing parameter upon initialization
  * Display the amortization table using `print_table()`, and access the principal, interest, etc. in each period.
  * Change attributes of the table, then calculate a parameter to update the table.
```python
>>> my_loan = Amortize(principal=10000, int_rate=.05, num_payments=5, future_value=0)
>>> print my_loan.payment
2309.75
>>> my_loan.print_table()
  Time  |  Payment  |  Principal Paid  |  Interest Paid  |  Principal Remaining
---------------------------------------------------------------------------------
   0    |  2309.75  |     1809.75      |      500.0      |        8190.25
   1    |  2309.75  |     1900.24      |     409.51      |        6290.01
   2    |  2309.75  |     1995.25      |      314.5      |        4294.76
   3    |  2309.75  |     2095.01      |     214.74      |        2199.75
   4    |  2309.74  |     2199.75      |     109.99      |           0
>>> print my_loan.interest_paid[2]
314.5
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
  can be seen using `ytm()` or `price()`.
  * Price and ytm can be updated with `price(new_price)` and `ytm(new_ytm)`.
  * A summary can be seen with `info()`. The attributes `macD`, `modD`, and `convexity` are calculated upon
  initialization and any change to price or ytm.
```python
>>> my_bond = Bond(length=5, par_value=1000, coupon_rate=.05, num_annual_coupons=2, ytm=.06)
>>> my_bond.price()
957.348985816121
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
>>> my_bond.price(1050)
>>> my_bond.ytm()
0.0388993771
```
* **Stock** - creates a stock object. Use `info()` to view the attributes.

* **Options - Binomial Models** - creates an option. The option is priced using a binomial tree. See the docstring
for each class for its specific use. Binary options can be cash-or-nothing or asset-or-nothing.
  * EuropeanCRR / EuropeanBinaryCRR
  * AmericanCRR / AmericanBinaryCRR
  * BermudanCRR / BermudanBinaryCRR
  * EuropeanJR / EuropeanBinaryJR
  * AmericanJR / AmericanBinaryJR
  * BermudanJR / BermudanBinaryJR
  * EuropeanTian / EuropeanBinaryTian
  * AmericanTian / AmericanBinaryTian
  * BermudanTian / BermudanBinaryTian
  
```python
>>> my_option = AmericanCRR(S=50, sigma=.3, K=50, T=1, n=4, r=.03, call=True)
>>> my_option.price
6.29
>>> my_option.print_tree() #Print option value tree and stock tree
6.29
1.98    10.91
0.0     4.09    18.24
0.0     0.0     8.46    28.79
0       0       0.0     17.49   41.11
>>> my_option.K = 60
>>> my_option.call=False
>>> my_option.calc_price()
11.93
>>> print my_option.delta, my_option.gamma, my_option.theta  #Get greeks
-0.668438538206 0.0139963646614 1.82
```
  
* **Options - Closed-Form Models** - creates an option priced using closed form solutions, such as Black-Scholes.
Binary options can be cash-or-nothing or asset-or-nothing. Implied volatility can be found for an option, given a
market price.
  * EuropeanBS / EuropeanBinaryBS

```python
>>> my_option = EuropeanBS(S=50, sigma=.4, K=50, T=2, r=.05, q=.03, call=True)
>>> my_option.price
11.229263763221908
>>> my_option.implied_vol(13)
0.4714627488000006
```

* **Options - Monte Carlo Models** - creates an option object priced using Monte Carlo methods. All of the methods
currently use antithetic path variance reduction.
  * EuropeanMCLogNormConstVol (European option, Monte Carlo method, lognormal constant vol. stock movement.)
  * AsianMCLogNormConstVol
  * LookbackFixedMCLogNormConstVol
  * LookbackFloatingLogNormConstVol
  
```python
>>> my_option = AsianMCLogNormConstVol(S=50, sigma=.2, K=50, T=1, r=.03, q=0.0, call=True, geometric=False)
>>> print my_option.calc_price(num_obs=50, num_sims=100000)
2.69153640404
```


####Retrieve Market Data from Yahoo Finance API

* **get_current_data** - retrieves up to 87 attributes of up to 50 symbols
* **get_historical_data** - retrieves historical open, high, low, close, volume, and adj. close for one stock during a 
given time period either daily, weekly, or monthly.
* **get_sector_data** - retrieves various attributes for all broad sectors (Financial, Utilities, etc.)
* **get_industry_data** - retrieves various attributes for a certain industry or sub-industry. Only argument is the 
industry code - 1 digit indicates a broad industry with data by sub-industry, and 3 digits indicates a sub-industry 
with data by company. Attributes are the same as those retrieved for `get_sector_data`.

```python
# Get the ask, bid, year range, and market cap for GOOG and LUV
>>> data = get_current_data(['GOOG', 'LUV'], ['a0','b0','w0','j1']) 
>>> print data['GOOG']
[556.0, 550.74, (502.8, 604.83), 375200000000.0]
# Get the monthly data for 'LUV' from 1/1/2010 to 1/1/2014
>>> data = get_historical_data('LUV', date(2010, 1, 1), date(2014, 1, 1), 'm')
>>> print data['headers']
['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
>>> print data[date(2012, 5, 1)]
[8.25, 9.14, 8.0, 9.03, 10193500.0, 8.89]
# Get the current sector data
>>>data = get_sector_data()
>>> print data['headers']
['Sectors', '1-Day Price Chg %', 'Market Cap', 'P/E', 'ROE %', 'Div. Yield %', 'Debt to Equity', 'Price to Book', 'Net Profit Margin (mrq)', 'Price To Free Cash Flow (mrq)']
>>> print data['Utilities']
[0.00942, 23779590000000.0, 25.098, 0.06472, 0.03252, 144.021, 2.413, 6.585, -62.248]
# Get data for the Basic Materials industry - code 1. Returns the same attributes as get_sector_data
>>> data = get_industry_data(1) 
>>> print data['Copper']
[-0.00597, 1606800000000.0, 14.7, 0.17, 0.02735, 71.392, 2.3, 17.0, -74.1]
# Get data for the Copper sub-industry - code 131. Returns the same attributes as get_sector_data
>>> data = get_industry_data(131)
>>> print data['Peak Resources Ltd']
[0.0, 19290000.0, 'NA', -0.06488, 'NA', 'NA', 0.556, 'NA', -17.024]
```
* **show_stock_field_codes** - displays the attribute codes used by the `get_current_data` query.
* **show_industry_detail_codes** - displays the codes for each sub-industry used by the `get_industry_data` query.
* The 1-digit codes for each broad industry can be found in the docstring for the `get_industry_data` function.

###Future Functionality:

* **Portfolio Analysis** - real-time quotes, market information, allocation optimization
* **Interest Rate Models** - HJM, BGM, CIR, etc.
* **Monte Carlo Simulation** - framework for option pricing
* **Finite Differences*** - framework for option pricing




