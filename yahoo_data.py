"""This program is used to collect data from the Yahoo Finance API.

Functions:
get_data - use this to get data form the Yahoo Finance API. Returns a dictionary of the symbol
           and resulting data.
show_yahoo_codes - use this to display the codes and field names accepted by Yahoo.
"""

import urllib
import urllib2
from datetime import date, time

#Yahoo API Info: https://code.google.com/p/yahoo-finance-managed/wiki/CSVAPI

#Create dictionary of the codes used by Yahoo Finance
get_yahoo_field = {
'c8':	'After Hours Change (Realtime)',
'g3':	'Annualized Gain',
'a0':	'Ask',
'b2':	'Ask (Realtime)',
'a5':	'Ask Size',
'a2':	'Average Daily Volume',
'b0':	'Bid',
'b3':	'Bid (Realtime)',
'b6':	'Bid Size',
'b4':	'Book Value Per Share',
'c1':	'Change',
'c0':	'Change Change In Percent',
'm7':	'Change From Fiftyday Moving Average',
'm5':	'Change From Two Hundredday Moving Average',
'k4':	'Change From Year High',
'j5':	'Change From Year Low',
'p2':	'Change In Percent',
'k2':	'Change In Percent (Realtime)',
'k5':	'Change In Percent From Year High',
'c6':	'Change (Realtime)',
'c3':	'Commission',
'c4':	'Currency',
'h0':	'Days High',
'g0':	'Days Low',
'm0':	'Days Range',
'm2':	'Days Range (Realtime)',
'w1':	'Days Value Change',
'w4':	'Days Value Change (Realtime)',
'r1':	'Dividend Pay Date',
'e0':	'Diluted EPS',
'j4':	'EBITDA',
'e7':	'EPS Estimate Current Year',
'e9':	'EPS Estimate Next Quarter',
'e8':	'EPS Estimate Next Year',
'q0':	'Ex Dividend Date',
'm3':	'Fiftyday Moving Average',
'l2':	'High Limit',
'g4':	'Holdings Gain',
'g1':	'Holdings Gain Percent',
'g5':	'Holdings Gain Percent (Realtime)',
'g6':	'Holdings Gain (Realtime)',
'v1':	'Holdings Value',
'v7':	'Last Trade Date',
'l1':	'Last Trade Price Only',
'k1':	'Last Trade (Realtime) With Time',
'k3':	'Last Trade Size',
't1':	'Last Trade Time',
'l0':	'Last Trade With Time',
'l3':	'Low Limit',
'j1':	'Market Capitalization',
'j3':	'Market Cap (Realtime)',
'i0':	'More Info DO NOT USE',
'n0':	'Name',
'n4':	'Notes',
't8':	'Oneyr Target Price',
'o0':	'Open',
'i5':	'Order Book (Realtime) DO NOT USE',
'r5':	'PEG Ratio',
'r0':	'PE Ratio',
'r2':	'PE Ratio (Realtime)',
'm8':	'Percent Change From Fiftyday Moving Average',
'm6':	'Percent Change From Two Hundredday Moving Average',
'j6':	'Percent Change From Year Low',
'p0':	'Previous Close',
'p6':	'Price Book',
'r6':	'Price EPS Estimate Current Year',
'r7':	'Price EPS Estimate Next Year',
'p1':	'Price Paid',
'p5':	'Price Sales',
's6':	'Revenue',
'f6':	'Shares Float DO NOT USE - Need to figure out parsing, number comes in as XXX,XXX,XXX so it gets split',
's1':	'Shares Owned',
'j2':	'Shares Outstanding DO NOT USE - Need to figure out parsing, number comes in as XXX,XXX,XXX so it gets split',
's7':	'Short Ratio',
'x0':	'Stock Exchange',
's0':	'Symbol',
't7':	'Ticker Trend DO NOT USE',
'd2':	'Trade Date',
't6':	'Trade Links DO NOT USE',
'f0':	'Trade Links Additional DO NOT USE',
'd0':	'Trailing Annual Dividend Yield',
'y0':	'Trailing Annual Dividend Yield In Percent',
'm4':	'Two Hundredday Moving Average',
'v0':	'Volume',
'k0':	'Year High',
'j0':	'Year Low',
'w0':	'Year Range'}

#Reverse get_yahoo_field dictionary to look up codes for desired fields.
get_yahoo_code = dict(zip(get_yahoo_field.values(),get_yahoo_field.keys()))


def show_yahoo_codes(by_field=True):
    if by_field:
        sorted_field_list = get_yahoo_field.values()
        sorted_field_list.sort()
        print '\n'.join(['%s: %s' % (get_yahoo_code[field], field) for field in sorted_field_list])
    else:
        sorted_code_list = get_yahoo_code.values()
        sorted_code_list.sort()
        print '\n'.join(['%s: %s' % (code, get_yahoo_field[code]) for code in sorted_code_list])


def get_data(symbols, codes=None):
    """get_data(symbols, codes=None)

Retrieves stock data from Yahoo Finance. Quotes may be delayed up to 15 minutes.

symbols - list of the stock symbols
codes - list of the codes for the desired fields. If None, will get all data. Use show_yahoo_codes()
        to see what each code is.
"""

    # Get all data if no codes are specified
    if codes is None:
        codes = get_yahoo_code.values()
    codes.sort()

    # Filter out codes that return bad or not useful data.
    _do_not_use_codes = ['i0','f6','j2','t7','t6','f0', 'i5']
    codes = filter(lambda x: x not in _do_not_use_codes, codes)

    # Set up url. urllib.quote used on symbols to ensure proper URL format.
    base_url = 'http://download.finance.yahoo.com/d/quotes.csv?'
    suffix = '&e=.csv'
    symbol_str = 's=' + urllib.quote(','.join(symbols), ',')
    code_str = '&f=' + ''.join(codes)

    # Retrieve and split the file
    raw_file = urllib2.urlopen(base_url+symbol_str+code_str+suffix).read().splitlines()
    split_file = [line.replace('"','').split(',') for line in raw_file]

    #Create dictionary of data for each symbol, format the data, then return it
    data = {}
    for i in xrange(len(symbols)):
        data[symbols[i]] = split_file[i]
    data = _format_data(codes, data)
    return data


def _format_data(codes, data):
    """Example pre- and post-formatting results for all codes for LUV:

Code    Unformatted         Formatted       Field Name
a0 	    N/A 	            None 	        Ask
a2 	    6254680	            6254680.0	    Average Daily Volume
a5 	    N/A 	            None 	        Ask Size
b0 	    N/A 	            None 	        Bid
b2 	    27.23	            27.23	        Ask (Realtime)
b3 	    27.08	            27.08	        Bid (Realtime)
b4 	    10.393	            10.393	        Book Value Per Share
b6 	    N/A 	            None 	        Bid Size
c0 	    +0.49 - +1.83% 	    (0.49, 0.0183) 	Change Change In Percent
c1 	    +0.49 	            0.49	        Change
c3 	    - 	                None 	        Commission
c4 	    USD 	            USD 	        Currency
c6 	    +0.49 	            0.49	        Change (Realtime)
c8 	    N/A - N/A 	        (None, None) 	After Hours Change (Realtime)
d0 	    0.18	            0.18	        Trailing Annual Dividend Yield
d2 	    - 	                None 	        Trade Date
e0 	    1.195	            1.195	        Diluted EPS
e7 	    1.50	            1.5	            EPS Estimate Current Year
e8 	    1.75	            1.75	        EPS Estimate Next Year
e9 	    0.43	            0.43	        EPS Estimate Next Quarter
f0 	    Missing Format Variable. 	Missing Format Variable. 	Trade Links Additional DO NOT USE
f6 	    Missing Format Variable. 	Missing Format Variable. 	Shares Float DO NOT USE - Need to figure out parsing, number comes in as XXX,XXX,XXX so it gets split
g0 	    26.74	            26.74	        Days Low
g1 	    - - - 	            (None, None) 	Holdings Gain Percent
g3 	    - 	                None 	        Annualized Gain
g4 	    - 	                None 	        Holdings Gain
g5 	    N/A - N/A 	        (None, None) 	Holdings Gain Percent (Realtime)
g6 	    N/A 	            None 	        Holdings Gain (Realtime)
h0 	    27.22	            27.22	        Days High
i0 	    Missing Format Variable. 	Missing Format Variable. 	More Info DO NOT USE
i5 	    Missing Format Variable. 	Missing Format Variable. 	Order Book (Realtime) DO NOT USE
j0 	    12.58	            12.58	        Year Low
j1 	    18.818B 	        18818000000.0	Market Capitalization
j2 	    Missing Format Variable. 	Missing Format Variable. 	Shares Outstanding DO NOT USE - Need to figure out parsing, number comes in as XXX,XXX,XXX so it gets split
j3 	    N/A 	            None 	        Market Cap (Realtime)
j4 	    2.392B 	            2392000000.0	EBITDA
j5 	    +14.62 	            14.62	        Change From Year Low
j6 	    +116.22% 	        1.1622	        Percent Change From Year Low
k0 	    26.78	            26.78	        Year High
k1 	    N/A - <b>27.20</b> 	(None, 27.2) 	Last Trade (Realtime) With Time
k2 	    N/A - +1.83% 	    (None, 0.0183) 	Change In Percent (Realtime)
k3 	    N/A 	            None 	        Last Trade Size
k4 	    +0.42 	            0.42	        Change From Year High
k5 	    +1.57% 	            0.0157	        Change In Percent From Year High
l0 	    Jun  3 - <b>27.20</b> 	(datetime.date(2014, 6, 3), 27.2) 	Last Trade With Time
l1 	    27.20	            27.2	        Last Trade Price Only
l2 	    - 	                None 	        High Limit
l3 	    - 	                None 	        Low Limit
m0 	    26.74 - 27.22 	    (26.74, 27.22) 	Days Range
m2 	    N/A - N/A 	        (None, None) 	Days Range (Realtime)
m3 	    24.6115	            24.6115	        Fiftyday Moving Average
m4 	    21.9118	            21.9118	        Two Hundredday Moving Average
m5 	    +5.2882 	        5.2882	        Change From Two Hundredday Moving Average
m6 	    +24.13% 	        0.2413	        Percent Change From Two Hundredday Moving Average
m7 	    +2.5885 	        2.5885	        Change From Fiftyday Moving Average
m8 	    +10.52% 	        0.1052	        Percent Change From Fiftyday Moving Average
n0 	    Southwest Airline 	Southwest Airline 	Name
n4 	    - 	                - 	            Notes
o0 	    26.75	            26.75	        Open
p0 	    26.71	            26.71	        Previous Close
p1 	    - 	                None 	        Price Paid
p2 	    +1.83% 	            0.0183	        Change In Percent
p5 	    1.04	            1.04	        Price Sales
p6 	    2.57	            2.57	        Price Book
q0 	    Jun  2 	            2014-06-02 	    Ex Dividend Date
r0 	    22.35	            22.35	        PE Ratio
r1 	    Jun 25 	            2014-06-25 	    Dividend Pay Date
r2 	    N/A 	            None 	        PE Ratio (Realtime)
r5 	    0.65	            0.65	        PEG Ratio
r6 	    17.81	            17.81	        Price EPS Estimate Current Year
r7 	    15.26	            15.26	        Price EPS Estimate Next Year
s0 	    LUV 	            LUV 	        Symbol
s1 	    - 	                None 	        Shares Owned
s6 	    17.781B 	        17781000000.0	Revenue
s7 	    2.60	            2.6	            Short Ratio
t1 	    4:00pm 	            16:00:00 	    Last Trade Time
t6 	    Missing Format Variable. 	Missing Format Variable. 	Trade Links DO NOT USE
t7 	    Missing Format Variable. 	Missing Format Variable. 	Ticker Trend DO NOT USE
t8 	    27.25	            27.25	        Oneyr Target Price
v0 	    8114240	            8114240.0	    Volume
v1 	    - 	                None 	        Holdings Value
v7 	    N/A 	            None 	        Last Trade Date
w0 	    12.58 - 26.78 	    (12.58, 26.78) 	Year Range
w1 	    - - +1.83% 	        (None, 0.0183) 	Days Value Change
w4  	N/A - N/A 	        (None, None) 	Days Value Change (Realtime)
x0  	NYSE 	            NYSE 	        Stock Exchange
y0  	0.67	            0.0067	        Trailing Annual Dividend Yield In Percent
"""

    # Values that will be integers (i.e. bid/ask volumes) are being formatted as floats to
    # prevent any calculation errors arising from using them, due to how integer division
    # works in Python.

    #All codes not specified here are formatted as floats
    range_codes = ['c0', 'c8', 'g1', 'g5', 'k1', 'k2', 'm0', 'm2', 'w0',
                   'w1', 'w4']
    date_codes = ['d2', 'q0', 'r1', 'v7']
    time_codes = ['t1']
    str_codes = ['c4', 'n0', 'n4', 's0', 'x0']
    do_not_use_codes = ['i0', 'f6', 'j2', 't7', 't6', 'f0', 'i5']

    for i, code in enumerate(codes):
        if code in time_codes:
            data = _format_field_as_time(data, i)
        elif code == 'l0':
            data = _format_l0(data, i)
        elif code in date_codes:
            if code == 'r1':
                data = _format_field_as_date(data, i, past=False)
            else:
                data = _format_field_as_date(data, i, past=True)
        elif code in str_codes or code in do_not_use_codes:
            #Formatted as string
            pass
        elif code in range_codes:
            data = _format_field_as_range(data, i)
        else:
            # Float code
            if code == 'y0': # This field doesn't end in %, even though it's a percentage
                for data_line in data.values():
                    data_line[i] += '%'
            data = _format_field_as_float(data, i)
    return data


#--------------------------------------------------------------------------------
# Code for formatting individual portions of each field
#--------------------------------------------------------------------------------

def _format_as_float(str_num):
    if str_num in ['N/A', '-']:
        return None
    else:
        factor = 1.0
        #Check for <b>num</b> which wraps some realtime results
        if str_num.find('<b>') != -1:
            str_num = str_num.replace('<b>','')
            str_num = str_num.replace('</b>','')
        #Check for leading +'s and -'s
        if str_num[0] == '-':
            factor *= -1
            str_num = str_num[1:]
        elif str_num[0] == '+':
            str_num = str_num[1:]
        #Check for trailing %'s, B's, M's, or K's
        if str_num[-1] == '%':
            str_num = str_num[:-1]
            factor /= 100
        elif str_num[-1] == 'B':
            str_num = str_num[:-1]
            factor *= (10**9)
        elif str_num[-1] == 'M':
            str_num = str_num[:-1]
            factor *= (10**6)
        elif str_num[-1] == 'K':
            str_num = str_num[:-1]
            factor *= (10**3)
        return float(str_num)*factor


def _format_as_date(str_date, past=True):
    if str_date in ['N/A', '-']:
        return None
    month = [0, 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    #Example date: 'Jun  3'
    data_mon = month.index(str_date[:3])
    data_day = int(str_date[str_date.rfind(' ')+1:])
    # Need to determine year since Yahoo doesn't provide it. Assumes that no date is
    # greater than 1 year away from the current date
    curr_date = date.today()
    data_year = curr_date.year
    if past:
        if data_mon > curr_date.month:
            data_year -= 1
        elif data_mon == curr_date.month and data_day > curr_date.day:
            data_year -= 1
    else:
        if data_mon < curr_date.month:
            data_year += 1
        elif data_mon == curr_date.month and data_day < curr_date.day:
            data_year += 1
    return date(data_year, data_mon, data_day)


def _format_as_time(str_time):
    hour = int(str_time[:str_time.find(':')])
    if str_time[-2:] == 'pm':
        hour += 12
    minute = int(str_time[str_time.find(':')+1:-2])
    return time(hour, minute)


#--------------------------------------------------------------------------------
# Code for formatting each field
#--------------------------------------------------------------------------------


def _format_field_as_float(data, i):
    for data_line in data.values():
        data_line[i] = _format_as_float(data_line[i])
    return data


def _format_field_as_range(data, i):
    # Example: '26.52 - 28.55'
    for data_line in data.values():
        rng = data_line[i]
        new_rng = (rng[:rng.find(' ')], rng[rng.rfind(' ')+1:])
        new_rng = (_format_as_float(new_rng[0]), _format_as_float(new_rng[1]))
        data_line[i] = new_rng
    return data


def _format_field_as_date(data, i, past):
    for data_line in data.values():
        data_line[i] = _format_as_date(data_line[i], past)
    return data


def _format_field_as_time(data, i):
    for data_line in data.values():
        data_line[i] = _format_as_time(data_line[i])
    return data


def _format_l0(data, i):
    #Example: 'Jun  2 - <b>26.71</b>'
    for data_line in data.values():
        temp = data_line[i]
        new_data = (temp[:temp.find('-')-1], temp[temp.rfind(' ')+1:])
        new_data = (_format_as_date(new_data[0], past=True), _format_as_float(new_data[1]))
        data_line[i] = new_data
    return data


if __name__ == '__main__':
    test_data = get_data(['HIG'], ['c0'])
    for item in test_data['HIG']:
        print item




















