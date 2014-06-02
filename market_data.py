import urllib
import urllib2
import sys

#https://code.google.com/p/yahoo-finance-managed/wiki/CSVAPI

base_url = 'http://download.finance.yahoo.com/d/quotes.csv?'
suffix = '&e=.csv'

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
'd0':	'Trailing Annual Dividend Yield',
'y0':	'Trailing Annual Dividend Yield In Percent',
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
'i5':	'Order Book (Realtime)',
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
'm4':	'Two Hundredday Moving Average',
'v0':	'Volume',
'k0':	'Year High',
'j0':	'Year Low',
'w0':	'Year Range'}

_do_not_use = ['i0','f6','j2','t7','t6','f0']

#Reverse code_meaning dictionary to look up codes for desired fields.
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


def get_data(symbols, fields=None):
    if fields is None:
        fields = ''.join(get_yahoo_code.values())
    fields = filter(lambda x: x not in _do_not_use, fields)
    symbol_str = 's=' + urllib.quote(','.join(symbols), ',')
    field_str = '&f=' + ''.join(fields)
    raw_file = urllib2.urlopen(base_url+symbol_str+field_str+suffix).read().splitlines()
    split_file = [line.replace('"','').split(',') for line in raw_file]
    data = {}
    for i in xrange(len(symbols)):
        data[symbols[i]] = split_file[i]
    return data


def _format_as_float(data, i):
    for data_line in data.values():
        if data_line[i] in ['N/A', '-']:
            data_line[i] = None
        else:
            data_line[i] = float(data_line[i])
    return data


def _format_as_num(data, i):
    for data_line in data.values():
        print data_line[i]
        if data_line[i] in ['N/A', '-']:
            data_line[i] = None
        else:
            factor = 1
            if data_line[i].find('-') != -1:
                factor *= -1
            data_line[i] = float(data_line[i][1:])*factor
    return data



def _format_data(fields, data):
    """Example non-formatted results for all codes for GOOG:

a0 N/A Ask
a2 6285880 Average Daily Volume
a5 N/A Ask Size
b0 N/A Bid
b2 26.72 Ask (Realtime)
b3 26.68 Bid (Realtime)
b4 10.393 Book Value Per Share
b6 N/A Bid Size
c0 +0.32 - +1.21% Change Change In Percent
c1 +0.32 Change
c3 - Commission
c4 USD Currency
c6 +0.32 Change (Realtime)
c8 N/A - N/A After Hours Change (Realtime)
d0 0.16 Trailing Annual Dividend Yield
d2 - Trade Date
e0 1.195 Diluted EPS
e7 1.50 EPS Estimate Current Year
e8 1.75 EPS Estimate Next Year
e9 0.43 EPS Estimate Next Quarter
f0 Missing Format Variable. Trade Links Additional DO NOT USE
f6 Missing Format Variable. Shares Float DO NOT USE - Need to figure out parsing, number comes in as XXX,XXX,XXX so it gets split
g0 26.18 Days Low
g1 - - - Holdings Gain Percent
g3 - Annualized Gain
g4 - Holdings Gain
g5 N/A - N/A Holdings Gain Percent (Realtime)
g6 N/A Holdings Gain (Realtime)
h0 26.78 Days High
i0 Missing Format Variable. More Info DO NOT USE
i5 N/A Order Book (Realtime)
j0 12.58 Year Low
j1 18.479B Market Capitalization
j2 Missing Format Variable. Shares Outstanding DO NOT USE - Need to figure out parsing, number comes in as XXX,XXX,XXX so it gets split
j3 N/A Market Cap (Realtime)
j4 2.392B EBITDA
j5 +14.13 Change From Year Low
j6 +112.32% Percent Change From Year Low
k0 26.64 Year High
k1 N/A - <b>26.71</b> Last Trade (Realtime) With Time
k2 N/A - +1.21% Change In Percent (Realtime)
k3 N/A Last Trade Size
k4 +0.07 Change From Year High
k5 +0.26% Change In Percent From Year High
l0 Jun  2 - <b>26.71</b> Last Trade With Time
l1 26.71 Last Trade Price Only
l2 - High Limit
l3 - Low Limit
m0 26.18 - 26.78 Days Range
m2 N/A - N/A Days Range (Realtime)
m3 24.456 Fiftyday Moving Average
m4 21.7638 Two Hundredday Moving Average
m5 +4.9462 Change From Two Hundredday Moving Average
m6 +22.73% Percent Change From Two Hundredday Moving Average
m7 +2.254 Change From Fiftyday Moving Average
m8 +9.22% Percent Change From Fiftyday Moving Average
n0 Southwest Airline Name
n4 - Notes
o0 26.48 Open
p0 26.39 Previous Close
p1 - Price Paid
p2 +1.21% Change In Percent
p5 1.03 Price Sales
p6 2.54 Price Book
q0 Mar  4 Ex Dividend Date
r0 22.08 PE Ratio
r1 Jun 25 Dividend Pay Date
r2 N/A PE Ratio (Realtime)
r5 0.64 PEG Ratio
r6 17.59 Price EPS Estimate Current Year
r7 15.08 Price EPS Estimate Next Year
s0 LUV Symbol
s1 - Shares Owned
s6 17.781B Revenue
s7 2.60 Short Ratio
t1 4:00pm Last Trade Time
t6 Missing Format Variable. Trade Links DO NOT USE
t7 Missing Format Variable. Ticker Trend DO NOT USE
t8 27.25 Oneyr Target Price
v0 4973583 Volume
v1 - Holdings Value
v7 N/A Last Trade Date
w0 12.58 - 26.64 Year Range
w1 - - +1.21% Days Value Change
w4 N/A - N/A Days Value Change (Realtime)
x0 NYSE Stock Exchange
y0 0.61 Trailing Annual Dividend Yield In Percent
"""

    # Values that will be integers (i.e. bid/ask volumes) are being formatted as floats to
    # prevent any calculation errors arising from using them, due to how integer division
    # works in Python.
    #
    # Could reduce code size by putting codes formatted same ways into a list and checking for
    # each field's inclusion, but currently, explicitly evaluating each field will help ensure
    # everything is being done properly. It can be refactored in the future.
    for i, field in enumerate(fields):
        if field == 'a0':
            data = _format_as_float(data, i)
        elif field == 'a2':
            data = _format_as_float(data, i)
        elif field == 'a5':
            data = _format_as_float(data, i)
        elif field == 'b0':
            data = _format_as_float(data, i)
        elif field == 'b2':
            data = _format_as_float(data, i)
        elif field == 'b3':
            data = _format_as_float(data, i)
        elif field == 'b4':
            data = _format_as_float(data, i)
        elif field == 'b6':
            data = _format_as_float(data, i)
        elif field == 'c0':
            for data_line in data.values():
                pass
                #TODO: Get data during market hours to see format.
        elif field == 'c1':
            data = _format_as_float(data, i)
        elif field == 'c3':
            for data_line in data.values():
                pass
                #TODO: Find stock with commission to see format.
        elif field == 'c4':
            pass
        elif field == 'c6':
            data = _format_as_float(data, i)
        elif field == 'c8':
            for data_line in data.values():
                pass
                #TODO: Get data during market hours to see format.
        elif field == 'd0':
            data = _format_as_float(data, i)
        elif field == 'd2':
            pass
            #TODO: See what this field looks like
        elif field == 'e0':
            data = _format_as_float(data, i)
        elif field == 'e7':
            data = _format_as_float(data, i)
        elif field == 'e8':
            data = _format_as_float(data, i)
        elif field == 'e9':
            data = _format_as_float(data, i)
        elif field == 'g0':
            data = _format_as_float(data, i)
        elif field == 'g1':
            pass
            #TODO: Figure out how to format this field
        elif field == 'g3':
            data = _format_as_float(data, i)
        elif field == 'g4':
            data = _format_as_float(data, i)
        elif field == 'g5':
            pass
            #TODO: Figure out how to format this field
        elif field == 'g6':
            data = _format_as_float(data, i)
        elif field == 'h0':
            data = _format_as_float(data, i)
        elif field == 'i5':
            pass
            #TODO: Run in real time, figure out how to format this field
        elif field == 'j0':
            data = _format_as_float(data, i)
        elif field == 'j1':
            pass
            #TODO: Figure out this field - get results for large and small caps to see range of results
        elif field == 'j3':
            pass
            #TODO: Run in real time, figure out how to format this field
        elif field == 'j4':
            pass
            #TODO: Figure out this field - get results for large and small caps to see range of results
        elif field == 'j5':
            pass
            #TODO: Figure out this field - number with +
        elif field == 'j6':
            pass
            #TODO: Figure out this field - percentage formatting
        elif field == 'k0':
            data = _format_as_float(data, i)
        elif field == 'k1':
            pass
            #TODO: Run in real time, figure out how to format this field
        elif field == 'k2':
            pass
            #TODO: Run in real time, figure out how to format this field
        elif field == 'k3':
            data = _format_as_float(data, i)
        elif field == 'k4':
            pass
            #TODO: Run in real time, figure out how to format this field
        elif field == 'k5':
            pass
            #TODO: Figure out this field - percentage formatting
        elif field == 'l0':
            pass
            #TODO: Figure out this field - date, remove <b>, etc
        elif field == 'l2':
            data = _format_as_float(data, i)
        elif field == 'l3':
            data = _format_as_float(data, i)
        elif field == 'm0':
            pass
            #TODO: Figure out this field - range
        elif field == 'm2':
            pass
            #TODO: Figure out this field - range
        elif field == 'm3':
            data = _format_as_float(data, i)
        elif field == 'm4':
            data = _format_as_float(data, i)
        elif field == 'm5':
            pass
            #TODO: Run in real time, figure out how to format this field
        elif field == 'm6':
            pass
            #TODO: Figure out this field - percentage formatting
        elif field == 'm7':
            pass
            #TODO: Run in real time, figure out how to format this field
        elif field == 'm8':
            pass
            #TODO: Figure out this field - percentage formatting
        elif field == 'n0':
            pass
        elif field == 'n4':
            pass
        elif field == 'o0':
            data = _format_as_float(data, i)
        elif field == 'p0':
            data = _format_as_float(data, i)
        elif field == 'p1':
            data = _format_as_float(data, i)
        elif field == 'p2':
            pass
            #TODO: Figure out this field - percentage formatting
        elif field == 'p5':
            data = _format_as_float(data, i)
        elif field == 'p6':
            data = _format_as_float(data, i)
        elif field == 'q0':
            pass
            #TODO: Figure out this field - date formatting
        elif field == 'r0':
            data = _format_as_float(data, i)
        elif field == 'r1':
            pass
            #TODO: Figure out this field - date formatting
        elif field == 'r2':
            data = _format_as_float(data, i)
        elif field == 'r5':
            data = _format_as_float(data, i)
        elif field == 'r6':
            data = _format_as_float(data, i)
        elif field == 'r7':
            data = _format_as_float(data, i)
        elif field == 's0':
            pass
        elif field == 's1':
            data = _format_as_float(data, i)
        elif field == 's6':
            pass
            #TODO: Figure out this field - number with letter
        elif field == 's7':
            data = _format_as_float(data, i)
        elif field == 't1':
            pass
            #TODO: Figure out this field - time formatting
        elif field == 't8':
            data = _format_as_float(data, i)
        elif field == 'v0':
            data = _format_as_float(data, i)
        elif field == 'v1':
            data = _format_as_float(data, i)
        elif field == 'v7':
            pass
            #TODO: Figure out this field - date formatting
        elif field == 'w0':
            pass
            #TODO: Figure out this field - range
        elif field == 'w1':
            pass
            #TODO: Figure out this field - range with percent
        elif field == 'w4':
            pass
            #TODO: Figure out this field - range
        elif field == 'x0':
            pass
        elif field == 'y0':
            pass
            #TODO: Figure out this field - percent



if __name__ == '__main__':
    """
    results = {}
    sort_codes = get_yahoo_code.values()
    sort_codes.sort()
    for code in sort_codes:
        test_data = get_data(['LUV'], [code])
        print code, test_data['LUV'][0], get_yahoo_field[code]
    """
    test_data = _format_as_num({'a':['+1.23']}, 0)
    print test_data
    print test_data['a'][0]




















