import urllib
import urllib2

#https://code.google.com/p/yahoo-finance-managed/wiki/CSVAPI

base_url = 'http://download.finance.yahoo.com/d/quotes.csv?'
suffix = '&e=.csv'

codes = {
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
'e0':	'Diluted E P S',
'j4':	'E B I T D A',
'e7':	'E P S Estimate Current Year',
'e9':	'E P S Estimate Next Quarter',
'e8':	'E P S Estimate Next Year',
'q0':	'Ex Dividend Date',
'm3':	'Fiftyday Moving Average',
'f6':	'Shares Float',
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
'i0':	'More Info',
'n0':	'Name',
'n4':	'Notes',
't8':	'Oneyr Target Price',
'o0':	'Open',
'i5':	'Order Book (Realtime)',
'r5':	'P E G Ratio',
'r0':	'P E Ratio',
'r2':	'P E Ratio (Realtime)',
'm8':	'Percent Change From Fiftyday Moving Average',
'm6':	'Percent Change From Two Hundredday Moving Average',
'k5':	'Change In Percent From Year High',
'j6':	'Percent Change From Year Low',
'p0':	'Previous Close',
'p6':	'Price Book',
'r6':	'Price E P S Estimate Current Year',
'r7':	'Price E P S Estimate Next Year',
'p1':	'Price Paid',
'p5':	'Price Sales',
's6':	'Revenue',
's1':	'Shares Owned',
'j2':	'Shares Outstanding',
's7':	'Short Ratio',
'x0':	'Stock Exchange',
's0':	'Symbol',
't7':	'Ticker Trend',
'd2':	'Trade Date',
't6':	'Trade Links',
'f0':	'Trade Links Additional',
'm4':	'Two Hundredday Moving Average',
'v0':	'Volume',
'k0':	'Year High',
'j0':	'Year Low',
'w0':	'Year Range'}


def get_info(symbols, fields):
    symbol_str = 's=' + urllib.quote(','.join(symbols), ',')
    field_str = '&f=' + fields
    raw_file = urllib2.urlopen(base_url+symbol_str+field_str+suffix)
    print raw_file.read()




if __name__ == '__main__':
    symbols = ['@^DJI','GOOG']
    fields = 'n0s0l1o0p0'
    get_info(symbols, fields)
    for i in xrange(5):
        print codes[fields[2*i:(2*i)+2]]