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







def get_current_data(symbols, codes=None):
    """get_current_data(symbols, codes=None)

Retrieves stock data from Yahoo Finance. Quotes may be delayed up to 15 minutes.

symbols - list of the stock symbols
codes - list of the codes for the desired fields. If None, will get all data. Use show_yahoo_codes()
        to see what each code is.
"""

    # Get all data if no codes are specified
    if codes is None:
        codes = get_stock_field_code.values()
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


def get_historical_data(symbol, from_date, to_date, interval):
    base_url = 'http://ichart.yahoo.com/table.csv?'
    s = 's=%s&' % symbol
    a = 'a=%s&' % str(from_date.month-1)
    b = 'b=%s&' % str(from_date.day)
    c = 'c=%s&' % str(from_date.year)
    d = 'd=%s&' % str(to_date.month-1)
    e = 'e=%s&' % str(to_date.day)
    f = 'f=%s&' % str(to_date.year)
    g = 'g=%s&' % interval
    suffix = 'ignore=.csv'
    total_url = base_url+s+a+b+c+d+e+f+g+suffix
    raw_file = urllib2.urlopen(total_url).read().splitlines()
    data = {'headers': raw_file[0].split(',')}
    for line in raw_file[1:]:
        line = line.split(',')
        data_date = date(int(line[0][:4]), int(line[0][5:7]), int(line[0][8:]))
        line = line[1:]
        data[data_date] = [float(num) for num in line]
    return data


def get_sector_data():
    """get_sector_data()

Returns a dictionary of data by sector. Use the key 'headers' to see what each item
in the data set represents. The other keys of the dictionary will be the sectors.
"""
    # Only options are sorting options, which doesn't matter since it's being thrown
    # into a dictionary, so this default URL is used.
    total_url = 'http://biz.yahoo.com/p/csv/s_conameu.csv'
    raw_file = urllib2.urlopen(total_url).read().splitlines()
    #First element is headers, last element is a blank list
    data = {'headers': raw_file[0].replace('"','').split(',')}
    for line in raw_file[1:-1]:
        line = line.replace('"','').split(',')
        data[line[0]] = [_format_as_float(item) for item in line[1:]]
        #Certain fields are percentage, so divide by 100
        for i in [0, 3, 4]:
            data[line[0]][i] /= 100
    return data


def get_industry_data(industry_num):
    """get_industry_data(industry_num)

Returns a dictionary of data by industry. Below is a key for the industry numbers:

Basic_Materials	 	1
Conglomerates	 	2
Consumer_Goods	 	3
Financial	    	4
Healthcare	    	5
Industrial_Goods	6
Services	    	7
Technology	    	8
Utilities	    	9
"""
    total_url = 'http://biz.yahoo.com/p/csv/%sconameu.csv' % industry_num
    raw_file = urllib2.urlopen(total_url).read().splitlines()
    print raw_file
    data = {'headers': raw_file[0].replace('"','').split(',')}
    for line in raw_file[1:-1]:
        old_co_name = line[:line.find('"',1)+1]
        new_co_name = old_co_name.replace(',','')
        line = line.replace(old_co_name, new_co_name)
        line = line.replace('"','').split(',')
        data[line[0]] = [_format_as_float(item) if item != 'NA' else item for item in line[1:]]
        #Certain fields are percentage, so divide by 100
        for i in [0, 3, 4]:
            if data[line[0]][i] != 'NA':
                data[line[0]][i] /= 100
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

def _format_data(codes, data):
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

#--------------------------------------------------------------------------------
# Displaying the codes for the Yahoo Finance API
#--------------------------------------------------------------------------------

def show_stock_field_codes(by_field=True):
    get_stock_field_name = {
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
    #Reverse get_stock_field_name dictionary to look up codes for desired fields.
    get_stock_field_code = dict(zip(get_stock_field_name.values(),get_stock_field_name.keys()))
    if by_field:
        sorted_field_list = get_stock_field_name.values()
        sorted_field_list.sort()
        print '\n'.join(['%s: %s' % (get_stock_field_code[field], field) for field in sorted_field_list])
    else:
        sorted_code_list = get_stock_field_code.values()
        sorted_code_list.sort()
        print '\n'.join(['%s: %s' % (code, get_stock_field_name[code]) for code in sorted_code_list])


def show_industry_detail_codes(by_field=True):
    get_industry_detail_code = {
    'Agricultural_Chemicals'                :112,
    'Aluminum'                              :132,
    'Chemicals-Major_Diversified'           :110,
    'Copper'                                :131,
    'Gold'                                  :134,
    'Independent_Oil&Gas'                   :121,
    'Industrial_Metals&Minerals'            :133,
    'Major_Integrated_Oil&Gas'              :120,
    'Nonmetallic_Mineral_Mining'            :136,
    'Oil&Gas_Drilling&Exploration'          :123,
    'Oil&Gas_Equipment&Services'            :124,
    'Oil&Gas_Pipelines'                     :125,
    'Oil&Gas_Refining&Marketing'            :122,
    'Silver'                                :135,
    'Specialty_Chemicals'                   :113,
    'Steel&Iron'                            :130,
    'Synthetics'                            :111,
    'Conglomerates'                         :210,
    'Appliances'                            :310,
    'Auto_Manufacturers-Major'              :330,
    'Auto_Parts'                            :333,
    'Beverages-Brewers'                     :346,
    'Beverages-Soft_Drinks'                 :348,
    'Beverages-Wineries&Distillers'         :347,
    'Business_Equipment'                    :313,
    'Cigarettes'                            :350,
    'Cleaning_Products'                     :326,
    'Confectioners'                         :345,
    'Dairy_Products'                        :344,
    'Electronic_Equipment'                  :314,
    'Farm_Products'                         :341,
    'Food-Major_Diversified'                :340,
    'Home_Furnishings&Fixtures'             :311,
    'Housewares&Accessories'                :312,
    'Meat_Products'                         :343,
    'Office_Supplies'                       :327,
    'Packaging&Containers'                  :325,
    'Paper&Paper_Products'                  :324,
    'Personal_Products'                     :323,
    'Photographic_Equipment&Supplies'       :318,
    'Processed&Packaged_Goods'              :342,
    'Recreational_Goods,_Other'             :317,
    'Recreational_Vehicles'                 :332,
    'Rubber&Plastics'                       :322,
    'Sporting_Goods'                        :316,
    'Textile-Apparel_Clothing'              :320,
    'Textile-Apparel_Footwear&Accessories'  :321,
    'Tobacco_Products,_Other'               :351,
    'Toys&Games'                            :315,
    'Trucks&Other_Vehicles'                 :331,
    'Accident&Health_Insurance'             :431,
    'Asset_Management'                      :422,
    'Closed-End_Fund-Debt'                  :425,
    'Closed-End_Fund-Equity'                :426,
    'Closed-End_Fund-Foreign'               :427,
    'Credit_Services'                       :424,
    'Diversified_Investments'               :423,
    'Foreign_Money_Center_Banks'            :417,
    'Foreign_Regional_Banks'                :418,
    'Insurance_Brokers'                     :434,
    'Investment_Brokerage-National'         :420,
    'Investment_Brokerage-Regional'         :421,
    'Life_Insurance'                        :430,
    'Money_Center_Banks'                    :410,
    'Mortgage_Investment'                   :447,
    'Property&Casualty_Insurance'           :432,
    'Property_Management'                   :448,
    'REIT-Diversified'                      :440,
    'REIT-Healthcare_Facilities'            :442,
    'REIT-Hotel/Motel'                      :443,
    'REIT-Industrial'                       :444,
    'REIT-Office'                           :441,
    'REIT-Residential'                      :445,
    'REIT-Retail'                           :446,
    'Real_Estate_Development'               :449,
    'Regional-Mid-Atlantic_Banks'           :412,
    'Regional-Midwest_Banks'                :414,
    'Regional-Northeast_Banks'              :411,
    'Regional-Pacific_Banks'                :416,
    'Regional-Southeast_Banks'              :413,
    'Regional-Southwest_Banks'              :415,
    'Savings&Loans'                         :419,
    'Surety&Title_Insurance'                :433,
    'Biotechnology'                         :515,
    'Diagnostic_Substances'                 :516,
    'Drug_Delivery'                         :513,
    'Drug_Manufacturers-Major'              :510,
    'Drug_Manufacturers-Other'              :511,
    'Drug_Related_Products'                 :514,
    'Drugs-Generic'                         :512,
    'Health_Care_Plans'                     :522,
    'Home_Health_Care'                      :526,
    'Hospitals'                             :524,
    'Long-Term_Care_Facilities'             :523,
    'Medical_Appliances&Equipment'          :521,
    'Medical_Instruments&Supplies'          :520,
    'Medical_Laboratories&Research'         :525,
    'Medical_Practitioners'                 :527,
    'Specialized_Health_Services'           :528,
    'Aerospace/Defense-Major_Diversified'   :610,
    'Aerospace/Defense_Products&Services'   :611,
    'Cement'                                :633,
    'Diversified_Machinery'                 :622,
    'Farm&Construction_Machinery'           :620,
    'General_Building_Materials'            :634,
    'General_Contractors'                   :636,
    'Heavy_Construction'                    :635,
    'Industrial_Electrical_Equipment'       :627,
    'Industrial_Equipment&Components'       :621,
    'Lumber,_Wood_Production'               :632,
    'Machine_Tools&Accessories'             :624,
    'Manufactured_Housing'                  :631,
    'Metal_Fabrication'                     :626,
    'Pollution&Treatment_Controls'          :623,
    'Residential_Construction'              :630,
    'Small_Tools&Accessories'               :625,
    'Textile_Industrial'                    :628,
    'Waste_Management'                      :637,
    'Advertising_Agencies'                  :720,
    'Air_Delivery&Freight_Services'         :773,
    'Air_Services,_Other'                   :772,
    'Apparel_Stores'                        :730,
    'Auto_Dealerships'                      :744,
    'Auto_Parts_Stores'                     :738,
    'Auto_Parts_Wholesale'                  :750,
    'Basic_Materials_Wholesale'             :758,
    'Broadcasting-Radio'                    :724,
    'Broadcasting-TV'                       :723,
    'Building_Materials_Wholesale'          :751,
    'Business_Services'                     :760,
    'CATV_Systems'                          :725,
    'Catalog&Mail_Order_Houses'             :739,
    'Computers_Wholesale'                   :755,
    'Consumer_Services'                     :763,
    'Department_Stores'                     :731,
    'Discount,_Variety_Stores'              :732,
    'Drug_Stores'                           :733,
    'Drugs_Wholesale'                       :756,
    'Education&Training_Services'           :766,
    'Electronics_Stores'                    :735,
    'Electronics_Wholesale'                 :753,
    'Entertainment-Diversified'             :722,
    'Food_Wholesale'                        :757,
    'Gaming_Activities'                     :714,
    'General_Entertainment'                 :716,
    'Grocery_Stores'                        :734,
    'Home_Furnishing_Stores'                :737,
    'Home_Improvement_Stores'               :736,
    'Industrial_Equipment_Wholesale'        :752,
    'Jewelry_Stores'                        :742,
    'Lodging'                               :710,
    'Major_Airlines'                        :770,
    'Management_Services'                   :769,
    'Marketing_Services'                    :721,
    'Medical_Equipment_Wholesale'           :754,
    'Movie_Production,_Theaters'            :726,
    'Music&Video_Stores'                    :743,
    'Personal_Services'                     :762,
    'Publishing-Books'                      :729,
    'Publishing-Newspapers'                 :727,
    'Publishing-Periodicals'                :728,
    'Railroads'                             :776,
    'Regional_Airlines'                     :771,
    'Rental&Leasing_Services'               :761,
    'Research_Services'                     :768,
    'Resorts&Casinos'                       :711,
    'Restaurants'                           :712,
    'Security&Protection_Services'          :765,
    'Shipping'                              :775,
    'Specialty_Eateries'                    :713,
    'Specialty_Retail,_Other'               :745,
    'Sporting_Activities'                   :715,
    'Sporting_Goods_Stores'                 :740,
    'Staffing&Outsourcing_Services'         :764,
    'Technical_Services'                    :767,
    'Toy&Hobby_Stores'                      :741,
    'Trucking'                              :774,
    'Wholesale,_Other'                      :759,
    'Application_Software'                  :821,
    'Business_Software&Services'            :826,
    'Communication_Equipment'               :841,
    'Computer_Based_Systems'                :812,
    'Computer_Peripherals'                  :815,
    'Data_Storage_Devices'                  :813,
    'Diversified_Communication_Services'    :846,
    'Diversified_Computer_Systems'          :810,
    'Diversified_Electronics'               :836,
    'Healthcare_Information_Services'       :825,
    'Information&Delivery_Services'         :827,
    'Information_Technology_Services'       :824,
    'Internet_Information_Providers'        :851,
    'Internet_Service_Providers'            :850,
    'Internet_Software&Services'            :852,
    'Long_Distance_Carriers'                :843,
    'Multimedia&Graphics_Software'          :820,
    'Networking&Communication_Devices'      :814,
    'Personal_Computers'                    :811,
    'Printed_Circuit_Boards'                :835,
    'Processing_Systems&Products'           :842,
    'Scientific&Technical_Instruments'      :837,
    'Security_Software&Services'            :823,
    'Semiconductor-Broad_Line'              :830,
    'Semiconductor-Integrated_Circuits'     :833,
    'Semiconductor-Specialized'             :832,
    'Semiconductor_Equipment&Materials'     :834,
    'Semiconductor-Memory_Chips'            :831,
    'Technical&System_Software'             :822,
    'Telecom_Services-Domestic'             :844,
    'Telecom_Services-Foreign'              :845,
    'Wireless_Communications'               :840,
    'Diversified_Utilities'                 :913,
    'Electric_Utilities'                    :911,
    'Foreign_Utilities'                     :910,
    'Gas_Utilities'                         :912,
    'Water_Utilities'                       :914
    }
    #Reverse the dictionary to look up industry names by key
    get_industry_detail_name = dict(zip(get_industry_detail_code.values(), get_industry_detail_code.keys()))
    if by_field:
        sorted_industry_list = get_industry_detail_code.keys()
        sorted_industry_list.sort()
        print '\n'.join(['%s: %s' % (get_industry_detail_code[key], key) for key in sorted_industry_list])
    else:
        sorted_code_list = get_industry_detail_name.keys()
        sorted_code_list.sort()
        print '\n'.join(['%s: %s' % (code, get_industry_detail_name[code]) for code in sorted_code_list])


#--------------------------------------------------------------------------------
# End
#--------------------------------------------------------------------------------


if __name__ == '__main__':
    show_industry_detail_codes()
    test_data = get_industry_data(112)
    for item in test_data.keys():
        print item, test_data[item]
    """
    test_data = get_historical_data('LUV', date(2014, 1, 1), date(2014, 6, 3), 'w')
    print test_data['headers']
    print test_data[date(2014, 5, 19)]
    """
