import datetime
import numpy as np
import  pandas as pd
datetime64Obj = np.datetime64('2002-07-04T02:55:41-0700')
date = pd.to_datetime(datetime64Obj)
print(date.year)

# string = "19 Nov 2015  18:45:00.000"
# date = datetime.datetime.strptime(string, "%d %b %Y  %H:%M:%S.%f")
#
# lis = str(date).split('-')
#
# print(lis[2].split( ))
#
# d = '2/18/2011 12:13' #month day year
#
# import re
# if isinstance(d, str) == True:
#     sp = re.split('/| ',d)
#     month = sp[0]
#     day = sp[1]
#     year = sp[2]
#     print(year)