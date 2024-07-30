#!/usr/bin/env python

"""download_active_sites_table.py: processes active SNOTEL site metadata table (imperial units) to csv""" 

import pandas as pd
from datetime import datetime


# specify out directory
snotel_dir = '/uufs/chpc.utah.edu/common/home/skiles-group3/SNOTEL'

# read the table from NRCS url
url = 'https://wcc.sc.egov.usda.gov/nwcc/yearcount?network=sntl&state=&counttype=statelist'
df = pd.read_html(url)[0]

# add date of download
datetime.today().strftime('%Y-%m-%d')
todays_date = datetime.today().strftime('%Y%m%d')

# out csv filename 
outfn = f'{snotel_dir}/active_snotel_sites_{todays_date}.csv'
print(outfn)

# add column for site numbers and populate
df['site_num'] = [int(site_name.split('(')[1][:-1]) for site_name in df['site_name']]

# no checks for overwrite
df.to_csv(outfn)
