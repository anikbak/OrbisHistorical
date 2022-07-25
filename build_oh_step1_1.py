##########################################################################
# ORBIS HISTORICAL DATA CONSTRUCTION: STEP 1.1
# --------------------------------------------
# * Combine Address, Industry Classification, Legal Information to make
#   details files.
# * Combine details files with financials to make complete datasets.
# * Implement Preliminary cleaning. 
##########################################################################
# Imports
import sys 
import os, itertools, fnmatch, time
import numpy as np 
import pandas as pd
sys.path.append("/oak/stanford/projects/econ/anikbak/bvdorbishistorical/code")
import routines_orbis_historical as prog

ymin,ymax = 1995,2015
years = np.arange(1995,2016,dtype=int)

countries = ['IT','ES','PT','SK','SI','BG','DK','HU','NO','PL','SE','AT','EE','FI','DE','GR','IE','NL','FR','BE','US','CN','IN']
'''
countries = ['AE', 'AF', 'AG', 'AI', 'AL', 'AM', 'AO', 'AR', 'AT', 'AU',
             'AW', 'AZ', 'BA', 'BB', 'BD', 'BE', 'BF', 'BG', 'BH', 'BI',
             'BJ', 'BM', 'BN', 'BO', 'BR', 'BS', 'BT', 'BW', 'BY', 'CA',
             'CD', 'CF', 'CG', 'CH', 'CI', 'CL', 'CM', 'CN', 'CO', 'CR',
             'CU', 'CV', 'CW', 'CY', 'CZ', 'DE', 'DJ', 'DK', 'DM', 'DO',
             'DZ', 'EC', 'EE', 'EG', 'ER', 'ES', 'ET', 'FI', 'FJ', 'FM',
             'FR', 'GA', 'GB', 'GD', 'GE', 'GH', 'GI', 'GM', 'GN', 'GQ',
             'GR', 'GT', 'GW', 'GY', 'HK', 'HN', 'HR', 'HU', 'ID', 'IE',
             'II', 'IL', 'IN', 'IQ', 'IR', 'IS', 'IT', 'JM', 'JO', 'JP',
             'KE', 'KG', 'KH', 'KM', 'KN', 'KR', 'KV', 'KW', 'KY', 'KZ',
             'LA', 'LB', 'LC', 'LI', 'LK', 'LR', 'LS', 'LT', 'LU', 'LV',
             'LY', 'MA', 'MC', 'MD', 'ME', 'MG', 'MH', 'MK', 'ML', 'MM',
             'MN', 'MR', 'MT', 'MU', 'MW', 'MX', 'MY', 'MZ', 'NA', 'NG',
             'NI', 'NL', 'NO', 'NP', 'NZ', 'OM', 'PA', 'PE', 'PG', 'PH',
             'PK', 'PL', 'PS', 'PT', 'PW', 'PY', 'QA', 'RO', 'RS', 'RU',
             'RW', 'SA', 'SC', 'SD', 'SE', 'SG', 'SI', 'SK', 'SL', 'SN',
             'SO', 'SS', 'ST', 'SV', 'SY', 'SZ', 'TD', 'TG', 'TH', 'TN',
             'TR', 'TT', 'TW', 'TZ', 'UA', 'UG', 'US', 'UY', 'UZ', 'VC',
             'VE', 'VG', 'VN', 'WS', 'YE', 'ZA', 'ZM', 'ZW']
'''

# Paths
datapath = "/oak/stanford/projects/econ/anikbak/bvdorbishistorical/"
savepath = datapath + 'data/'
blockpath = savepath + 'MakeBlocksByCountry/'

if blockpath != prog.blockpath:
    print("something's wrong... paths misaligned.")

# Clean Financials Data
for country in countries:
    # Check if Files Exist
    check = os.path.isfile(savepath+country+'/'+country+'_master.csv')
    if check == False:
        print(f"COULDN'T FIND MASTER FILE")
    else:
        prog.CleanFinancialsData_unitfix_imputeL(country,savepath)
