##########################################################################
# ORBIS HISTORICAL DATA CONSTRUCTION: STEP 0
# ------------------------------------------
# * Get data from Raw files
# * Construct country-level datasets, with minimal cleaning
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

#countries = ['IT','ES','PT','SK','SI','BG','DK','HU','NO','PL','SE','AT','EE','FI','DE','GR','IE','NL','FR','BE','US','CN','IN']
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

# Paths
datapath = "/oak/stanford/projects/econ/anikbak/bvdorbishistorical/"
savepath = datapath + 'data/'
blockpath = savepath + 'MakeBlocksByCountry/'

if blockpath != prog.blockpath:
    print("something's wrong... paths misaligned.")

filename_fin = datapath + "Industry-Global_financials_and_ratios.txt"
filename_add = datapath + "Contact_info.txt" 
filename_ind = datapath + "Industry_classifications.txt"
filename_leg = datapath + "Legal_info.txt"

# Create Directories and Preallocate Datasets
if os.path.isdir(blockpath):
    print(f'Block Path exists...')
else:
    os.mkdir(blockpath)
    print(f'Made Block Path')

for country in countries:
    if os.path.isdir(blockpath+country):
        print(f'directory {blockpath+country} exists already.')
        continue
    else:
        os.mkdir(blockpath+country)
        print(f'directory {blockpath+country} created.')
    if os.path.isdir(savepath+country):
        print(f'directory {savepath+country} exists already.')
        continue
    else:
        os.mkdir(savepath+country)
        print(f'directory {savepath+country} created.')

# Extract data from file in chunks
chunkN = 3_000_000
chunks_add = pd.read_csv(filename_add,sep='\t',chunksize=chunkN,usecols=prog.varlist_add)
chunks_ind = pd.read_csv(filename_ind,sep='\t',chunksize=chunkN,usecols=prog.varlist_ind)
chunks_leg = pd.read_csv(filename_leg,sep='\t',chunksize=chunkN,usecols=prog.varlist_leg)
chunks_fin = pd.read_csv(filename_fin,sep='\t',chunksize=chunkN,usecols=prog.varlist_fin)

summary_add = prog.MakeBlocks(chunks_add,'add',countries,blockpath)
summary_add.to_csv(savepath+'summary_add.csv')
for country in countries:
    prog.MakeWhole(country,'add',savepath,blockpath)

summary_ind = prog.MakeBlocks(chunks_ind,'ind',countries,blockpath)
summary_ind.to_csv(savepath+'summary_ind.csv')
for country in countries:
    prog.MakeWhole(country,'ind',savepath,blockpath)

summary_leg = prog.MakeBlocks(chunks_leg,'leg',countries,blockpath)
summary_leg.to_csv(savepath+'summary_leg.csv')
for country in countries:
    prog.MakeWhole(country,'leg',savepath,blockpath)

summary_fin = prog.MakeBlocks(chunks_fin,'fin',countries,blockpath)
summary_fin.to_csv(savepath+'summary_fin.csv')
for country in countries:
    prog.MakeWhole(country,'fin',savepath,blockpath)