##########################################################################################################################    
# Main Objects
##########################################################################################################################

import numpy as np
import pandas as pd
import os, fnmatch

# Paths
datapath = "/oak/stanford/projects/econ/anikbak/bvdorbishistorical/"
savepath = datapath + 'data/'
blockpath = savepath + 'MakeBlocksByCountry/'

filename_fin = datapath + "Industry-Global_financials_and_ratios.txt"
filename_add = datapath + "Contact_info.txt" 
filename_ind = datapath + "Industry_classifications.txt"
filename_leg = datapath + "Legal_info.txt"

# Variable Lists
varlist_add = ['BvD ID number', 'NAME_INTERNAT', 'NAME_NATIVE', 'Postcode', 'City', 'City (native)', 'Country', 'Country ISO code', 'Metropolitan area (in US)', 'State or province (in US or Canada)', 'County (in US or Canada)', 'Telephone number', 'Fax number', 'Region in country', 'Type of region in country', 'NUTS1', 'NUTS2', 'NUTS3']
varlist_ind = ['BvD ID number', 'National industry classification used by the IP', 'Primary code(s) in this classification', 'Primary code in national industry classification, text description', 'Secondary code(s) in this classification', 'Secondary code in national industry classification, text description', 'NACE Rev. 2 main section', 'NACE Rev. 2, Core code (4 digits)', 'NACE rev.2, core code text description', 'NACE Rev. 2, Primary code(s)', 'NACE rev.2, primary code , text description', 'NACE Rev. 2, Secondary code(s)', 'NACE rev.2, secondary code , text description', 'NAICS 2012, Core code (4 digits)', 'NAICS 2012, core code, text description', 'NAICS 2012, Primary code(s)', 'NAICS 2012, primary code, text description', 'NAICS 2012, Secondary code(s)', 'NAICS 2012, secondary code, text description', 'US SIC, Core code (3 digits)', 'US SIC core code, text description', 'US SIC, Primary code(s)', 'US SIC primary code, text description', 'US SIC, Secondary code(s)', 'US SIC secondary code, text description', 'BvD major sector']
varlist_leg = ['BvD ID number', 'Status', 'Status date', 'Standardised legal form', 'National legal form', 'Date of incorporation', 'Type of entity', 'Category of the company', 'Listed/Delisted/Unlisted', 'Delisted date', 'Main exchange', 'IPO date', 'Information provider']
varlist_fin = ['BvD ID number', 'Consolidation code', 'Filing type', 'Closing date', 'Number of months', 'Audit status', 'Accounting practice', 'Source (for publicly quoted companies)', 'Original units', 'Original currency', 'Fixed assets', 'Intangible fixed assets', 'Tangible fixed assets', 'Other fixed assets', 'Current assets', 'Stock', 'Debtors', 'Other current assets', 'Cash & cash equivalent', 'Total assets', 'Shareholders funds', 'Capital', 'Other shareholders funds', 'Non-current liabilities', 'Long term debt', 'Other non-current liabilities', 'Provisions', 'Current liabilities', 'Loans', 'Creditors', 'Other current liabilities', 'Total shareh. funds & liab.', 'Working capital', 'Net current assets', 'Enterprise value', 'Number of employees', 'Operating revenue (Turnover)', 'Sales', 'Costs of goods sold', 'Gross profit', 'Other operating expenses', 'Operating P/L [=EBIT]', 'Financial revenue', 'Financial expenses', 'Financial P/L', 'P/L before tax', 'Taxation', 'P/L after tax', 'Extr. and other revenue', 'Extr. and other expenses', 'Extr. and other P/L', 'P/L for period [=Net income]', 'Export revenue', 'Material costs', 'Costs of employees', 'Depreciation & Amortization', 'Interest paid', 'Research & Development expenses', 'Cash flow', 'Added value', 'EBITDA', 'ROE using P/L before tax (%)', 'ROCE using P/L before tax (%)', 'ROA using P/L before tax (%)', 'ROE using Net income (%)', 'ROCE using Net income (%)', 'ROA using Net income (%)', 'Profit margin (%)', 'Gross margin (%)', 'EBITDA margin (%)', 'EBIT margin (%)', 'Cash flow / Operating revenue (%)', 'Enterprise value / EBITDA (x)', 'Market cap / Cash flow from operations (x)', 'Net assets turnover (x)', 'Interest cover (x)', 'Stock turnover (x)', 'Collection period (days)', 'Credit period (days)', 'Export revenue / Operating revenue (%)', 'R&D expenses / Operating revenue (%)', 'Current ratio (x)', 'Liquidity ratio (x)', 'Shareholders liquidity ratio (x)', 'Solvency ratio (Asset based) (%)', 'Solvency ratio (Liability based) (%)', 'Gearing (%)', 'Profit per employee (th)', 'Operating revenue per employee (th)', 'Costs of employees / Operating revenue (%)', 'Average cost of employee (th)', 'Shareholders funds per employee (th)', 'Working capital per employee (th)', 'Total assets per employee (th)']

varlist_details = varlist_add + varlist_ind + varlist_leg
varlist_details_important = ['BvD ID number', 'NAME_INTERNAT', 'NAME_NATIVE', 'Postcode', 'City', 'City (native)', 'Country', 'Country ISO code', 'Region in country', 'Type of region in country', 'NUTS1', 'NUTS2', 'NUTS3', 'National industry classification used by the IP', 'Primary code(s) in this classification', 'Primary code in national industry classification, text description', 'Secondary code(s) in this classification', 'Secondary code in national industry classification, text description', 'NACE Rev. 2 main section', 'NACE Rev. 2, Core code (4 digits)', 'NACE rev.2, core code text description', 'NACE Rev. 2, Primary code(s)', 'NACE rev.2, primary code , text description', 'NACE Rev. 2, Secondary code(s)', 'NACE rev.2, secondary code , text description', 'NAICS 2012, Core code (4 digits)', 'NAICS 2012, core code, text description', 'NAICS 2012, Primary code(s)', 'NAICS 2012, primary code, text description', 'NAICS 2012, Secondary code(s)', 'NAICS 2012, secondary code, text description', 'US SIC, Core code (3 digits)', 'US SIC core code, text description', 'US SIC, Primary code(s)', 'US SIC primary code, text description', 'US SIC, Secondary code(s)', 'US SIC secondary code, text description', 'BvD major sector', 'Status', 'Status date', 'Standardised legal form', 'National legal form', 'Date of incorporation', 'Type of entity', 'Category of the company', 'Listed/Delisted/Unlisted', 'Delisted date', 'Main exchange', 'IPO date', 'Information provider']
varlist_reduced = varlist_details_important+varlist_fin

varlist_TFP_sample = [  'BvD ID number', 'Closing date', 'year', 'NAME_INTERNAT', 'Postcode', 'City', 'City (native)', 'Country', 'Country ISO code', 'Region in country', 'Type of region in country', 'NUTS1', 'NUTS2', 'NUTS3', 
                        'National industry classification used by the IP', 'Primary code(s) in this classification', 'Secondary code(s) in this classification', 
                        'NACE Rev. 2 main section', 'NACE Rev. 2, Core code (4 digits)', 'NACE Rev. 2, Primary code(s)', 'NACE Rev. 2, Secondary code(s)', 
                        'NAICS 2012, Core code (4 digits)', 'NAICS 2012, Primary code(s)', 'NAICS 2012, Secondary code(s)', 
                        'US SIC, Core code (3 digits)', 'US SIC, Primary code(s)', 'US SIC, Secondary code(s)', 'BvD major sector', 
                        'Status', 'Status date', 'Standardised legal form', 'National legal form', 'Date of incorporation', 'Type of entity', 'Category of the company',
                        'Consolidation code', 'Filing type', 'Number of months', 'Audit status', 'Accounting practice', 'Source (for publicly quoted companies)', 'Original units', 'Original currency', 'Fixed assets', 'Intangible fixed assets', 'Tangible fixed assets', 'Other fixed assets', 'Current assets', 'Stock', 'Debtors', 'Other current assets', 'Cash & cash equivalent', 'Total assets', 'Shareholders funds', 'Capital', 'Other shareholders funds', 'Non-current liabilities', 'Long term debt', 'Other non-current liabilities', 'Provisions', 'Current liabilities', 'Loans', 'Creditors', 'Other current liabilities', 'Total shareh. funds & liab.', 'Working capital', 'Net current assets', 'Enterprise value', 'Number of employees', 'Operating revenue (Turnover)', 'Sales', 'Costs of goods sold', 'Gross profit', 'Other operating expenses', 'Operating P/L [=EBIT]', 'Financial revenue', 'Financial expenses', 'Financial P/L', 'P/L before tax', 'Taxation', 'P/L after tax', 'Extr. and other revenue', 'Extr. and other expenses', 'Extr. and other P/L', 'P/L for period [=Net income]', 'Export revenue', 'Material costs', 'Costs of employees', 'Depreciation & Amortization', 'Interest paid', 'Research & Development expenses', 'Cash flow', 'Added value', 'EBITDA', 
                        'ROE using P/L before tax (%)', 'ROCE using P/L before tax (%)', 'ROA using P/L before tax (%)', 'ROE using Net income (%)', 'ROCE using Net income (%)', 'ROA using Net income (%)', 'Profit margin (%)', 'Gross margin (%)', 'EBITDA margin (%)', 'EBIT margin (%)', 'Cash flow / Operating revenue (%)', 'Enterprise value / EBITDA (x)', 'Market cap / Cash flow from operations (x)', 'Net assets turnover (x)', 'Interest cover (x)', 'Stock turnover (x)', 
                        'Collection period (days)', 'Credit period (days)', 'Export revenue / Operating revenue (%)', 'R&D expenses / Operating revenue (%)', 'Current ratio (x)', 'Liquidity ratio (x)', 'Shareholders liquidity ratio (x)', 'Solvency ratio (Asset based) (%)', 'Solvency ratio (Liability based) (%)', 'Gearing (%)', 'Profit per employee (th)', 'Operating revenue per employee (th)', 'Costs of employees / Operating revenue (%)', 'Average cost of employee (th)', 'Shareholders funds per employee (th)', 'Working capital per employee (th)', 'Total assets per employee (th)', 
                        'Year incorporated']

##########################################################################################################################    
# Main Routines: Dataset Construction
##########################################################################################################################

def Importer(chunk,stub,countries):
    # Define Country
    chunk = chunk[pd.notna(chunk['BvD ID number'])]
    chunk['country'] = chunk['BvD ID number'].str[:2]
    chunk = chunk.loc[chunk['country'].isin(countries)]
    # Perform Management
    if stub == 'add':
        chunk = chunk.loc[chunk['country']==chunk['Country ISO code']]
    elif stub == 'ind':
        chunk = chunk[(pd.notna(chunk['NACE Rev. 2, Core code (4 digits)']))]
        chunk = clean_nace(chunk)
    elif stub == 'leg':
        chunk = make_years_incorporated(chunk)
    elif stub == 'fin':
        chunk['day'] = (chunk['Closing date']%100).astype(int)
        chunk['month'] = (((chunk['Closing date']-chunk['day'])/100) % 100).astype(int)
        chunk['year'] = (((chunk['Closing date']-chunk['month']*100 - chunk['day'])/10_000) % 10_000).astype(int)
        chunk.loc[chunk['month']<6,'year'] = chunk.loc[chunk['month']<6,'year']-1
        chunk = chunk.loc[(chunk['year']>1994) & (chunk['year']<2016)]
        chunk = chunk.drop(columns=['day','month'])
        chunk = make_years(chunk)
    # Return
    return chunk

def MakeBlocks(chunks,stub,countries,blockpath):
    chunkno = 0
    Summary = pd.DataFrame(columns=countries+['chunkno','Total'])
    Summary['chunkno'] = np.arange(1,501,dtype=int)
    for country in countries:
        Summary[country] = 0
    print('*******************************************************')
    print('Making Blocks')
    for chunk in chunks:
        chunkno += 1
        print(f'    chunk {chunkno} imported')
        chunk = Importer(chunk,stub,countries)
        chunk = chunk.drop_duplicates()
        countries_chunk = chunk['country'].unique()
        for country in countries_chunk:
            print(f'        saving {country}')
            temp = chunk.loc[chunk['country']==country]
            Summary.loc[(Summary['chunkno']==chunkno),country] = len(temp.index)
            temp.to_csv(blockpath+country+'/'+country+'_'+stub+'_'+str(chunkno)+'.csv')
            chunk = chunk.loc[chunk['country']!=country]
        chunk.to_csv(blockpath+'OtherCountries_'+stub+'_'+str(chunkno)+'.csv')
    Summary['Total'] = 0
    for country in countries:
        Summary['Total'] = Summary['Total']+Summary[country]
    return Summary

def MakeWhole(country,stub,savepath,blockpath):
    print('*******************************************************')
    print(f'Stitching {stub} Blocks for {country}')
    blockpath_c = blockpath+country+'/'
    files_list = [blockpath_c + filename for filename in fnmatch.filter(os.listdir(blockpath_c),country+'_'+stub+'*.csv')]
    files_list.sort()
    # Loop over files for stub
    cols = list(pd.read_csv(files_list[0],nrows=2).columns)
    temp = pd.DataFrame(columns = cols)
    for file in files_list:
        print(f'    on file {file}')
        temp = temp.append(pd.read_csv(file,low_memory=False),sort=True)
    # Save
    temp.to_csv(savepath+country+'/'+country+'_'+stub+'.csv')
    return None

def MakeDetails(country,savepath,varlist_details):
    countrypath = savepath+country+'/'
    print('*******************************************************')
    print(f'Details Dataset for {country}')
    
    # Check if all datasets exist
    has_add = os.path.isfile(countrypath+country+'_add.csv')
    has_ind = os.path.isfile(countrypath+country+'_ind.csv')
    has_leg = os.path.isfile(countrypath+country+'_leg.csv')
    has_all = has_add * has_ind * has_leg

    if has_all != 1:

        if has_add == False:
            print("    can't find Addresses file.")
        if has_ind == False:
            print("    can't find Industries file.")
        if has_leg == False:
            print("    can't find Legals file.")
        return None
    
    else:

        # Import Datasets
        print('    importing add')
        add = pd.read_csv(countrypath+country+'_add.csv',usecols=varlist_add)
        print('    importing ind')
        ind = pd.read_csv(countrypath+country+'_ind.csv',usecols=varlist_ind)
        print('    importing leg')
        leg = pd.read_csv(countrypath+country+'_leg.csv',usecols=varlist_leg+['Year incorporated'])
        
        # Ensure Uniqueness by BvD ID
        add = KeepMostCompleteObs(add,'rowcount','BvD ID number')
        ind = KeepMostCompleteObs(ind,'rowcount','BvD ID number')
        leg = KeepMostCompleteObs(leg,'rowcount','BvD ID number')
        
        # Combine Files
        print('    combining add,ind,leg')
        details = add.merge(ind,on='BvD ID number',how='outer',indicator='merge_add_ind')
        details = details.merge(leg,on='BvD ID number',how='outer',indicator='merge_add_ind_leg')
        details = details.iloc[:,~details.columns.duplicated()]
        details = details[varlist_details+['Year incorporated','merge_add_ind','merge_add_ind_leg']]
        details = details.drop_duplicates()

        # Tag Duplicate Observations
        details = details.set_index('BvD ID number')
        details['ii'] = 1
        details['NDuplicates'] = details.groupby('BvD ID number')['ii'].sum()
        details = details.reset_index()

        # Save
        details.to_csv(countrypath+country+'_details.csv')
        return None

def CombineDetailsFinancials(country,savepath):
    countrypath = savepath+country+'/'
    print('*******************************************************')
    print(f'Combining Datasets for {country}')

    # Check availability
    has_det = os.path.isfile(countrypath+country+'_details.csv')
    has_fin = os.path.isfile(countrypath+country+'_fin.csv')
    has_all = has_det * has_fin

    if has_all:

        # Import Datasets
        details = pd.read_csv(countrypath+country+'_details.csv',usecols=varlist_details+['Year incorporated','merge_add_ind','merge_add_ind_leg'])
        fin = pd.read_csv(countrypath+country+'_fin.csv',usecols = varlist_fin+['year'])

        # Fix something weird - The Triple Merge creates a Tuple BvD ID Number. Not sure why this is happening.
        details[['1','2','3']] = details['BvD ID number'].str.split(",",expand=True)
        for v in ['1','2','3']:
            details[v] = details[v].str.replace("\(","").str.replace("\)","").str.replace("'","").str.strip()
        details['consistent_match'] = (details['1'] == details['2']) & (details['2']==details['3'])
        details = details.loc[details['consistent_match']==True]
        details = details.drop(columns=['consistent_match','2','3','BvD ID number'])
        details = details.rename(columns={'1':'BvD ID number'})

        # Begin Merge
        fin = fin.loc[(fin['year']>=1995)|(fin['year']<=2015)]
        for year in range(1995,2016):
            print(f'    on {year}')
            temp = fin.loc[fin['year'] == year].copy()
            temp = temp.merge(details,on='BvD ID number',how='left',indicator='merge_add_ind_leg_fin')
            temp.to_csv(countrypath+country+'_master_'+str(year)+'.csv')
        
        return None
    
    else:
        if has_det == False:
            print("    can't combine files: no details file exists.")
        if has_fin == False:
            print("    can't combine files: no financials file exists.")
        
        return None

##########################################################################################################################    
# Main Routines: Cleaning
##########################################################################################################################

def assign_industry(df):

    # Preallocate
    nace,sic,naics,nuts,idnr = 'NACE Rev. 2, Core code (4 digits)', 'US SIC, Core code (3 digits)', 'NAICS 2012, Core code (4 digits)', 'NUTS3', 'BvD ID number'
    df['has_nace'] = pd.notna(df[nace])
    df['has_sic'] = pd.notna(df[sic])
    df['has_nuts'] = pd.notna(df[nuts])
    df['has_naics'] = pd.notna(df[naics])
    df['has_industry_any'] = (df['has_nace']==True)|(df['has_nuts']==True)|(df['has_naics']==True)

    # Lengths of NACE, NAICS and SIC codes
    Lnace,Lnaics,Lsic = len(df.loc[df['has_nace']].index),len(df.loc[df['has_naics']].index),len(df.loc[df['has_sic']].index)
    Lmax = max(Lnace,Lnaics,Lsic)
    
    # Assign 
    if Lmax == Lsic:
        industry = 'US SIC, Core code (3 digits)'
    if Lmax == Lnaics:
        industry = 'NAICS 2012, Core code (4 digits)'
    if Lmax == Lnace:
        industry = 'NACE Rev. 2, Core code (4 digits)'

    # New Variables
    df['industry'] = df[industry]
    df['has_industry'] = pd.notna(df['industry']) 
    df['industry_type'] = industry
    return df,industry

def assign_geography(df):

    # Check if dataset has a NUTS3 Region or a Region Code
    has_nuts = len(df['NUTS3'].unique()) > 1
    has_reg = len(df['Region in country'].unique()) > 1 | has_nuts == 1

    # Create
    if has_nuts == True:
        df['geo'] = df['NUTS3']
        df.loc[pd.isna(df['geo']),'geo'] = df.loc[pd.isna(df['geo']),'Region in country'] + ', ' + df.loc[pd.isna(df['geo']),'Country'] 
        geotype = 'nuts'
    
    elif (has_nuts == False) & (has_reg == True):
        df['geo'] = df['Region in country'] + ', ' + df['Country']
        geotype = 'region+country'

    else:
        df['geo'] == df['Country']
        geotype = 'country'
    
    # Has Data indicator
    df['has_geo'] = pd.notna(df['geo'])

    return df,geotype

def UnitFixOneStep(df,var,stepno):
    
    print('**************')
    print(f'VARIABLE: {var}, iter: {stepno}')
    
    # Create Lags and Leads
    print('    make lags, leads and growth rates')
    df = df.sort_values(['BvD ID number','year'])
    df['L'+var] = df.groupby(['BvD ID number'])[var].shift(1)
    df['F'+var] = df.groupby(['BvD ID number'])[var].shift(-1)

    # Enforce Positive
    df[var] = np.maximum(df[var],0)
    df['L'+var] = np.maximum(df[var],0)
    df['F'+var] = np.maximum(df[var],0)

    # Create Growth Variables
    df['Lg'+var] = (df[var]-df['L'+var])/df['L'+var]
    df['Fg'+var] = (df['F'+var]-df[var])/df[var]

    # Identify Corrections
    print('    identify corrections and enforce')
    df['correct_1'],df['correct_2'],df['correct_3'] = 0,0,0
    df.loc[(df['Original units']=='thousands') & (df['Lyear'] == df['yr']-1) & (df['Lg'+var] < -0.99),'correct_1'] = 1
    df.loc[(df['Original units']=='thousands') & (df['Fyear'] == df['yr']+1) & (df['Fg'+var] > 198),'correct_2'] = 1
    df.loc[(df['Lyear']==df['yr']-1) & (df['Fyear']==df['yr']+1) & (df['Lg'+var] < -0.99) & (df['Fg'+var] > 198),'correct_3'] = 1

    # Account for 0's
    df.loc[df[var] <= 0,'correct_1'] = 0
    df.loc[df[var] <= 0,'correct_2'] = 0
    df.loc[df[var] <= 0,'correct_3'] = 0
    
    # Identify elements for which correction is to be performed
    df['legit'] = 0
    df['WEIRD: Jumps in value unexplained by unit changes'] = 0
    df[var+'corrected'] = 0
    df.loc[(1000*np.abs(df[var])/np.abs(df['L'+var]) < 2) | (1000*np.abs(df[var])/np.abs(df['L'+var]) > 0.5),'legit'] = 1
    df.loc[(df['correct_1']==0) & (df['correct_2']==0) & (df['correct_3']==0) & (df['legit'] == 0),'legit'] = 1
    df.loc[(df['correct_3']==1)&(df['legit']==0),'WEIRD: Jumps in value unexplained by unit changes'] = 1
    
    # Enforce Correction
    df.loc[(df['correct_1']==1) | (df['correct_2']==1) | ((df['correct_3']==1) & (df['legit']==1)),var+'corrected'] = 1
    #df.loc[((df['correct_3']==1) & (df['legit']==1)),var+'corrected'] = 1

    df.loc[(df['correct_1']==1) | (df['correct_2']==1) | ((df['correct_3']==1) & (df['legit']==1)),var] = df.loc[(df['correct_1']==1) | (df['correct_2']==1) | ((df['correct_3']==1) & (df['legit']==1)),var]*1000
    #df.loc[(df['correct_3']==1) & (df['legit']==1),var] = df.loc[(df['correct_3']==1) & (df['legit']),var]*1000
    
    df.loc[(df['correct_1']==1) | (df['correct_2']==1) | ((df['correct_3']==1) & (df['legit']==1)),'new'+var] = df.loc[(df['correct_1']==1) | (df['correct_2']==1) | ((df['correct_3']==1) & (df['legit']==1)),var]*1000
    #df.loc[(df['correct_3']==1) & (df['legit']==1),var] = df.loc[(df['correct_3']==1) & (df['legit']),var]*1000
    
    # Cleanup
    nchanges = len(df.loc[(df['correct_1']==1) | (df['correct_2']==1)].index) + len(df.loc[(df['correct_3']==1) & (df['legit']==1)].index)
    df = df.drop(columns=['L'+var,'F'+var,'Lg'+var,'Fg'+var,'correct_1','correct_2','correct_3','legit'])
    df = df.rename(columns={var+'corrected':var+'corrected_step'+str(stepno)})
    df[var+'_iter_'+str(stepno)] = df[var]
    return df, nchanges

def UnitFix(df,varlist_o,iterates=True,replace=False):

    print('Unit Fixing Algorithm')
    print('***********************************************************************************')
    df = df.sort_values(by=['BvD ID number','year'])
    df = df.set_index(['BvD ID number'])
    
    # Lead/Lag Years
    df['Lyear'] = df.groupby(['BvD ID number'])['year'].shift(1)
    df['Fyear'] = df.groupby(['BvD ID number'])['year'].shift(-1)
    df['yr'] = df['year']
    df = df.reset_index()
    df = df.set_index(['BvD ID number','year'])

    # Define Variable Names
    if replace == True:
        varlist = varlist_o
    else:
        varlist = ['new'+var_o for var_o in varlist_o]
        for var_o in varlist_o:
            df['new'+var_o] = df[var_o] 

    # Initialize Step No.
    iter,maxiter,nchanges = 0,21,{var:10 for var in varlist}
    nchanges_max = 10

    # Main Loop over Iterations
    while (iter<=maxiter) & (nchanges_max>0):
        iter += 1
        nchanges_max = 0
        print(f'ITERATION: {iter}')
        # Loop over Variables
        for var in varlist:
            print(f'    on {var}')
            # If no changes were made in the last iteration, make no more changes.
            if nchanges[var] == 0:
                print(f'    no more changes.')
                continue
            else:
                df,nchanges[var] = UnitFixOneStep(df,var,iter)
                nchanges_max = max(nchanges_max,nchanges[var])
                print(f'        made {nchanges} changes.')
                if iterates == False:
                    df = df.drop(columns=[var+'_iter_'+str(iter)])
    
    df = df.reset_index()
    df = df.drop(columns=['yr','Lyear','Fyear'])
    return df

def MakeVariables(df):
    
    # Construct Financials
    df['Kt'] = df['Tangible fixed assets']
    df['Ki'] = df['Intangible fixed assets']
    df['KT'] = df['Fixed assets'] 
    df['L'] = df['Number of employees']
    df['wL'] = df['Costs of employees']
    df['M'] = df['Material costs']
    df['OPRE'] = df['Operating revenue (Turnover)']

    # Construct Drop Indicators and define TFP Sample
    df['tfpsample'] = 1
    for var in ['KT','Kt','Ki','L','wL','M','OPRE']:
        df[var+'_missing'],df[var+'_weaknegative'] = 0,0
        df.loc[(pd.isna(df[var])),var+'_missing'] = 1
        df.loc[(df[var] <= 0),var+'_weaknegative'] = 1
        df['tfpsample'] = df['tfpsample'] * (1-df[var+'_missing']) * (1-df[var+'_weaknegative'])

    return df

def ImputeWages(df):

    nace,sic,naics,nuts,idnr = 'NACE Rev. 2, Core code (4 digits)', 'US SIC, Core code (3 digits)', 'NAICS 2012, Core code (4 digits)', 'geo', 'BvD ID number'

    # Assign Industry Data and Geography Data
    df,industry = assign_industry(df)
    df,geotype = assign_geography(df)

    # If no industry * geography data, stop right here
    L0 = len(df.loc[(df['has_industry']==True)&(df['has_geo']==True)].index)

    if L0 == 0:
        imputeflag = 0
        print(f'Industry or Geography ill-defined. Wage Imputation NOT performed.')
        df['industry'] = 'NA'
        return df,imputeflag
    
    else:
        imputeflag = 1

        # Define Temporary Dataset
        temp = df.loc[(df['wL'] > 0) & (df['L'] > 0) & (pd.notna(df['wL'])) & (pd.notna(df['L'])) & (pd.notna(df[industry])) & (pd.notna(df[nuts])) & (pd.notna(df['year'])) & (pd.notna(df[idnr])),['year',industry,nuts,idnr,'wL','L','Number of employees']]
        temp = temp.set_index([idnr,industry,nuts,'year'])

        temp['w'] = temp['wL']/temp['L']
        temp['w_f'] = temp.groupby(idnr)['w'].mean()
        temp['WEIRD: implied wage outside 5x bounds of firm-level mean'] = 0
        temp.loc[(temp['w']>=5*temp['w_f']) | (temp['w']<=0.2*temp['w_f']),'WEIRD: implied wage outside 5x bounds of firm-level mean'] = 1
        temp.loc[(temp['w']>=5*temp['w_f']) | (temp['w']<=0.2*temp['w_f']),'w'] = temp.loc[(temp['w']>=5*temp['w_f']) | (temp['w']<=0.2*temp['w_f']),'w_f']
        
        # Remove Region*industry*year FE
        temp['w'] = np.log(temp['w'])
        temp['w_iry'] = temp.groupby([industry,nuts,'year'])['w'].mean()
        temp['w_res'] = temp['w'] - temp['w_iry']

        # Remove Firm FE
        temp['w_iry_f'] = temp.groupby([idnr])['w_res'].mean()
        temp['w_pred'] = temp['w_iry'] + temp['w_iry_f']
        temp = temp.reset_index()
        
        # Imputed Employment
        temp['L_imputed'] = temp['wL']/np.exp(temp['w_pred'])
        temp.loc[pd.isna(temp['L']),'L'] = temp.loc[pd.isna(temp['L']),'L_imputed']
        temp['Indicator: Imputed Labor'] = 0
        temp.loc[(pd.isna(temp['Number of employees'])) & (pd.notna(temp['L'])),'Indicator: Imputed Labor'] = 1

        # Cleanup
        temp = temp[[idnr,'year',industry,nuts,'L_imputed','Indicator: Imputed Labor','WEIRD: implied wage outside 5x bounds of firm-level mean']]
        df = df.merge(temp,how='left',on=[idnr,industry,nuts,'year'],indicator='merge_wage_imputation')
        return df,imputeflag

def CleanFinancialsData(country,savepath):
    countrypath = savepath+country+'/'
    print('***********************************************************************************')
    print(f'Clean Financials for {country}')
    df = pd.DataFrame()

    # Create Summary Dataset
    summary = pd.DataFrame(columns=['year','raw','hasregion','hasindustry','manufacturing','tfpsample','manufacturing_tfpsample'])
    summary['year'] = np.arange(1995,2016,dtype=int)
    nace,nuts = 'NACE Rev. 2, Core code (4 digits)', 'NUTS3'

    # Import Datasets and combine them
    for year in range(1995,2016):
        print(f'    importing {year}')
        temp = pd.read_csv(countrypath+country+'_master_'+str(year)+'.csv')
        print(f'    ... raw length: {len(temp.index)}')

        if len(temp.index) == 0:
            continue
        else:

            # Drop Consolidated Firms if Unconsolidated statements are available, and drop firms with "Limited Financials" or "No recent financials"
            temp = temp.loc[temp['Consolidation code'].isin(['LF','NF'])==False]
            temp['isU'] = temp['Consolidation code'].str.contains("U")
            temp = temp.set_index('BvD ID number')
            temp['hasU'] = temp.groupby(['BvD ID number'])['isU'].max()
            temp = temp.reset_index()
            temp['drop_U'] = (temp['hasU']==True) & (temp['isU']==False)
            temp = temp.loc[temp['drop_U']==False]
            temp = temp.drop(columns=['isU','hasU','drop_U'])
            print(f'    ... keep only unconsolidated firms: {len(temp.index)}')

            # Keep Most Complete Observations
            temp = KeepMostCompleteObs(temp,'sssss','BvD ID number')

            # Make Variables
            temp = MakeVariables(temp)

            # Summarize
            summary.loc[summary['year']==year,'raw'] = len(temp.index)
            summary.loc[summary['year']==year,'hasregion'] = len(temp.loc[pd.notna(temp[nuts])].index)
            summary.loc[summary['year']==year,'hasindustry'] = len(temp.loc[pd.notna(temp[nace])].index)
            summary.loc[summary['year']==year,'manufacturing'] = len(temp.loc[temp['NACE Rev. 2 main section'].astype(str).str[:1]=='C'].index)
            for var in ['Kt','Ki','KT','L','wL','M','OPRE']:
                summary.loc[summary['year']==year,'positive_'+var] = len(temp.loc[(temp[var+'_missing'] != 1) & (temp[var+'_weaknegative'] != 1)].index)
            summary.loc[summary['year']==year,'tfpsample'] = temp['tfpsample'].sum()
            
            # Retain ONLY data points of value
            temp['drop'] = 0
            for var in ['KT','OPRE']:
                temp.loc[(temp[var+'_missing']==True)|(temp[var+'_weaknegative']==True),'drop'] = 1
            temp.loc[temp['wL_weaknegative']==True,'drop'] = 1
            temp = temp.loc[temp['drop']==False]
            print(f'    ... after dropping bad observations, adding {len(temp.index)} observations.')
            df = df.append(temp,sort=True)
        
    df = df.drop(columns=df.filter(like='Unnamed'))
    summary.to_csv(countrypath+country+'summary_financials_precleaning.csv')

    # Count Min/Max no. of observations per year
    maxcount,nyears = 0,0
    for year in range(1995,2016):
        maxcount = np.maximum(maxcount,len(df.loc[df['year']==year].index))
        nyears = nyears + (len(df.loc[df['year']==year].index) > 0)
    
    # Assert Dataset is OK, Pass 1: Datasets must have at least 1000 observations, pre-clean
    if len(df.index) < 1000:
        print("    Fewer than 1000 observations overall in raw dataset.")
        return None

    # Assert Dataset is OK, Pass 2: Datasets must have at least one year with >= 50 observations 
    elif maxcount < 50:
        print("    Fewer than 50 observations in each year.") 
        return None

    else:

        print(f'    {nyears} years of data.')

        # Summarize
        for year in range(1995,2016):
            temp = df.loc[df['year']==year]
            summary.loc[summary['year']==year,'raw'] = len(temp.index)
            for var in ['Kt','Ki','KT','L','wL','M','OPRE']:
                summary.loc[summary['year']==year,'positive_'+var] = len(temp.loc[(temp[var+'_missing'] != 1) & (temp[var+'_weaknegative'] != 1)].index)
            summary.loc[summary['year']==year,'tfpsample'] = temp['tfpsample'].sum()

        # Save
        # note that this program represents the final part of step 1. 
        df.to_csv(countrypath+country+'_master.csv')
        summary.to_csv(countrypath+country+'summary_financials_cleaning.csv')
        return None

def CleanFinancialsData_unitfix(country,savepath):
    countrypath = savepath+country+'/'
    print('***********************************************************************************')
    print(f'Clean Financials for {country}')

    # Create Summary Dataset
    summary = pd.DataFrame(columns=['year','raw','hasregion','hasindustry','manufacturing','tfpsample','manufacturing_tfpsample'])
    summary['year'] = np.arange(1995,2016,dtype=int)

    try:
        df = pd.read_csv(countrypath+country+'_master.csv')

        # Unit Fixing
        vars_uf = ['Kt','Ki','wL','M','OPRE','Total shareh. funds & liab.','Shareholders funds','Total assets','Long term debt','Loans','Creditors','Debtors']
        
        # Ensure no negatives in unit fix
        for v in vars_uf:
            df.loc[df[v]<0,v] = np.nan
        
        df = UnitFix(df,vars_uf,iterates=False,replace=True)

        # Calculate some objects
        df['K'] = df['Kt'] + df['Ki']
        df['VA'] = df['OPRE'] - df['M']
        print(f'    DF has {len(df.index)} Observations.')
        df = df.loc[(df['VA']>0)|(pd.isna(df['VA']))]
        print(f'    ... after dropping bad Value added: {len(df.index)}')

        # Summarize
        for year in range(1995,2016):
            temp = df.loc[df['year']==year]
            summary.loc[summary['year']==year,'raw'] = len(temp.index)
            for var in ['Kt','Ki','KT','L','wL','M','OPRE']:
                summary.loc[summary['year']==year,'positive_'+var] = len(temp.loc[(temp[var+'_missing'] != 1) & (temp[var+'_weaknegative'] != 1)].index)
            summary.loc[summary['year']==year,'tfpsample'] = temp['tfpsample'].sum()

        # Save
        # note that this program represents the final part of step 1. 
        df.to_csv(countrypath+country+'_master_unitfix.csv')
        summary.to_csv(countrypath+country+'summary_financials_cleaning_unitfix.csv')
        return None

    except:
        print(f'NO CHANGES MADE: either df not found or an error was thrown somewhere.')
        return None

##########################################################################################################################    
# Analysis Routines: Study Leverage by Country
##########################################################################################################################

def duplicates_list_by_year(df):
    df2 = df[['year','BvD ID number']].copy()
    df2['d'] = df2.duplicated(subset=['year','BvD ID number'],keep=False)
    dups,nodups = {},{}
    dupcount = 0
    for y in range(1995,2016):
        dups[y] = len(df2.loc[(df2['year']==y) & (df2['d']==True)])
        dupcount = dupcount + dups[y]
        nodups[y] = len(df2.loc[(df2['year']==y) & (df2['d']==False)])
        print(f'{y}: {dups[y]} duplicates and {nodups[y]} unique.')
    return dups,nodups,dupcount

def make_quantiles_by_year(df,varlist,nq=100):
    
    # Preallocate
    nq_vy = {y: {v:{} for v in varlist} for y in range(1995,2016)}
    mq_vy = {y: {v:{} for v in varlist} for y in range(1995,2016)}

    # Check that there are enough nonmissing points
    for y in range(1995,2016):
        temp = df.loc[df['year']==y,varlist].copy()
        for v in varlist:
            nq_vy[y][v] = len(temp.loc[pd.notna(temp[v])].index)
            mq_vy[y][v] = nq_vy[y][v] > nq + 1 

    # Start Computing Quantiles
    for y in range(1995,2016):
        for v in varlist:
            if mq_vy[y][v] == 0:
                print(f'TOO FEW OBS FOR {v} in {y}, only {nq_vy[y][v]} observations for {nq} quantiles.')
                continue
            else:
                df.loc[(df['year']==y) & (pd.notna(df[v])), v+'_quantile'] = pd.qcut(df.loc[(df['year']==y) & (pd.notna(df[v])), v],nq,labels=False,duplicates='drop')
    
    return df 

def make_size_classes(df,size_metrics,nq=100):

    # Step 0: Separate Treatment for size classes by Labor
    size_metrics = list(set(size_metrics) - set(['L','Number of employees','L_imputed']))

    # Step 1: Size Classes by Number of Employees -> Always
    if 'L' in list(df.columns):
        df['Size Class (imputed)'] = ''
        df.loc[(df['L']>=0  )&(df['L']<10 ),'Size Class (imputed)'] = "0-9"
        df.loc[(df['L']>=10 )&(df['L']<20 ),'Size Class (imputed)'] = "10-19"
        df.loc[(df['L']>=20 )&(df['L']<50 ),'Size Class (imputed)'] = "20-49"
        df.loc[(df['L']>=50 )&(df['L']<250),'Size Class (imputed)'] = "50-249"
        df.loc[(df['L']>=250),'Size Class (imputed)'] = "GE250"

    if 'Number of employees' in list(df.columns):
        df['Size Class'] = ''
        df.loc[(df['Number of employees']>=0  )&(df['Number of employees']<10 ),'Size Class'] = "0-9"
        df.loc[(df['Number of employees']>=10 )&(df['Number of employees']<20 ),'Size Class'] = "10-19"
        df.loc[(df['Number of employees']>=20 )&(df['Number of employees']<50 ),'Size Class'] = "20-49"
        df.loc[(df['Number of employees']>=50 )&(df['Number of employees']<250),'Size Class'] = "50-249"
        df.loc[(df['Number of employees']>=250),'Size Class'] = "GE250"

    # Step 2: For all other metrics, assign quantiles to each firm by year
    df = make_quantiles_by_year(df,size_metrics,nq)

    return df

def make_herfindahls(df,varlist,byvars=['year']):
    
    # Count Uniques
    byvars = list(set(byvars) - set(['BvD ID number']))
    df2 = df[byvars+varlist+['BvD ID number']].copy()
    
    df2 = df2.set_index(byvars)
    df2['N_data'] = df2.groupby(byvars)['BvD ID number'].nunique()

    for v in varlist:
        # Count the Number of Firms that are entering this average
        df2['N_'+v] = df2.loc[pd.notna(df2[v])].groupby(byvars)['BvD ID number'].nunique()
        # Totals
        df2['s2_'+v] = df2.groupby(byvars)[v].sum()
        # Shares - Squared
        df2['s2_'+v] = (df2[v]/df2['s2_'+v]) ** 2
        # Sum
        df2[v+'_herfindahl'] = df2.groupby(byvars)['s2_'+v].sum()
    
    df2 = df2.reset_index()
    df2 = df2.drop(columns='BvD ID number')
    df2 = df2[byvars+[v+'_herfindahl' for v in varlist]+['N_'+ v for v in varlist]+['N_data']].drop_duplicates()

    # Return only herfindahl data out
    return df2

def make_top_shares(df,varlist,top_num_list=[5],byvars=['year']):
    
    # Dummy Dataset
    byvars = list(set(byvars) - set(['BvD ID number']))
    df2 = df[byvars+varlist+['BvD ID number']].copy()
    df2 = df2.set_index(byvars)

    # Count Uniques
    df2['N_data'] = df2.groupby(byvars)['BvD ID number'].nunique()

    for v in varlist:

        # Create Ranks and Totals
        df2[v+'_rank'] = df2.groupby(byvars)[v].rank(method='max',ascending=False)
        df2[v+'_total'] = df2.groupby(byvars)[v].sum()

        # Create Top Sums and Shares
        for i in top_num_list:
            
            # Count the Number of Firms that are entering this average
            df2['N_'+ str(i) + '_' + v] = (df2[v+'_rank'] <= i)
            df2['N_'+ str(i) + '_' + v] = df2.groupby(byvars)['N_'+ str(i) + '_' + v].sum()

            # Calculate Totals and Shares
            df2[v+'_top_'+str(i)] = df2[v] * (df2[v+'_rank'] <= i)
            df2[v+'_tot_'+str(i)] = df2.groupby(byvars)[v+'_top_'+str(i)].sum()
            df2[v+'_top_'+str(i)] = df2[v+'_tot_'+str(i)]/df2[v+'_total']
        
    # Keep only the Top
    df2 = df2.reset_index()
    df2 = df2[byvars+fnmatch.filter(df2.columns,'N_*')+fnmatch.filter(df2.columns,'*_top_*')+fnmatch.filter(df2.columns,'*_tot_*')].drop_duplicates()
    return df2

def chunkyfunc_null(chunk,chunkno=0,chunk_args=[]):
    return chunk 

def chunkyfunc_financials(chunk,chunkno=0,chunk_args=[]):

    # Require Some Key Financials (else the observation is too incomplete to do anything)
    chunk['incomplete'] = False
    if ['L_imputed'] in list(chunk.columns):
        chunk.loc[(pd.isna(chunk['OPRE'])) & (pd.isna(chunk['L_imputed'])) & (pd.isna(chunk['K'])) & (pd.isna(chunk['Sales'])) & (pd.isna(chunk['M'])),'incomplete'] = True 
    else:
        chunk.loc[(pd.isna(chunk['OPRE'])) & (pd.isna(chunk['L'])) & (pd.isna(chunk['K'])) & (pd.isna(chunk['Sales'])) & (pd.isna(chunk['M'])),'incomplete'] = True 
    chunk = chunk.loc[chunk['incomplete']==False]
    chunk = chunk.drop(columns=['incomplete'])

    # Ensure Positivity: Capital Stock and Labor
    chunk['positive'] = False
    chunk.loc[(chunk['K']>0) & ((chunk['L'])>0 | pd.isna(chunk['L'])),'positive'] = True
    chunk = chunk.loc[chunk['positive']==True]
    chunk = chunk.drop(columns=['positive'])

    # Compute Some Variables
    chunk['Firm age'] = chunk['year'] - chunk['Year incorporated'] + 1    
    chunk['Liabilities'] = chunk['Total shareh. funds & liab.'] - chunk['Shareholders funds']
    chunk['Net worth'] = chunk['Total assets'] - chunk['Liabilities']
    chunk['K by L ratio'] = chunk['K']/chunk['L']
    if list(chunk.columns).count('L_imputed')>0:
        chunk.loc[pd.isna(chunk['L']),'K by L ratio'] = chunk.loc[pd.isna(chunk['L']),'K']/chunk.loc[pd.isna(chunk['L']),'L_imputed']
    chunk['K by Wage Bill'] = chunk['K']/chunk['wL']

    # Construct Leverage Measures
    chunk['leverage_1'] = chunk['Kt']/chunk['Shareholders funds']
    chunk['leverage_2'] = chunk['Total assets']/chunk['Shareholders funds']
    chunk['leverage_3'] = (chunk['Long term debt'] + chunk['Loans'])/chunk['Shareholders funds']

    # Clean up a few values
    for lev in range(1,4):
        chunk.loc[(pd.isna(chunk['leverage_'+str(lev)])) | (chunk['leverage_'+str(lev)] == np.inf) ,'leverage_'+str(lev)] = np.nan

    return chunk

def analyze_import_complete_observations(country,savepath,chunksize,varlist,chunkyfunc=chunkyfunc_null,chunk_args=[]):

    # Set paths
    countrypath = savepath+country+'/'
    print(f'Import Data for {country}')

    # Chunk Import
    df_chunks = pd.read_csv(countrypath+country+'_master_unitfix.csv',chunksize=chunksize,usecols=varlist)

    # Import and apply chunkyfunc
    df = pd.DataFrame(columns=varlist)
    chunkno = 0
    for chunk in df_chunks:
        
        chunkno += 1
        chunk = chunkyfunc(chunk,chunkno,chunk_args)
        
        print(f'    on chunk {chunkno}, adding {len(chunk.index)} observations.')

        df = df.append(chunk,sort=True)
    
    return df

def analyze_financials_firm_sizes(country,savepath,size_metrics,return_df=False):

    # Set path
    countrypath = savepath+country+'/'
    varlist_imports = list(set(varlist_TFP_sample) | set(size_metrics) | set(['industry','has_industry','OPRE','M','VA','K','Ki','Kt','KT','L','wL']))

    # Imports
    try:
        df = analyze_import_complete_observations(country,savepath,chunksize=2_000_000,varlist=varlist_imports,chunkyfunc=chunkyfunc_financials)
        import_ok = 1

    except:
        print(f'COULDNT DO IMPORT, something is wrong.')
        import_ok = 0

    # Begin Analysis
    if import_ok == 0:
        return None

    else:
        # Try to impute Labor
        df,imputeflag = ImputeWages(df)
        print(f'impute flag {imputeflag}')

        # Step 0: Characterize Duplication
        _,_,dupc = duplicates_list_by_year(df)
        if dupc > 0:
            print(f'WARNING: DUPLICATES BY BvD ID NUMBER AND YEAR')

        # Step 1: Calculate Size Class For all Firms by Each Size Metric, Deciles only
        df = make_size_classes(df,size_metrics,nq = 10) 

        # Step 2: Calculate Herfindahls
        df_h_t = make_herfindahls(df,size_metrics)
        df_h_ind_t = make_herfindahls(df,size_metrics,['year','industry'])

        # Step 3: Calculate Top Shares
        df_shr_t = make_top_shares(df,size_metrics,top_num_list=[2,5,10],byvars=['year'])
        df_shr_ind_t = make_top_shares(df,size_metrics,top_num_list=[2,5,10],byvars=['year','industry'])

        # Step 4: Calculate Leverage Means/Medians by year and industry
        df = df.set_index(['year','industry'])
        print(f'Find leverage by sector x time')
        for lev in range(1,4):

            # Mean Leverage
            df['mean_leverage_st_'+str(lev)] = df.groupby(['year','industry'])['leverage_'+str(lev)].mean()

            # Median Leverage
            df['median_leverage_st_'+str(lev)] = df.groupby(['year','industry'])['leverage_'+str(lev)].median()

        # Step 5: Compute Sectoral x Regional measures of leverage
        df = df.reset_index()
        has_nuts = len(df['NUTS3'].unique()) > 1
        has_reg = len(df['Region in country'].unique()) > 1 | has_nuts == 1

        if has_nuts == True:
            df['geo'] = df['NUTS3']

        elif (has_nuts == False) & (has_reg == True):
            df['geo'] = df['Region in country'] + ', ' + df['Country']

        else:
            df['geo'] == df['Country']

        if len(df['geo'].unique()) > 1:
            print(f'Find leverage by sector x time x geography')
            df = df.set_index(['year','industry','geo'])
            
            for lev in range(1,4):

                # Mean Leverage
                df['mean_leverage_gst_'+str(lev)] = df.groupby(['year','industry','geo'])['leverage_'+str(lev)].mean()

                # Median Leverage
                df['median_leverage_gst_'+str(lev)] = df.groupby(['year','industry','geo'])['leverage_'+str(lev)].median()

            df = df.reset_index()

        # Residuals
        resvars = ['leverage_'+str(i) for i in range(1,4)]

        if has_reg == True:
            onvars = ['year','industry','geo']
        else:
            onvars = ['year','industry']
        
        print(f'Calculating Leverage residuals on {onvars}')
        df,v_orig,v_res = Residualize(df,resvars,onvars,variances=True)

        print(f'***********************************************************************************')
        print(f'Reduction in Variance after Residualizing on {onvars}')
        print(f'***********************************************************************************')
        print(f"leverage 1 (Tangible Assets/Book Equity): Res/Raw = {100*v_res['leverage_1']/v_orig['leverage_1'] : 5.2n} %")
        print(f"leverage 2 (Total Assets/Book Equity):    Res/Raw = {100*v_res['leverage_2']/v_orig['leverage_2'] : 5.2n} %")
        print(f"leverage 3 (Total Debt/Book Equity):      Res/Raw = {100*v_res['leverage_3']/v_orig['leverage_3'] : 5.2n} %")

        # Step 5: Save
        df.to_csv(countrypath+country+'_firm_size_classes.csv')
        df_h_t.to_csv(countrypath+country+'_herf_by_time.csv')
        df_h_ind_t.to_csv(countrypath+country+'_herf_by_time_industry.csv')
        df_shr_t.to_csv(countrypath+country+'_top_shrs_by_time.csv')
        df_shr_ind_t.to_csv(countrypath+country+'_top_shrs_by_time_industry.csv')

        if return_df == True:
            return df
        else:
            return None

##########################################################################################################################    
# Auxiliary Routines
##########################################################################################################################

def clean_nace(dataset):
    # Construct Improved Industry Match == Assign to First 4-digit code for any codes at 3 digit level
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 100 ,['NACE Rev. 2, Core code (4 digits)']] = 111
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 110 ,['NACE Rev. 2, Core code (4 digits)']] = 111
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 120 ,['NACE Rev. 2, Core code (4 digits)']] = 121
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 140 ,['NACE Rev. 2, Core code (4 digits)']] = 141
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 160 ,['NACE Rev. 2, Core code (4 digits)']] = 161
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 200 ,['NACE Rev. 2, Core code (4 digits)']] = 210
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 300 ,['NACE Rev. 2, Core code (4 digits)']] = 311
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 310 ,['NACE Rev. 2, Core code (4 digits)']] = 311
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 320 ,['NACE Rev. 2, Core code (4 digits)']] = 321
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 700 ,['NACE Rev. 2, Core code (4 digits)']] = 710
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 800 ,['NACE Rev. 2, Core code (4 digits)']] = 811
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 810 ,['NACE Rev. 2, Core code (4 digits)']] = 811
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 890 ,['NACE Rev. 2, Core code (4 digits)']] = 891
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 1000,['NACE Rev. 2, Core code (4 digits)']] = 1011
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 1010,['NACE Rev. 2, Core code (4 digits)']] = 1011
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 1030,['NACE Rev. 2, Core code (4 digits)']] = 1031
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 1040,['NACE Rev. 2, Core code (4 digits)']] = 1041
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 1050,['NACE Rev. 2, Core code (4 digits)']] = 1051
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 1060,['NACE Rev. 2, Core code (4 digits)']] = 1061
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 1070,['NACE Rev. 2, Core code (4 digits)']] = 1071
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 1080,['NACE Rev. 2, Core code (4 digits)']] = 1081
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 1090,['NACE Rev. 2, Core code (4 digits)']] = 1091
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 1100,['NACE Rev. 2, Core code (4 digits)']] = 1101
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 1300,['NACE Rev. 2, Core code (4 digits)']] = 1310
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 1390,['NACE Rev. 2, Core code (4 digits)']] = 1391
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 1400,['NACE Rev. 2, Core code (4 digits)']] = 1411
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 1410,['NACE Rev. 2, Core code (4 digits)']] = 1411
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 1430,['NACE Rev. 2, Core code (4 digits)']] = 1431
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 1500,['NACE Rev. 2, Core code (4 digits)']] = 1511
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 1510,['NACE Rev. 2, Core code (4 digits)']] = 1511
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 1600,['NACE Rev. 2, Core code (4 digits)']] = 1610
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 1610,['NACE Rev. 2, Core code (4 digits)']] = 1610
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 1620,['NACE Rev. 2, Core code (4 digits)']] = 1621
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 1700,['NACE Rev. 2, Core code (4 digits)']] = 1711
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 1710,['NACE Rev. 2, Core code (4 digits)']] = 1711
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 1720,['NACE Rev. 2, Core code (4 digits)']] = 1721
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 1800,['NACE Rev. 2, Core code (4 digits)']] = 1811
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 1810,['NACE Rev. 2, Core code (4 digits)']] = 1811
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 1900,['NACE Rev. 2, Core code (4 digits)']] = 1910
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 1911,['NACE Rev. 2, Core code (4 digits)']] = 1910
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 2000,['NACE Rev. 2, Core code (4 digits)']] = 2011
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 2010,['NACE Rev. 2, Core code (4 digits)']] = 2011
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 2040,['NACE Rev. 2, Core code (4 digits)']] = 2041
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 2050,['NACE Rev. 2, Core code (4 digits)']] = 2051
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 2100,['NACE Rev. 2, Core code (4 digits)']] = 2110
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 2200,['NACE Rev. 2, Core code (4 digits)']] = 2211
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 2210,['NACE Rev. 2, Core code (4 digits)']] = 2211
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 2220,['NACE Rev. 2, Core code (4 digits)']] = 2221
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 2300,['NACE Rev. 2, Core code (4 digits)']] = 2311
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 2310,['NACE Rev. 2, Core code (4 digits)']] = 2311
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 2330,['NACE Rev. 2, Core code (4 digits)']] = 2331
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 2340,['NACE Rev. 2, Core code (4 digits)']] = 2341
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 2350,['NACE Rev. 2, Core code (4 digits)']] = 2351
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 2360,['NACE Rev. 2, Core code (4 digits)']] = 2361
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 2390,['NACE Rev. 2, Core code (4 digits)']] = 2391
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 2400,['NACE Rev. 2, Core code (4 digits)']] = 2410
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 2411,['NACE Rev. 2, Core code (4 digits)']] = 2410
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 2430,['NACE Rev. 2, Core code (4 digits)']] = 2431
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 2440,['NACE Rev. 2, Core code (4 digits)']] = 2441
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 2450,['NACE Rev. 2, Core code (4 digits)']] = 2451
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 2500,['NACE Rev. 2, Core code (4 digits)']] = 2511
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 2510,['NACE Rev. 2, Core code (4 digits)']] = 2511
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 2520,['NACE Rev. 2, Core code (4 digits)']] = 2521
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 2560,['NACE Rev. 2, Core code (4 digits)']] = 2561
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 2570,['NACE Rev. 2, Core code (4 digits)']] = 2571
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 2590,['NACE Rev. 2, Core code (4 digits)']] = 2591
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 2600,['NACE Rev. 2, Core code (4 digits)']] = 2611
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 2610,['NACE Rev. 2, Core code (4 digits)']] = 2611
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 2650,['NACE Rev. 2, Core code (4 digits)']] = 2651
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 2700,['NACE Rev. 2, Core code (4 digits)']] = 2711
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 2710,['NACE Rev. 2, Core code (4 digits)']] = 2711
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 2730,['NACE Rev. 2, Core code (4 digits)']] = 2731
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 2750,['NACE Rev. 2, Core code (4 digits)']] = 2751
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 2800,['NACE Rev. 2, Core code (4 digits)']] = 2811
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 2810,['NACE Rev. 2, Core code (4 digits)']] = 2811
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 2820,['NACE Rev. 2, Core code (4 digits)']] = 2821
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 2840,['NACE Rev. 2, Core code (4 digits)']] = 2841
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 2890,['NACE Rev. 2, Core code (4 digits)']] = 2891
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 2900,['NACE Rev. 2, Core code (4 digits)']] = 2910
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 2930,['NACE Rev. 2, Core code (4 digits)']] = 2931
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 3000,['NACE Rev. 2, Core code (4 digits)']] = 3011
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 3010,['NACE Rev. 2, Core code (4 digits)']] = 3011
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 3090,['NACE Rev. 2, Core code (4 digits)']] = 3091
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 3100,['NACE Rev. 2, Core code (4 digits)']] = 3101
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 3200,['NACE Rev. 2, Core code (4 digits)']] = 3211 
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 3210,['NACE Rev. 2, Core code (4 digits)']] = 3211
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 3290,['NACE Rev. 2, Core code (4 digits)']] = 3291
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 3300,['NACE Rev. 2, Core code (4 digits)']] = 3311 
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 3310,['NACE Rev. 2, Core code (4 digits)']] = 3311
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 3500,['NACE Rev. 2, Core code (4 digits)']] = 3511
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 3510,['NACE Rev. 2, Core code (4 digits)']] = 3511
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 3520,['NACE Rev. 2, Core code (4 digits)']] = 3521
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 3800,['NACE Rev. 2, Core code (4 digits)']] = 3811
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 3810,['NACE Rev. 2, Core code (4 digits)']] = 3811
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 3820,['NACE Rev. 2, Core code (4 digits)']] = 3821
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 3830,['NACE Rev. 2, Core code (4 digits)']] = 3831
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 4100,['NACE Rev. 2, Core code (4 digits)']] = 4110
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 4210,['NACE Rev. 2, Core code (4 digits)']] = 4211
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 4220,['NACE Rev. 2, Core code (4 digits)']] = 4221
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 4290,['NACE Rev. 2, Core code (4 digits)']] = 4291
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 4300,['NACE Rev. 2, Core code (4 digits)']] = 4311
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 4310,['NACE Rev. 2, Core code (4 digits)']] = 4311
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 4320,['NACE Rev. 2, Core code (4 digits)']] = 4321
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 4330,['NACE Rev. 2, Core code (4 digits)']] = 4331
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 4390,['NACE Rev. 2, Core code (4 digits)']] = 4391
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 4500,['NACE Rev. 2, Core code (4 digits)']] = 4511
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 4510,['NACE Rev. 2, Core code (4 digits)']] = 4511
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 4530,['NACE Rev. 2, Core code (4 digits)']] = 4531
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 4600,['NACE Rev. 2, Core code (4 digits)']] = 4611
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 4610,['NACE Rev. 2, Core code (4 digits)']] = 4611
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 4620,['NACE Rev. 2, Core code (4 digits)']] = 4621
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 4630,['NACE Rev. 2, Core code (4 digits)']] = 4631
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 4640,['NACE Rev. 2, Core code (4 digits)']] = 4641
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 4650,['NACE Rev. 2, Core code (4 digits)']] = 4651
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 4660,['NACE Rev. 2, Core code (4 digits)']] = 4661
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 4670,['NACE Rev. 2, Core code (4 digits)']] = 4671
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 4700,['NACE Rev. 2, Core code (4 digits)']] = 4711
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 4710,['NACE Rev. 2, Core code (4 digits)']] = 4711
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 4720,['NACE Rev. 2, Core code (4 digits)']] = 4721
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 4740,['NACE Rev. 2, Core code (4 digits)']] = 4741
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 4750,['NACE Rev. 2, Core code (4 digits)']] = 4751
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 4760,['NACE Rev. 2, Core code (4 digits)']] = 4761
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 4770,['NACE Rev. 2, Core code (4 digits)']] = 4771
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 4780,['NACE Rev. 2, Core code (4 digits)']] = 4781
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 4790,['NACE Rev. 2, Core code (4 digits)']] = 4791
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 4900,['NACE Rev. 2, Core code (4 digits)']] = 4910
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 4930,['NACE Rev. 2, Core code (4 digits)']] = 4931
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 4940,['NACE Rev. 2, Core code (4 digits)']] = 4941
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 5000,['NACE Rev. 2, Core code (4 digits)']] = 5010
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 5100,['NACE Rev. 2, Core code (4 digits)']] = 5110
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 5200,['NACE Rev. 2, Core code (4 digits)']] = 5210
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 5220,['NACE Rev. 2, Core code (4 digits)']] = 5221
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 5300,['NACE Rev. 2, Core code (4 digits)']] = 5310
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 5500,['NACE Rev. 2, Core code (4 digits)']] = 5510
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 5600,['NACE Rev. 2, Core code (4 digits)']] = 5610
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 5620,['NACE Rev. 2, Core code (4 digits)']] = 5621
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 5800,['NACE Rev. 2, Core code (4 digits)']] = 5811
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 5810,['NACE Rev. 2, Core code (4 digits)']] = 5811
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 5820,['NACE Rev. 2, Core code (4 digits)']] = 5821
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 5910,['NACE Rev. 2, Core code (4 digits)']] = 5911
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 6000,['NACE Rev. 2, Core code (4 digits)']] = 6010
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 6100,['NACE Rev. 2, Core code (4 digits)']] = 6110
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 6200,['NACE Rev. 2, Core code (4 digits)']] = 6201
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 6300,['NACE Rev. 2, Core code (4 digits)']] = 6311
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 6310,['NACE Rev. 2, Core code (4 digits)']] = 6311
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 6390,['NACE Rev. 2, Core code (4 digits)']] = 6391
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 6400,['NACE Rev. 2, Core code (4 digits)']] = 6411
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 6410,['NACE Rev. 2, Core code (4 digits)']] = 6411
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 6490,['NACE Rev. 2, Core code (4 digits)']] = 6491
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 6500,['NACE Rev. 2, Core code (4 digits)']] = 6511
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 6510,['NACE Rev. 2, Core code (4 digits)']] = 6511
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 6600,['NACE Rev. 2, Core code (4 digits)']] = 6611
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 6610,['NACE Rev. 2, Core code (4 digits)']] = 6611
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 6620,['NACE Rev. 2, Core code (4 digits)']] = 6621
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 6800,['NACE Rev. 2, Core code (4 digits)']] = 6810
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 6830,['NACE Rev. 2, Core code (4 digits)']] = 6831
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 6900,['NACE Rev. 2, Core code (4 digits)']] = 6910
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 7000,['NACE Rev. 2, Core code (4 digits)']] = 7010
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 7020,['NACE Rev. 2, Core code (4 digits)']] = 7021
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 7100,['NACE Rev. 2, Core code (4 digits)']] = 7111
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 7110,['NACE Rev. 2, Core code (4 digits)']] = 7111
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 7200,['NACE Rev. 2, Core code (4 digits)']] = 7211
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 7210,['NACE Rev. 2, Core code (4 digits)']] = 7211
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 7300,['NACE Rev. 2, Core code (4 digits)']] = 7311
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 7310,['NACE Rev. 2, Core code (4 digits)']] = 7311
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 7400,['NACE Rev. 2, Core code (4 digits)']] = 7410
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 7700,['NACE Rev. 2, Core code (4 digits)']] = 7711
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 7710,['NACE Rev. 2, Core code (4 digits)']] = 7711
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 7720,['NACE Rev. 2, Core code (4 digits)']] = 7721
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 7730,['NACE Rev. 2, Core code (4 digits)']] = 7731
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 7800,['NACE Rev. 2, Core code (4 digits)']] = 7810
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 7900,['NACE Rev. 2, Core code (4 digits)']] = 7911
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 7910,['NACE Rev. 2, Core code (4 digits)']] = 7911
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 8000,['NACE Rev. 2, Core code (4 digits)']] = 8010
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 8120,['NACE Rev. 2, Core code (4 digits)']] = 8121
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 8200,['NACE Rev. 2, Core code (4 digits)']] = 8211
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 8210,['NACE Rev. 2, Core code (4 digits)']] = 8211
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 8290,['NACE Rev. 2, Core code (4 digits)']] = 8291
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 8400,['NACE Rev. 2, Core code (4 digits)']] = 8411
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 8500,['NACE Rev. 2, Core code (4 digits)']] = 8510
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 8530,['NACE Rev. 2, Core code (4 digits)']] = 8531
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 8540,['NACE Rev. 2, Core code (4 digits)']] = 8541
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 8550,['NACE Rev. 2, Core code (4 digits)']] = 8551
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 8600,['NACE Rev. 2, Core code (4 digits)']] = 8610
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 8620,['NACE Rev. 2, Core code (4 digits)']] = 8621
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 8700,['NACE Rev. 2, Core code (4 digits)']] = 8710
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 8800,['NACE Rev. 2, Core code (4 digits)']] = 8810
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 8890,['NACE Rev. 2, Core code (4 digits)']] = 8891
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 9000,['NACE Rev. 2, Core code (4 digits)']] = 9001
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 9100,['NACE Rev. 2, Core code (4 digits)']] = 9101
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 9300,['NACE Rev. 2, Core code (4 digits)']] = 9311
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 9310,['NACE Rev. 2, Core code (4 digits)']] = 9311
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 9320,['NACE Rev. 2, Core code (4 digits)']] = 9321
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 9400,['NACE Rev. 2, Core code (4 digits)']] = 9411
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 9510,['NACE Rev. 2, Core code (4 digits)']] = 9511
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 9520,['NACE Rev. 2, Core code (4 digits)']] = 9521
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 9600,['NACE Rev. 2, Core code (4 digits)']] = 9601
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 9510,['NACE Rev. 2, Core code (4 digits)']] = 9511
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 9520,['NACE Rev. 2, Core code (4 digits)']] = 9521
    dataset.loc[dataset['NACE Rev. 2, Core code (4 digits)'] == 9600,['NACE Rev. 2, Core code (4 digits)']] = 9601
    return dataset

def make_years(dataset):
    # Get date, month, year
    dataset['day'] = (dataset['Closing date']%100).astype(int)
    dataset['month'] = (((dataset['Closing date']-dataset['day'])/100) % 100).astype(int)
    dataset['year'] = (((dataset['Closing date']-dataset['month']*100 - dataset['day'])/10_000) % 10_000).astype(int)
    dataset.loc[dataset['month']<6,'year'] = dataset.loc[dataset['month']<6,'year']-1
    return dataset

def make_years_incorporated(dataset):
    dataset['Date of incorporation'] = pd.to_numeric(dataset['Date of incorporation'],errors='coerce')
    dataset.loc[ dataset['Date of incorporation']<9_999,'Year incorporated'] = dataset['Date of incorporation']
    dataset.loc[(dataset['Date of incorporation']>=10_000) & (dataset['Date of incorporation']<999_999),'Year incorporated'] = np.floor(dataset['Date of incorporation']/100)
    dataset.loc[ dataset['Date of incorporation']>=1_000_000,'Year incorporated'] = np.floor(dataset['Date of incorporation']/10_000)
    return dataset

def KeepMostCompleteObs(df,newname,byvars):
    # Count Number of not-na entries in each row
    cols = df.columns
    df[newname] = 0
    for col in cols:
        df[newname] = df[newname] + pd.notna(df[col])
    # Find Max
    df = df.set_index(byvars)
    df[newname+'max'] = df.groupby(byvars)[newname].max()
    df = df.reset_index()
    df = df.loc[df[newname]==df[newname+'max']]
    df = df.drop(columns=[newname,newname+'max'])
    # Drop Duplicates
    df = df.drop_duplicates(subset=byvars,keep='first')
    return df

def Residualize(df,resvars,onvars,stub=', demeaned',keepmeans=True,variances=False):
    
    # Create Group Index
    df = df.set_index(onvars)
    
    # Create Averages and Demeaned Vars
    for var in resvars:
        df[var+'mm'] = df.groupby(onvars)[var].mean()
        df[var+stub] = df[var] - df[var+'mm']
        if keepmeans == False:
            df = df.drop(columns = var+'mm')

    # Compute Changes in Variance if required
    if variances == False:
        return df 

    else:
        variances_o = {}
        variances_r = {}
        df = df.reset_index()

        for var in resvars:
            variances_o[var] = df.loc[pd.notna(df[var+stub]),var].var()
            variances_r[var] = df.loc[pd.notna(df[var+stub]),var+stub].var()

        return df,variances_o,variances_r

#########################################################################################################################################################
# OBSOLETE CODE
#########################################################################################################################################################

def UnitFixOld(df,varlist,iterates=True,replace=False):
    print('Unit Fixing Algorithm')
    print('***********************************************************************************')
    df = df.sort_values(by=['BvD ID number','year'])
    df = df.set_index(['BvD ID number'])
    
    # Lead/Lag Years
    df['Lyear'] = df.groupby(['BvD ID number'])['year'].shift(1)
    df['Fyear'] = df.groupby(['BvD ID number'])['year'].shift(-1)
    df['yr'] = df['year']
    df = df.reset_index()
    df = df.set_index(['BvD ID number','year'])

    # Iterate over Variables
    for var_o in varlist:
        print('**************')
        print(f'VARIABLE: {var_o}')

        # Define Variable being manipulated
        if replace == True:
            var = var_o 
        else:
            df['new'+var_o] = df[var_o]
            var = 'new'+var_o
        
        iter,maxiter,nchanges = 0,21,10
        
        # Iterate as long as a positive number of changes are being made
        while (iter<=maxiter) & (nchanges>0):
            iter += 1
            df,nchanges = UnitFixOneStep(df,var,iter)
            print(f'    Changes made: {nchanges}')
            if iterates == False:
                df = df.drop(columns=[var+'_iter_'+str(iter)])
    
    df = df.reset_index()
    df = df.drop(columns=['yr','Lyear','Fyear'])
    return df 

def analyze_financials_leverage(country,savepath):
    
    # Set path
    countrypath = savepath+country+'/'

    # Imports
    try: 
        df = analyze_import_complete_observations(country,savepath,chunksize=2_000_000,varlist=varlist_TFP_sample+['OPRE','M','VA','K','Ki','Kt','KT','L','wL','L_imputed'],chunkyfunc=chunkyfunc_financials)
        import_ok = 1
    except:
        try:
            df = analyze_import_complete_observations(country,savepath,chunksize=2_000_000,varlist=varlist_TFP_sample+['OPRE','M','VA','K','Ki','Kt','KT','L','wL'],chunkyfunc=chunkyfunc_financials)
            df['L_imputed'] = df['L']
            import_ok = 1
        except:
            print(f'COULDNT DO IMPORT, something is wrong.')
            import_ok = 0
    
    if import_ok == 0:
        return None
    else:
        
        # New Industry Measures
        df = assign_industry(df)

        # Compute Sectoral measures of leverage
        df = df.set_index(['year','industry'])
        print(f'Find leverage by sector x time')
        for lev in range(1,4):

            # Mean Leverage
            df['mean_leverage_st_'+str(lev)] = df.groupby(['year','industry'])['leverage_'+str(lev)].mean()

            # Median Leverage
            df['median_leverage_st_'+str(lev)] = df.groupby(['year','industry'])['leverage_'+str(lev)].median()

        # Compute Sectoral x Regional measures of leverage
        df = df.reset_index()
        has_nuts = len(df['NUTS3'].unique()) > 1
        has_reg = len(df['Region in country'].unique()) > 1 | has_nuts == 1
        if has_nuts == True:
            df['geo'] = df['NUTS3']

        elif (has_nuts == False) & (has_reg == True):
            df['geo'] = df['Region in country'] + ', ' + df['Country']

        if len(df['geo'].unique()) > 1:
            print(f'Find leverage by sector x time x geography')
            df = df.set_index(['year','industry','geo'])
            
            for lev in range(1,4):

                # Mean Leverage
                df['mean_leverage_gst_'+str(lev)] = df.groupby(['year','industry','geo'])['leverage_'+str(lev)].mean()

                # Median Leverage
                df['median_leverage_gst_'+str(lev)] = df.groupby(['year','industry','geo'])['leverage_'+str(lev)].median()

            df = df.reset_index()

        # Residuals
        resvars = ['leverage_'+str(i) for i in range(1,4)]

        if has_reg == True:
            onvars = ['year','industry','geo']
        else:
            onvars = ['year','industry']
        
        print(f'Calculating Leverage residuals on {onvars}')
        df,v_orig,v_res = Residualize(df,resvars,onvars,variances=True)
        print(f'*************************')
        print(f"leverage 1: Raw: {v_orig['leverage_1'] : 5.2n}, Res: {v_res['leverage_1'] : 5.2n}")
        print(f"leverage 2: Raw: {v_orig['leverage_2'] : 5.2n}, Res: {v_res['leverage_2'] : 5.2n}")
        print(f"leverage 3: Raw: {v_orig['leverage_3'] : 5.2n}, Res: {v_res['leverage_3'] : 5.2n}")

        # Save
        df = df.reset_index()
        df = df.to_csv(countrypath+country+'_leverage_analysis.csv')
        return None
