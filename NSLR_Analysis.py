# -*- coding: utf-8 -*-
"""
Date: 9/16/2022

Author: Jackson Hawkins, Mount Washington Observatory Fall Intern

Goals:
    -Read in ARVP data, plot all data in boxplots
    -Calculate monthly & seasonal NSLRs based on average of daily NSLRs
    -Statistically test to see if seasons are different from each other for max, min, & avg data
    -Statistically test to see if monthly NSLRs are different from the ELR

To run different datasets:
    1) Change filepath(s) (duh)
    2) Change start date & number of days to appropriate in cell around line 155 (to calc. daily NSLR)
    3) Change order of months at line 279 (currently Dec first to reflect accurate ordering for this 1 year)

"""

#Import packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
import datetime
from scipy import stats
from pandas.api.types import CategoricalDtype



#%% Read in data and do initial cleanup/prep

# Read in .csv with data
dfRaw = pd.read_csv('S:\\Summit\\Interns\\JacksonHawkins\\NSLR_Project\\Data\\CleanedData2.csv')

#Exclude temp values that are way too high or too low
dfRaw = dfRaw.drop(dfRaw[dfRaw['TempF'] > 110].index) # 106*F is highest temp recorded in NH
dfRaw = dfRaw.drop(dfRaw[dfRaw['TempF'] < -60].index) # -50*F is lowest temp recorded in NH


# Drop any days for which summit or base data are misssing

# Split into max/min/mean
TmaxRaw = dfRaw[dfRaw['Ttype'] == 'Tmax'].reset_index(drop = True)
TminRaw = dfRaw[dfRaw['Ttype'] == 'Tmin'].reset_index(drop = True)
TavgRaw = dfRaw[dfRaw['Ttype'] == 'Tavg'].reset_index(drop = True)

# Make list of those 3 data types
TtypesRaw = [TmaxRaw, TminRaw, TavgRaw]

# Initiate an empty df
df21 = pd.DataFrame()

# Loop through the raw data for 3 types of temp
for ttype in TtypesRaw:

    # Select only summit or base data
    ends = ttype[ttype['Elev'].isin([6288, 1600])]
    
    # Get null values for the summit/base data
    nulls = ends[ends['TempF'].isna()].reset_index(drop = True)
    
    # Make list of dates with null values
    null_dates = nulls['date'].values.tolist()
    
    # Drop those dates from the overall dataset
    ttype = ttype[~ttype['date'].isin(null_dates)]
    
    #Add temporary (within loop) df to original df
    df21 = pd.concat([df21, ttype],
                            ignore_index = True)


#Drop any other NULL/NaN values
df21 = df21.dropna()

#Add Celsius temperature column
df21 = df21.assign(TempC = (df21['TempF']-32)/1.8)

#Add meters elevation column
df21 = df21.assign(ElevM = (df21['Elev']/3.28))

#Make date column into date format & add month column
df21['date'] = pd.to_datetime(df21['date'])
df21['Month'] = df21['date'].apply(lambda x: x.strftime('%b'))


#Write a function to interpret month to be season
def whatseason(row): 
    if row['Month'] == 'Jan' or row['Month'] == 'Feb' or row['Month'] == 'Dec':
        return 'Winter'
    if row['Month'] == 'Mar' or row['Month'] == 'Apr' or row['Month'] == 'May':
        return 'Spring'
    if row['Month'] == 'Jun' or row['Month'] == 'Jul' or row['Month'] == 'Aug':
        return 'Summer'
    if row['Month'] == 'Sep' or row['Month'] == 'Oct' or row['Month'] == 'Nov':
        return 'Fall'
    else:
        return 'ERROR'

#Use above function to add season column to df21
df21['season'] = df21.apply(lambda row: whatseason(row), axis = 1)

#Subset df into Tmax, Tmin, Tmean
Tmax = df21[df21['Ttype'] == 'Tmax'].reset_index(drop = True)
Tmin = df21[df21['Ttype'] == 'Tmin'].reset_index(drop = True)
Tavg = df21[df21['Ttype'] == 'Tavg'].reset_index(drop = True)

#Create list of 3 subset dfs
Ttypes = [Tmax, Tmin, Tavg]


#%% Make box and whisker plots for the full dataset

# This is the desired order of Ttypes
type_order = ['Tmax', 'Tavg', 'Tmin']

# Create a category type along which to order data; in this case, ttype order
cat_type_order = CategoricalDtype(type_order, ordered=True)
df21['Ttype'] = df21['Ttype'].astype(cat_type_order)

# Sort the data frame according to that recently ordered category
df21 = df21.sort_values('Ttype')


# Set up grid formating, using entire available dataset
g = sns.FacetGrid(df21, col = 'season', 
                  col_wrap = 2, 
                  col_order = ['Winter', 'Spring', 'Summer', 'Fall'],
                  aspect = 1.25)

# Add boxplots to the grid & format axis labels 
g.map(sns.boxplot, 'Elev', 'TempC', 'Ttype',
                  palette = ['#31688e', '#b5de2b', '#440154'], boxprops=dict(alpha=.8), saturation = .9).set(
            xlabel = 'Elevation (Feet)',
            ylabel = 'Temperature (°C)')
                      
# Format titles for subplots and plot
g.set_titles('{col_name}')
g.fig.subplots_adjust(top = 0.9)
g.fig.suptitle('ARVP Temperature Data', fontweight = 'bold')

# Add a legend
g.add_legend()

# Export figure
# plt.savefig('S:/Summit/Interns/JacksonHawkins/NSLR_Project/Figures/DataDist2021.png', dpi = 300)


#%% Create a list of K days starting from a date specified in first row

# Start from the date specificed here (inclusive)
test_date = datetime.datetime.strptime('2020-12-01', '%Y-%m-%d')

# Number of days to list
K = 365

# Create the list and format it appropriately
date_generated = pd.date_range(test_date, periods = K)
date_generated = date_generated.strftime("%Y-%m-%d") # Drop the timestamp


#%% Loop thru list of dates to calculate NSLR for each date

# Initiate empty df that daily NSLRs will wind up in
DailyNSLR = pd.DataFrame()

# Loop through Tmax/Tmin/Tavg
for t in Ttypes:
    
    # Loop through each date in the year of observation
    for date in date_generated:
        
        # Make a dataframe that's only data from the current date in the loop
        df = t.loc[t['date'] == date].reset_index(drop = True)
    
        #Create NumPy arrays for data being regressed
        X = df.iloc[:, 5].values.reshape(-1, 1)
        Y = df.iloc[:, 4].values.reshape(-1, 1)
    
        #Choose which model to use
        lin = LinearRegression()
    
        #Fit the model to the data
        lin.fit(X, Y)
        Y_pred = lin.predict(X)
        
        #Calculate NSLR based on lin reg
        nslr = lin.coef_[0][0] * 1000
        nslr = round(nslr, 2)
        
        
        #Create temporary df with this round's value
        dict1 = {'Month' : [df['Month'][0]], #add loop's values to a dictionary
                 'Date' : [df['date'][0]],
                'Ttype' : [df['Ttype'][0]],
                'nslr' : nslr,
                'Season' : [df['season'][0]]}
        df1 = pd.DataFrame(data = dict1) #Create a df from the dictionary
        
        #Add temporary (within loop) df to original df
        DailyNSLR = pd.concat([DailyNSLR, df1],
                                ignore_index = True)
    
    
    
#%% Create df with seasonal mean NSLRs for Tmax/Tmin/Tavg by averaging daily NSLRs

AvgSeasonalNSLR = DailyNSLR.groupby(['Ttype', 'Season'])['nslr'].agg(['mean', 'sem'])

# rename mean column to be nslr
AvgSeasonalNSLR = AvgSeasonalNSLR.rename(columns = {'mean' : 'nslr'})

AvgSeasonalNSLR.reset_index(inplace = True)

#%% Plot seasonal NSLRs

# Desired order of Ttypes
type_order = ['Tmax', 'Tavg', 'Tmin']

# Create a category type along which to order data; in this case, Ttype order
cat_type_order = CategoricalDtype(type_order, ordered=True)
AvgSeasonalNSLR['Ttype'] = AvgSeasonalNSLR['Ttype'].astype(cat_type_order)

# Sort the data frame according to that recently ordered category
AvgSeasonalNSLR = AvgSeasonalNSLR.sort_values('Ttype')


# Reformat seasonal NSLR df to more easily plot a grouped bar chart, order by season
SeasonalNSLR_pivot = AvgSeasonalNSLR.pivot('Season', 'Ttype', 'nslr').loc[['Winter', 'Spring', 'Summer', 'Fall']]

# Pivot df to include standard error of the mean (SEM) error bars
SeasonalSEM_pivot = AvgSeasonalNSLR.pivot('Season', 'Ttype', 'sem').loc[['Winter', 'Spring', 'Summer', 'Fall']]

# Plot a grouped bar chart for seasonal NSLR
SeasonalNSLR_pivot.plot(kind = 'bar', yerr = SeasonalSEM_pivot, capsize = 3,
                        color = {'Tmax' : '#31688e', 'Tavg' : '#b5de2b', 'Tmin': '#440154'},
                        alpha = 0.8)

# Add single dashed line at -6.5 to show ELR
plt.hlines(-6.5, -.3, 3.3, '#d44842', linestyles = (0, (5, 10)), label = 'ELR')

#Flip y axis
plt.ylim(0, -9) 

#Label that ish
plt.xlabel('Season')
plt.xticks(rotation = 0)
plt.ylabel('Near-Surface Lapse Rate (°C/Km)')
plt.title('Seasonal NSLRs', fontweight = 'bold')

# Remove legend title
l = plt.legend()
l.set_title('')


# Export figure
# plt.savefig('S:/Summit/Interns/JacksonHawkins/NSLR_Project/Figures/SeasonalNSLR2021_ELR.png', dpi = 300)

#%% Create df with monthly mean NSLRs as above for seasonal NSLRs

AvgMonthlyNSLR = DailyNSLR.groupby(['Month', 'Ttype']).agg(nslr = ('nslr', 'mean'))

AvgMonthlyNSLR.reset_index(inplace = True)

#%% Re-order AvgMonthlyNSLR df so that data are in logical max/avg/min order for key

# This is the desired order of Ttypes
type_order = ['Tmax', 'Tavg', 'Tmin']

# Create a category type along which to order data; in this case, ttype order
cat_type_order = CategoricalDtype(type_order, ordered=True)
AvgMonthlyNSLR['Ttype'] = AvgMonthlyNSLR['Ttype'].astype(cat_type_order)

# Sort the data frame according to that recently ordered category
AvgMonthlyNSLR = AvgMonthlyNSLR.sort_values('Ttype')


#%% Re-order AvgMonthlyNSLR df so that data are chronological and can be plotted logically

# This is the order of months, tweaked to better represent this meteorological year
month_order = ['Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']

# Create a category type along which to order data; in this case, monthly order
cat_month_order = CategoricalDtype(month_order, ordered=True)
AvgMonthlyNSLR['Month'] = AvgMonthlyNSLR['Month'].astype(cat_month_order)

# Sort the data frame according to that recently ordered category
AvgMonthlyNSLR = AvgMonthlyNSLR.sort_values('Month')

#%% Plot monthly NSLR (based on averaged daily data) for each Ttype


# Plot the data as a line graph
plot2 = sns.lineplot(x = 'Month', y = 'nslr', data = AvgMonthlyNSLR, 
                    hue = 'Ttype', palette = ['#31688e', '#b5de2b', '#440154'])


# Shade seasonal periods
plot2.fill_between(['Dec', 'Feb'], y1 = -9, y2 = -3, alpha = 0.2, color = 'gray')

plot2.fill_between(['Mar', 'May'], y1 = -9, y2 = -3, alpha = 0.5, color = 'gray')

plot2.fill_between(['Jun', 'Aug'], y1 = -9, y2 = -3, alpha = 0.2, color = 'gray')

plot2.fill_between(['Sep', 'Nov'], y1 = -9, y2 = -3, alpha = 0.5, color = 'gray')


# Plot points for monthly data
plot2 = sns.scatterplot(x = 'Month', y = 'nslr', data = AvgMonthlyNSLR, 
                color = 'black')

# Add single dashed line at -6.5 to show ELR
plot2 = sns.lineplot(x = 'Month', y = -6.5, data = AvgMonthlyNSLR, 
                    color = '#d44842', linestyle = (0, (5, 20)), linewidth = .8, label = 'ELR')


# Add labels
plot2.set_xlabel('Month')
plot2.set_ylabel('Near-Surface Lapse Rate (°C/Km)')
plot2.set_title('Monthly NSLR', fontweight = 'bold')

# Flip the Y axis
plot2.invert_yaxis()

# Export figure
# plt.savefig('S:/Summit/Interns/JacksonHawkins/NSLR_Project/Figures/NSLR_Monthly2021_Shaded.png', dpi = 300)


#%% Create list of Tmax, Tmin, and Tavg NSLRs for stat tests, define function to return significance as Y/N

# Subset avg nslr df into avg/max/min
TavgNSLR = DailyNSLR.loc[DailyNSLR['Ttype'] == 'Tavg'].reset_index(drop = True)
TmaxNSLR = DailyNSLR.loc[DailyNSLR['Ttype'] == 'Tmax'].reset_index(drop = True)
TminNSLR = DailyNSLR.loc[DailyNSLR['Ttype'] == 'Tmin'].reset_index(drop = True)

# Create list of 3 subsetted dfs
NSLR_ByType = [TmaxNSLR, TavgNSLR, TminNSLR]

# Write a function to discern if a value is significant or not (95% confidence)
def is_sig(row):
    if row['p'] > 0.05:
        return 'N'
    if row['p'] <= 0.05:
        return 'Y'
    else:
        return 'ERROR'

#%% Plot histograms of data broken out by analysis types, to show non-normality of data

# List of colors to keep schema consistent
wanted_palette = ['#31688e', '#b5de2b', '#440154']

# List of seasons
seasons = ['Winter', 'Spring', 'Summer', 'Fall']

# To help iterate through color scheme
c = 0

for df in NSLR_ByType:
    
    # Set up grid formating, using max/avg/min dataset, breaking out by season
    g = sns.FacetGrid(df, col = 'Season', 
                      col_wrap = 2, 
                      col_order = ['Winter', 'Spring', 'Summer', 'Fall'],
                      aspect = 1.25)
    
    # Add histograms to the grid formatting, pick color, format axis labels 
    g.map(plt.hist, 'nslr', bins = 30, color = wanted_palette[c], alpha = 0.8).set(
                xlabel = 'NSLR (°C/Km)')
    
    #Format titles for subplots and plot
    g.set_titles('{col_name}')
    g.fig.subplots_adjust(top = 0.9)
    g.fig.suptitle('Daily ' + df['Ttype'][0] + ' Near-Surface Lapse Rate Distribution', fontweight = 'bold')
    
    # To pick the right color in the next round
    c = c + 1
    
    # Numerically test for normal distribution in each season
    
    for season in seasons:
        
        # Use a Shapiro-Wilk test to test for normality in each season
        stat, p = stats.shapiro(df[df['Season'] == season]['nslr'])
       
        # Report Shapiro-Wilk test results
        print(season + ' ' + df['Ttype'][0] + '\n' + 'stat=%.3f, p=%.3f' % (stat, p))

        # Interpert Shapiro-Wilk results
        if p > 0.05:
            print('Probably Gauss\n')
        else:
            print('Probably not Gauss\n')

        
    
    # Export figure
    # plt.savefig('S:/Summit/Interns/JacksonHawkins/NSLR_Project/Figures/%s_NSLR_hist.png' %df['Ttype'][0], dpi = 300)


#%% Statistically test difference between all seasons for every Ttype using non-parametric test

# Create empty df with column headers
StatOutput = pd.DataFrame(columns = ['s1', 's2', 'p', 'Ttype'])

# Loop through the list of Ttypes
for Ttype in NSLR_ByType:
    
    # Subset monthly NSLRs into seasons for just nslr
    WtrData = Ttype[Ttype['Season'] == 'Winter']
    WtrData = WtrData['nslr']
    
    SpgData = Ttype[Ttype['Season'] == 'Spring']
    SpgData = SpgData['nslr']
    
    SumData = Ttype[Ttype['Season'] == 'Summer']
    SumData = SumData['nslr']
    
    FallData = Ttype[Ttype['Season'] == 'Fall']
    FallData = FallData['nslr']
    
    # Perform stat tests for all 6 seasonal combinations, and add output to StatOutput df
    
    # Winter vs. Spring
    p1 = stats.mannwhitneyu(WtrData, SpgData).pvalue
    StatOutput.loc[len(StatOutput)] = ['Winter', 'Spring', p1, Ttype['Ttype'][0]]
    
    # Winter vs. Summer
    p1 = stats.mannwhitneyu(WtrData, SumData).pvalue
    StatOutput.loc[len(StatOutput)] = ['Winter', 'Summer', p1, Ttype['Ttype'][0]]
    
    # Winter vs. Fall
    p1 = stats.mannwhitneyu(WtrData, FallData).pvalue
    StatOutput.loc[len(StatOutput)] = ['Winter', 'Fall', p1, Ttype['Ttype'][0]]
    
    # Summer vs. Spring
    p1 = stats.mannwhitneyu(SumData, SpgData).pvalue
    StatOutput.loc[len(StatOutput)] = ['Summer', 'Spring', p1, Ttype['Ttype'][0]]
    
    # Summer vs. Fall
    p1 = stats.mannwhitneyu(SumData, FallData).pvalue
    StatOutput.loc[len(StatOutput)] = ['Summer', 'Fall', p1, Ttype['Ttype'][0]]
    
    # Spring vs. Fall
    p1 = stats.mannwhitneyu(SpgData, FallData).pvalue
    StatOutput.loc[len(StatOutput)] = ['Spring', 'Fall', p1, Ttype['Ttype'][0]]

# Round StatOutput df to get P values as a decimal, not sci. notation
StatOutput = StatOutput.round(5)

# Add significance column
StatOutput['sig'] = StatOutput.apply(lambda row: is_sig(row), axis = 1)


#%% Statistically test to see if each season is different from -6.5C/Km

# Environmental Lapse Rate (global avg)
elr = -6.5


# Create empty df with column headers for output
ELRStat = pd.DataFrame(columns = ['season', 'p', 'Ttype'])


# Loop through the list of Ttypes
for Ttype in NSLR_ByType:

    
    # Subset monthly NSLRs into seasons for just nslr
    WtrData = Ttype[Ttype['Season'] == 'Winter']
    WtrData = WtrData['nslr']
    
    SpgData = Ttype[Ttype['Season'] == 'Spring']
    SpgData = SpgData['nslr']
    
    SumData = Ttype[Ttype['Season'] == 'Summer']
    SumData = SumData['nslr']
    
    FallData = Ttype[Ttype['Season'] == 'Fall']
    FallData = FallData['nslr']
    
    
    # Winter comparison
    x = stats.ttest_1samp(WtrData, elr).pvalue
    ELRStat.loc[len(ELRStat)] = ['Winter', x, Ttype['Ttype'][0]]
    
    
    # Spring comparison
    x = stats.ttest_1samp(SpgData, elr).pvalue
    ELRStat.loc[len(ELRStat)] = ['Spring', x, Ttype['Ttype'][0]]
    
    
    # Summer comparison
    x = stats.ttest_1samp(SumData, elr).pvalue
    ELRStat.loc[len(ELRStat)] = ['Summer', x, Ttype['Ttype'][0]]
    
    
    # Fall comparison
    x = stats.ttest_1samp(FallData, elr).pvalue
    ELRStat.loc[len(ELRStat)] = ['Fall', x, Ttype['Ttype'][0]]


# Add significance column
ELRStat['sig'] = ELRStat.apply(lambda row: is_sig(row), axis = 1)

# Round data to ditch sci. notation
ELRStat = ELRStat.round(5)


#%% Report Summary Data Frames, save to system

# Seasonal Near-Surface Lapse Rates
print('Seasonal NSLRs')
print(AvgSeasonalNSLR)
AvgSeasonalNSLR.to_excel('S:/Summit/Interns/JacksonHawkins/NSLR_Project/Results/SeasonalNSLR.xlsx')

# Monthly Near-Surface Lapse Rates
print('\nMonthly NSLRs')
print(AvgMonthlyNSLR)
AvgMonthlyNSLR.to_excel('S:/Summit/Interns/JacksonHawkins/NSLR_Project/Results/MonthlyNSLR.xlsx')

# Statistical difference between seasons
print('\nSeasonal Statistical Differences')
print(StatOutput)
StatOutput.to_excel('S:/Summit/Interns/JacksonHawkins/NSLR_Project/Results/SeasonalStatDif.xlsx')

# Statistical difference from ELRs
print('\nSeasonal Statistical Differences from ELR')
print(ELRStat)
ELRStat.to_excel('S:/Summit/Interns/JacksonHawkins/NSLR_Project/Results/StatDifFromELR.xlsx')