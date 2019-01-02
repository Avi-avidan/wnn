import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from geopy import distance
import math
import datetime
import random
import os
import seaborn as sns
import seaborngrid 
from seaborngrid import SeabornGrid

data_root = '/Users/aviavidan/data/kaggle/wnv'
weather_cols = ['Date', 'Station', 'WetBulb', 'Tavg', 'Tmax', 'Tmin', 'Cool','Sunrise', 'Sunset', 
                'PrecipTotal', 'AvgSpeed']

weather_features = ['dl_hrs', 'avg_tmp', 'avg_wind', 'PrecipTotal_3wks', 'dl_2wks_ago', 'dl_4wks_ago']
relevant_cols = ['Date', 'Species', 'Trap', 'Latitude', 'Longitude', 'AddressAccuracy', 'NumMosquitos', 'WnvPresent']

final_cols = ['Latitude', 'Longitude', 'Trap', 'AddressAccuracy',                # original features
              'Duplicats', 'day_count', 'day_of_week', 'week', 'month',          # engineered seasonality features
              'ohare_3', 'ohare_5', 'mckin_3', 'mckin_5', 'aubur_3', 'aubur_5',  # engineered location features
              'CULEX PIPIENS/RESTUANS', 'CULEX PIPIENS', 'CULEX RESTUANS',       # species features
              'avg_tmp', 'dl_hrs', 'dl_2wks_ago', 'dl_4wks_ago', 'avg_wind', 
              'PrecipTotal_3wks', 'WetBulb',                                     # weather features
              'Tavg', 'Tmin', 'Tmax', 'Cool', 'PrecipTotal', 'AvgSpeed',
              'effective_spray']         # engineered weather features

contiuous_cols = ['avg_tmp', 'dl_hrs', 'avg_wind', 'PrecipTotal_3wks', 'effective_spray',
                  'Tavg', 'Tmin', 'Tmax', 'Cool', 'PrecipTotal', 'AvgSpeed', 'WetBulb']

discrete_cols = ['AddressAccuracy', 'Duplicats', 'day_of_week', 'month', 'ohare_3', 'ohare_5', 'mckin_3', 
                 'mckin_5', 'aubur_3', 'aubur_5', 'CULEX PIPIENS', 'CULEX RESTUANS'] #'CULEX PIPIENS/RESTUANS'
colors = ['y', 'm', 'g', 'c', 'b', 'w']


def drop_irrelevant_cols(df, keep_cols):
    df.drop(columns=[col for col in df.columns.tolist() if col not in keep_cols], inplace=True)

def preprocess_dataset(df, name, weather_stat1, weather_stat2, all_traps, 
                       wnv_species, spray, final_cols=final_cols):
    if 'test' in name:
        drop_irrelevant_cols(df, relevant_cols[:-2])
    else:
        drop_irrelevant_cols(df, relevant_cols)
    #add_effct_spray(spray, df)
    add_dist_from_int_points(df, interest_points)
    add_weighted_weather_cols(df, weather_stat1, weather_stat2)
    traps_to_categorical(df, all_traps) # validated all train traps are in test set
    add_date_features(df)
    add_dups(df, name)
    wnvdetect_by_specie(df, wnv_species)
    drop_irrelevant_cols(df, final_cols)
    df.to_csv(name +'.csv', columns=df.columns.tolist())
    print('saved, done.')
    return df


def fill_missing_vals(df, columns=weather_cols):
    for col in columns:
        for row in range(df.shape[0]):
            if df.loc[row, col] in ['T','  T']:
                df.loc[row, col] = 0
            elif df.loc[row, col] in ['-', 'M']:
                df.loc[row, col] = df.loc[row - 1, col]
    df.loc[:, columns[2:]] = df[columns[2:]].astype(dtype='float32')
    return df

def float_time(time_o):
    return int(time_o)%100/60+int(time_o)//100

def offset_feature(df, col, offset, col_name):
    for row in range(df.shape[0]):   
        if row > offset:
            df.loc[row, col_name] = df[col][row-offset:row-offset+7].mean()
        else:
            df.loc[row, col_name] = df.loc[0, col]

            
def add_weather_features(df):
    df.loc[:,'dl_hrs'] = df.apply(lambda row: float_time(row.Sunset) - float_time(row.Sunrise), axis=1)
    df.loc[:,'avg_tmp'] = df['Tavg'].rolling(2*7).mean()
    df.loc[:,'avg_wind'] = df['AvgSpeed'].rolling(3*7).mean()
    df.loc[:,'PrecipTotal_3wks'] = df['PrecipTotal'].rolling(3*7).sum()/2
    four_weeks_line_count = 4*7
    offset_feature(df, 'dl_hrs', four_weeks_line_count, 'dl_4wks_ago')
    offset_feature(df, 'dl_hrs', four_weeks_line_count//2, 'dl_2wks_ago')
    df.loc[:, weather_features] = df[weather_features].fillna(df.mean())
    return df


def add_weather_features(df):
    df['dl_hrs'] = df.apply(lambda row: float_time(row.Sunset) - float_time(row.Sunrise), axis=1)
    df['avg_tmp'] = df['Tavg'].rolling(2*7).mean()
    df['avg_wind'] = df['AvgSpeed'].rolling(3*7).mean()
    df['PrecipTotal_3wks'] = df['PrecipTotal'].rolling(3*7).sum()/2
    four_weeks_line_count = 4*7
    offset_feature(df, 'dl_hrs', four_weeks_line_count, 'dl_4wks_ago')
    offset_feature(df, 'dl_hrs', four_weeks_line_count//2, 'dl_2wks_ago')
    df[weather_features] = df[weather_features].fillna(df.mean())
    return df
   
    
def get_weighted_wfeature(feat1, feat2, dist1, dist2):
    return (feat1*1/dist1+feat2*1/dist2)/(1/dist1+1/dist2)


def split_weather_by_station(weather):
    weather = fill_missing_vals(weather)
    weather_stat1 = add_weather_features(weather[weather['Station'] ==1].reset_index(drop = True))
    weather_stat2 = add_weather_features(weather[weather['Station'] ==2].reset_index(drop = True))
    return weather_stat1, weather_stat2

def add_weighted_weather_cols(df, weather_stat1, weather_stat2, weather_features=weather_features):
    add_features = weather_features + ['PrecipTotal', 'AvgSpeed', 'WetBulb', 'Tavg', 'Tmax', 'Tmin', 'Cool']
    for col in add_features:
        print('adding feature to data set:', col)
        for i, row in enumerate(df.index):
            if i%(df.shape[0]//10) == 0:
                print('{:.2f}% complete'.format(i/df.shape[0]*100))
            date = df.loc[row, 'Date']
            try:
                feat1 = float(weather_stat1[weather_stat1['Date'] == date][col])
                feat2 = float(weather_stat2[weather_stat2['Date'] == date][col])
                dist1, dist2 = df.loc[row, ['stat1_dist', 'stat2_dist']]
                df.loc[i, col] = get_weighted_wfeature(feat1, feat2, dist1, dist2)
            except:
                print(weather_stat1[weather_stat1['Date'] == date][col])
                print(weather_stat2[weather_stat2['Date'] == date][col])
                print(df.loc[row, ['stat1_dist', 'stat2_dist']])

                
def reformat_enrich(df):
    rename_dict = {'TEST DATE':'Date', 'RESULT':'WnvPresent', 'NUMER OF MOSQUITOS':'NumMosquitos',
                  'TRAP':'Trap', 'LONGITUDE':'Longitude', 'LATITUDE':'Latitude', 'SPECIES':'Species'}
    df.rename(columns=rename_dict, inplace=True)
    df.loc[:, 'WnvPresent'] = df.WnvPresent.apply(lambda x: int(x == 'positive'))


def load_data(csv_list=['train', 'spray', 'weather', 'test', 'enrich']):
    dfs = [pd.read_csv(os.path.join(data_root, csv+'.csv')) for csv in csv_list]
    reformat_enrich(dfs[-1])
    [df_scope(dfs[i], csv_list[i]) for i in range(len(csv_list))]
    [print_dates_min_max(dfs[i], csv_list[i]) for i in range(len(csv_list))]
    [date_reformat(dfs[i]) for i in range(len(csv_list[:-1]))]
    enrich_date_reformat(dfs[-1])
    return dfs

def load_map(mapdata, legend=[[]*2]):
    patches = []
    aspect = mapdata.shape[0] / mapdata.shape[1]
    lon_lat_box = (-88, -87.5, 41.6, 42.1)
    plt.figure(figsize=(15,15))
    plt.imshow(mapdata, 
               cmap=plt.get_cmap('gray'), 
               extent=lon_lat_box, 
               aspect=aspect)
    for c, text in legend:
        patches.append(mpatches.Patch(color=c, label=text))
    plt.legend(handles=patches)
    
def df_scope(df, name):
    nulls = df.isnull().sum().sum() 
    print('\nname: {}, rows: {}, columns: {}, null values: {}'.format(name, df.shape[0], df.shape[1], nulls))

def print_dates_min_max(df, name):
    print('\nname: {}, date range: {} - {}'.format(name, df['Date'].min(), df['Date'].max()))
    
def date_features(date, dformat='%Y-%m-%d'):
    date = datetime.datetime.strptime(date, dformat)
    day_count = get_day_count(date)
    day_of_week = date.weekday()
    week_of_year = day_count//7
    return day_count, day_of_week, week_of_year, date.month

def get_day_count(date):
    first_day_of_year = datetime.datetime(year=date.year, month=1, day=1)
    return (date - first_day_of_year).days + 1

def date_reformat(df, date_format='%Y-%m-%d'):
    #df.loc[:, 'Date'] = pd.to_datetime(df[['Date']]).date()
    df.loc[:, 'Date'] = pd.to_datetime(df['Date'], format=date_format)

def enrich_date_reformat(df, date_format='%Y-%m-%d'):
    df.loc[:, 'Date'] = df['Date'].apply(lambda x: x.split()[0])
    date_reformat(df, date_format='%m/%d/%Y')
    
def add_date_features(df):
    df.loc[:,'day_count'] = df.apply(lambda row: get_day_count(row.Date), axis=1)
    df.loc[:,'day_of_week'] = df.apply(lambda row: row.Date.weekday(), axis=1)
    df.loc[:,'week'] = df['Date'].dt.week
    df.loc[:,'month'] = df['Date'].dt.month

def add_time_features(df):
    df.loc[:, 'Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df.loc[:, 'Time'] = df['Time'].apply(lambda row: pd.to_datetime(row).strftime('%H:%M:%S'))
    df.loc[:, 'hour_of_day'] = df.apply(
        lambda row: float(row.Time.split(':')[0])+float(row.Time.split(':')[1])/60, axis=1)

def get_dist(pointa, pointb):
    return distance.great_circle(pointa, pointb).kilometers

def wnvdetect_by_specie(df, wnv_species):
    for i, specie in enumerate(wnv_species):
        df[wnv_species[i]] = 0
        # df.loc[:, wnv_species[i]][df.Species.isin([wnv_species[i]])] = 1
        df.loc[df.Species.isin([wnv_species[i]]), wnv_species[i]] = 1
    df.drop(columns=['Species'], inplace=True)
    
def traps_to_categorical(df, all_traps):
    df.loc[:, 'Trap'] = df['Trap'].apply(lambda x: all_traps.index(x))


# Station 1: CHICAGO O'HARE INTERNATIONAL AIRPORT Lat: 41.995 Lon: -87.933 Elev: 662 ft. above sea level
# Station 2: CHICAGO MIDWAY INTL ARPT Lat: 41.786 Lon: -87.752 Elev: 612 ft. above sea level

interest_points = {'stat1': (41.995, -87.933), 'stat2': (41.786, -87.68),
                   'ohare': (41.9742, -87.9073), 'mckin': (41.835, -87.682), 'aubur': (41.75, -87.64),
                   'river': (41.673, -87.6)
                  }


def add_dist_from_int_points(df, interest_points):
    for key, point in interest_points.items():
        df.loc[:, key+'_dist'] = df.apply(lambda row: get_dist((row.Latitude, row.Longitude), point), axis=1)
        if key in ['ohare', 'mckin', 'aubur']:
            df.loc[:, key+'_3'] = df[key+'_dist'].apply(lambda x: int(x < 2.5))
            df.loc[:, key+'_5'] = df[key+'_dist'].apply(lambda x: int(x < 4.5))

def add_effct_spray(spray, df):
    df.loc[:, 'effective_spray'] = df.apply(lambda row: spray_factor(row, spray), axis=1)
            
def spray_factor(row, spray):
    rel_spray = spray[(spray['Date'] < row.Date) & ((row.Date - spray['Date']) < pd.Timedelta(('10 days')))]
    factor = 0
    for spray_row in rel_spray.iterrows():
        #print(spray_row)
        try:
            distance = get_dist((row.Latitude, row.Longitude), (spray_row.Latitude, spray_row.Longitude))
            if distance < 0.3:
                t = (Date - spray_row['Date']).days
                factor += 2 ** (1/t/dis)
        except:
            try:
                print('df:', row.Date)#, (row.Latitude, row.Longitude))
            except:
                pass
            try:
                print('spray:', spray_row.Date, (spray_row.Latitude, spray_row.Longitude))
            except:
                pass
    return factor   

def plot_weather_features(df, features, wnv_by_year):
    ax2 = []
    f, ax1 = plt.subplots(len(features), 1, figsize=(18,4*len(features)))
    for i, feat in enumerate(features):
        color = 'tab:blue'
        ax1[i].tick_params(axis='y', labelcolor=color)
        ax1[i].set_ylabel(feat, color=color)
        ax1[i].plot(df['Date'], df[feat], marker='o', linestyle='', c=color,)
        ax2.append(ax1[i].twinx())

        color = 'tab:red'
        ax2[i].set_ylabel('WnvPresent', color=color)
        ax2[i].tick_params(axis='y', labelcolor=color)
        ax2[i].plot(wnv_by_year.index, wnv_by_year['WnvPresent'], marker='o', c=color, linestyle='')

    f.tight_layout()     
    plt.show()
    
def plot_detect_by_specie(df):
    detect_species = df[df['WnvPresent'] == 1].Species.unique().tolist()
    f, ax = plt.subplots(len(detect_species), 1, figsize=(16,3*len(detect_species)))
    for i in range(len(detect_species)):
        wnv_by_spe = df[df.Species.isin([detect_species[i]])][['Date', 'Trap', 'WnvPresent']].groupby(
        by = ['Date','Trap'])['Date','Trap','WnvPresent'].sum().reset_index().sort_values(
        'WnvPresent', ascending = False).set_index('Date')
        wnv_by_spe.plot(style='.', c=colors[i], grid=True, ax=ax[i], ylim=[0,12])
        ax[i].set_title('# of {} Detections by Year'.format(detect_species[i]), fontsize=16)
        ax[i].set_xlabel('Year', fontsize=14)
        ax[i].set_ylabel(detect_species[i] + ' Detections')
    plt.tight_layout()
    plt.show()

    


def plot_lat_lng_dist(wnv_species, wnv, wnv_neg):
    fig = plt.figure(figsize=(15,8))
    title = '{}, {} & {}'.format(wnv_species[0], wnv_species[1], wnv_species[2]) + ' [row 1: positive row 2: negative]'
    fig.suptitle(title, fontsize=14)
    #fig.subplots_adjust(top=2)
    gs = gridspec.GridSpec(2, 3)
    pos, neg, m = [], [], []
    sns.set_style("whitegrid")
    for i in range(len(wnv_species)):
        spe_wnv = wnv[wnv['Species'] == wnv_species[i]].values
        spe_wnv_neg = wnv_neg[wnv_neg['Species'] == wnv_species[i]].values
        print(len(spe_wnv), wnv_species[i])
        pos.append(sns.jointplot(x=np.float32(spe_wnv[:,0]), y=np.float32(spe_wnv[:,1]),
                                 col=wnv_species[i], row="WnvPresent=1", color=colors[i], kind="kde", space=0))

        neg.append(sns.jointplot(x=np.float32(spe_wnv_neg[:,0]), y=np.float32(spe_wnv_neg[:,1]), #xlim=[41.6,42.1], ylim=[-88,-87.5],
                      color=colors[i], kind="kde", space=0))
        m.append(SeabornGrid(pos[i], fig, gs[0, i]))
        m.append(SeabornGrid(neg[i], fig, gs[1, i]))

    gs.tight_layout(fig)
    plt.show()
    
    
def add_dups(df, name):
    if 'test' in name:
        df['Duplicats'] = df.duplicated(['Date', 'Species', 'Trap', 'Latitude', 'Longitude', 'AddressAccuracy'])
        df.loc[:, 'Duplicats'] = df.Duplicats.apply(lambda x : 1 if x else 0)
    else:
        df.loc[:, 'Duplicats'] = df.NumMosquitos.apply(lambda x : round(x/50))