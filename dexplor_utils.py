import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from geopy import distance
import math
import datetime
import random
import os

data_root = '/Users/aviavidan/data/kaggle/wnv'
mapdata = np.loadtxt(data_root + '/mapdata_copyright_openstreetmap_contributors.txt')
weather_cols = ['Date', 'Station', 'Tavg', 'Sunrise', 'Sunset', 'PrecipTotal', 'AvgSpeed']
weather_features = ['dl_hrs', 'dl_2wks_ago', 'dl_4wks_ago', 'avg_tmp', 'avg_wind', 'preciptotal_3wks']
relevant_cols = ['Date', 'Species', 'Trap', 'Latitude', 'Longitude', 'AddressAccuracy', 'NumMosquitos', 'WnvPresent']


colors = ['y', 'm', 'g', 'c', 'b', 'w']
final_cols = []

def load_preprocessed(csv_list=['x_train.csv', 'y_train.csv', 'test.csv']):
    return [pd.read_csv(ds, index_col=0) for ds in csv_list]


def preprocess_dataset(df, name, weather_stat1, weather_stat2, all_traps, wnv_species):
    df = df[relevant_cols]
    add_dist_from_weather_stations(df)
    add_weighted_weather_cols(df, weather_stat1, weather_stat2)
    traps_to_categorical(df, all_traps)
    add_date_features(df)
    wnvdetect_by_specie(df, wnv_species)
    df.drop(columns=['Date', 'stat1_dist', 'stat2_dist'], inplace = True)
    df = df[final_cols]
    df.to_csv(name +'.csv', columns=df.columns.tolist())

def fill_missing_vals(df, columns=weather_cols):
    for col in columns:
        for row in range(df.shape[0]):
            if df.loc[row, col] in ['T','  T']:
                df.loc[row, col] = 0
            elif df.loc[row, col] in ['-', 'M']:
                df.loc[row, col] = df.loc[row - 1, col]
    df[columns[2:]] = df[columns[2:]].astype(dtype='float32')
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
    df['dl_hrs'] = df.apply(lambda row: float_time(row.Sunset) - float_time(row.Sunrise), axis=1)
    df['avg_tmp'] = df['Tavg'].rolling(2*7).mean()
    df['avg_wind'] = df['AvgSpeed'].rolling(3*7).mean()
    df['preciptotal_3wks'] = df['PrecipTotal'].rolling(3*7).sum()/2
    four_weeks_line_count = 4*7
    offset_feature(df, 'dl_hrs', four_weeks_line_count, 'dl_4wks_ago')
    offset_feature(df, 'dl_hrs', four_weeks_line_count//2, 'dl_2wks_ago')
    df[weather_features] = df[weather_features].fillna(df.mean())
    return df
   
    
def get_weighted_wfeature(feat1, feat2, dist1, dist2):
    return (feat1*1/dist1+feat2*1/dist2)/(1/dist1+1/dist2)


def split_weather_by_station(weather):
    weather = fill_missing_vals(weather)
    weather_stat1 = weather[weather['Station'] ==1].reset_index(drop = True)
    weather_stat2 = weather[weather['Station'] ==2].reset_index(drop = True)
    weather_stat1 = add_weather_features(weather_stat1)
    weather_stat2 = add_weather_features(weather_stat2)
    return weather_stat1, weather_stat2

def add_weighted_weather_cols(df, weather_stat1, weather_stat2, weather_features=weather_features):
    add_features = weather_features + ['Tavg', 'PrecipTotal', 'AvgSpeed']
    for col in add_features:
        print('adding feature to data set:', col)
        for i, row in enumerate(df.index):
            if i%(df.shape[0]//10) == 0:
                print('{:.2f}% complete'.format(i/df.shape[0]*100))
            date = df.loc[row, 'Date']
            try:
                feat1 = float(weather_stat1[col][weather_stat1['Date'] == date])
                feat2 = float(weather_stat2[col][weather_stat2['Date'] == date])
                dist1, dist2 = df.loc[row, ['stat1_dist', 'stat2_dist']]
                df.loc[i, col] = get_weighted_wfeature(feat1, feat2, dist1, dist2)
            except:
                print(weather_stat1[col][weather_stat1['Date'] == date])
                print(weather_stat2[col][weather_stat2['Date'] == date])
                print(df.loc[row, ['stat1_dist', 'stat2_dist']])

def reformat_enrich(df):
    rename_dict = {'TEST DATE':'Date', 'RESULT':'WnvPresent', 'NUMER OF MOSQUITOS':'NumMosquitos',
                  'TRAP':'Trap', 'LONGITUDE':'Longitude', 'LATITUDE':'Latitude', 'SPECIES':'Species'}
    df.rename(columns=rename_dict, inplace=True)
    df.WnvPresent.loc[df.WnvPresent == 'negative'] = 0
    df.WnvPresent.loc[df.WnvPresent == 'positive'] = 1

    
# loading original datasets
def load_data(csv_list=['train', 'spray', 'weather', 'test', 'enrich'], pprint=True):
    dfs = [pd.read_csv(os.path.join(data_root, csv+'.csv')) for csv in csv_list]
    reformat_enrich(dfs[-1])
    
    if pprint:
        [df_scope(dfs[i], csv_list[i]) for i in range(len(dfs))]
        [print_dates_min_max(dfs[i], csv_list[i]) for i in range(len(dfs))]
        
    [date_reformat(dfs[i]) for i in range(len(csv_list[:-1]))]
    enrich_date_reformat(dfs[-1], date_format='%m/%d/%Y')
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
    df['Date'] = pd.to_datetime(df['Date'], format=date_format)

def enrich_date_reformat(df, date_format='%Y-%m-%d'):
    df['Date'] = df['Date'].apply(lambda x: x.split()[0])
    date_reformat(df, date_format='%m/%d/%Y')
    
def add_date_features(df):
    df['day_count'] = df.apply(lambda row: get_day_count(row.Date), axis=1)
    df['day_of_week'] = df.apply(lambda row: row.Date.weekday(), axis=1)
    df['week'] = df['Date'].dt.week
    df['month'] = df['Date'].dt.month

def add_time_features(df):
    df.loc['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df['Time'] = df['Time'].apply(lambda row: pd.to_datetime(row).strftime('%H:%M:%S'))
    df['hour_of_day'] = df.apply(
        lambda row: float(row.Time.split(':')[0])+float(row.Time.split(':')[1])/60, axis=1)

def get_dist(pointa, pointb):
    return distance.great_circle(pointa, pointb).kilometers

def wnvdetect_by_specie(df, wnv_species):
    for i, specie in enumerate(wnv_species):
        df.loc[wnv_species[i]] = 0
        df[wnv_species[i]][df.Species.isin([wnv_species[i]])] = 1
    df.drop(columns=['Species'], inplace=True)
    
def traps_to_categorical(df, all_traps):
    df.loc['Trap'] = df['Trap'].apply(lambda x: all_traps.index(x))


# Station 1: CHICAGO O'HARE INTERNATIONAL AIRPORT Lat: 41.995 Lon: -87.933 Elev: 662 ft. above sea level
# Station 2: CHICAGO MIDWAY INTL ARPT Lat: 41.786 Lon: -87.752 Elev: 612 ft. above sea level
def add_dist_from_weather_stations(df):
    stat1_lat_lng = (41.995, -87.933)
    stat2_lat_lng = (41.786, -87.752)
    df['stat1_dist'] = df.apply(lambda row: get_dist((row.Latitude, row.Longitude), stat1_lat_lng), axis=1)
    df['stat2_dist'] = df.apply(lambda row: get_dist((row.Latitude, row.Longitude), stat2_lat_lng), axis=1)

def plot_weather_features(df, features):
    f, ax = plt.subplots(len(features), 1, figsize=(18,4*len(features)))
    for i, feat in enumerate(features):
        ax[i].plot(df['Date'], df[feat], marker='o', linestyle='')
        ax[i].set_title(feat)
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
    
def plot_detect_by_year(all_train):
    num_by_year = all_train[['Date', 'Trap', 'NumMosquitos']].groupby(
        by = ['Date','Trap'])['Date','Trap','NumMosquitos'].sum().reset_index().sort_values(
        'NumMosquitos', ascending = False).set_index('Date')

    wnv_by_year = all_train[['Date', 'Trap', 'WnvPresent']].groupby(
        by = ['Date','Trap'])['Date','Trap','WnvPresent'].sum().reset_index().sort_values(
        'WnvPresent', ascending = False).set_index('Date')

    plt.clf()
    f, ax = plt.subplots(2, 1, figsize=(18,8))
    num_by_year.plot(style='.', c='m', grid=True, ax=ax[0])
    ax[0].set_title('# Mosquitos by Trap by Year', fontsize=24)
    ax[0].set_xlabel('Year', fontsize=18)
    ax[0].set_ylabel('Mosquitos at Traps', fontsize=18)

    wnv_by_year.plot(style='.', c='r', grid=True, ax=ax[1], ylim=[0,20])
    ax[1].set_title('# Detections by Trap by Year', fontsize=24)
    ax[1].set_xlabel('Year', fontsize=18)
    ax[1].set_ylabel('Detections at Traps', fontsize=18)
    plt.tight_layout()
    plt.show()
    
    
def plot_on_map(all_train, test, spray):
    traps = all_train[['Date', 'Trap','Longitude', 'Latitude', 'WnvPresent', 'Species']]
    locations = traps[['Longitude', 'Latitude', 'Species']].drop_duplicates().values
    wnv = traps[traps['WnvPresent'] == 1][['Longitude', 'Latitude', 'Species']].drop_duplicates()

    spray_locations = spray[spray['Latitude'] <42.1][['Longitude', 'Latitude']].drop_duplicates().values
    test_locations = test[['Longitude', 'Latitude']].drop_duplicates().values
    wnv_species = all_train[all_train['WnvPresent'] == 1].Species.unique()
    
    legend = [[colors[i], wnv_species[i]] for i in range(len(wnv_species))] +\
    [['w', 'spray'], ['c', 'test'], ['b', 'traps']]
    load_map(mapdata, legend=legend)
    plt.scatter(spray_locations[:,0], spray_locations[:,1], marker='o', c=colors[-1], s=2)
    plt.scatter(test_locations[:,0], test_locations[:,1], marker='o', c=colors[3])
    plt.scatter(locations[:,0], locations[:,1], marker='x', c=colors[4])

    tot_wnv_locations = []
    for i in range(len(wnv_species)):
        spe_wnv = wnv[wnv['Species'] == wnv_species[i]].values
        tot_wnv_locations.append(len(spe_wnv))
        print(len(spe_wnv), wnv_species[i])
        plt.scatter(spe_wnv[:,0], spe_wnv[:,1], marker='o', c=colors[i])
    plt.show()