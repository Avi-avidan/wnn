import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import random
from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

cont_feat = ['Latitude', 'Longitude', 'avg_tmp', 'dl_hrs', 'avg_wind', 'PrecipTotal_3wks', 'PrecipTotal', 
             'AvgSpeed', 'Tavg', 'Tmin', 'Tmax', 'Cool', 'WetBulb']

disc_feat = ['Trap', 'AddressAccuracy', 'day_count', 'day_of_week', 'week', 'month', 'CULEX PIPIENS/RESTUANS',
 'CULEX PIPIENS', 'CULEX RESTUANS', 'Duplicats', 'ohare_3', 'ohare_5', 'mckin_3', 'mckin_5', 'aubur_3', 'aubur_5']

augment_cols_disc = ['Trap', 'AddressAccuracy']
augment_cols_cont = ['Tavg', 'Tmin', 'Tmax', 'Cool', 'AvgSpeed', 'PrecipTotal', 'dl_2wks_ago', 'dl_4wks_ago', 
                     'avg_wind', 'PrecipTotal_3wks', 'WetBulb', 'Latitude', 'Longitude']

def scale_features(train_df, test_df):
    all_data = pd.concat([train_df, test_df])
    scaler = StandardScaler()
    scaler.fit(all_data)
    x_train_scaled = scaler.transform(train_df)
    x_test_scaled = scaler.transform(test_df)
    x_test = pd.DataFrame(x_test_scaled, columns=train_df.columns.tolist())
    x_train = pd.DataFrame(x_train_scaled, columns=train_df.columns.tolist())
    return x_train, x_test

def augment_col(x, col, augment_dict, prec_aug):
    if random.uniform(0,1) < prec_aug:
        if col in augment_cols_cont:
            x = random.uniform(augment_dict[col]['min'], augment_dict[col]['max']) 
        elif col in augment_cols_disc:
            x = augment_dict[col]['vals'][random.randint(0, len(augment_dict[col]['vals'])-1)]
    return x

def augment_df(df, augment_dict, inplace=False, columns=[], prec_aug=0.2):
    target_df = df if inplace else pd.DataFrame()
    if len(columns) < 1:
        columns = df.columns.tolist()
    for i, col in enumerate(columns):
        # print('augmenting ', col)
        target_df[col] = df[col].apply(lambda x: augment_col(x, col, augment_dict, prec_aug))
    return target_df

def create_augment_dict(df):
    augment_dict = {}
    for i, col in enumerate(df.columns.tolist()):
        if col in augment_cols_cont:
            augment_dict[col] = {'min': df[col].min(), 'max': df[col].max()}
        elif col in augment_cols_disc:
            augment_dict[col] = {'vals': np.unique(df[col])}
    return augment_dict

def resample_augment_split(x_train_scaled, y_train_df, prec_aug, columns, state=0, pprint=True):
    if state == 0:
        state = random.randint(0, 1234)
        print('random_state:', state)
    
    
    x_train_scaled['WnvPresent'] = y_train_df.values
    train_majority = x_train_scaled[y_train_df['WnvPresent'] == 0]
    train_minority = x_train_scaled[y_train_df['WnvPresent'] == 1]
    
    if pprint:
        print('before augmentation train dist - \n{:.2f}% ({}) positive samples, {:.2f}% ({}) negative samples'.format(
            train_minority.shape[0]/x_train.shape[0]*100, train_minority.shape[0],
            train_majority.shape[0]/x_train.shape[0]*100, train_majority.shape[0]))
        
        print('val set class dist - \n{:.2f}%({}) positive samples, {:.2f}%({}) negative samples,'.format(
            np.sum([y_val['WnvPresent'] == 1])/x_val.shape[0]*100, np.sum([y_val['WnvPresent'] == 1]),
            np.sum([y_val['WnvPresent'] == 0])/x_val.shape[0]*100, np.sum([y_val['WnvPresent'] == 0])))

    #upsample minority class
    train_minority_upsampled = resample(train_minority, 
                                         replace = True, 
                                         n_samples = train_majority.shape[0],
                                         random_state = state)
    
    augment_dict = create_augment_dict(train_minority)
    augmented_minority = augment_df(train_minority_upsampled, augment_dict, prec_aug=prec_aug)
    
    train_data_upsampled = pd.concat([train_majority, augmented_minority])
    #split back into X_train and y_train
    train_data_upsampled = shuffle(train_data_upsampled, random_state=state)
    y_train = train_data_upsampled['WnvPresent']
    x_train = train_data_upsampled.drop(columns='WnvPresent')
    
    if pprint:
        for i, col in enumerate(columns):
            prec_aug = np.sum(
                augmented_minority[col] == train_minority_upsampled[col])/train_minority_upsampled.shape[0]
            print('{}, {:.2f}% augmented'.format(col, 100*(1 - prec_aug)))
            
    print('after augment train dist - \n{:.2f}%({}) positive samples, {:.2f}%({}) negative samples '.format(
        np.sum([y_train == 1])/x_train.shape[0]*100, np.sum([y_train == 1]),
        np.sum([y_train == 0])/x_train.shape[0]*100, np.sum([y_train == 0])))
    
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.33, random_state=state)
    x_val = pd.DataFrame(x_val, columns=columns)
    x_train = pd.DataFrame(x_train, columns=columns)
    return x_train, x_val, y_train, y_val, state



def train_val_data_split(x_train_scaled, y_train_df, prec_aug, columns, state=0, pprint=True):
    if state == 0:
        state = random.randint(0, 1234)
        print('random_state:', state)
    x_train, x_val, y_train, y_val = train_test_split(x_train_scaled, y_train_df, test_size=0.33, random_state=state)
    traindata = pd.DataFrame(x_train, columns=columns)
    traindata['WnvPresent'] = y_train.values

    x_val = pd.DataFrame(x_val, columns=columns)
    
    train_majority = traindata[traindata['WnvPresent'] == 0]
    train_minority = traindata[traindata['WnvPresent'] == 1]
    
    if pprint:
        print('before augmentation train dist - \n{:.2f}% ({}) positive samples, {:.2f}% ({}) negative samples'.format(
            train_minority.shape[0]/x_train.shape[0]*100, train_minority.shape[0],
            train_majority.shape[0]/x_train.shape[0]*100, train_majority.shape[0]))
        
        print('val set class dist - \n{:.2f}%({}) positive samples, {:.2f}%({}) negative samples,'.format(
            np.sum([y_val['WnvPresent'] == 1])/x_val.shape[0]*100, np.sum([y_val['WnvPresent'] == 1]),
            np.sum([y_val['WnvPresent'] == 0])/x_val.shape[0]*100, np.sum([y_val['WnvPresent'] == 0])))

    #upsample minority class
    train_minority_upsampled = resample(train_minority, 
                                         replace = True, 
                                         n_samples = train_majority.shape[0],
                                         random_state = state)
    
    augment_dict = create_augment_dict(train_minority)
    augmented_minority = augment_df(train_minority_upsampled, augment_dict, prec_aug=prec_aug)
    
    train_data_upsampled = pd.concat([train_majority, augmented_minority])
    #split back into X_train and y_train
    train_data_upsampled = shuffle(train_data_upsampled, random_state=state)
    y_train = train_data_upsampled['WnvPresent']
    y_val = y_val['WnvPresent']
    x_train = train_data_upsampled.drop(columns='WnvPresent')
    
    if pprint:
        for i, col in enumerate(columns):
            prec_aug = np.sum(
                augmented_minority[col] == train_minority_upsampled[col])/train_minority_upsampled.shape[0]
            print('{}, {:.2f}% augmented'.format(col, 100*(1 - prec_aug)))
            
    print('after augment train dist - \n{:.2f}%({}) positive samples, {:.2f}%({}) negative samples '.format(
        np.sum([y_train == 1])/x_train.shape[0]*100, np.sum([y_train == 1]),
        np.sum([y_train == 0])/x_train.shape[0]*100, np.sum([y_train == 0])))
    
    return x_train, x_val, y_train.values, y_val.values, state


def print_classes_ratio(df):
    classes = df.WnvPresent.unique()
    print('number of classes:{}'.format(len(classes)))
    amount_neg = np.sum([df['WnvPresent'] == 1])
    total = df.shape[0]
    print('classes ratios: {:.2f}% ({}) negative samps, {:.2f}% ({}) positive samps'.format(
        amount_neg/total*100, amount_neg, (total-amount_neg)/total*100, total-amount_neg))


def validate_uniques_vals(test, train, cols=['Species', 'Block', 'Street', 'Trap', 'AddressAccuracy']):
    oos_dict = {}
    mask_out_df = pd.DataFrame()
    for col in cols:
        missing = [val for val in test[col].unique() if val not in train[col].unique()]
        print('\nfeature:{}, # uniques in testset:{}, # uniques in trainset:{}'. format(
            col, len(test[col].unique()), len(train[col].unique())))
        print('feature:{}, # of missing uniques in train:{}'.format(col, len(missing)))
        if len(missing) > 0:
            print(missing)
            oos_dict[col] = missing
            mask_out_df[col] = np.int32(test[col].isin(missing))
    return oos_dict, mask_out_df
    
    

def validate_lat_lng(test, all_train, cols=['Longitude', 'Latitude']):    
    oos_list = []
    for col in cols:
        train_col_min = all_train[col].min()
        train_col_max = all_train[col].max()
        test_col_min = test[col].min()
        test_col_max = test[col].max()

        print('\nfeature:{}, train_min_val:{}, train_max_val:{}'.format(col, train_col_min, train_col_max))
        print('feature:{}, test_min_val:{}, test_max_val:{}'.format(col, test_col_min, test_col_max))

        if train_col_min > test_col_min:
            print('{} out of scope vals in test for feature:{}'.format(np.sum([test[col]<train_col_min]), col))
            oos_list.append(col)
        if train_col_max < test_col_max:
            print('{} out of scope vals in test for feature:{}'.format(np.sum([test[col]>train_col_max]), col))
            if col not in oos_list:
                oos_list.append(col)
    return oos_list


def add_dist_compare(ax, df1, df2, names, prec=True):
    bins = min(max(len(df1.unique()), len(df1.unique())), 150)
    rwidth=0.25 if bins == 2 else None
    weights1 = np.ones(len(df1))/len(df1) if prec else None
    weights2 = np.ones(len(df2))/len(df2) if prec else None
    ax.hist(df1, color='r', bins=bins, weights=weights1, label=names[0], alpha=0.5, rwidth=rwidth)
    ax.hist(df2, color='b', bins=bins, weights=weights2, label=names[1], alpha=0.5, rwidth=rwidth)
    ax.legend(loc='best')
    ax.set_title(names[2])

def plot_dist_compare(cols_list, test_df, train_df, names, prec=True):
    in_row = 4
    rows = len(cols_list)//4 + 1
    f, ax = plt.subplots(rows, in_row, figsize=(18, 3.5*rows))
    for i, col in enumerate(cols_list):
        add_dist_compare(ax[i//in_row, i%in_row], test_df[col], train_df[col], names+[col], prec=prec)
    plt.show()
    
    
def plot_pca(df):
    pca = PCA()
    pca.fit(df)
    cumsum = np.cumsum(pca.explained_variance_ratio_)*100
    d = [n for n in range(len(cumsum))]
    f, ax = plt.subplots(figsize=(9, 7))
    plt.plot(d,cumsum, color = 'red',label='cumulative explained variance', linewidth=2.5)
    plt.title('Cumulative Explained Variance as a Function of the Number of Components')
    plt.ylabel('Cumulative Explained variance')
    plt.xlabel('Principal components')
    plt.axhline(y = 95, color='k', linestyle='--', label = '95% Explained Variance')
    y = np.cumsum(pca.explained_variance_ratio_)*100
    x = np.linspace(0,len(y), len(y))
    plt.bar(x, y, alpha=0.5)
    
    plt.legend(loc='best')
    return pca