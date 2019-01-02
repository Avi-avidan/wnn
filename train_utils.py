from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score, classification_report
from sklearn import tree
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from xgboost import plot_importance
from xgboost.sklearn import XGBClassifier
import keras
import keras.layers as KL
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

def train_xgb(x_train, y_train, lr=0.001, max_depth=4, n_estimators=50):
    xgb1 = XGBClassifier(learning_rate=lr, silent=1, max_depth=max_depth, n_estimators=n_estimators)
    xgb1.fit(x_train, y_train)
    return xgb1

def plot_xgb_importance(xgb, figsize=(10, 15)):
    fig, ax = plt.subplots(figsize=figsize)
    plot_importance(xgb, ax=ax)
    plt.show()

def randomized_search_xgb(x_train, x_val, y_train, y_val, xgb_params):
    all_train_data = pd.concat([x_train, x_val])
    all_y = y_train.tolist()
    all_y.extend(y_val)
    
    xgb = XGBClassifier(learning_rate=0.02, n_estimators=400, objective='binary:logistic',
                        silent=True, nthread=1)
    skf = StratifiedKFold(n_splits=3, shuffle = True, random_state = 1001)

    xgb_rsearch = RandomizedSearchCV(
        xgb, param_distributions=xgb_params, n_iter=5, scoring='roc_auc', n_jobs=-1, 
        cv=skf.split(all_train_data, all_y), verbose=1)
    xgb_rsearch.fit(all_train_data, all_y)
    
    print(xgb_rsearch.best_params_)
    
    xgb_rs = xgb_rsearch.best_estimator_.fit(x_train, y_train)
    return xgb_rs


def randomized_search_rf(x_train, x_val, y_train, y_val, rs_params, n_iter_search=40, cv=3, scoring='roc_auc'):
    all_train_data = pd.concat([x_train, x_val])
    all_y = y_train.tolist()
    all_y.extend(y_val)

    rf_clf_rs = RandomForestClassifier(n_estimators=20)
    random_search = RandomizedSearchCV(rf_clf_rs, param_distributions=rs_params, scoring=scoring,
                                       n_iter=n_iter_search, cv=cv)
    start = time.time()
    random_search.fit(all_train_data, all_y)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time.time() - start), n_iter_search))
    report(random_search.cv_results_)
    rf_clf_rs = random_search.best_estimator_.fit(all_train_data, all_y)
    return rf_clf_rs

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))

def get_model_checkp(root_dir, name):
    return keras.callbacks.ModelCheckpoint(root_dir + name + '_{val_binary_accuracy:.4f}.h5',
                save_best_only=True,
                save_weights_only=False,
                period=1, verbose=1)

def fc(depths, use_bn=False):
    inp = KL.Input(shape=(depths[0],), name='input')
    x = KL.Dense(depths[1], name='d1', kernel_regularizer=keras.regularizers.l2(0.001))(inp)
    x = KL.advanced_activations.LeakyReLU(alpha=0.1, name='lr1')(x)
    if use_bn:
        x = KL.normalization.BatchNormalization(momentum=0.39)(x)
    x = KL.Dense(depths[2], name='d2', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = KL.advanced_activations.LeakyReLU(alpha=0.1, name='lr2')(x)
    if use_bn:
        x = KL.normalization.BatchNormalization(momentum=0.39)(x)
    x = KL.Dense(depths[3], name='d3', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = KL.advanced_activations.LeakyReLU(alpha=0.1, name='lr3')(x)
    if use_bn:
        x = KL.normalization.BatchNormalization(momentum=0.39)(x)
    x = KL.Dropout(0.4, name='do1')(x)
    x = KL.Dense(depths[4], name='d4', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = KL.advanced_activations.LeakyReLU(alpha=0.1, name='lr4')(x)
    x = KL.Dropout(0.4, name='do2')(x)
    x = KL.Dense(depths[5], name='d5', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = KL.advanced_activations.LeakyReLU(alpha=0.1, name='lr5')(x)
    x = KL.Dropout(0.2, name='do3')(x)
    x = KL.Dense(depths[-1], name='output', activation='sigmoid')(x)
    return keras.models.Model(inp, x), keras.optimizers.Adam(lr=1e-4)

def simple_fc(depths):
    inp = KL.Input(shape=(depths[0],), name='input')
    x = KL.Dense(depths[1], name='fc1', activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(inp)
    x = KL.Dense(depths[2], name='fc2', activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = KL.Dense(depths[-1], name='output', activation='sigmoid')(x)
    return keras.models.Model(inputs=[inp], outputs=[x])

def plot_loss(history, keys=['loss', 'val_loss'], metric='binary_accuracy'):
    for key in keys:
        plt.plot(history.history[key])
        plt.title(metric + ' chart')
    plt.ylabel(metric + ' ' + keys[0])
    plt.xlabel('epoch')
    plt.legend(['train set', 'validation set'], loc='lower left')
    plt.show()
    
def eval_fc(model, x_train_scaled, y_train_df):
    pred_val = model.predict(x_val)
    #pred_val = refine_predict(pred_val, x_val)
    fpr, tpr, _ = roc_curve(y_val, pred_val)
    roc_auc = auc(fpr, tpr)
    print('val mean auc: {:.5f}'.format(roc_auc))
    return roc_auc

def plot_fc_results(x_train, y_train, x_val, y_val, load_model_path, name='fc1.0'):
    model = keras.models.load_model(load_model_path)
    keras.backend.set_learning_phase(0)
    model.compile(optimizer='Adam')
    name = name + ' ' + os.path.basename(load_model_path)
    plot_nn_roc(x_train, y_train, x_val, y_val, model, name)
    return model


def fit_dt(x_train, y_train, x_val, y_val, min_imp_rng, clf_type='dt'):
    best_score = 0
    for val in np.linspace(min_imp_rng[0], min_imp_rng[1], min_imp_rng[2]):
        if clf_type == 'dt':
            clf = tree.DecisionTreeClassifier(min_impurity_decrease=val)
        elif clf_type == 'rf':
            clf = RandomForestClassifier(min_impurity_decrease=val)
        elif clf_type == 'et':
            clf = ExtraTreesClassifier(min_impurity_decrease=val)
            
        clf.fit(x_train, y_train)
        pred_val = clf.predict(x_val)
        # pred_val = refine_predict(pred_val, x_val)
        #f1 = f1_score(y_val, pred_val, average='binary')
        fpr, tpr, _ = roc_curve(y_val, pred_val)
        roc_auc = auc(fpr, tpr)
    
        if roc_auc > best_score:
            best_score = roc_auc
            best_model = clf
            best_pred_val = pred_val
    print('best auc: {:.4f}, val_score: {:.4f}, min_impurity_decrease:{}'.format(
        best_score, best_model.score(x_val, y_val), val))
    return best_model, best_pred_val

def get_fpr_tpr_auc(label, pred):
    fpr, tpr, _ = roc_curve(label, pred)
    return fpr, tpr, auc(fpr, tpr)


def add_roc_to_plot(clf, samp_list, label, name, refine=False):
    pred_val = clf.predict(samp_list) 
    color = 'red' if (name == 'val') else 'blue'
    fpr, tpr, roc_auc = get_fpr_tpr_auc(label, pred_val)
    label = name + ' ROC curve (area = %0.4f)' % roc_auc
    plt.plot(fpr, tpr, label=label, color=color)
    return pred_val

def init_roc_plot():
    plt.figure()
    plt.plot([0, 1], [0,1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    
def plot_roc(x_train, y_train, x_val, y_val, clf, name):
    init_roc_plot()
    pred_val = add_roc_to_plot(clf, x_val, y_val, 'val')
    _ = add_roc_to_plot(clf, x_train, y_train, 'train')    
    plt.title('ROC curve ' + name)
    plt.legend(loc="lower right")
    plt.show()
    print(classification_report(y_val, pred_val))
    
    
def plot_nn_roc(x_train, y_train, x_val, y_val, model, name):
    init_roc_plot()
    pred_val = get_fc_model_predict(model, x_val)
    fpr, tpr, roc_auc = get_fpr_tpr_auc(y_val, pred_val)
    label = 'val ROC curve (area = %0.4f)' % roc_auc
    plt.plot(fpr, tpr, label=label, color='red')
    
    pred_train = get_fc_model_predict(model, x_train)
    fpr, tpr, roc_auc = get_fpr_tpr_auc(y_train, pred_train)
    label = 'train ROC curve (area = %0.4f)' % roc_auc
    plt.plot(fpr, tpr, label=label, color='blue') 
    plt.title('ROC curve ' + name)
    plt.legend(loc="lower right")
    plt.show()
    
def refine_predict(test_pred, test_df):
    test_pred[test_df.loc[(test_df['CULEX PIPIENS/RESTUANS'] == 0) &\
                (test_df['CULEX PIPIENS'] == 0) & test_df['CULEX RESTUANS'] == 0]] = 0
    return test_pred

def get_fc_model_predict(model, x_test):
    return np.squeeze(model.predict(x_test))

def get_en_pred_df(x_test, models, names):
    en_df = pd.DataFrame()
    for i, mod in enumerate(models):
        en_df[names[i]] = np.squeeze(mod.predict(x_test))
    en_df['en_avg'] = en_df.apply(lambda row: np.mean([row[name] for name in names]), axis=1)
    return en_df

def save_submission(name, test_pred, x_test, refine=False, data_root='/Users/aviavidan/data/kaggle/wnv'):
    submission = pd.read_csv(os.path.join(data_root, 'test.csv'), usecols=[0])
    test_pred = np.squeeze(test_pred)
    if refine:
        test_pred = refine_predict(test_pred, x_test)
    submission['WnvPresent'] = test_pred
    submission.to_csv(name+'.csv', index=False, columns=submission.columns.tolist())