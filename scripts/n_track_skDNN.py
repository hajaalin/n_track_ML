"""
This is to try keras models in sklearn environment (i.e. to use GroupKFold or grid search)
as here
https://machinelearningmastery.com/use-keras-deep-learning-models-scikit-learn-python/
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.wrappers.scikit_learn import KerasClassifier
from keras import models
from keras import layers
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from datetime import datetime

import shap

''' 
read the data 

'''
data = pd.read_csv('data/a286935_data_chromatin_live.csv')
data = data[~data["comment"].isin(["stress_control"])]
data = data[~data["comment"].isin(["H2B"])]
data = data[data["guide"].str.contains('1398') | data["guide"].str.contains('1514')]
data = data[data["time"] < 40]
data.set_index(['file', 'particle'], inplace=True)
# original data before aggregation

fr_in_ind = data.index.value_counts()
dots_to_drop = fr_in_ind[fr_in_ind < 30]
data.drop(dots_to_drop.index, inplace=True)
# check for dots with less than 30 frames

data.set_index('frame', append=True, inplace=True)
data = data[['diff_xy_micron', 'area_micron', 'perimeter_au_norm', 'min_dist_micron']]
# filter features

data = data.unstack()  # reshape
data.drop(('diff_xy_micron', 0), axis=1, inplace=True)  # drop first delta 'diff_xy_micron', which is NaN
data.columns = data.columns.to_flat_index()  # needed for concat

data_sterile = pd.read_csv('data/data_sterile_PCA_92ba95d.csv')
data_sterile.set_index(['file', 'particle'], inplace=True)
features = data_sterile.columns[4:]
data_sterile = data_sterile[features]
# data after aggregation and feature engineering

data_sterile.drop(dots_to_drop.index, inplace=True)
# drop those with less than 30 frames

data_raw = data_sterile.join(data)
# concatenate


''' 
data preprocessing 
'''

# X = data_raw[data_raw.columns[1:]]
X = data_raw[data_raw.columns[1:21]]  # these are engineered features only
# X = data_raw[data_raw.columns[37:]] #  raw features only
X_norm = X / X.max(axis=0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

y = ((data_raw['t_serum_conc_percent']) / 10).astype('int')

''' 
building keras model

technically it is possible to gridsearch through hidden layers number as hyperparameter
https://stackoverflow.com/questions/47788799/grid-search-the-number-of-hidden-layers-with-keras

'''


def create_model():
    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


''' 
use model with sklearn cross-val
'''
for k in range(5, 600, 5):  # takes an eternity, needed to check for accuracy 'degradation' due to overfitting
    model = KerasClassifier(build_fn=create_model, epochs=k, verbose=0)
    gkf = StratifiedGroupKFold(n_splits=4, shuffle=True)
    results = pd.DataFrame(columns=['spl1', 'spl2', 'spl3', 'spl4'])
    for i in range(10):  # repeated CV, since one iteration gives too unstable results
        scores = cross_val_score(model, X_norm, y, cv=gkf, groups=data_raw.reset_index()['file'])
        results = results.append(pd.Series(scores, index=results.columns), ignore_index=True)

    print(str(k) + ' epochs: ' + str(results.mean().mean()))

''' 
(one time version) use model with sklearn cross-val 
'''

model = KerasClassifier(build_fn=create_model, epochs=100, verbose=0)
gkf = StratifiedGroupKFold(n_splits=4, shuffle=True)
results = pd.DataFrame(columns=['spl1', 'spl2', 'spl3', 'spl4'])
for i in range(10):  # repeated CV, since one iteration gives too unstable results
    scores = cross_val_score(model, X_norm, y, cv=gkf, groups=data_raw.reset_index()['file'])
    results = results.append(pd.Series(scores, index=results.columns), ignore_index=True)

print(results.mean().mean())

'''
pipeline version, to avoid data leakage from scaling before splitting 
'''


def train_dnn(iters):
    gkf = StratifiedGroupKFold(n_splits=4, shuffle=True)
    model_pipe = KerasClassifier(build_fn=create_model, epochs=100, verbose=0)
    DNN_pipeline = Pipeline(
        steps=[('scaler', StandardScaler()),
               ('DNN', model_pipe)]
    )
    results = pd.DataFrame(columns=['spl1', 'spl2', 'spl3', 'spl4'])
    for i in range(iters):  # repeated CV, since one iteration gives too unstable results
        print('iter ' + str(i) + ' start, time:', datetime.now().strftime("%H:%M:%S"))
        scores = cross_val_score(DNN_pipeline, Xsp, y, cv=gkf, groups=data_raw.reset_index()['file'])
        results = results.append(pd.Series(scores, index=results.columns), ignore_index=True)

    print(results.mean().mean())
    return results.mean().mean()


'''
feature knockout to estimate feature importances for DNN
'''
knocks = {}
for feature in X.columns:
    Xsp = X.copy()
    Xsp[feature] = 0
    print(feature)
    knocks[feature] = train_dnn(30)

# subsets of features to ko redundant

speed = ['f_mean_diff_xy_micron',
         'f_max_diff_xy_micron',
         'f_var_diff_xy_micron',
         'f_Rvar_diff_xy_micron',
         ]

morphology = ['f_area_micron',
              'f_perimeter_au_norm',
              'f_slope_area_micron',
              'f_slope_perimeter_au_norm',
              ]

mindist = ['f_min_dist_micron',
           'f_var_dist_micron',
           'f_Rvar_dist_micron',
           'f_min_dist_range',
           'f_total_min_dist',
           'f_slope_min_dist_micron',
           ]

persistence = ['f_total_displacement',
               'f_persistence',
               'f_outliers2SD_diff_xy',
               'f_outliers3SD_diff_xy',
               ]

homologs = ['f_fastest_mask',
            'f_most_central_mask',
            ]

gr_list = [speed, morphology, mindist, persistence, homologs]

gr_knocks = {}
for gr in gr_list:
    Xsp = X.copy()
    Xsp[gr] = 0
    print(str(gr))
    gr_knocks[str(gr)] = train_dnn(30)

# todo: do same KO wiht learning curve n_track_DNN