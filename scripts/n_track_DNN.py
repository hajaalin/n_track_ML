import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import linregress
from matplotlib import pyplot as plt

from keras import models
from keras import layers
from keras import regularizers
from keras.layers import Dropout

# from sklearn.model_selection import GroupKFold
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import GridSearchCV
# from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold

''' 
read the data 
'''

data = pd.read_csv('data/a286935_data_chromatin_live.csv')
data = data[~data["comment"].isin(["stress_control"])]
data = data[~data["comment"].isin(["H2B"])]
data = data[data["guide"].str.contains('1398') | data["guide"].str.contains('1514')]
data = data[data["time"] < 40]

# initial filtering based on experimental setup

''' 
add features 

In a resulting table target column names start with a 't', while features to be used in training start with 'f'.
'''

data_agg = data.groupby(['file', 'particle']).agg(t_guide=('guide', 'first'),
                                                  t_time=('time', 'first'),
                                                  t_serum_conc_percent=('serum_conc_percent', 'first'),

                                                  f_mean_diff_xy_micron=('diff_xy_micron', 'mean'),
                                                  # average displacement
                                                  f_max_diff_xy_micron=('diff_xy_micron', 'max'),
                                                  # maximal displacement
                                                  f_sum_diff_xy_micron=('diff_xy_micron', 'sum'),
                                                  # total trajectory length
                                                  f_var_diff_xy_micron=('diff_xy_micron', 'var'),
                                                  # variance in displacements

                                                  sum_diff_x_micron=('diff_x_micron', 'sum'),
                                                  sum_diff_y_micron=('diff_y_micron', 'sum'),

                                                  f_area_micron=('area_micron', 'mean'),
                                                  f_perimeter_au_norm=('perimeter_au_norm', 'mean'),
                                                  # morphology

                                                  f_min_dist_micron=('min_dist_micron', 'mean'),
                                                  # minimal distance to edge averaged for each timelapse
                                                  min_min_dist_micron=('min_dist_micron', 'min'),
                                                  max_min_dist_micron=('min_dist_micron', 'max'),
                                                  beg_min_dist_micron=('min_dist_micron', 'first'),
                                                  end_min_dist_micron=('min_dist_micron', 'last'),
                                                  f_var_dist_micron=('min_dist_micron', 'var'),
                                                  )

data_agg['f_Rvar_diff_xy_micron'] = data_agg['f_var_diff_xy_micron'] / data_agg['f_mean_diff_xy_micron']
data_agg['f_Rvar_dist_micron'] = data_agg['f_var_dist_micron'] / data_agg['f_min_dist_micron']
# Relative variance

data_agg['f_total_displacement'] = np.sqrt((data_agg['sum_diff_x_micron']) ** 2 + (data_agg['sum_diff_y_micron']) ** 2)
# distance from first to last coordinate
data_agg['f_persistence'] = data_agg['f_total_displacement'] / data_agg['f_sum_diff_xy_micron']
# shows how directional the movement is

data_agg['file_mean_diff_xy_micron'] = data_agg.groupby('file')['f_mean_diff_xy_micron'].transform(np.max)
data_agg['f_fastest_mask'] = np.where((data_agg['f_mean_diff_xy_micron'] == data_agg['file_mean_diff_xy_micron']), 1, 0)
# DO NOT USE FOR guide AS TARGET (telo!)
# the fastest (or the only available) dot in the nucleus is 1, the rest is 0

data_agg['f_min_dist_range'] = data_agg['max_min_dist_micron'] - data_agg['min_min_dist_micron']
# min_dist change within timelapse (max-min) for each dot
data_agg['f_total_min_dist'] = data_agg['end_min_dist_micron'] - data_agg['beg_min_dist_micron']
# how distance changed within timelapse (frame29-frame0)

data_agg['file_max_min_dist_micron'] = data_agg.groupby('file')['f_min_dist_micron'].transform(np.max)
data_agg['f_most_central_mask'] = np.where((data_agg['f_min_dist_micron'] == data_agg['file_max_min_dist_micron']), 1,
                                           0)
# DO NOT USE FOR guide AS TARGET (telo!)
# the most central (or the only available) dot in the nucleus is 1, the rest is 0

data_slope = data.groupby(['file', 'particle']).apply(lambda x: linregress(x['frame'], x['min_dist_micron'])[0])
data_agg['f_slope_min_dist_micron'] = data_slope
# slope for minimal distance to edge; how distance to edge changes within the timelapse?


data_slope_area = data.groupby(['file', 'particle']).apply(lambda x: linregress(x['frame'], x['area_micron'])[0])
data_agg['f_slope_area_micron'] = data_slope_area
# slope for nucleus area; how area changes within the timelapse?

data_slope_perimeter = data.groupby(['file', 'particle']).apply(lambda x: linregress(x['frame'],
                                                                                     x['perimeter_au_norm'])[0])
data_agg['f_slope_perimeter_au_norm'] = data_slope_perimeter
# slope for nucleus perimeter

data_SD_diff_xy_micron = data.groupby(['file', 'particle']).agg(SD_diff=('diff_xy_micron', 'std'))
data_i = data.set_index(['file', 'particle'])
data_i['SD_diff_xy_micron'] = data_SD_diff_xy_micron
data_i['f_mean_diff_xy_micron'] = data_agg['f_mean_diff_xy_micron']
data_i['outliers2SD_diff_xy'] = np.where((data_i['diff_xy_micron'] >
                                          (data_i['f_mean_diff_xy_micron'] + 2 * data_i['SD_diff_xy_micron'])), 1, 0)
data_i['outliers3SD_diff_xy'] = np.where((data_i['diff_xy_micron'] >
                                          (data_i['f_mean_diff_xy_micron'] + 3 * data_i['SD_diff_xy_micron'])), 1, 0)
data_agg['f_outliers2SD_diff_xy'] = data_i.groupby(['file', 'particle']) \
    .agg(f_outliers2SD_diff_xy=('outliers2SD_diff_xy', 'sum'))
data_agg['f_outliers3SD_diff_xy'] = data_i.groupby(['file', 'particle']) \
    .agg(f_outliers3SD_diff_xy=('outliers3SD_diff_xy', 'sum'))
# is there a displacement larger than mean plus 2SD or 3SD (SD calculated for each dot, 29xy pairs) respectively

data_sterile = data_agg.drop(['sum_diff_x_micron',
                              'sum_diff_y_micron',
                              'min_min_dist_micron',
                              'max_min_dist_micron',
                              'beg_min_dist_micron',
                              'end_min_dist_micron',
                              'file_mean_diff_xy_micron',
                              'file_max_min_dist_micron',
                              'f_sum_diff_xy_micron',  # proportional to f_mean_diff_xy_micron, thus, useless
                              ], axis=1)
data_sterile.reset_index(inplace=True)
corr_features = data_sterile.corr()

''' 
Train / test split
'''

features = [
    'f_mean_diff_xy_micron', 'f_max_diff_xy_micron', 'f_var_diff_xy_micron',
    'f_area_micron', 'f_perimeter_au_norm', 'f_min_dist_micron',
    'f_var_dist_micron', 'f_Rvar_diff_xy_micron', 'f_Rvar_dist_micron',
    'f_total_displacement', 'f_persistence', 'f_fastest_mask',
    'f_min_dist_range', 'f_total_min_dist', 'f_most_central_mask',
    'f_slope_min_dist_micron', 'f_slope_area_micron',
    'f_slope_perimeter_au_norm', 'f_outliers2SD_diff_xy',
    'f_outliers3SD_diff_xy'
]

tst = int((data_sterile['file'].unique().shape[0]) / 4)
# nuclei number to choose for testing

# test_choice = np.random.RandomState(7).choice(data_sterile['file'].unique(), tst, replace=False)

results = pd.DataFrame()
ix = 0 # index for columns in results
for i in range(20):

    #X = data_sterile[features]
    labels = ((data_sterile['t_serum_conc_percent']) / 10).astype('int')
    groups = data_sterile['file']
    gkf = StratifiedGroupKFold(n_splits=4, shuffle=True)
    for train_idxs, test_idxs in gkf.split(data_sterile[features], labels, groups):


        X = data_sterile[features].loc[train_idxs]
        X_test = data_sterile[features].loc[test_idxs]
        y = labels.loc[train_idxs]
        y_test = labels.loc[test_idxs]


        scaler = StandardScaler()
        scaler.fit(X)
        X_norm = scaler.transform(X)
        X_test_norm = scaler.transform(X_test)

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

        history = model.fit(X_norm,
                            y,
                            epochs=500,
                            # batch_size=50,
                            validation_data=(X_test_norm, y_test),
                            verbose=0,
                            )

        results["acc" + str(ix)] = history.history['accuracy']
        results["val_acc" + str(ix)] = history.history['val_accuracy']
        ix += 1

        # print(1-y.sum()/len(y))
        # print(1 - y_test.sum() / len(y_test))
        print(ix)

dnn_results = results.iloc[:, range(1, len(results.columns), 2)]
dnn_results['mean'] = dnn_results.mean(axis=1)
plt.plot(dnn_results['mean'])
plt.show()

# this moved to n_track_DNN_nff:
# todo normalization
# todo cv as in sgkf but no pipeline
# todo repeat 63+acc
# todo repeat Harris shap (monitor acc?)
# todo make SHAP aggregated from repeats (monitor acc?)