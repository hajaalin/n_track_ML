import pandas as pd
import numpy as np
import shap
from scipy.stats import linregress
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedGroupKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, make_scorer, f1_score, precision_score, recall_score
import seaborn as sns
from matplotlib import pyplot as plt
from datetime import datetime
from random import randint

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
# cleaning up

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

X = data_sterile[features]
y = data_sterile['t_serum_conc_percent']  # .astype('str')
y = (y / 10).astype('int')  # '10% serum' = 1, '0.3% serum' = 0

''' 
Gradient boosting trees
'''

metric = 'accuracy'  # 'accuracy', 'precision', 'recall', 'f1' see suffixes
# metric = make_scorer(recall_score, pos_label=0)
pivots = []
grids = []
baselines = []
baselines_free = []

iterations = 1

for i in range(iterations):
    print('iter ' + str(i) + ' start, time:', datetime.now().strftime("%H:%M:%S"))
    rand_state = randint(1, 100)  # not sure that this is needed
    gkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=rand_state)

    b_param_grid = {'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1, 10],
                    'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30]}

    grid_b_forest = GradientBoostingClassifier(n_estimators=1000)
    b_grid_search = GridSearchCV(grid_b_forest, b_param_grid, cv=gkf,
                                 scoring=metric,
                                 refit=False)
    b_grid_search.fit(X, y, groups=data_sterile['file'])
    grids.append(b_grid_search.cv_results_)

    b_pvt = pd.pivot_table(pd.DataFrame(b_grid_search.cv_results_),
                           values='mean_test_score',
                           index='param_learning_rate',
                           columns='param_max_depth')

    pivots.append(b_pvt)

    # fast baseline with the same cv:
    tree = DecisionTreeClassifier(max_depth=1)
    baselines_free.append(np.mean(cross_val_score(tree, X, y, cv=gkf, groups=data_sterile['file'], scoring=metric)))
    # tree can't choose the feature, why?

    baselines.append(np.mean(cross_val_score(tree, X['f_slope_area_micron'].values.reshape(-1, 1),
                                             y, cv=gkf, groups=data_sterile['file'], scoring=metric)))

mpvts = pd.concat(pivots).mean(level=0)
pvt_sem = pd.concat(pivots).sem(level=0)  # standard error of mean, element-wise
sns.heatmap(mpvts, annot=True)
plt.title(str(metric) + ' , ' + str(iterations) + ' cv reps')
plt.show()
plt.close()

mpvts.plot(kind='bar',
           yerr=pvt_sem / 100,
           ylim=(0.4, 0.65),
           figsize=(14, 9),
           colormap='magma',
           width=1).legend(loc='best')

plt.title(str(metric) + ' , ' + str(iterations) + ' cv reps')
plt.show()
plt.close()

print(metric)
print('baseline (area slope): ' + str(np.mean(baselines)))
print('baseline (free feature choice): ' + str(np.mean(baselines_free)))

'''
check for acc with hyperparameters selected from grid
'''
fixed_hyper_acc = []
for i in range(10):
    gbc = GradientBoostingClassifier(n_estimators=1000, learning_rate=1, max_depth=7)
    # gkf = GroupKFold(n_splits=4)
    gkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=None)
    score = cross_val_score(gbc, X, y, groups=data_sterile['file'], cv=gkf)
    fixed_hyper_acc.append(np.mean(score))
print(np.mean(fixed_hyper_acc))

'''
SHAP
'''

X = data_sterile[features]
y = data_sterile['t_serum_conc_percent']  # .astype('str')
y = (y / 10).astype('int')  # '10% serum' = 1, '0.3% serum' = 0
gkf = StratifiedGroupKFold(n_splits=4, shuffle=True)
l_rate = 1
depth = 7

pred_list = []
pred_proba_list = []
shap_vs_list = []
sX_test_list = []
sy_test_list = []
s_id_list = []

for strain, stest in gkf.split(X, y, data_sterile['file']):
    train_data = data_sterile.iloc[strain, :]
    test_data = data_sterile.iloc[stest, :]
    sX = train_data[features]
    sy = train_data['t_serum_conc_percent']
    sy = (sy / 10).astype('int')
    sX_test = test_data[features]
    sy_test = test_data['t_serum_conc_percent']
    sy_test = (sy_test / 10).astype('int')

    sX_test_list.append(sX_test)
    sy_test_list.append(sy_test)
    s_id_list.append(test_data[['file', 'particle']])

    gbc = GradientBoostingClassifier(n_estimators=1000, learning_rate=l_rate, max_depth=depth)
    gbc.fit(sX, sy)

    pred = gbc.predict(sX_test)
    pred_list.append(pred)
    pred_proba = gbc.predict_proba(sX_test)
    pred_proba_list.append(pred_proba)

    explainer = shap.TreeExplainer(gbc)
    shap_values = explainer.shap_values(sX_test)
    shap_vs_list.append(shap_values)

    # shap.summary_plot(shap_values, sX_test, sort=False, color_bar=False, plot_size=(10,10))

all_sX_test = pd.concat(sX_test_list)
all_sy_test = pd.concat(sy_test_list)
all_splits_shap = np.concatenate(shap_vs_list)
all_pred = np.concatenate(pred_list)
all_pred_proba = np.concatenate(pred_proba_list)
all_s_id = pd.concat(s_id_list)

plt.title('aggregated')
shap.summary_plot(all_splits_shap, all_sX_test, sort=False, color_bar=False, plot_size=(10, 10))

df_all_splits_shap = pd.DataFrame(all_splits_shap, columns=all_sX_test.columns).add_prefix('shap_')

list_to_concat = [all_sX_test.reset_index(),
                  all_sy_test.reset_index(),
                  df_all_splits_shap.reset_index(),
                  pd.DataFrame(all_pred, columns=['predicted']).reset_index(),
                  pd.DataFrame(all_pred_proba).add_prefix('proba_').reset_index(),
                  all_s_id.reset_index()]

df_all = pd.concat(list_to_concat, axis=1)
df_all['correct'] = (df_all['t_serum_conc_percent'] == df_all['predicted'])
print((np.sum(df_all['correct'])) / (len(df_all)))

'''
Small block to check if SHAP correlations are meaningful (spoiler alert: no), and to study long jumpers
'''


df_all.to_csv('C:/Users/redchuk/python/plot_trajectory_test/df_all.csv')
jumpers = df_all[df_all['f_outliers3SD_diff_xy']==1]

jumpers = jumpers[['file',
         'particle',
         'f_mean_diff_xy_micron',
         'f_max_diff_xy_micron',
         'f_min_dist_range',
         'f_total_min_dist',
         'f_outliers2SD_diff_xy',
         't_serum_conc_percent',
         'proba_0',
         'proba_1',
         'correct', ]]

jumpers.to_csv('C:/Users/redchuk/python/plot_trajectory_test/jumpers.csv')

# correlation for features
plt.figure(figsize=(8, 7))
ax = sns.heatmap(all_sX_test.corr())
ax.figure.tight_layout()
plt.show()
plt.close()

# correlation for shap values
plt.figure(figsize=(8, 7))
ax = sns.heatmap(df_all_splits_shap.corr())
ax.figure.tight_layout()
plt.show()
plt.close()

# correlation for shap values (absolute)
plt.figure(figsize=(8, 7))
ax = sns.heatmap(df_all_splits_shap.corr().abs())
ax.figure.tight_layout()
plt.show()
plt.close()

# shap.dependence_plot('f_mean_diff_xy_micron', all_splits_shap, all_sX_test, interaction_index='f_most_central_mask')
# x_jitter=0.3
# https://towardsdatascience.com/you-are-underutilizing-shap-values-feature-groups-and-correlations-8df1b136e2c2
# https://shap-lrjball.readthedocs.io/en/latest/generated/shap.dependence_plot.html

'''
Decision tree as a baseline issue (solved)
'''

for feature in X.columns:
    gkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=None)
    # acc = cross_val_score(tree, X[feature].values.reshape(-1, 1), y, cv=gkf, groups=data_sterile['file'])
    # print('baseline ('+feature+'): ' + str(np.mean(acc)))
    score_test = []
    score_train = []
    for strain, stest in gkf.split(X, y, data_sterile['file']):
        train_data = data_sterile.iloc[strain, :]
        test_data = data_sterile.iloc[stest, :]
        sX = train_data[feature].values.reshape(-1, 1)
        sy = train_data['t_serum_conc_percent']
        sy = (sy / 10).astype('int')
        sX_test = test_data[feature].values.reshape(-1, 1)
        sy_test = test_data['t_serum_conc_percent']
        sy_test = (sy_test / 10).astype('int')

        tree = DecisionTreeClassifier(max_depth=1)
        tree.fit(sX, sy)
        score_train.append(tree.score(sX, sy))
        # print(tree.score(sX, sy))
        score_test.append(tree.score(sX_test, sy_test))
        # print(tree.score(sX_test, sy_test))

    print(feature + ': train ' + str(np.mean(score_train)) + ' test ' + str(np.mean(score_test)))
