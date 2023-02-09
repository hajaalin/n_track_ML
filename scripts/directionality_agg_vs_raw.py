# this is auxiliary script to check which directionality is right for raw data (inceptiontime)
import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
import seaborn as sns

path = 'to_become_public/tracking_output/data_47091baa.csv'
data = pd.read_csv(path)

data['dX'] = data['x_micron'].diff()
data['dY'] = data['y_micron'].diff()
data['t'] = data['dY'] / data['dX']  # t for tangent
data['dt'] = data['t'].diff()

'''
print(data['dt_abs'].max())
print(data['dt_abs'].mean())
# tan(theta) = dy/dx seems a bad choice for directionality estimation, +/- infinity, if dx is close to 0
'''

data['theta'] = np.arctan(data['t'])
data['theta2'] = np.arctan2(data['dY'], data['dX'])
# arctan() doesn't work either, since it ignores actually direction 
# (dy=1, dx=1) and (dy=-1, dx=-1) have the same arctan()

# arctan2 knows the direction, but behaves weird when change in theta crosses pi.
# following is to test and prevent the change in angles from being above pi or below -pi
test_data = pd.DataFrame({'theta_f0': [45, 45, 45, 45, -135, -135, -135, -135, 90, 90],
                          'theta_f1': [0, 90, -45, 135, 0, 90, 150, -60, -135, -45]})
test_data_radian = test_data * math.pi / 180  # to radians as in np trigonometry
test_data_radian['d_theta'] = test_data_radian['theta_f1'] - test_data_radian['theta_f0']

test_data_radian['d_theta_minabs'] = test_data_radian['d_theta']
test_data_radian.loc[test_data_radian['d_theta'] > math.pi, 'd_theta_minabs'] = test_data_radian[
                                                                                    'd_theta'] - 2 * math.pi
test_data_radian.loc[test_data_radian['d_theta'] < -math.pi, 'd_theta_minabs'] = test_data_radian[
                                                                                     'd_theta'] + 2 * math.pi
print(test_data_radian)

# since it seems working, below is the same on real data
data['d_theta'] = data['theta'].diff()
data['d_theta2'] = data['theta2'].diff()

data['d_theta2_minabs'] = data['d_theta2']
data.loc[data['d_theta2'] > math.pi, 'd_theta2_minabs'] = data['d_theta2'] - 2 * math.pi
data.loc[data['d_theta2'] < -math.pi, 'd_theta2_minabs'] = data['d_theta2'] + 2 * math.pi

data['d_theta2_minabs_abs'] = data['d_theta2_minabs'].abs()

'''
['Unnamed: 0', 'file', 'particle', 'script_version_git', 'date', 'frame',
'y_micron', 'x_micron', 'D', 'A', 'P', 'Dist', 'serum', 'dX', 'dY', 't',
'dt', 'theta', 'theta2', 'd_theta', 'd_theta2', 'd_theta2_minabs',
'd_theta2_minabs_abs']
'''

# small changes in angle = high directionality, so we have to use reciprocal angle change
for i in ['dt', 'd_theta', 'd_theta2', 'd_theta2_minabs', 'd_theta2_minabs_abs']:
    data['1/' + i] = 1 / data[i]


'''
The part below is to check whether one of the angle change representations will correlate with 
persistence as it is estimated in aggregated data (Pers = temp_TD / temp_sum_D)

Although there is no correlation, I still think that d_theta2_minabs or d_theta2_minabs_abs 
represent directionality better than tan- or vanilla arctan-derived functions
'''







# normalized to displacement to check if correlated to persistence
for i in ['dt', 'd_theta', 'd_theta2', 'd_theta2_minabs', 'd_theta2_minabs_abs']:
    data['D_norm_'+'1/'+ i] = data['1/' + i]*data['D']

# now to aggregate data and get persistence as it is in aggregated version elsewhere
data = data[data['frame'] > 1]


data_agg = data.groupby(['file', 'particle']).agg(rec_dt=('1/dt', 'mean'),
                                                  rec_d_theta=('1/d_theta', 'mean'),
                                                  rec_d_theta2=('1/d_theta2', 'mean'),
                                                  rec_d_theta2_minabs=('1/d_theta2_minabs', 'mean'),
                                                  rec_d_theta2_minabs_abs=('1/d_theta2_minabs_abs', 'mean'),

                                                  D_norm_rec_dt=('D_norm_1/dt', 'mean'),
                                                  D_norm_rec_d_theta=('D_norm_1/d_theta', 'mean'),
                                                  D_norm_rec_d_theta2=('D_norm_1/d_theta2', 'mean'),
                                                  D_norm_rec_d_theta2_minabs=('D_norm_1/d_theta2_minabs', 'mean'),
                                                  D_norm_rec_d_theta2_minabs_abs=('D_norm_1/d_theta2_minabs_abs',
                                                                                  'mean'),

                                                  temp_sum_dX=('dX', 'sum'),
                                                  temp_sum_dY=('dY', 'sum'),
                                                  temp_sum_D=('D', 'sum')
                                                  )

data_agg['temp_TD'] = np.sqrt((data_agg['temp_sum_dX']) ** 2 + (data_agg['temp_sum_dY']) ** 2)
data_agg['Pers'] = data_agg['temp_TD'] / data_agg['temp_sum_D']

data_agg.drop(['temp_sum_dX', 'temp_sum_dY', 'temp_sum_D', 'temp_TD'], axis=1, inplace=True)

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
corr = data_agg.corr()
sns.heatmap(data=corr)
plt.show()
