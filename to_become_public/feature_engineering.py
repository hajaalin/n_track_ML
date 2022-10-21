import pandas as pd
import numpy as np
from scipy.stats import linregress

data = pd.read_csv('to_become_public/tracking_output/data_47091baa.csv')

data['dX'] = data['x_micron'].diff()
data['dY'] = data['y_micron'].diff()

data_agg = data.groupby(['file', 'particle']).agg(serum=('serum', 'first'),

                                                  MD=('D', 'mean'),
                                                  # Mean locus displacement
                                                  MaxD=('D', 'max'),
                                                  # Maximal locus displacement
                                                  temp_sum_D=('D', 'sum'),
                                                  VarD=('D', 'var'),
                                                  # Variance of locus displacement

                                                  temp_sum_dX=('dX', 'sum'),
                                                  temp_sum_dY=('dY', 'sum'),

                                                  MA=('A', 'mean'),
                                                  # Mean nuclear area
                                                  MP=('P', 'mean'),
                                                  # Mean nuclear perimeter

                                                  MDist=('Dist', 'mean'),
                                                  # Mean locus distance to nuclear periphery
                                                  temp_min_Dist=('Dist', 'min'),
                                                  temp_max_Dist=('Dist', 'max'),
                                                  temp_beg_Dist=('Dist', 'first'),
                                                  temp_end_Dist=('Dist', 'last'),
                                                  VarDist=('Dist', 'var'),
                                                  # Variance of locus distance to nuclear periphery
                                                  )

data_agg['rVarD'] = data_agg['VarD'] / data_agg['MD']
# Relative variance of locus displacement
data_agg['rVarDist'] = data_agg['VarDist'] / data_agg['MDist']
# Relative variance of locus distance to nuclear periphery

data_agg['TD'] = np.sqrt((data_agg['temp_sum_dX']) ** 2 + (data_agg['temp_sum_dY']) ** 2)
# Total locus displacement
data_agg['Pers'] = data_agg['TD'] / data_agg['temp_sum_D']
# Persistence of locus movement

data_agg['temp_file_MD'] = data_agg.groupby('file')['MD'].transform(np.max)
data_agg['ifFast'] = np.where((data_agg['MD'] == data_agg['temp_file_MD']), 1, 0)
# True for the fastest homologous locus in the nucleus (highest MD)

data_agg['DistR'] = data_agg['temp_max_Dist'] - data_agg['temp_min_Dist']
# Range of locus distance to nuclear periphery
data_agg['TDist'] = data_agg['temp_end_Dist'] - data_agg['temp_beg_Dist']
# Total locus radial displacement

data_agg['temp_file_max_Dist'] = data_agg.groupby('file')['MDist'].transform(np.max)
data_agg['ifCentr'] = np.where((data_agg['MDist'] == data_agg['temp_file_max_Dist']), 1, 0)
# True for the homologous locus furthest from nuclear rim (highest MDist)

data_slope = data.groupby(['file', 'particle']).apply(lambda x: linregress(x['frame'], x['Dist'])[0])
data_agg['sDist'] = data_slope
# Trend of change of locus distance to nuclear periphery

data_sA = data.groupby(['file', 'particle']).apply(lambda x: linregress(x['frame'], x['A'])[0])
data_agg['sA'] = data_sA
# Trend of change of nuclear area

data_sP = data.groupby(['file', 'particle']).apply(lambda x: linregress(x['frame'], x['P'])[0])
data_agg['sP'] = data_sP
# Trend of change of nuclear perimeter

data_SD_D = data.groupby(['file', 'particle']).agg(SD_diff=('D', 'std'))
data_i = data.set_index(['file', 'particle'])
data_i['SD_D'] = data_SD_D
data_i['MD'] = data_agg['MD']
data_i['out2sd'] = np.where((data_i['D'] > (data_i['MD'] + 2 * data_i['SD_D'])), 1, 0)
data_i['out3sd'] = np.where((data_i['D'] > (data_i['MD'] + 3 * data_i['SD_D'])), 1, 0)
data_agg['out2sd'] = data_i.groupby(['file', 'particle']).agg(out2sd=('out2sd', 'sum'))
# Number of displacements 2SD longer than MD
data_agg['out3sd'] = data_i.groupby(['file', 'particle']).agg(out3sd=('out3sd', 'sum'))
# Number of displacements 3SD longer than MD

data = data_agg.drop(['temp_sum_dX',
                      'temp_sum_dY',
                      'temp_min_Dist',
                      'temp_max_Dist',
                      'temp_beg_Dist',
                      'temp_end_Dist',
                      'temp_file_MD',
                      'temp_file_max_Dist',
                      'temp_sum_D',
                      ], axis=1)
data.reset_index(inplace=True)

features = ['MD', 'MaxD', 'VarD', 'MA', 'MP', 'MDist', 'VarDist', 'rVarD', 'rVarDist',
            'TD', 'Pers', 'ifFast', 'DistR', 'TDist', 'ifCentr', 'sDist', 'sA', 'sP', 'out2sd', 'out3sd']

# todo: pack into a function to use in gbc
