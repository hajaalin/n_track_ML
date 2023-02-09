#!/usr/bin/env python
# coding: utf-8

import math
import numpy as np
import pandas as pd
from pathlib import Path
from sktime.datatypes._panel._convert import (
    from_multi_index_to_nested,
    from_multi_index_to_3d_numpy,
    from_nested_to_multi_index,
    from_nested_to_3d_numpy,
)
import sklearn
import sktime
import sys

from utility import parse_config

print('sklearn ' + sklearn.__version__)
print('sktime ' + sktime.__version__)


def nested_mean(row, col_name='col'):
    return row[col_name].mean()

def displacement(row):
    return math.sqrt(row['dx']**2 + row['dy']**2)

def angle(row):
    return math.atan2(row['dy'], row['dx'])

def add_features(df):
    df = df.copy()

    # displacement
    df['dx'] = df['x_micron'].diff()
    df['dy'] = df['y_micron'].diff()
    df['D'] = df.apply(displacement, axis=1)
    df = df[df['frame']!=0]
    #df = df.reset_index()

    # direction (t for theta)
    df['t'] = df.apply(angle, axis=1)
    df['dt'] = df['t'].diff()

    # fix weirdness if change in theta is more than pi
    df.loc[df['dt'] > math.pi, 'dt'] = df['dt'] - 2 * math.pi
    df.loc[df['dt'] < -math.pi, 'dt'] = df['dt'] + 2 * math.pi

    df = df[df['frame']!=1]
  
    return df
    

# feature sets
# 'file' is included because it is needed to create groups
fsets = {}
fs = ['file', 'serum']
fsets['D'] =             fs + ['D']
fsets['D_Dist'] =        fs + ['D', 'Dist']
fsets['D_A_P'] =         fs + ['D', 'A', 'P']
fsets['D_t'] =           fs + ['D', 't']
fsets['D_t_dt'] =        fs + ['D', 'dt']
fsets['D_Dist_t_dt_A'] = fs + ['D', 'Dist', 't','dt', 'A']
fsets['all'] =           fs + ['D', 'A', 'P', 'Dist', 'dt']


def load_data(path, f_set_name, X_in_dataframe=False, debug=False):
    data = pd.read_csv(path)
    data = add_features(data)
    print('Loaded data shape: ' + str(data.shape))

    # combine file and particle columns for using as instance index later on
    data['fp'] = data['file'] + '__' + data['particle'].astype(str)

    # add column for number of frames
    data['nframes'] = data.groupby('fp')['frame'].transform('count')    
    debug0 = data.copy()
    #print(data.nframes.unique())
    #print(data.groupby(by=['nframes']).count())
    # find the maximum number of frames
    idxmax = data.groupby(by=['nframes']).count()['file'].idxmax()
    # drop all series where frame count is not idxmax
    data = data[data['nframes']==idxmax]
    #print(data.nframes.unique())

    # multi-index dataframe to get all frames under one index level
    datam = data.set_index(['fp','frame'])
    datam.replace(to_replace=pd.NA, value=None, inplace=True)

    # select columns
    cols = fsets[f_set_name]
    datam = datam[cols]
    debug1 = datam.copy()

    # nested dataframe where file/particle define an instance
    datan = from_multi_index_to_nested(datam, instance_index='fp')
    debug2 = datan.copy()

    # read group name from the last element of the series
    # index of first element might be 0,1 or 2, depending on how many elements
    # have been dropped because of adding features with df.diff()
    print(datan.columns)
    groups = datan['file'].apply(lambda x: x.iloc[-1])
    datan = datan.drop(columns=['file'])

    # collapse class vector to a single value
    datan['class'] = datan.apply(nested_mean, axis=1, col_name='serum')
    # separate class vector...
    y = datan['class'].values
    
    dfX = datan.drop(columns=['class', 'serum'])
    features = dfX.columns
    X = from_nested_to_3d_numpy(dfX)
    print('X.shape: ' + str(X.shape))

    # X is now shaped (instance, feature, timestep)
    # reshape to (instance, timestep, feature)
    X = np.transpose(X,(0,2,1))
    print('X.shape: ' + str(X.shape))

    if X_in_dataframe:
        return dfX, y, groups
    elif debug:
        return X, y, groups, features, debug1, debug2

    return X, y, groups, features


def test():
    # open configuration
    prog_dir = Path(__file__).parent.absolute()
    paths_file = 'paths.yml'
    paths = parse_config(prog_dir / paths_file)

    # read data file name
    data_dir = paths['data']['dir']
    raw_data_file = paths['data']['raw_data_file']
    raw_data_file = Path(data_dir) / raw_data_file

    # load data
    fset = 'all'
    X, y, groups, features, d1, d2 = load_data(raw_data_file, fset, debug=True)

    #print('X.shape')
    print('X.shape: ' + str(X.shape))
    print('y.shape: ' + str(y.shape))
    print(features)
    print(groups.shape)

    
# https://github.com/hajaalin/InceptionTime/blob/d518ebe04c4404e49bdb056aa4aa00ea6ba9bd26/utils/utils.py#L30
def readucr(filename, delimiter=','):
    data = np.loadtxt(filename, delimiter=delimiter)
    Y = data[:, 0]
    X = data[:, 1:]
    return X, Y


def test_ucr():
    file_name = \
    '/proj/hajaalin/Projects/UCR_TS_Archive_2015/InlineSkate/InlineSkate_TRAIN'
    print('test_ucr: ' + file_name)
    x_train, y_train = readucr(file_name)
    print('x_train.shape: ' + str(x_train.shape))
    print('y_train.shape: ' + str(y_train.shape))
# https://github.com/hajaalin/InceptionTime/blob/d518ebe04c4404e49bdb056aa4aa00ea6ba9bd26/main.py#L35
    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    print('x_train.shape: ' + str(x_train.shape))

    
if __name__ == "__main__":
#    etl()
    test()
    test_ucr()
    

