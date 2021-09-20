import pandas as pd
import dabest

data_to_plot = pd.read_csv('scripts/data_sterile_449e453.csv')


def gen_set(data, parameter='', chr_name='', guides=''):
    """

    :param data: df
    :param parameter: feature to plot, str
    :param chr_name: chr name for labeling, str
    :param guides: guides to aggregate, str, use upright slash for list of guides
    :return: tuple for Dabest plotting, format:
                        idx=(("Control 1", "Test 1",),
                             ("Control 2", "Test 2")
                             ))

    """

    data = data[data["t_guide"].str.contains(guides, regex=True)]#.dropna()
    before = pd.Series(data=data[data["t_time"] == 0][parameter], name=chr_name + ', 10%')
    after = pd.Series(data=data[(data["t_time"] < 40) & (data["t_time"] > 0)][parameter], name=chr_name + ', 0.3%')
    frame = pd.concat([before, after], axis=1)
    return frame


'''
Guides:
'pl_1521_chr10', 'pl_1522_chr10', 'pl_1404_chr13', 'pl_1406_chrx',
'pl_1398_chr1', 'pl_1532_chr18', 'pl_1362_telo', 'pl_1514_chr1',
'pl_1403_chr13'

what is wrong with chr18?
'''

chr1 = gen_set(data_to_plot, chr_name='chr1', parameter='f_min_dist_micron', guides='1398|1514')
chr10 = gen_set(data_to_plot, chr_name='chr10', parameter='f_min_dist_micron', guides='1521|1522')
chr13 = gen_set(data_to_plot, chr_name='chr13', parameter='f_min_dist_micron', guides='1403|1404')
chrX = gen_set(data_to_plot, chr_name='chrX', parameter='f_min_dist_micron', guides='1406')
telo = gen_set(data_to_plot, chr_name='telo', parameter='f_min_dist_micron', guides='1362')


all_chr = pd.concat([chr1, chr10, chr13, chrX, telo], axis=1)
# all_chr.columns

multi_2group = dabest.load(all_chr, idx=(('chr1, 10%', 'chr1, 0.3%'),
                                         ('chr10, 10%', 'chr10, 0.3%'),
                                         ('chr13, 10%', 'chr13, 0.3%'),
                                         ('chrX, 10%', 'chrX, 0.3%'),
                                         ('telo, 10%', 'telo, 0.3%'),
                                         ))

multi_2group.mean_diff.plot(raw_marker_size=3,
                            es_marker_size=6)