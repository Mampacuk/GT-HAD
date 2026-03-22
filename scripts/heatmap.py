import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.io as sio

save_dir = '../heat_map/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

method_list = [
    # 'LRASR'
    # 'GT-HAD',
    #            'Auto-AD'
# 'KIFD'
#     'GTVLRR',
'FEBPAD',
# 'PTA',
#     'PCA-TLRSR',
#     'UNRS',
#     'LSAD-CR-IDW',
#     'AED',
#     'PAB-DC_2d',
#     'PAB-DC_3dmean',
#     'RXD',
#  'RSAD'
# 'LAD_Q_S'
#     'RXD'        ,
#     'WSCF'       ,
    # 'RSAD'       ,
    # 'LAD_Q'      ,
    # 'LAD_Q_S'    ,
    # 'LAD_C'      ,
    # 'LAD_C_S'    ,
    # 'RXD_PCA'    ,
    # 'LAD_Q_PCA'  ,
    # 'LAD_Q_PCA_S',
    # 'LAD_C_PCA'  ,
    # 'LAD_C_PCA_S',
    # 'CSD'
    # 'Beta'
    # 'SLW_LRRSTO'
    # 'SSFAD',
    # 'CRDBPSW'
]

file_list =  [
    # 'porcine4_348x260x31',
# 'porcine3_696x520x31',
# 'porcine2_696x520x31',
# 'porcine1_696x520x31',
#     'porcine4_1392x1040x31',
#         'porcine3_174x130x31',
    'bovine5.3_174x128x31',
# 'bovine3_174x130x31',
# 'bovine4_174x130x31'
#     'porcine4_174x130x31',
    # 'porcine3_174x130x31',
    #     'porcine4_174x130x31',
# 'porcine1_348x260x31',
#         'porcine2_348x260x31',
#     'porcine4_348x260x31',
    #     'porcine3_696x520x31',
# 'porcine4_696x520x31',
#     'porcine4_348x260x31',
    # 'porcine2_696x520x31',
    #     'pavia',
# 'los-angeles-1',
    # 'los-angeles-2', 'gulfport', 'texas-goast', 'cat-island'
]

VMAX = 0.5

def heatmap(img, save_name):
    h, w = img.shape

    # choose a base height (in inches)
    base_height = 4
    fig_width = base_height * (w / h)
    plt.figure(figsize=(fig_width, base_height))
    _plt = sns.heatmap(img, cmap='turbo', vmax=1.0 if VMAX is None else VMAX, annot=False, xticklabels=False,
        yticklabels=False, cbar=False, linewidths=0.0, rasterized=True)
    _plt.figure.savefig(save_name, format='pdf', bbox_inches='tight', pad_inches=0.0)
    plt.close()


# file_list =  ['pavia']
for file in file_list:
    mat_dir = os.path.join('../results/', file)
    for method in method_list:
        mat_name = os.path.join(mat_dir, method + '_map.mat')
        mat = sio.loadmat(mat_name)
        img = mat['show']
        # norm
        if VMAX is not None:
            img = img - img.min()
            img = img / img.max()
        # save fig
        save_subdir = os.path.join(save_dir, file)
        if not os.path.exists(save_subdir):
            os.makedirs(save_subdir)
        save_name = os.path.join(save_subdir, method + '.pdf')
        heatmap(img, save_name)