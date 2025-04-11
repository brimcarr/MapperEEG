#%% Load packages
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import kmapper as km
from kmapper.plotlyviz import *
from sklearn.decomposition import PCA
# from umap.umap_ import UMAP
import warnings
warnings.filterwarnings("ignore")
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import Isomap


#%% Function to load in data (Sandra's stuff)
def load_data(NAME):
    if NAME == 'clean':
        data = scipy.io.loadmat("data/clean_20220713_EEG_EMG.mat")
    else:
        print('File not found')
    ### dict_keys(['__header__', '__version__', '__globals__', 'EMG_filtered_L', 
    ###             'EMG_filtered_R', 'bpchan', 'channels', 'conditionNames', 
    ###             'conditions', 'dataL', 'dataR', 'intervals', 'labels', 
    ###             'samples', 'seed', 'session', 'sessionTypes', 'sr' = 2000, 'tooshort'])
    chan_labels = []
    for j in range(np.shape(data['labels'])[1]):
        chan_labels.append((((data['labels'])[0])[j])[0])
    chan_labels.append('response') # Makes the len of the header match the size of the df
    # Makes the dataframes for EEG data
    dfs = {}
    for i in range(np.shape(data['dataL'])[1]):
        dfs['data_l_{0}'.format(i+1)] = pd.DataFrame(data['dataL'][0][i])
        dfs['data_l_{0}'.format(i+1)].columns = chan_labels
        dfs['data_r_{0}'.format(i+1)] = pd.DataFrame(data['dataR'][0][i])
        dfs['data_r_{0}'.format(i+1)].columns = chan_labels
    cond = data['conditions']
    return dfs, chan_labels, cond

#%% Load up the data and choose trial
dfs, chan_labels, cond = load_data('clean')  
  
trial_num = 2
l_or_r = 'r'

tl_name = 'data_l_' + str(trial_num)
tr_name = 'data_r_' + str(trial_num)

### Entire trial
# tl_data = dfs[tl_name]
# tr_data = dfs[tr_name]

### Subset
tl_data = dfs[tl_name].iloc[100000:125000]
tr_data = dfs[tr_name].iloc[100000:125000]

respon_l = tl_data['response'].to_numpy(copy=True)
respon_r = tr_data['response'].to_numpy(copy=True)
# print(np.unique(respon_l,  return_counts=True))

len_o_trial_l = len(tl_data)
len_o_trial_r = len(tr_data)

#%% Plot/visualize the data    
# ## Prints out the headers
# print(tl_data.head())
# ## Prints out the data description; count, mean, std dev, quartiles, max
# print(tl_data.describe())
# ## Prints out the info for the dataframe entry
# tl_data.info()

# channels = chan_labels[0:32]
# ## Plot the channel activity for each channel (spread out)
# tl_data[channels].plot(subplots=True, figsize=(15, 20), layout=(32, 1), sharex=True, sharey=True);
# plt.savefig('all_channels_' + l_or_r + '_trial_' + str(trial_num) + '.png')
# ## Plot the channel activity for each channel (condensed)
# tl_data[channels].plot(subplots=True, figsize=(15, 20), layout=(8, 4), sharex=True, sharey=False);
# plt.savefig('stacked_all_channels_' + l_or_r + '_trial_' + str(trial_num) + '.png')


#%% PCA for 2 participants, 1 trial, all 32 channels

# apply_pca = PCA(n_components = 2)

# trial_l_pca = apply_pca.fit_transform(tl_data)
# trial_r_pca = apply_pca.fit_transform(tr_data)

# print(trial_r_pca)
#%% PCA for 2 participants, 1 trial, single channel (not working)
'''Note, if  we want to do one channel, we can just throw it straight to mapper.
Mapper will take the lower dimensional signal, embed it up, and then bring it 
back down.'''

# band_name = ' 8-F8'

# trial_l = tl_data[band_name]
# trial_r = tr_data[band_name]

# apply_pca = PCA(n_components = 2)

# trial_l_pca = apply_pca.fit_transform(trial_l)
# trial_r_pca = apply_pca.fit_transform(trial_r)


#%% Plot 2D PCA
# Xaxl = trial_l_pca[:,0]
# Yaxl = trial_l_pca[:,1]

# Xaxr = trial_r_pca[:,0]
# Yaxr = trial_r_pca[:,1]


# fig = plt.figure(figsize=(16,9))
# axl = fig.add_subplot(121)
# plt.title('PCA left activity projection, trial ' + str(trial_num))
# axr = fig.add_subplot(122)
# plt.title('PCA right activity projection, trial ' + str(trial_num))

# scatterl = axl.scatter(Xaxl, Yaxl, s=5, c = respon_l, cmap = 'Set1')
# scatterr = axr.scatter(Xaxr, Yaxr, s=5, c = respon_r, cmap = 'Set1')

# axl.set_xlabel("PC1", fontsize=12)
# axl.set_ylabel("PC2", fontsize=12)

# axr.set_xlabel("PC1", fontsize=12)
# axr.set_ylabel("PC2", fontsize=12)
 
# # axl.view_init(30, 125)
# axl.legend(*scatterl.legend_elements(),loc="lower left", title="Time")
# # axr.view_init(30, 125)
# axr.legend(*scatterr.legend_elements(),loc="lower left", title="Time")
# plt.savefig('pca_2d_pair_figures/test_' + str(trial_num) + '.png')
# # plt.show()


#%% Plot 3D PCA
# Xaxl = trial_l_pca[:,0]
# Yaxl = trial_l_pca[:,1]
# Zaxl = trial_l_pca[:,2]

# Xaxr = trial_r_pca[:,0]
# Yaxr = trial_r_pca[:,1]
# Zaxr = trial_r_pca[:,2]

# fig = plt.figure(figsize=(16,9))
# axl = fig.add_subplot(121, projection='3d')
# plt.title('PCA left activity projection, trial ' + str(trial_num))
# axr = fig.add_subplot(122, projection='3d')
# plt.title('PCA right activity projection, trial ' + str(trial_num))
 
# scatterl = axl.scatter(Xaxl, Yaxl, Zaxl, s=60, c = range(len_o_trial_l), cmap = 'plasma')
# scatterr = axr.scatter(Xaxr, Yaxr, Zaxr, s=60, c = range(len_o_trial_r), cmap = 'plasma')
 
# axl.set_xlabel("PC1", fontsize=12)
# axl.set_ylabel("PC2", fontsize=12)
# axl.set_zlabel("PC3", fontsize=12)

# axr.set_xlabel("PC1", fontsize=12)
# axr.set_ylabel("PC2", fontsize=12)
# axr.set_zlabel("PC3", fontsize=12)
 
# # axl.view_init(30, 125)
# axl.legend(*scatterl.legend_elements(),loc="lower left", title="Blocks")
# # axr.view_init(30, 125)
# axr.legend(*scatterr.legend_elements(),loc="lower left", title="Blocks")

# plt.show()

  
#%% Apply Mapper
## Choose left or right person
if l_or_r == 'l':
    # tpca = trial_l_pca
    res = respon_l
elif l_or_r == 'r':
    # tpca = trial_r_pca
    res = respon_r
    

    
tpca = tl_data

# tap_trials = []
# res_list = res.tolist()
# print(type(res_list[0]))
# for count, ele in enumerate(res_list):
#     if ele == 1:
#         tap_trials.append(count)
#     else:
#         pass
# print(tap_trials)




mapper = km.KeplerMapper(verbose=2)

### Fit to and transform the data
projected_data = mapper.fit_transform(tpca, 
                                      projection=sklearn.manifold.TSNE())
### projections: [“sum”, “mean”, “median”, “max”, “min”, 
###                 “std”, “dist_mean”, “l2norm”, “knn_distance_n”, [0,1]
###                 sklearn.manifold.TSNE()]

### Create dictionary called 'graph' with nodes, edges and meta-information
# graph = mapper.map(projected_data, tpca, cover=km.Cover(n_cubes = 15, perc_overlap=0.7))
graph = mapper.map(projected_data, 
                    # tpca, 
                    # clusterer=AgglomerativeClustering(n_clusters=3,
                    #                                          linkage="complete",
                    #                                          affinity="cosine"),
                    clusterer=sklearn.cluster.DBSCAN(eps=0.3, min_samples=15),
                    cover=km.Cover(35, 0.5))    
print("Output graph examples to html")
# Visualize it
mapper.visualize(graph, 
                  title = "All Data, Trial " + str(trial_num) + ' Subject ' + l_or_r, 
                  path_html = "mapper_outputs/subset_" + str(trial_num) + '_' + l_or_r + ".html",
                  color_values = res,
                  node_color_function = ['max'],
                  color_function_name = 'Tap (1) vs. No Tap (0)')







