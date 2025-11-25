#%% Load packages
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from functions import file_names, dwnsmpl, load_data_h5py, compile_data, calc_errors
from sklearn.cluster import KMeans
import numpy as np

#%% Load up the data and choose trial
### Pull all datafile names
fl = file_names('data')
pairs = [[0,1],[2,3],[4,5],[6,7],[8,9],[10,11]]
parent = []
for leftright in ['l', 'r']:
    print(leftright)
    for i in range(6):
        print(i)
        ind1 = (pairs[i])[0]
        ind2 = (pairs[i])[1]
        
        t_type1 = fl[ind1]
        t_type2 = fl[ind2]
        t_type = t_type1 + ' and ' + t_type2
        
        dfs1, chan_labels1, cond1, sess1 = load_data_h5py(t_type1, fl)
        dfs2, chan_labels2, cond2, sess2 = load_data_h5py(t_type2, fl)
        
        ### Gather appropriate data
        ### Downsample before compiling
        reduction = [10,2] 
        dwn_dfs1 = dwnsmpl(dfs1, reduction)
        dwn_dfs2 = dwnsmpl(dfs2, reduction)
        
        ### Compile the data you want.  
        ### Choices for amount_data: 
        ### single_trial, part_trial, concat, part_concat, lr_concat, pick2
        amount_data = 'concat'
        l_or_r = leftright
        ### Choose band
        ### Choose band
        band_dict = {'alpha': [8,13],
                     'beta': [13, 30],
                     'gamma': [30,50],
                     'delta': [1, 5],
                     'theta': [5, 8]}
        
        km_stat_array = []
        for bd in band_dict.keys():
            band = band_dict[bd]  
        ### Used in full or partial
            trial_num = 1
            ### Used in partial and partial_concat only
            se = [1,2]
            
            ######################
            trial_data1, trial_length1 = compile_data(dwn_dfs1, cond1, sess1, 
                                                         amount_data, band, l_or_r)
            trial_data2, trial_length2 = compile_data(dwn_dfs2, cond2, sess2, 
                                                         amount_data, band, l_or_r)
            trial_data = pd.concat([trial_data1, trial_data2], ignore_index=True)
            trial_length = trial_length1 + trial_length2
            trial_labels = trial_data['session'].tolist()
            
            param_dict = {'lr': l_or_r,
                            'tn': trial_num,
                            'se': se,
                            'tl': trial_length,
                            'bd': band,
                            'ad': amount_data,
                            'tt': t_type}
            
            #%% k-Means
            
            for clus in range(2, 11):
                print(clus)
                t_data = trial_data.iloc[:,0:32]
                km = KMeans(n_clusters = clus).fit(t_data)
                km_labels = (km.labels_+1).tolist()            
                errs = calc_errors(t_data, trial_labels, km_labels)
    
                km_stat_array.append([clus, errs[0], errs[1], errs[2], errs[3], 
                                          errs[4], errs[5]]) 
    
        parent.append(km_stat_array)    

#%% Make error array
parent_np = np.array(parent)
pnp = np.hstack(parent_np)
np.savetxt('errors/all_errors_kmeans.csv', pnp, delimiter=',')     
