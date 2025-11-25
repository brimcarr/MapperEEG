#%% Load packages
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from functions import file_names, dwnsmpl, load_data_h5py, compile_data, calc_errors
from sklearn.cluster import DBSCAN
import numpy as np

#%% Load up the data and choose trial
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
        
        ### Downsample before compiling
        reduction = [10,2] 
        dwn_dfs1 = dwnsmpl(dfs1, reduction)
        dwn_dfs2 = dwnsmpl(dfs2, reduction)
        
        ### Compile the data you want.  Choices for amount_data:
        ### single_trial, part_trial, concat, part_concat, lr_concat, pick2
        amount_data = 'concat'
        l_or_r = leftright
        ### Choose band
        band_dict = {'alpha': [8,13],
                     'beta': [13, 30],
                     'gamma': [30,50],
                     'delta': [1, 5],
                     'theta': [5, 8]}
        db_stat_array = []
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
            
#%% Run DBSCAN across a range of parameters        
            for ep in range(5, 21):
                for ms in range(3, 21):
                    print(bd, ep, ms)
                    t_data = trial_data.iloc[:,0:32]
                    db = DBSCAN(eps=ep, min_samples=ms).fit(t_data)
            
                    n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
                    n_noise_ = list(db.labels_).count(-1)
                    
                    ### Labels are [0, 1, ...] Points considered noise are labeled with -1
                    ### Note, we shift each entry by +1 to match the trial_labels list.
                    db_labels = (db.labels_+1).tolist()
                    
                    if db_labels.count(0) == 0 or db_labels.count(1) == 0:
                        errs = [100]*6
                    else:
                        errs = calc_errors(t_data, trial_labels, db_labels)
                    rw = [ep, ms, n_clusters_, n_noise_, errs[0], 
                                          errs[1], errs[2], errs[3], errs[4], errs[5]]
                    db_stat_array.append(rw)    
        parent.append(db_stat_array)    

#%% Make error array
parent_np = np.array(parent)
pnp = np.hstack(parent_np)
np.savetxt('errors/all_errors_DBSCAN.csv', pnp, delimiter=',')    
    



