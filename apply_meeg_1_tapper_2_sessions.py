#%% Load packages
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import time
# import matplotlib.pyplot as plt
from functions import file_names, load_data_h5py, dwnsmpl, compile_data
from functions import apply_mapper, make_networkx_graph, make_pie_graph, calc_qmod

#%% Establish hyperparameters and initialize choices
band_dict = {'alpha': [8,13],
              'beta': [13, 30],
              'gamma': [30,50],
              'delta': [1, 4],
              'theta': [4, 8]}

### Band: alpha, beta, gamma, delta, theta
time_list = []
band = 'gamma'
for lr in ['l', 'r']:
    for num in range(0,6):
        st_time = time.perf_counter()
    # lr = 'r'
    # num = 5
            
        p_d = {'dyad':   num,         ### [0-5]
               'lr':     lr,       ### Left or right: 'l' or 'r'
               'bd':     band,      ### Band: alpha, beta, gamma, delta, theta
               'bd_range': band_dict[band], ### Band range
               'nc':     20,        ### Number of cubes
               'po':     0.35,       ### Percent overlap
               'tr':     1,         ### Trial: 1-12 (if needed)
               'ad':     'concat',  ### Data amt: single_trial, part_trial, concat, part_concat, lr_concat, pick2
               'st_ed':  [0,1],     ### If ad is part, need start and end
               'pf':     'umap',    ### Proj funcs: ['tsne', 'pca', 'umap', 'combo']
               'cf':     'dbscan',  ### Cluster funcs: ['dbscan']
               'labels': 'ss'}      ### Label options: ['ss', 'four']
        
        #%% Select data
        fl = file_names('data')
        pairs = [[0,1],[2,3],[4,5],[6,7],[8,9],[10,11]]
        ind1 = (pairs[p_d['dyad']])[0]
        ind2 = (pairs[p_d['dyad']])[1]
        
        t_type1 = fl[ind1]
        t_type2 = fl[ind2]
        t_type = t_type1 + ' and ' + t_type2
        
        dfs1, chan_labels1, cond1, sess1 = load_data_h5py(t_type1, fl)
        dfs2, chan_labels2, cond2, sess2 = load_data_h5py(t_type2, fl)
        
        
        
        ### Downsample before compiling
        reduction = [10,2] 
        dwn_dfs1 = dwnsmpl(dfs1, reduction)
        dwn_dfs2 = dwnsmpl(dfs2, reduction)
        
        
        ###################### 
        p_d['sess'] = sess1
        trial_data1, trial_length1 = compile_data(dwn_dfs1, cond1, p_d)
        p_d['sess'] = sess2
        trial_data2, trial_length2 = compile_data(dwn_dfs2, cond2, p_d)
        trial_data = pd.concat([trial_data1, trial_data2])
        trial_length = trial_length1 + trial_length2
        p_d['tl'] = trial_length
        
        graph_data = apply_mapper(trial_data, p_d)
        
        #% Construct static pie graph
        
        loaded_graph, nx_graph = make_networkx_graph(graph_data)
        #%%
        #% Calculate Q-mod of graph
        qmod = calc_qmod(loaded_graph, nx_graph, trial_data, p_d)
        p_d['qmod'] = qmod
                
        #%% Plot the graph
        fsize = (10,10)
        piesize = 0.07
        
        make_pie_graph(nx_graph, trial_data, loaded_graph, fsize, piesize, p_d)
        end_time = time.perf_counter()
        tot_time = end_time-st_time
        time_list.append(tot_time)


#%% avg time

avg_time = sum(time_list)/len(time_list)

    
    
