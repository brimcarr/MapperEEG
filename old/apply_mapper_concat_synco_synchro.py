#%% Load packages
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import numpy as np
# import matplotlib.pyplot as plt
from functions import file_names, load_data_h5py, dwnsmpl, compile_data
from functions import apply_mapper, make_networkx_graph, make_pie_graph, calc_qmod

#%% Run all tappers over all lists of Mapper parameters
### Load up the data and choose trial
### Pull all datafile names
array_o_qmods_four = []
array_o_qmods_ss =[]
fl = file_names('data')
pairs = [[0,1],[2,3],[4,5],[6,7],[8,9],[10,11]]

for leftright in ['l', 'r']:
    for i in range(0,6):
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
        lr = leftright
        ### Choose band
        band_dict = {'alpha': [8,13],
                      'beta': [13, 30],
                      'gamma': [30,50],
                      'delta': [1, 5],
                      'theta': [5, 8]}
        band = band_dict['theta']
        
        ### Used in full or partial
        trial_num = 1
        ### Used in partial and partial_concat only
        se = [1,2]
        
        ######################
        trial_data1, trial_length1 = compile_data(dwn_dfs1, cond1, sess1, 
                                                      amount_data, band, lr)
        trial_data2, trial_length2 = compile_data(dwn_dfs2, cond2, sess2, 
                                                      amount_data, band, lr)
        trial_data = pd.concat([trial_data1, trial_data2])
        trial_length = trial_length1 + trial_length2
        
        param_dict = {'lr': lr,
                        'tn': trial_num,
                        'se': se,
                        'tl': trial_length,
                        'bd': band,
                        'ad': amount_data,
                        'tt': t_type}
        
        ### Apply Mapper
        ### Projection functions: ['tsne', 'pca', 'umap', 'combo']
        proj_func = 'umap'
        ### Clusterers: ['dbscan']
        clus_func = 'dbscan'
        
        list_o_qmods_ss = []
        list_o_qmods_four = []
        for perc in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
            for cube in [10, 15, 20, 25, 30, 35, 40]:
                ### Adjusting the mapper parameters
                num_cubes = cube
                perc_overlap = perc
                
                graph_data = apply_mapper(trial_data, proj_func, clus_func, num_cubes, 
                                          perc_overlap, param_dict)
                
                #% Construct static pie graph
                
                loaded_graph, nx_graph = make_networkx_graph(graph_data)
                
                ### Plot the graph
                fsize = (10,10)
                piesize = 0.07
                
                make_pie_graph(nx_graph, trial_data, loaded_graph, fsize, piesize,
                                  lr + ' Overlap: ' + str(perc) + ' Num of Cubes: ' 
                                  + str(cube), ss_or_four = 'four')
                make_pie_graph(nx_graph, trial_data, loaded_graph, fsize, piesize,
                                  lr + ' Overlap: ' + str(perc) + ' Num of Cubes: ' 
                                  + str(cube), ss_or_four = 'ss')
                
                #% Calculate Q-mod of graph
                qmod_four = calc_qmod(loaded_graph, nx_graph, trial_data, 
                                      ss_or_four = 'four')
                qmod_ss = calc_qmod(loaded_graph, nx_graph, trial_data, 
                                    ss_or_four = 'ss')
                print('four: ', qmod_four, 'ss: ', qmod_ss)
                list_o_qmods_ss.append(qmod_ss)
                list_o_qmods_four.append(qmod_four)
        array_o_qmods_four.append(list_o_qmods_four)
        array_o_qmods_ss.append(list_o_qmods_ss)
list_o_qmods_ss.append(qmod_ss)
list_o_qmods_four.append(qmod_four)
array_o_qmods_four.append(list_o_qmods_four)
array_o_qmods_ss.append(list_o_qmods_ss)

### Save qmod values
np_qmod_four = np.array(array_o_qmods_four).T
np_qmod_ss = np.array(array_o_qmods_ss).T
   
np.savetxt('Qmod/' + lr + '_qmod_four.csv', np_qmod_four, delimiter=',')
np.savetxt('Qmod/' + lr + '_qmod_ss.csv', np_qmod_ss, delimiter=',')


#%% Run one Tapper pair through one set of Mapper parameter 
ind1 = 0
ind2 = 1

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
lr = 'l'
### Choose band
band_dict = {'alpha': [8,13],
              'beta': [13, 30],
              'gamma': [30,50],
              'delta': [1, 5],
              'theta': [5, 8]}
band = band_dict['theta']

### Used in full or partial
trial_num = 1
### Used in partial and partial_concat only
se = [1,2]

######################
trial_data1, trial_length1 = compile_data(dwn_dfs1, cond1, sess1, 
                                              amount_data, band, lr)
trial_data2, trial_length2 = compile_data(dwn_dfs2, cond2, sess2, 
                                              amount_data, band, lr)
trial_data = pd.concat([trial_data1, trial_data2])
trial_length = trial_length1 + trial_length2

param_dict = {'lr': lr,
                'tn': trial_num,
                'se': se,
                'tl': trial_length,
                'bd': band,
                'ad': amount_data,
                'tt': t_type}

### Projection functions: ['tsne', 'pca', 'umap', 'combo']
proj_func = 'umap'
### Clusterers: ['dbscan']
clus_func = 'dbscan'

### Adjusting the mapper parameters
cube = 35
perc = .35
num_cubes = 35
perc_overlap = 0.35

graph_data = apply_mapper(trial_data, proj_func, clus_func, num_cubes, 
                          perc_overlap, param_dict)

#% Construct static pie graph

loaded_graph, nx_graph = make_networkx_graph(graph_data)

### Plot the graph (from Italo)
fsize = (10,10)
piesize = 0.07

make_pie_graph(nx_graph, trial_data, loaded_graph, fsize, piesize,
                 lr + ' Overlap: ' + str(perc_overlap) + ' Num of Cubes: ' 
                 + str(num_cubes), ss_or_four = 'four')
make_pie_graph(nx_graph, trial_data, loaded_graph, fsize, piesize,
                 lr + ' Overlap: ' + str(perc_overlap) + ' Num of Cubes: ' 
                 + str(num_cubes), ss_or_four = 'ss')

#% Calculate Q-mod of graph
qmod_four = calc_qmod(loaded_graph, nx_graph, trial_data, 
                      ss_or_four = 'four')
qmod_ss = calc_qmod(loaded_graph, nx_graph, trial_data, 
                    ss_or_four = 'ss')
print('four: ', qmod_four, 'ss: ', qmod_ss)

    
    
    
    
