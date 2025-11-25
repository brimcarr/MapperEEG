#%% Load packages
import warnings
warnings.filterwarnings("ignore")
from functions import file_names, load_data_h5py, dwnsmpl, compile_data
from functions import make_networkx_graph, make_pie_graph, apply_mapper, calc_qmod

#%% Initialize all needed things
band_dict = {'alpha': [8, 13],
             'beta': [13, 30],
             'gamma': [30, 80],
             'delta': [1, 4],
             'theta': [4, 8]}

### Band: alpha, beta, gamma, delta, theta
band = 'gamma'

### Establish the parameter dictionary
    
p_d = {'lr':     'l',       ### Left or right: 'l' or 'r'
       'dyad':   1,
       'bd':     band,      ### Band: alpha, beta, gamma, delta, theta
       'bd_range': band_dict[band], ### Band range
       'nc':     10,        ### Number of cubes
       'po':     0.15,       ### Percent overlap
       'tr':     1,         ### Trial: 1-12 (if needed)
       'ad':     'concat',  ### Data amt: single_trial, part_trial, concat, part_concat, lr_concat, pick2
       'st_ed':  [0,1],     ### If ad is part, need start and end
       'pf':     'umap',    ### Proj funcs: ['tsne', 'pca', 'umap', 'combo']
       'cf':     'dbscan',  ### Cluster funcs: ['dbscan']
       'labels': 'ss'}      ### Label options: ['ss', 'four']

                
#%% Load up the data and choose trial
### Pull all datafile names
folder = 'data'
fl = file_names(folder)
qmod_list = []

dfs, chan_labels, cond, sess = load_data_h5py(fl[p_d['dyad']], fl)
p_d['sess'] = sess

### Downsample before compiling
reduction = [10,2] 
dwn_dfs = dwnsmpl(dfs, reduction)


######################
trial_data, trial_length = compile_data(dwn_dfs, cond, p_d)

p_d['tl'] = trial_length


#%% Apply Mapper
graph_data = apply_mapper(trial_data, p_d)


#%% Construct static pie graph
loaded_graph, nx_graph = make_networkx_graph(graph_data)

### Plot the graph 
fsize = (10,10)
piesize = 0.07

qmod = calc_qmod(loaded_graph, nx_graph, trial_data, p_d)
p_d['qmod'] = qmod
qmod_list.append(qmod)

make_pie_graph(nx_graph, 
               trial_data, 
               loaded_graph, 
               fsize, 
               piesize, 
               p_d)




        
        
    
        
        
        
        
        
        
