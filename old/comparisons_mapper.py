#%% Load packages
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from functions import file_names, load_data_h5py, dwnsmpl, compile_data, apply_mapper
from functions import make_networkx_graph, make_pie_graph, calc_errors
import networkx as nx

#%% Load up the data and choose trial
### Pull all datafile names
fl = file_names('data')
pairs = [[0,1],[2,3],[4,5],[6,7],[8,9],[10,11]]
parent = []

for leftright in ['l', 'r']:
    for i in range(0,6):
        print(leftright, i)

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
        
        ### Compile the data you want. Choices for amount_data:  
        ### single_trial, part_trial, concat, part_concat, lr_concat, pick2
        amount_data = 'concat'
        l_or_r = 'r'
        
        ### Choose band
        band_dict = {'alpha': [8,13],
                      'beta': [13, 30],
                      'gamma': [30,50],
                      'delta': [1, 5],
                      'theta': [5, 8]}
        mapper_sa = []
        
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
            trial_data = pd.concat([trial_data1, trial_data2])
            trial_length = trial_length1 + trial_length2
            trial_labels = trial_data['session'].tolist()
            
            param_dict = {'lr': l_or_r,
                            'tn': trial_num,
                            'se': se,
                            'tl': trial_length,
                            'bd': band,
                            'ad': amount_data,
                            'tt': t_type}
            
            #%% Apply Mapper
            ### Projection functions: ['tsne', 'pca', 'umap', 'combo']
            proj_func = 'umap'
            ### Clusterers: ['dbscan']
            clus_func = 'dbscan'
            t_data = trial_data.iloc[:,0:32]
            
            for perc in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
                for cube in [10, 15, 20, 25, 30, 35, 40]:
                    ### Adjusting the mapper parameters
                    num_cubes = cube
                    perc_overlap = perc
                    print(num_cubes, perc_overlap)
                    
                    graph_data = apply_mapper(trial_data, proj_func, clus_func, num_cubes, 
                                              perc_overlap, param_dict)
                    
                    #% Construct static pie graph
                    loaded_graph, nx_graph = make_networkx_graph(graph_data)
                    
                    ### Plot the graph (from Italo)
                    fsize = (10,10)
                    piesize = 0.07
                    make_pie_graph(nx_graph, trial_data, loaded_graph, fsize, piesize,
                                     l_or_r + ' Overlap: ' + str(perc) + ' Num of Cubes: ' 
                                     + str(cube), ss_or_four = 'ss')
                    
                    ### Create clusters based off of number of connected components
                    cc = nx.connected_components(nx_graph)  
                    list_o_cc = [c for c in sorted(cc, key=len, reverse=True)] 
                    gnl = []
                    gnl_miss = []
                    ### Make label list for each point by id'ing cluster it belongs to
                    for ind, cluster in enumerate(list_o_cc): 
                        for node in cluster:
                            ccp = loaded_graph['nodes'].get(node)
                            for element in ccp:
                                gnl.append((element, ind))
                                gnl_miss.append(element)
                    
                    ''' Not all nodes get assigned a cluster, so we create a new class
                    with a different label that id's all of these points as noise. 
                    (Similar to DBSCAN)'''
                    missing = [ele for ele in range(len(trial_labels)) if ele not in gnl_miss]
                    for m in missing:
                        gnl.append((m, (len(list_o_cc)+1)))
    
                    sorted_gnl= sorted(set(gnl), key=lambda x: x[0])
                    mapper_lbls = [x[1] for x in sorted_gnl]
                    ### Flip the label numbers if necessary
                    if mapper_lbls[0] == 1:
                        a = np.array(mapper_lbls)
                        m_labels = a^(a&1==a).tolist()
                        m_labels = [x+1 for x in m_labels]
                    else:
                        m_labels = [x+1 for x in mapper_lbls]
                        
                    if mapper_lbls.count(0) == 0 or mapper_lbls.count(1) == 0:
                        errs = [100]*6
                    else:
                        errs = calc_errors(t_data, trial_labels, m_labels)
                    mapper_sa.append([num_cubes, perc_overlap, errs[0], errs[1], 
                                      errs[2], errs[3], errs[4], errs[5]])
        parent.append(mapper_sa)

#%% Save error values
parent_np = np.array(parent)
pnp = np.hstack(parent_np)
np.savetxt('errors/all_errors_mapper.csv', pnp, delimiter=',') 
