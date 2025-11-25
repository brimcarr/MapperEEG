#%% Load packages
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import numpy as np
# import matplotlib.pyplot as plt
from functions import file_names, apply_mapper, calc_qmod
from functions import make_networkx_graph, make_pie_graph, calc_errors
import networkx as nx
from functions_comp import data_pull, make_error_array

#%% Load up the data and choose trial
fl = file_names('data')
pairs = [[0,1],[2,3],[4,5],[6,7],[8,9],[10,11]]
band_dict = {'delta': [1, 4],
             'theta': [4, 8],
             'alpha': [8,13],
             'beta': [13, 30],
             'gamma': [30,50]}

p_d = {}
band_name = 'gamma'
p_d['bd_range'] = band_dict[band_name]
p_d['bd'] = band_name
p_d['ad'] = 'concat'
p_d['clus_alg'] = 'meeg'
p_d['hp_1'] = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5] # % overlap
p_d['hp_2'] = [10, 15, 20, 25, 30, 35, 40] # of cubes
p_d['pf'] = 'tsne'
p_d['cf'] = 'dbscan'
p_d['labels'] = 'ss'

csv_list = []
nec_array = []
nec_header = ['Dyad', 'lr', '% overlap', 'cubes', 'f1_micro', 'f1_macro', 'f1_weight', 'sil_score', 'l_loss', 'db_score'] 
   
#%% Run GMM
def comp_meeg(trial_data, trial_labels, p_d, nec_array, csv_list):
    error_array = []
    t_data = trial_data.iloc[:,0:32]
    for cube in p_d['hp_2']:
        p_d['nc'] = cube
        for perc in p_d['hp_1']:  
            p_d['po'] = perc
            graph_data = apply_mapper(trial_data, p_d)
            
            #% Construct static pie graph
            loaded_graph, nx_graph = make_networkx_graph(graph_data)
            
            #% Calculate qmod of the graph
            qmod = calc_qmod(loaded_graph, nx_graph, trial_data, p_d)
            p_d['qmod'] = qmod
            
            ### Plot the graph 
            fsize = (10,10)
            piesize = 0.07
            make_pie_graph(nx_graph, trial_data, loaded_graph, fsize, piesize, p_d)
            
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
                errs = [-100]*6
            else:
                errs = calc_errors(t_data, trial_labels, m_labels)
                    
            err_str = [p_d['dyad'], p_d['lr'], cube, perc, errs[0], errs[1], errs[2], errs[3], errs[4], errs[5]]
            error_array.append(err_str)
            nec_array.append(err_str)
            
        df = pd.DataFrame(np.array(error_array))
        df.columns = nec_header
        
        csv_path = 'errors/' + p_d['clus_alg'] +'/' + str(p_d['dyad']) + str(p_d['lr']) + '.csv'
        df.to_csv(csv_path, index=False)
        csv_list.append(csv_path) 
    return nec_array

#%% Load in data
for leftright in ['l', 'r']:
    p_d['lr'] = leftright
    print(leftright)
    for i in range(6):
        p_d['dyad'] = i
        print(i)
        t_d, t_l = data_pull(p_d, pairs, fl)
        nec_arr = comp_meeg(t_d, t_l, p_d, nec_array, csv_list)
#%% Make error array
make_error_array(p_d, nec_arr, nec_header)


#%% Test Mapper errors

# ### Adjusting the mapper parameters
# num_cubes = 25
# perc_overlap = 0.05
# print(num_cubes, perc_overlap)

# graph_data = apply_mapper(trial_data, proj_func, clus_func, num_cubes, 
#                           perc_overlap, param_dict)

# #% Construct static pie graph

# loaded_graph, nx_graph = make_networkx_graph(graph_data)

# ### Plot the graph (from Italo)
# fsize = (10,10)
# piesize = 0.07

# # make_pie_graph_ss(nx_graph, trial_data, loaded_graph, 
# #                 fsize, piesize, l_or_r + ' Overlap: ' + str(perc) + ' Num of Cubes: ' + str(cube), ss_or_four = 'four')
# make_pie_graph_ss(nx_graph, trial_data, loaded_graph, 
#                 fsize, piesize, l_or_r + ' Overlap: ' + str(perc) + ' Num of Cubes: ' + str(cube), ss_or_four = 'ss')



# cc = nx.connected_components(nx_graph)  
# list_o_cc = [c for c in sorted(cc, key=len, reverse=True)] 
# gnl = []
# gnl_miss = []
# for ind, cluster in enumerate(list_o_cc): 
#     for node in cluster:
#         ccp = loaded_graph['nodes'].get(node)
#         for element in ccp:
#             gnl.append((element, ind))
#             gnl_miss.append(element)
            
# missing = [ele for ele in range(len(trial_labels)) if ele not in gnl_miss]
# for m in missing:
#     gnl.append((m, (len(list_o_cc)+1)))

# sorted_gnl = sorted(set(gnl), key=lambda x: x[0])
# mapper_lbls = [x[1] for x in sorted_gnl]
# if mapper_lbls[0] == 1:
#     a = np.array(mapper_lbls)
#     m_labels = a^(a&1==a).tolist()
#     m_labels = [x+1 for x in m_labels]
# else:
#     m_labels = [x+1 for x in mapper_lbls]
    
    
        
# errs = calc_errors(trial_data, trial_labels, m_labels)

# print([num_cubes, perc_overlap, errs[0], errs[1], errs[2], errs[3], 
#                   errs[4], errs[5]])

    
    
    
