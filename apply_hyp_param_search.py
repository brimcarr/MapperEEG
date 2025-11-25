#%% Load packages
import pandas as pd
import warnings
import os
warnings.filterwarnings("ignore")
import numpy as np
# import matplotlib.pyplot as plt
from functions import file_names, load_data_h5py, dwnsmpl, compile_data
from functions import apply_mapper, make_networkx_graph, make_pie_graph
from functions import calc_qmod

#%% Establish hyperparameters and initialize choices
band_dict = {'delta': [1, 4],
             'theta': [4, 8],
             'alpha': [8, 13],
             'beta' : [13, 30],
             'gamma': [30, 50]}
for bdtime in band_dict.keys():
    p_d = {}
    array_o_qmods = []
    band_name = bdtime
    p_d['bd_range'] = band_dict[band_name]
    p_d['bd'] = band_name
    p_d['ad'] = 'concat'
    ### Projection functions: ['tsne', 'pca', 'umap', 'combo']
    p_d['pf'] = 'tsne'
    ### Clusterers: ['dbscan']
    p_d['cf'] = 'dbscan'
    p_d['labels'] = 'ss'
    
    csv_list = []
    best_params = []
    
    
    #%% Load up the data and choose trial
    ### Pull all datafile names
    fl = file_names('data')
    pairs = [[0,1],[2,3],[4,5],[6,7],[8,9],[10,11]]
    
    for lr in ['l', 'r']:
        p_d['lr'] = lr
        
        for i in range(0,6):
            print('Left or Right: ' + lr + ' Dyad: ' + str(i))
            p_d['dyad'] = i
            
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
            
            ######################
            p_d['sess'] = sess1
            trial_data1, trial_length1 = compile_data(dwn_dfs1, cond1, p_d)
            p_d['sess'] = sess2
            trial_data2, trial_length2 = compile_data(dwn_dfs2, cond2, p_d)
            trial_data = pd.concat([trial_data1, trial_data2])
            
            trial_length = trial_length1 + trial_length2
            p_d['tl'] = trial_length
            qmod_array = []
            ### Apply Mapper
            for perc in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
                p_d['po'] = perc
                for cube in [10, 15, 20, 25, 30, 35, 40]:
                    p_d['nc'] = cube
                    graph_data = apply_mapper(trial_data, p_d)
                    
                    #% Construct static pie graph
                    loaded_graph, nx_graph = make_networkx_graph(graph_data)
                    
                    #% Calculate Q-mod of graph
                    qmod = calc_qmod(loaded_graph, nx_graph, trial_data, p_d)
                    p_d['qmod'] = qmod
                    ### Plot the graph
                    fsize = (10,10)
                    piesize = 0.07
                    
                    make_pie_graph(nx_graph, trial_data, loaded_graph, fsize, 
                                   piesize, p_d)
    
                    qmod_array.append([p_d['lr'], p_d['dyad'], p_d['bd'], 
                                       p_d['nc'], p_d['po'], p_d['qmod']])
    
    
            db = str(str(p_d['dyad']) + p_d['lr'])
            df = pd.DataFrame(np.array(qmod_array))
            
            df.columns = ['Left/Right', 'Dyad', 'Band', '# of Cubes', '% overlap', 'Q-mod']
            mx = max(df['Q-mod'])
            row_with_max = df.loc[df['Q-mod'] == mx]
            print(row_with_max)
            np_row_with_max = row_with_max.to_numpy()
            best_params.append(np_row_with_max[0].tolist())
            
            csv_path = 'hyp_param_search/' + p_d['bd'] + '/' + db + '_' + p_d['pf'] +'_'+ p_d['cf'] +'_' + p_d['bd'] + '.csv'
            df.to_csv(csv_path, index=False)
        
            csv_list.append(csv_path)
    
    
    #%%
    best_csv_path = 'hyp_param_search/' + p_d['bd'] + '/best_params_' + p_d['bd'] + '.csv'
        
    best_df = pd.DataFrame(best_params)
    best_df.columns = ['Left/Right', 'Dyad', 'Band', '# of Cubes', '% overlap', 'Q-mod']
    best_df.to_csv(best_csv_path, index=False)
    
    csv_list.append(best_csv_path)
    
    #%% Make the combined Excel document
    excel_writer = pd.ExcelWriter('hyp_param_search/'+ p_d['bd'] +'/' + p_d['bd']+'_compiled_qmod.xlsx', engine='xlsxwriter')
    
    for csv_file in csv_list:
    
        df = pd.read_csv(csv_file)
    
        sheet_name = os.path.splitext(os.path.basename(csv_file))[0]
        df.to_excel(excel_writer, sheet_name=sheet_name, index=False)
    
    excel_writer.close()                                


 
    
