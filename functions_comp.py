#%% Load packages
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from functions import dwnsmpl, load_data_h5py, compile_data

#%% Pull in data for comparison
def data_pull(p_d, pairs, fl):
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
    trial_data = pd.concat([trial_data1, trial_data2], ignore_index=True)
    trial_length = trial_length1 + trial_length2
    trial_labels = trial_data['session'].tolist()
    p_d['tl'] = trial_length

    return trial_data, trial_labels

#%% Make error array
def make_error_array(p_d, nec_array, nec_header):
    nec_csv_path = 'errors/' + p_d['clus_alg'] + '/accuracy_params.csv'
    nec_df = pd.DataFrame(nec_array)
    nec_df.columns = nec_header
    nec_df.to_csv(nec_csv_path, index=False)
    
    
    
#%% Proper labels
def make_proper_labels(output_labels, true_labels):
    t_labels = [x-1 for x in true_labels]
    # print(output_labels)
    # print(true_labels)
    clusters = list(set(output_labels))
    clus_array = np.zeros([len(clusters), 2])
    for i, ele in enumerate(output_labels):
        tl = t_labels[i]
        clus_array[ele][tl] += 1
    print(clus_array)
    bes_zero = np.argmax(clus_array[:,0])
    bes_one = np.argmax(clus_array[:,1])
    new_label_list = []
    for x in output_labels:
        if x == bes_zero:
            new_label_list.append(1)
        elif x == bes_one:
            new_label_list.append(2)
        else:
            new_label_list.append(3)
    return new_label_list

#%% Proper labels DBSCAN
def make_proper_labels_db(output_labels, true_labels):
    true_labels = [x-1 for x in true_labels]
    print(true_labels)
    clusters = list(set(output_labels))
    clus_array = np.zeros([len(clusters), 2])
    for i, ele in enumerate(output_labels):
        tl = true_labels[i]
        clus_array[ele][tl] += 1
    print(clus_array)
    if -1 not in clusters:
        bes_zero = np.argmax(clus_array[:,0])
        bes_one = np.argmax(clus_array[:,1])
    else:
        try:
            bes_zero = np.argmax(clus_array[1:,0])
            bes_one = np.argmax(clus_array[1:,1])
        except:
            bes_zero = -100
            bes_one = -100
        
    new_label_list = []
    for x in output_labels:
        if x == bes_zero:
            new_label_list.append(1)
        elif x == bes_one:
            new_label_list.append(2)
        else:
            new_label_list.append(3)
    return new_label_list


#%%   
# t1 = [0,1,2,2,5,5,3,3,3,3,3,4]
# tru = [0,0,0,0,0,0,1,1,1,1,1,1]
# make_proper_labels(t1, tru)



