#%% Load packages
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from functions import file_names, calc_errors
from functions_comp import data_pull, make_error_array, make_proper_labels_db
from sklearn.cluster import DBSCAN
import numpy as np
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

#%% Load up the data and choose trial
fl = file_names('data')
pairs = [[0,1],[2,3],[4,5],[6,7],[8,9],[10,11]]
band_dict = {'alpha': [8,13],
              'beta': [13, 30],
              'gamma': [30,50],
              'delta': [1, 4],
              'theta': [4, 8]}

p_d = {}
band_name = 'gamma'
p_d['bd_range'] = band_dict[band_name]
p_d['band_name'] = band_name
p_d['ad'] = 'concat'
p_d['clus_alg'] = 'DBSCAN'
p_d['hp_1'] = np.arange(0.5,10,0.5) # EP
p_d['hp_2'] = np.arange(4, 64, 4) ## MS

csv_list = []
#### Define necessary array as [num_participants, clus_range*5]
nec_array = []
nec_header = ['Dyad', 'lr', 'ep', 'ms', '# of clusters', 'number of noise clusters', 'f1_micro', 'f1_macro', 'f1_weight', 'sil_score', 'l_loss', 'db_score'] 

#%% Run DBSCAN across a range of parameters
def comp_DBSCAN(trial_data, trial_labels, p_d, nec_array, csv_list):
    t_data = trial_data.iloc[:,0:32]
    error_array = []
    for ep in p_d['hp_1']:
        for ms in p_d['hp_2']:
            db = DBSCAN(eps=ep, min_samples=ms).fit(t_data)
            
            n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
            n_noise = list(db.labels_).count(-1)
            
            ### Labels are [0, 1, ...] Points considered noise are labeled with -1
            ### Note, we shift each entry by +1 to match the trial_labels list.
            db_labels = db.labels_
            prop_labels = make_proper_labels_db(db_labels, trial_labels)
            print(prop_labels)
            try:
                errs = calc_errors(t_data, trial_labels, prop_labels)
                
            except:
                errs = [100]*6

            err_str = [p_d['dyad'], p_d['lr'], ep, ms, n_clusters, n_noise, 
                       errs[0], errs[1], errs[2], errs[3], errs[4], errs[5]]
            error_array.append(err_str) 
            nec_array.append(err_str) 
            
    df = pd.DataFrame(np.array(error_array))
    df.columns = nec_header
    
    csv_path = 'errors/'+p_d['clus_alg']+'/' + str(p_d['dyad']) + str(p_d['lr']) + '.csv'
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
        nec_arr = comp_DBSCAN(t_d, t_l, p_d, nec_array, csv_list)
        

#%% Make error array
make_error_array(p_d, nec_arr, nec_header)


    
#%% Choosing eps
# 3. Calculate and plot the k-distance graph
# For eps, we plot the distance to the k-th nearest neighbor (k = min_samples)
# The NearestNeighbors calculation includes the point itself, so we use min_samples - 1.
