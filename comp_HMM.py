#%% Load packages
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
# import matplotlib.pyplot as plt
from functions import file_names, calc_errors
from functions_comp import data_pull, make_error_array, make_proper_labels
from hmmlearn.hmm import GaussianHMM
import numpy as np

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
p_d['clus_alg'] = 'hmm'
p_d['hp_1'] = range(2,11) # clusters
p_d['hp_2'] = ['full', 'diag', 'spherical', 'tied']

csv_list = []
#### Define necessary array as [num_participants, clus_range*5]
nec_array = []
nec_header = ['Dyad', 'lr', 'comp', 'cover_type', 'f1_micro', 'f1_macro', 'f1_weight', 'sil_score', 'l_loss', 'db_score']


           
#%% Run HMM
def comp_hmm(trial_data, trial_labels, p_d, nec_array, csv_list):
    error_array = []
    t_data = trial_data.iloc[:,0:32]
    for comp in p_d['hp_1']:
        for cov in p_d['hp_2']:
            ### HMM 
            observations = trial_data.iloc[:,0:32]
            # n_observations = observations.shape[0] 
            model = GaussianHMM(n_components=comp, covariance_type=cov, n_iter=100)
            
            try:
                model.fit(observations)
                hmm_labels = model.predict(observations)
                prop_labels = make_proper_labels(hmm_labels, trial_labels) 
                errs = calc_errors(t_data, trial_labels, prop_labels)
            except:
                errs = [-100]*6
            ###
            rw = [p_d['dyad'], p_d['lr'], str(comp), str(cov), errs[0], 
                                errs[1], errs[2], errs[3], errs[4], errs[5]]
            error_array.append(rw)  
            nec_array.append(rw)
            
        
    df = pd.DataFrame(np.array(error_array))
    df.columns = nec_header
    
    csv_path = 'errors/' + p_d['clus_alg'] +'/' + str(p_d['dyad']) + str(p_d['lr']) + '.csv'
    df.to_csv(csv_path, index=False)
    csv_list.append(csv_path)
    
    return nec_array
    
#%% Hidden Markov Model (HMM)
for leftright in ['l', 'r']:
    p_d['lr'] = leftright
    print(leftright)
    for i in range(6):
        p_d['dyad'] = i
        print(i)
        t_d, t_l = data_pull(p_d, pairs, fl)
        nec_arr = comp_hmm(t_d, t_l, p_d, nec_array, csv_list)
#%% Make error array
make_error_array(p_d, nec_arr, nec_header)   

