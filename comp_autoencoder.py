#%% Load packages
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
# import matplotlib.pyplot as plt
from functions import file_names, calc_errors
from functions_comp import data_pull, make_error_array, make_proper_labels
from functions_autoencoder import build_ae, unwindow, make_ae_data
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping
from math import sqrt
from sklearn.metrics import mean_squared_error as mse
from sklearn.cluster import KMeans
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
p_d['clus_alg'] = 'autoencoder'
p_d['hp_1'] = [20, 40, 60, 80, 100] # num of epochs
p_d['hp_2'] = [2,3,4] # number of layers

csv_list = []
#### Define necessary array as [num_participants, clus_range*5]
nec_array = []
nec_header = ['Dyad', 'lr', 'epochs', 'layers', 'f1_micro', 'f1_macro', 'f1_weight', 'sil_score', 'l_loss', 'db_score'] 
   

#%% Run ae
def comp_ae(trial_data, trial_labels, p_d, nec_array, csv_list):
    error_array = []
    subset = trial_data.iloc[:, 0:32].to_numpy()
    seq_size = 2
    for epo in p_d['hp_1']:
        for sz in p_d['hp_2']:
            ae_data = make_ae_data(subset, 'dense', seq_size)
            ### Build Dense or LSTM autoencoder
            input_seq, ae, encoded, decoded = build_ae(ae_data, 2, 'dense', sz)
            callback = EarlyStopping(monitor='loss',
                                     patience=10)
            ae.compile(optimizer='adam', 
                       loss='mean_squared_error')
            ae.fit(ae_data['xtr'], 
                   ae_data['ytr'], 
                   validation_data = (ae_data['xte'], ae_data['yte']),
                   verbose = 2, 
                   epochs = epo, 
                   callbacks = [callback])
            ### Normalized predictions for the whole model
            trainPredict_norm = ae.predict(ae_data['xtr'])
            testPredict_norm = ae.predict(ae_data['xte'])

            #% Use encoder to project data
            encoder = Model(input_seq, encoded)
            encoder.compile(optimizer='adam', loss='mean_squared_error')
            encoder.summary()
            
            ### Normalized projections from the encoder
            trainPredict_normEncode = encoder.predict(ae_data['xtr'])
            testPredict_normEncode = encoder.predict(ae_data['xte'])

            # if ae_type == 'lstm':
            #     ### Full autoencoder unwindowing and de-normalizing
            #     trainPredict_unwindow = unwindow(trainPredict_norm, ae_data['ss'], 'lstm')
            #     testPredict_unwindow = unwindow(testPredict_norm, ae_data['ss'], 'lstm') 
                
            #     trainPredict = ae_data['scaler'].inverse_transform(trainPredict_unwindow)
            #     testPredict = ae_data['scaler'].inverse_transform(testPredict_unwindow)
                
            #     ### Encoder unwindowing and de-normalizing
            #     trainPredict_unwindowEncode = unwindow(trainPredict_normEncode, ae_data['ss'], 'lstm')
            #     testPredict_unwindowEncode = unwindow(testPredict_normEncode, ae_data['ss'], 'lstm') 
                
            #     trainPredict_Encode = ae_data['scaler'].inverse_transform(trainPredict_unwindowEncode)
            #     testPredict_Encode = ae_data['scaler'].inverse_transform(testPredict_unwindowEncode)

            #     projected_data = np.concatenate((trainPredict_Encode, 
            #                                      testPredict_Encode), 
            #                                      axis = 0) 
                

            # elif ae_type == 'dense':
            ### Autoencoder de-normalizing
            trainPredict = ae_data['scaler'].inverse_transform(trainPredict_norm)
            testPredict = ae_data['scaler'].inverse_transform(testPredict_norm)
            
            ### Encoder dataset
            projected_data = np.concatenate((trainPredict_normEncode, 
                                             testPredict_normEncode), 
                                             axis = 0)
                
            ### Evaluate error
            trainScore = sqrt(mse(ae_data['train'], 
                                  trainPredict))
            print('Train score: %.2f RMSE' % (trainScore))

            testScore = sqrt(mse(ae_data['test'], 
                                  testPredict))
            print('Test score: %.2f RMSE' % (testScore))
            
            clus = 2
            t_data = projected_data
            km = KMeans(n_clusters = clus).fit(t_data)
            km_labels = (km.labels_).tolist() 
            prop_labels = make_proper_labels(km_labels, trial_labels) 
            errs = calc_errors(t_data, trial_data.iloc[:, 34], prop_labels)

            error_array.append([p_d['dyad'], p_d['lr'], str(epo), str(sz), 
                                errs[0], errs[1], errs[2], errs[3], errs[4], errs[5]])
                   
            
            nec_array.append([p_d['dyad'], p_d['lr'], str(epo), str(sz), 
                                errs[0], errs[1], errs[2], errs[3], errs[4], errs[5]])
            
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
        nec_arr = comp_ae(t_d, t_l, p_d, nec_array, csv_list)
        
        
#%% Make error array
make_error_array(p_d, nec_arr, nec_header)
   


#%% Generate the predictions/projections by de-normailizing


### Plot encoded data
# fig = plt.figure()

# if plot_dim == 3:
#     ### 3D
#     ax = plt.axes(projection = '3d') 
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(projected_data[:,0], 
#                projected_data[:,1],
#                projected_data[:,2],
#                c = trial_data.iloc[:, 34])
# elif plot_dim == 2:
#     ### 2D
#     ax = fig.add_subplot() 
#     ax.scatter(projected_data[:,0], 
#                projected_data[:,1],
#                c = trial_data.iloc[:, 34])
# plt.plot()
# plt.legend()
# plt.show()
    

