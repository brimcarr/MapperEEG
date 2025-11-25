import numpy as np
from keras.models import Sequential, Model
from keras.layers import Input, LSTM, Bidirectional, ConvLSTM2D, Flatten, Dense, Conv2D, Dropout
from keras.layers import RepeatVector, TimeDistributed, Reshape, Conv2DTranspose, MaxPooling2D
from sklearn.preprocessing import MinMaxScaler

#%% Generates sequences for LSTM autoencoder
def to_seqs_lstm(dataset, seq_size = 2):
    x = []
    y = []
    for j in range(dataset.shape[1]):
        xj = []
        yj = []
        for i in range(dataset.shape[0]-seq_size):
            xj.append(dataset[i:(i + seq_size), j]) ### Reconstruct original entry
            yj.append(dataset[i:(i + seq_size), j]) 
        x.append(xj)
        y.append(yj)
    return np.transpose(np.array(x),(1,2,0)), np.transpose(np.array(y),(1,2,0))

#%% Unwindow data (this is for LSTM and CNN autoencoder)
### data: [# of samples, window_size, # of channels]
def unwindow(data, seq_size, network_type):
    if network_type == 'cnn':
        data = np.transpose(data,(0,2,1))
    output_array = []
    print(np.shape(data))
    for x in range(data.shape[0] + seq_size):
        b = []
        c = 0
        for i in range(seq_size-1):
            if x-i >= 0 and x < data.shape[0]:
                a = [data[x-i][i][j] for j in range(data.shape[2])]
                c = c+1
                b.append(a)
            elif x-i >= 0 and x >= data.shape[0] and i+x-data.shape[0] < seq_size:
                shift = x - data.shape[0]
                a = [data[-1-i][i + shift][j] for j in range(data.shape[2])]
                c = c+1
                b.append(a)
            else:
                a = [0]*data.shape[2]
                b.append(a)
        pt = sum(np.array(b))/c
        output_array.append(pt)
    return np.array(output_array)

#%% Unwindow data (this is for LSTM and CNN autoencoder)
### data: [# of samples, window_size, # of channels]
def unwindow_noverlap(data, seq_size, network_type):
    if network_type == 'cnn':
        data = np.transpose(data, (0,2,1))
        dim1 = data.shape[0] * data.shape[1]
        print(dim1)
        dim2 = data.shape[2]
    unstacked = np.reshape(data, (dim1, dim2))
    return unstacked

#%%% Make ae dataset
def make_ae_data(subset, type_o_network, seq_size = 2):
    ae_data = {}
    scaler = MinMaxScaler(feature_range=(0,1))
    X = scaler.fit_transform(subset)
    X_shape_x = np.shape(X)[0]
    X_shape_y = np.shape(X)[1]
    # X_shape_z = np.shape(X)[2]
            
    train_size = int(X_shape_x*0.7)
    train = X[0:train_size, :]
    test = X[train_size:len(X),:]
    if type_o_network == 'dense':
        trainX = train
        testX = test
        trainY = train
        testY = test
    if type_o_network == 'lstm':
        trainX, trainY = to_seqs_lstm(train, seq_size)
        testX, testY = to_seqs_lstm(test, seq_size)

    ae_data['scaler'] = scaler
    ae_data['xtr'] = trainX
    ae_data['xte'] = testX
    ae_data['ytr'] = trainY
    ae_data['yte'] = testY
    ae_data['xshpx'] = X_shape_x
    ae_data['xshpy'] = X_shape_y
    ae_data['ss'] = seq_size
    ae_data['train'] = train
    ae_data['test'] = test
    
    return ae_data

#%% Build network
def build_ae(ae_data, enddim, type_o_network, sz):
    
    if type_o_network == 'lstm':
        trainX = ae_data['xtr']
        input_seq = Input(shape = (trainX.shape[1], trainX.shape[2]))
        # "encoded" is the encoded representation of the input
        encoded = LSTM(32, 
                       return_sequences = True, 
                       input_shape = (trainX.shape[1], trainX.shape[2]),
                       activation = 'relu')(input_seq)
        encoded = Dropout(0.2)(encoded)
        encoded = LSTM(16, 
                       return_sequences = True, 
                       input_shape = (trainX.shape[1], 32),
                       activation = 'relu')(encoded)
        encoded = Dropout(0.2)(encoded)
        encoded = TimeDistributed(Dense(8))(encoded)
        encoded = Dropout(0.2)(encoded)
        encoded = LSTM(enddim, 
                        return_sequences = True, 
                        input_shape = (trainX.shape[1], 8),
                        activation = 'relu')(encoded)
        ### "decoded" is the lossy reconstruction of the input
        decoded = LSTM(8, 
                        return_sequences = True, 
                        input_shape = (trainX.shape[1], enddim),
                        activation = 'relu')(encoded)
        decoded = Dropout(0.2)(decoded)
        decoded = LSTM(16, 
                       return_sequences = True, 
                       input_shape = (trainX.shape[1], 8),
                       activation = 'relu')(encoded)
        decoded = Dropout(0.2)(decoded)
        decoded = LSTM(32, 
                       return_sequences = True, 
                       input_shape = (trainX.shape[1], 16),
                       activation = 'relu')(decoded)
        decoded = Dropout(0.2)(decoded)
    
        ### This model maps an input to its reconstruction
        autoencoder = Model(input_seq, decoded)
        
    elif type_o_network == 'dense':
        tX = ae_data['xtr']
        trainX = tX[:, :, np.newaxis]
        print(np.shape(trainX))
        ### Dense Autoencoder
        input_seq = Input(shape = (trainX.shape[1], ))
        # "encoded" is the encoded representation of the input
        if sz == 2:
            encoded = Dense(16)(input_seq)
            encoded = Dense(2)(encoded)
            decoded = Dense(16)(encoded)
            decoded = Dense(32)(decoded)
        if sz == 3:
            encoded = Dense(16)(input_seq)
            encoded = Dense(8)(encoded)
            encoded = Dense(2)(encoded)
            decoded = Dense(8)(encoded)
            decoded = Dense(16)(decoded)
            decoded = Dense(32)(decoded)
        if sz == 4:
            encoded = Dense(16)(input_seq)
            encoded = Dense(8)(encoded)
            encoded = Dense(4)(encoded)
            encoded = Dense(2)(encoded)
            decoded = Dense(4)(encoded)
            decoded = Dense(8)(decoded)
            decoded = Dense(16)(decoded)
            decoded = Dense(32)(decoded)




        ### This model maps an input to its reconstruction
        autoencoder = Model(input_seq, decoded)
        
    autoencoder.summary()
    
        
    return input_seq, autoencoder, encoded, decoded









