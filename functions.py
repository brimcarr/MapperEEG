#%% Import packages
import os
import numpy as np
import pandas as pd
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")
import scipy.signal as ss
from scipy.integrate import simpson
from collections import defaultdict
import h5py
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import umap.umap_ as umap
import random
# from datetime import datetime
import kmapper as km
from kmapper.plotlyviz import *
from sklearn.metrics import f1_score, silhouette_score, log_loss, davies_bouldin_score

#%% Function to pull in data file names
def file_names(folder):
    file_list = sorted(os.listdir(folder))
    for i in file_list:
        if i[0] == '.':
            file_list.remove(i)
    return file_list

#%% Function to load matlab files
def load_data_h5py(NAME, fl):
    if NAME in fl:
        with h5py.File('data/'+NAME, 'r') as f:
            ### dict_keys([ '#refs#', 'bpchan', 'channels', 'conditionNames', 
            ###             'conditions', 'dataL', 'dataR', 'intervals', 
            ###             'labels', 'samples', 'seed', 'session', 
            ###             'sessionTypes', 'sr' = 2000, 'tooshort'])
            
            ### Pull the session from the data
            sess = int(pd.DataFrame((f['session'])[:]).iloc[0,0])
            
            ### Pull the channel labels from the data (note, since strings, need the weird join statement)
            h5_labels = pd.DataFrame((f['labels'])[:])[0]
            chan_labels = []
            for ref in h5_labels:
                chan_labels.append("".join(chr(c.item()) for c in f[ref][:]))
            chan_labels.append('response') 
            
            ### Make dataframe with all 24 trials (12 left 12 right)
            dfs = {}
            ### Left tapper
            h5_dataL = pd.DataFrame((f['dataL'])[:])[0]
            for i, ref in enumerate(h5_dataL):
                dfs['data_l_{0}'.format(i+1)] = pd.DataFrame(f[ref]).T
                dfs['data_l_{0}'.format(i+1)].columns = chan_labels
            ### Right tapper    
            h5_dataR = pd.DataFrame((f['dataR'])[:])[0]
            for i, ref in enumerate(h5_dataR):
                dfs['data_r_{0}'.format(i+1)] = pd.DataFrame(f[ref]).T
                dfs['data_r_{0}'.format(i+1)].columns = chan_labels
            
            ### Pull trial conditions from the data
            cond = list(pd.DataFrame((f['conditions'])[:]).iloc[:,0])
            cond = [int(c) for c in cond]
    else:
        raise Exception('File not found.')

    ### Returns the dataframe, channel labels, trial conditions, and the session,
    return dfs, chan_labels, cond, sess


#%% Function to downsample the data (as recommended by Piotr)
def dwnsmpl(data_to_dwnspl, reduction):
    dnsp_data = {}
    lr_keys = list(data_to_dwnspl.keys())
    red_fac = int(np.prod(reduction))
    for lr_key in lr_keys:
        chan_keys = list(data_to_dwnspl[lr_key].keys())
        downsampled_data = {}
        for key in chan_keys:
            ### Pull the proper tapping response for the downsampled data
            if key == 'response':
                res = (data_to_dwnspl[lr_key])[key].to_numpy()
                res_index = np.where(res == 1)
                new_res_index = np.round(np.divide(res_index, red_fac))
                a = np.shape(res[::red_fac])[0]
                new_res = [1 if x in new_res_index else 0 for x in range(a)]
                downsampled_data[key] = new_res
            elif key == 'condition':
                pass
            else: 
                sig = (data_to_dwnspl[lr_key])[key].to_numpy()
                ### Downsample the data
                sig_dnspl = sig[::red_fac]
                downsampled_data[key] = sig_dnspl
            ### Turn newly minted dictionary into a DataFrame
            dwnsp_df = pd.DataFrame(downsampled_data)
        dnsp_data[lr_key] = dwnsp_df
    return dnsp_data

#%% Function to calculate band power (Piotr)
def calc_band_pwr(data, band, window_sz=10*100):
    band_array = {}
    for ch in data.keys():
        channel_data = data[ch].values
        aa = []
        for s in range(0, int(channel_data.size-window_sz), int(window_sz/2)):
            samp_freq, psd = ss.welch(channel_data[s:s+window_sz])
            a_power = simpson(psd[band], dx=1/100)
            aa.append(a_power)
        band_array[ch] = aa
        smp_len = len(aa)
    return band_array, smp_len

#%% Function to compile data
def compile_data(dfs, cond, p_d):
    t_name = 'data_' + p_d['lr']
    ### Single trial
    if p_d['ad'] =='single_trial':
        t_data = dfs[t_name + '_' + str(p_d['tr'])]
        band_array, smp_len = calc_band_pwr(t_data, p_d['bd_range'])
        if p_d['lr'] == 'l':
            band_array['trial_condition'] = [cond[p_d['tr']]]*smp_len
        elif p_d['lr'] == 'r':
            if cond[p_d['tr']] == 2:
                band_array['trial_condition'] = [3]*smp_len
            elif cond[p_d['tr']] == 3:
                band_array['trial_condition'] = [2]*smp_len
            else: 
                band_array['trial_condition'] = [cond[p_d['tr']]]*smp_len
        band_array['session'] = [p_d['sess']]*smp_len
        t_length = 'Single trial'

    elif p_d['ad'] =='part_trial':
        t_data = dfs[t_name + '_' + str(p_d['tr'])].iloc[p_d['st_ed'][0]:p_d['st_ed'][1]]
        band_array, smp_len = calc_band_pwr(t_data, p_d['bd_range'])
        if p_d['lr'] == 'l':
            band_array['trial_condition'] = [cond[p_d['tr']]]*smp_len
        elif p_d['lr'] == 'r':
            if cond[p_d['tr']] == 2:
                band_array['trial_condition'] = [3]*smp_len
            elif cond[p_d['tr']] == 3:
                band_array['trial_condition'] = [2]*smp_len
            else: 
                band_array['trial_condition'] = [cond[p_d['tr']]]*smp_len
        band_array['session'] = [p_d['sess']]*smp_len
        t_length = 'Partial data, single trial'
    
    elif p_d['ad'] == 'concat':
        mba = []
        for i in range(1,13):
            t_data = dfs[t_name + '_' + str(i)]
            band_array, smp_len = calc_band_pwr(t_data, p_d['bd_range'])
            if p_d['lr'] == 'l':
                band_array['trial_condition'] = [cond[i-1]]*smp_len
            elif p_d['lr'] == 'r':
                if cond[i-1] == 2:
                    band_array['trial_condition'] = [3]*smp_len
                elif cond[i-1] == 3:
                    band_array['trial_condition'] = [2]*smp_len
                else: 
                    band_array['trial_condition'] = [cond[i-1]]*smp_len
            
            band_array['session'] = [p_d['sess']]*smp_len
            mba.append(band_array)
        result = defaultdict(list)
        for j in range(len(mba)):
            current = mba[j]
            for key, value in current.items():
                # print(value)
                for k in range(len(value)):
                    result[key].append(value[k])
        band_array = pd.DataFrame.from_dict(dict(result))
        t_length = 'Concantenated data, all trials'
    
    elif p_d['ad'] == 'temporal':
        mba = []
        for i in range(1,13):
            t_data = dfs[t_name + '_' + str(i)]
            # band_array, smp_len = calc_band_pwr(t_data, p_d['bd_range'])
            band_array = t_data
            smp_len = len(band_array)
            if p_d['lr'] == 'l':
                band_array['trial_condition'] = [cond[i-1]]*smp_len
            elif p_d['lr'] == 'r':
                if cond[i-1] == 2:
                    band_array['trial_condition'] = [3]*smp_len
                elif cond[i-1] == 3:
                    band_array['trial_condition'] = [2]*smp_len
                else: 
                    band_array['trial_condition'] = [cond[i-1]]*smp_len
            
            band_array['session'] = [p_d['sess']]*smp_len
            mba.append(band_array)
        result = defaultdict(list)
        for j in range(len(mba)):
            current = mba[j]
            for key, value in current.items():
                # print(value)
                for k in range(len(value)):
                    result[key].append(value[k])
        band_array = pd.DataFrame.from_dict(dict(result))
        t_length = 'Concantenated temporal data, all trials'

    elif p_d['ad'] == 'part_concat':
        mba = []
        for i in range(p_d['st_ed'][0], p_d['st_ed'][1]):
            t_data = dfs[t_name + '_' + str(i)]
            band_array, smp_len = calc_band_pwr(t_data, p_d['bd_range'])
            if p_d['lr'] == 'l':
                band_array['trial_condition'] = [cond[i-1]]*smp_len
            elif p_d['lr'] == 'r':
                if cond[i-1] == 2:
                    band_array['trial_condition'] = [3]*smp_len
                elif cond[i-1] == 3:
                    band_array['trial_condition'] = [2]*smp_len
                else: 
                    band_array['trial_condition'] = [cond[i-1]]*smp_len
            band_array['session'] = [p_d['sess']]*smp_len
            mba.append(band_array)
        result = defaultdict(list)
        for j in range(len(mba)):
            current = mba[j]
            for key, value in current.items():
                for k in range(len(value)):
                    result[key].append(value[k])
        band_array = pd.DataFrame.from_dict(dict(result))
        t_length = 'Concantenated data, 4 trials, 1 trial of each type.'
   
    elif p_d['ad'] == 'pick2':
        lab = [2,3]
        cond = np.array(cond)
        indlab1 = np.where(cond == lab[0])
        indlab2 = np.where(cond == lab[1])
        indlab = np.sort(np.concatenate([indlab1, indlab2], axis = None))
        mba = []
        for i in indlab:
            t_data = dfs[t_name + '_' + str(i+1)]
            band_array, smp_len = calc_band_pwr(t_data, p_d['bd_range'])
            if p_d['lr'] == 'l':
                band_array['trial_condition'] = [cond[i]]*smp_len
            elif p_d['lr'] == 'r':
                if cond[i] == 2:
                    band_array['trial_condition'] = [3]*smp_len
                elif cond[i] == 3:
                    band_array['trial_condition'] = [2]*smp_len
                else: 
                    band_array['trial_condition'] = [cond[i]]*smp_len
            band_array['session'] = [p_d['sess']]*smp_len
            mba.append(band_array)
        result = defaultdict(list)
        for j in range(len(mba)):
            current = mba[j]
            for key, value in current.items():
                for k in range(len(value)):
                    result[key].append(value[k])
        band_array = pd.DataFrame.from_dict(dict(result))
        t_length = 'Concantenated data, all trials of 2 trial types.'   
                  
    elif p_d['ad'] == 'lr_concat':
        ### Make left dataset
        mba1 = []
        for i in range(1,13):
            t_data = dfs['data_l' + '_' + str(i)]
            band_array, smp_len = calc_band_pwr(t_data, p_d['bd_range'])
            if p_d['lr'] == 'l':
                band_array['trial_condition'] = [cond[i-1]]*smp_len
            elif p_d['lr'] == 'r':
                if cond[i-1] == 2:
                    band_array['trial_condition'] = [3]*smp_len
                elif cond[i-1] == 3:
                    band_array['trial_condition'] = [2]*smp_len
                else: 
                    band_array['trial_condition'] = [cond[i-1]]*smp_len
            band_array['left or right'] = ['l']*smp_len
            band_array['session'] = p_d['sess']*smp_len
            mba1.append(band_array)
        result1 = defaultdict(list)        
        for j in range(len(mba1)):
            current = mba1[j]
            for key, value in current.items():
                for k in range(len(value)):
                    result1[key].append(value[k])
        band_array1 = pd.DataFrame.from_dict(dict(result1))
        ### Make right dataset
        mba2 = []    
        for i in range(1,13):
            t_data = dfs['data_r' + '_' + str(i)]
            band_array, smp_len = calc_band_pwr(t_data, p_d['bd_range'])
            if p_d['lr'] == 'l':
                band_array['trial_condition'] = [cond[i-1]]*smp_len
            elif p_d['lr'] == 'r':
                if cond[i-1] == 2:
                    band_array['trial_condition'] = [3]*smp_len
                elif cond[i-1] == 3:
                    band_array['trial_condition'] = [2]*smp_len
                else: 
                    band_array['trial_condition'] = [cond[i-1]]*smp_len
            band_array['left or right'] = ['r']*smp_len
            band_array['session'] = [p_d['sess']]*smp_len
            mba2.append(band_array)
        result2 = defaultdict(list)
        for jj in range(len(mba2)):
            current = mba2[jj]
            for key, value in current.items():
                for k in range(len(value)):
                    result2[key].append(value[k])
        band_array2 = pd.DataFrame.from_dict(dict(result2))
        ### Combine them
        band_array = pd.concat([band_array1, band_array2], ignore_index = True)
        t_length = 'Concantenated left and right data, all trials'
        
    elif p_d['ad'] == 'fake':
        mba = []
        for i in range(1,13):
            t_data = dfs['data_r_' + str(i)]
            band_array, smp_len = calc_band_pwr(t_data, p_d['bd_range'])
            if p_d['lr'] == 'l':
                band_array['trial_condition'] = [cond[i-1]]*smp_len
            elif p_d['lr'] == 'r':
                if cond[i-1] == 2:
                    band_array['trial_condition'] = [3]*smp_len
                elif cond[i-1] == 3:
                    band_array['trial_condition'] = [2]*smp_len
                else: 
                    band_array['trial_condition'] = [cond[i-1]]*smp_len
            band_array['session'] = [p_d['sess']]*smp_len
            mba.append(band_array)
        result = defaultdict(list)
        for j in range(len(mba)):
            current = mba[j]
            for key, value in current.items():
                for k in range(len(value)):
                    result[key].append(value[k])
        band_array = pd.DataFrame.from_dict(dict(result))
        t_length = 'Fake data for validation, equivalent to concat in length.'
   
    else: 
        raise Exception('Not a valid option. Please choose single_trial, part_trial, concat, part_concat or lr_concat.')
    
    return band_array, t_length


#%% Function to apply mapper and save all parameters
def apply_mapper(trial_data, param_dict):
    ### Choose projection function
    if param_dict['pf'] == 'tsne':
        proj = TSNE(n_components=2) #works for dim 2, 3 makes weird graphs
    elif param_dict['pf'] == 'pca':
        proj = PCA(n_components=2, random_state=12)
    elif param_dict['pf'] == 'umap':
        proj = umap.UMAP(n_components=2, random_state = 12)
    elif param_dict['pf'] == 'combo':
        proj = [PCA(n_components=16, random_state=12),
                    umap.UMAP(n_components=2, random_state = 12)]
    else: 
        print('Option not available.')
    ### Choose clustering function
    if param_dict['cf'] == 'dbscan':
        clus = DBSCAN(eps=1, min_samples=2)
    else: 
        print('Option not available.')
    cov = km.Cover(param_dict['nc'], param_dict['po'])
    ### Apply mapper
    mapper = km.KeplerMapper(verbose=2)
    dim = trial_data.shape[1]-3
    print(dim)
    
    data_4_mapper = trial_data.iloc[:, 0:dim].to_numpy()
    projected_data = mapper.fit_transform(data_4_mapper, projection=proj)
    ### Create dictionary called 'graph' with nodes, edges and meta-information
    graph = mapper.map(projected_data, clusterer=clus, cover=cov)    
    
    ### Visualize it with Mapper
    col_val = trial_data['trial_condition']
    col_val = list(range(len(trial_data['trial_condition'])))
    fl_name = str(param_dict['dyad']) + '_' + str(param_dict['lr']) + '_' + param_dict['bd'] + '_' + param_dict['cf'] + '_' + str(param_dict['nc']) + '_' + str(int(param_dict['po']*100))
    mapper.visualize(graph, 
                     title = 'Dyad: ' + str(param_dict['dyad']) + ' Subject ' + param_dict['lr'], 
                     path_html = 'Figures/mapper_time_graphs/' + fl_name + ".html",
                     color_values = col_val,
                     color_function_name = 'Tapping Trial Type')
    ### Save trial info
    graph_data = 'Figures/mapper_pkl_files/' + fl_name + '.pkl'
    with open(graph_data, 'wb') as f:
        pickle.dump(graph, f)
    return graph_data

#%% Function to make the graph with the pie chart visualization
def make_networkx_graph(graph_data):
    with open(graph_data, 'rb') as f:
        loaded_graph = pickle.load(f)
    nodes_mapper = loaded_graph['nodes'].keys()  
    edges_mapper = []
    for key in loaded_graph['links']:
        stop_list = (loaded_graph['links'])[key]
        for x in stop_list:
            edges_mapper.append((key, x))       
    ### Build the graph in networkx
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(nodes_mapper)
    nx_graph.add_edges_from(edges_mapper)
    return loaded_graph, nx_graph



#%% Plot the synchronize/syncopated graph (from Italo)
def make_pie_graph(nx_graph, tdt, loaded_graph, fsize, piesize, param_dict):
    fig=plt.figure(figsize=fsize)
    ax1 =  fig.add_axes([0, 0, 2, 2])
    ### Works with UMAP
    posi = nx.spring_layout(nx_graph, seed=29)
    ### Works with TSNE
    # posi = nx.kamada_kawai_layout(nx_graph)
    trans=ax1.transData.transform # Change from your data coord. to display coord.
    trans2=fig.transFigure.inverted().transform # Goes from display to data coord.
    nx.draw_networkx(nx_graph, pos=posi, with_labels=False, ax = ax1, node_size=piesize)
    plt.title('Synchro/Synco -- '  
              # + ', Trial Length: ' + str(param_dict['tl']) 
              + 'Band: ' + str(param_dict['bd']) 
              + ', Cubes: ' + str(param_dict['nc']) 
              + ', Overlap: ' + str(param_dict['po']) 
              + ', Q-mod: ' + str(np.round(param_dict['qmod'],3)), 
              loc = 'left', fontsize =30)
    if param_dict['labels'] == 'ss':
    ### Plot legend (synchro vs synco)
        pi_patch = mpatches.Patch(color='pink', label='Synchronized')
        bl_patch = mpatches.Patch(color='mediumblue', label='Syncopated')
        plt.legend(handles=[pi_patch, bl_patch], fontsize = 30)
    elif param_dict['labels'] == 'four':
    ### Plot legend (four trials)
        rd_patch = mpatches.Patch(color='red', label='No lead')
        or_patch = mpatches.Patch(color='orange', label='Lead')
        gy_patch = mpatches.Patch(color='grey', label='Follow')
        bl_patch = mpatches.Patch(color='blue', label='Bidirectional')
        plt.legend(handles=[rd_patch, or_patch, gy_patch, bl_patch], fontsize = 30)
    # elif ss_or_four == 'time':
    #     plt.legend()
    ### Plot the pie charts at each node
    # Get the size of the cluster with the most elements
    big_node = max(len(loaded_graph['nodes'].get(key)) for key in nx_graph)
    cond_dict = tdt['trial_condition']
    cond_np = cond_dict.to_numpy()
    sess_dict = tdt['session']
    sess_np = sess_dict.to_numpy()
    # List of edges
    connected_nodes = list(nx_graph.edges)
    # Flatten the list of tuples into a single list
    flat_con_nodes = [item for sublist in connected_nodes for item in sublist]
    # Get unique elements using set()
    edge_elements = list(set(flat_con_nodes))
    ### Plotting the pies, if unconnected to other pies, plot at the bottom
    count = 0
    for key in nx_graph:
        if key in edge_elements:
            xx, yy = trans((posi[key][0], posi[key][1]))
        else:
            count = count + 1
            x_coord = -1.2 + (count*2.5*piesize)
            xx,yy=trans((x_coord,-0.7))
        xa,ya=trans2((xx,yy)) # axes coordinates (change back to data coord.)
        nd_size = len(loaded_graph['nodes'].get(key))/big_node
        p1 = piesize*nd_size
        a = plt.axes([xa-p1/2, ya-p1/2, p1, p1])
        a.set_aspect('equal')
        current_cluster_pts = loaded_graph['nodes'].get(key)
        current_labels = []

        ### Label based off of synchro vs synco
        if param_dict['labels'] == 'ss':
            for thing in current_cluster_pts:
                current_labels.append(sess_np[thing])
            c1 = sum(1 for x in current_labels if x==1)
            c2 = sum(1 for x in current_labels if x==2)
            fracs = [c1, c2]
            a.pie(fracs,  
                  colors = ['pink', 'mediumblue'])
        elif param_dict['labels'] == 'four':
            for thing in current_cluster_pts:
                current_labels.append(cond_np[thing])
            c1 = sum(1 for x in current_labels if x==1 )
            c2 = sum(1 for x in current_labels if x==2 )
            c3 = sum(1 for x in current_labels if x==3 )
            c4 = sum(1 for x in current_labels if x==4 )
            fracs = [c1, c2, c3, c4]
            a.pie(fracs,  
                  colors = ['red', 'orange', 'grey', 'blue'])
        # elif ss_or_four == 'time':
        

    return plt.show()

#%% Function to calculate Qmod
def calc_qmod(loaded_graph, nx_graph, trial_data, p_d):
    A = nx.adjacency_matrix(nx_graph).toarray()
    m = len(nx_graph.edges())
    I = nx.incidence_matrix(nx_graph).toarray()
    # nodes = list(nx.nodes(nx_graph))
    
    community_labels = {}
    cl = []
    if p_d['labels'] == 'four':
        label_np = trial_data['trial_condition'].to_numpy()
    elif p_d['labels'] == 'ss':
        label_np = trial_data['session'].to_numpy()
        
    for node in nx_graph:
        current_cluster_pts = loaded_graph['nodes'].get(node)
        current_labels = []
        ### Label based off of task type
        if p_d['labels'] == 'four':
            for timestep in current_cluster_pts:
                current_labels.append(label_np[timestep])
            c1 = sum(1 for x in current_labels if x==1 )
            c2 = sum(1 for x in current_labels if x==2 )
            c3 = sum(1 for x in current_labels if x==3 )
            c4 = sum(1 for x in current_labels if x==4 )
            fracs = [c1, c2, c3, c4]
        
        elif p_d['labels']  == 'ss':
            for timestep in current_cluster_pts:
                current_labels.append(label_np[timestep])
            c1 = sum(1 for x in current_labels if x==1 )
            c2 = sum(1 for x in current_labels if x==2 )
            fracs = [c1, c2]  
            
        m_list = []
        for l, x in enumerate(fracs):
            if x == max(fracs):
                m_list.append(l+1)
        if len(m_list) == 1:
            community_labels[node]=m_list[0]
            cl.append(m_list[0])
        else:
            community_labels[node]=random.choice(m_list)
            cl.append(random.choice(m_list))
### Note, I went with random for now. We could probably choose the community
### based on the membership of it's neighbors, but that seems more complicated, 
### so for now, random it is.
    qmod = []
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            Ci = cl[i]
            Cj = cl[j]
            if Ci == Cj and m != 0:
                Aij = A[i,j]
                ki = np.sum(I[[i],:])
                kj = np.sum(I[[j],:])
                qmod.append(1/(2*m)*(Aij - (ki*kj/(2*m))))
            elif Ci != Cj and m!=0:
                qmod.append(0)
    sqmod = sum(qmod)
     
    
    # Initialize modularity
    modularity = 0.0
    
    # Iterate over all pairs of nodes
    for i in nx_graph.nodes():
        for j in nx_graph.nodes():
            # Check if nodes i and j are in the same community
            if community_labels[i] == community_labels[j]:
                # Adjacency value (1 if connected, 0 otherwise)
                A_ij = 1 if nx_graph.has_edge(i, j) else 0
                
                # Degree of nodes i and j
                k_i = nx_graph.degree(i)
                k_j = nx_graph.degree(j)
                
                # Modularity contribution
                modularity += (A_ij - (k_i * k_j) / (2 * m))
    
    # Normalize by 1 / (2 * m)
    modularity /= (2 * m)
    # return modularity, sqmod
    return modularity


#%% Isolate nodes of subgraph
def iso_nodes(rng, len_vec, loaded_graph, nx_graph):
    ### Get subnodes
    sub_nodes = {}
    for key_n in loaded_graph['nodes'].keys():
        cur_node = loaded_graph['nodes'].get(key_n)
        mem_list=[]
        for mem in cur_node:
            for r in rng:
                if int(mem) >= len_vec[r[0]] and mem <= len_vec[r[1]]:
                    mem_list.append(mem)
        if len(mem_list) == 0:
            pass
        else:
            sub_nodes[key_n] = mem_list
     ### Get subedges       
    sub_edges = []
    # print(list(nx_graph.edges))
    # print(loaded_graph['links'])
    for edge in list(nx_graph.edges):
        # print(cn)
        if edge[0] in sub_nodes and edge[1] in sub_nodes:
            sub_edges.append(edge)
    return sub_nodes, sub_edges




#%% Plot the subgraphs (from Italo)
def make_sub_pie_graph(nx_graph, tdt, loaded_graph, fsize, piesize, p_d, scale_x, scale_y, sub_graph, len_vec, rng, sub_edges):
    fig=plt.figure(figsize=fsize)
    ax=plt.axes()
    ax.set_aspect('equal')
    ### Plot the graph edges
    posi = nx.kamada_kawai_layout(nx_graph)
    trans=ax.transData.transform # Change from your data coord. to display coord.
    trans2=fig.transFigure.inverted().transform # Goes from display to data coord.
    nx.draw_networkx_edges(nx_graph, edgelist = sub_edges, node_size=piesize, pos=posi)
    if p_d['sess'] == 1:
        ts = 'Synchronized'
    elif p_d['sess'] == 2:
        ts = 'Syncopated'
    plt.title(ts + ' ' + p_d['lr'] +' ' + 'Trials ' + str(rng), loc = 'left')
    
    rd_patch = mpatches.Patch(color='red', label='No lead')
    or_patch = mpatches.Patch(color='orange', label='L lead')
    gy_patch = mpatches.Patch(color='grey', label='R lead')
    bl_patch = mpatches.Patch(color='blue', label='Bidirectional')

    plt.legend(handles=[rd_patch, or_patch, gy_patch, bl_patch], 
                fontsize = 14)
    ### Plot the pie charts at each node
    # Get the size of the cluster with the most elements
    big_node = max(len(loaded_graph['nodes'].get(key)) for key in nx_graph)
    cond_dict = tdt['trial_condition']
    cond_np = cond_dict.to_numpy()
    # List of edges
    connected_nodes = list(nx_graph.edges)
    # Flatten the list of tuples into a single list
    flat_con_nodes = [item for sublist in connected_nodes for item in sublist]
    # Get unique elements using set()
    edge_elements = list(set(flat_con_nodes))
    ### Plotting the pies, if unconnected to other pies, plot at the bottom
    # big_sub_node = max(len(sub_graph[key]) for key in sub_graph)
    count = 0
    for key in sub_graph:
        if key in edge_elements:
            xx, yy = trans((posi[key][0]*scale_x, posi[key][1]*scale_y))
            # xx,yy=trans(posi[key]) # figure coordinates (display coord.)
        else:
            count = count + 1
            x_coord = -0.7 + (count*2.5*piesize)
            xx,yy=trans((x_coord,-1.1))
        xa,ya=trans2((xx,yy)) # axes coordinates (change back to data coord.)
        nd_size = len(sub_graph[key])/big_node
        p1 = piesize*nd_size
        a = plt.axes([xa-p1/2, ya-p1/2, p1, p1])
        a.set_aspect('equal')
        current_cluster_pts = sub_graph[key]
        current_labels = []
        for thing in current_cluster_pts:
            current_labels.append(cond_np[thing])
        c1 = sum(1 for x in current_labels if x==1 )
        c2 = sum(1 for x in current_labels if x==2 )
        c3 = sum(1 for x in current_labels if x==3 )
        c4 = sum(1 for x in current_labels if x==4 )
        fracs = [c1, c2, c3, c4]
        a.pie(fracs,  
              colors = ['red', 'orange', 'grey', 'blue'])
    return plt.show()



#%% Plot the projection of the data
def proj_plot(trial_data, data, reduction, n_comp_and_plot_dim, t_type, lr):
    if reduction == 'pca':
        red_alg = PCA(n_components=n_comp_and_plot_dim, 
                  random_state=12)
    elif reduction == 'tsne':
        red_alg = TSNE(n_components=n_comp_and_plot_dim, 
                   learning_rate='auto',
                   init='random', 
                   perplexity=3,
                   random_state = 12)
    elif reduction == 'umap':
        red_alg = umap.UMAP(n_components=n_comp_and_plot_dim, 
                   random_state = 12)
    trial_data_embed = red_alg.fit_transform(data)
    label_vec = trial_data.iloc[:, 33].to_numpy()
    if n_comp_and_plot_dim == 2:
        fig, ax = plt.subplots()
        scatter = ax.scatter(trial_data_embed[:,0], 
                             trial_data_embed[:,1], 
                             c = label_vec)
    elif n_comp_and_plot_dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection = '3d')
        scatter = ax.scatter(trial_data_embed[:,0], 
                             trial_data_embed[:,1], 
                             trial_data_embed[:,2], 
                             c = label_vec)
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="lower left", 
                        title="Classes")
    ax.add_artist(legend1)
    plt.title(reduction + ' for ' + t_type + ' ' + lr)
    return plt.show()

#%% Function for comparing clustering algorithms
def calc_errors(t_data, true_labels, new_labels):
    ### F1-scores (note F1 micro calculates accuracy)
    f1_micro = f1_score(true_labels, new_labels, average = 'micro')
    f1_macro = f1_score(true_labels, new_labels, average = 'macro')
    f1_weight = f1_score(true_labels, new_labels, average = 'weighted')
    ### Silhouette score
    try:
        sil_score = silhouette_score(t_data, new_labels)
    except:
        sil_score = 500
    ### Log loss
    l_loss = log_loss(true_labels, new_labels)
    ### Davies-Bouldin score
    try:
        db_score = davies_bouldin_score(t_data, new_labels)
    except:
        db_score = 500
    return [f1_micro, f1_macro, f1_weight, sil_score, l_loss, db_score]











