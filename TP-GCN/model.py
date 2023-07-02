import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import scipy.sparse as sp
from time2vec import Time2Vec

class Model(nn.Module):
    """
        temporal propagation
    """
    def __init__(self, input_size,time_dim,device,update_method):
        super(Model, self).__init__()
        self.input_size = input_size
        self.device = device
        self.time_dim=time_dim
        self.time2vec=Time2Vec('sin',time_dim).to(device)
        # self.Linear=nn.Linear(input_size+time_dim, input_size)
        self.update_method=update_method
        if self.update_method=='gru':
            self.updater=nn.GRU(input_size=input_size+time_dim, hidden_size=input_size, num_layers=1, batch_first=True)


    def forward(self, raw_features,dict_edge):
        if self.update_method == 'gru':
            feat_data= self.get_nodes_embedding_GRU(raw_features,dict_edge)# Get the feature of each node after aggregation:
        elif self.update_method == 'sum':
            feat_data= self.get_nodes_time_embedding(raw_features, dict_edge)
        elif self.update_method == 'matrix':
            feat_data = self.get_nodes_embedding_matrix(raw_features, dict_edge)
        else:
            feat_data=self.random(raw_features, dict_edge)

        output=torch.tanh(feat_data)
        return  output

    def get_nodes_embedding_GRU(self, self_feats, edge_order):
        for edge in edge_order:
            source = int(edge[0])
            target = int(edge[1])
            time= int(edge[2])
            source_feat=self_feats[source].clone().unsqueeze(dim=0)
            source_feat=F.normalize(source_feat)

            time_feature = self.time2vec(torch.Tensor([[time]]).to(self.device))

            target_feat = self_feats[target].clone().unsqueeze(dim=0)
            unpate_feat=torch.cat((source_feat,time_feature),dim=1)
            output, h_n=self.updater(unpate_feat,target_feat)
            self_feats[target] = F.normalize(h_n).squeeze(dim=0)
        return self_feats


    def get_nodes_embedding_matrix(self, self_feats, edge_order):
        dict_time = {}  # Record the time of the last interaction
        time_mx = torch.zeros(len(self_feats), self.time_dim)  # Initial temporal matrix

        for node in range(len(self_feats)):
            dict_time[node] = 0  # Initialize the interaction time

        dict_edge={}
        for edge in edge_order:
            time = int(edge[2])
            if int(time) not in dict_edge:
                dict_edge[int(time)] = [edge]
            else:
                dict_edge[int(time)].append(edge)

        for time, edges in dict_edge.items():
            a_t = torch.eye(len(self_feats))  # Define the initial adjacency matrix
            target_nodes = []
            for edge in edges:
                a_t[edge[1], edge[0]] += 1
                target_nodes.append(int(edge[1]))

            a_t = torch.tensor(normalize(a_t))
            self_feats = torch.mm(a_t, self_feats)

            for item in target_nodes:
                time_feature = self.time2vec(torch.Tensor([[time]]))
                time_feature = F.normalize(time_feature).squeeze(dim=0)
                time_mx[item] +=time_feature
                dict_time[item] = time  # Update interaction time


        feat_new = torch.cat((self_feats, time_mx), dim=1)
        return feat_new

    def get_nodes_time_embedding(self, ori_feats, edge_order):
        dict_time={}
        self_feats=ori_feats.clone()
        zeros=torch.zeros([self_feats.shape[0],self.time_dim])
        self_feats=torch.cat((self_feats,zeros),dim=1)
        new_feats = self_feats.clone()
        for node in range(len(self_feats)):
            dict_time[node] =0
        for edge in edge_order:
            source = int(edge[0])
            target = int(edge[1])
            time= int(edge[2])
            source_feat=self_feats[source].clone().unsqueeze(dim=0)
            source_feat=F.normalize(source_feat).squeeze(dim=0)
            self_feats[target] +=source_feat

            time_feature = self.time2vec(torch.Tensor([[time]]).to(self.device))
            time_feature = F.normalize(time_feature).squeeze(dim=0)

            source_feat[self.input_size:] = time_feature  # relative temporal information embedding
            new_feats[target] += source_feat  # Update node feature

        return new_feats

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
