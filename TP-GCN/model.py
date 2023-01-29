import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Layer(nn.Module):
    """
        Node embedding dimension transform
    """
    def __init__(self, input_size, out_size):
        super(Layer, self).__init__()
        self.input_size = input_size
        self.out_size = out_size
        self.weight = nn.Parameter(torch.FloatTensor(out_size, self.input_size) ) # Initialize the weight parameter
        self.init_params()

    def init_params(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def forward(self, node_data):
        output=self.weight.mm(node_data.t()).t()
        output=output.squeeze(dim=0)

        return output

class Model(nn.Module):
    """
        temporal propagation
    """
    def __init__(self, input_size, out_size):
        super(Model, self).__init__()
        self.input_size = input_size
        self.out_size = out_size
        setattr(self, 'layer',Layer(input_size, out_size))

    def forward(self, raw_features,dict_edge,dict_degree):
        feat_data=self.get_nodes_time_embedding(raw_features,dict_edge,dict_degree) # Get the feature of each node after aggregation
        layer = getattr(self, 'layer')
        terminal_embedding=[]
        for item in feat_data:
            item=item.unsqueeze(dim=0)
            node_embedding = layer(item)
            terminal_embedding.append(node_embedding.data.numpy().tolist())
        terminal_embedding=torch.tanh(torch.FloatTensor(np.asarray(terminal_embedding,dtype=np.float64)))
        return terminal_embedding

    def get_nodes_time_embedding(self, self_feats, dict_edge):
        all_time=0
        feat_dict = {}
        for node, feat in self_feats.items():
            feat_dict[node]=feat.clone()
        # Each node is aggregated in edge order
        for time, edge in dict_edge.items():
            all_time = (time+1)
        for time, edges in dict_edge.items():
            for edge in edges:
                source = int(edge[0].data)
                target = int(edge[1].data)
                source_feat=self_feats[source].clone()
                source_feat=F.normalize(source_feat)
                self_feats[target] +=source_feat
                result = source_feat.data
                result[:,[-1]] = ((time + 1) / all_time) # relative temporal information embedding
                feat_dict[target] += result  # Update node feature

        feat_list = []
        for node, feat in feat_dict.items(): #Pooling
            feat=feat.squeeze(dim=0)
            result = feat.data.numpy()
            feat_list.append(result)

        feat_list = torch.FloatTensor(np.asarray(feat_list, dtype=np.float64))
        return feat_list