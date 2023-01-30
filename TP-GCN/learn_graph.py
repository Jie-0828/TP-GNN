# In[1]
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import *



def evaluate(file,dict_test_file,model_time, classification,time_model, edge, device, hidden_size):
    """
    Test the performance of the model
    Parameters:
        datacenter: datacenter object
        graphSage：Well trained model object
        classification: Well trained classificator object
    """
    feature_data =dict_test_file[file]['_feats']
    labels_test =dict_test_file[file]['_labels']
    edge_test =dict_test_file[file]['_edge']


    # graph label
    graph_label = torch.tensor(np.asarray(labels_test))

    edge_aggregator = edge
    models = [model_time, classification,time_model]
    params = []
    for model in models:
        for param in model.parameters():
            if param.requires_grad:
                param.requires_grad = False
                params.append(param)

    nodes_embedding = model_time(feature_data, edge_test)

    # Convert to edge embedding
    edge_embeds = []
    for key, edges in edge_test.items():
        for value in edges:
            if edge_aggregator == 'cat':
                edge_embed = torch.cat(
                    (nodes_embedding[value[0]].unsqueeze(dim=0), nodes_embedding[value[1]].unsqueeze(dim=0)), dim=1)
            elif edge_aggregator == 'mean':
                edge_embed = (nodes_embedding[value[0]] + nodes_embedding[value[1]]) / 2  # 依次得到每条时序边的嵌入
            elif edge_aggregator == 'had':
                edge_embed = nodes_embedding[value[0]].mul(nodes_embedding[value[1]])  # had
            elif edge_aggregator == 'w1':
                edge_embed = torch.abs(nodes_embedding[value[0]] - nodes_embedding[value[1]])  # w1
            elif edge_aggregator == 'w2':
                edge_embed = torch.abs(nodes_embedding[value[0]] - nodes_embedding[value[1]]).mul(
                    torch.abs(nodes_embedding[value[0]] - nodes_embedding[value[1]]))  # w2
            elif edge_aggregator == 'activate':
                edge_embed = torch.cat(
                    (nodes_embedding[value[0]].unsqueeze(dim=0), nodes_embedding[value[1]].unsqueeze(dim=0)), dim=1)
                edge_embed = F.relu(edge_embed)
            edge_embed = edge_embed.data.squeeze()
            edge_embed = edge_embed.data.numpy().tolist()
            edge_embeds.append(edge_embed)
    # Feed each edge in turn to the RNN
    edge_embeds = np.asarray(edge_embeds)
    edge_embeds = torch.tensor(edge_embeds, dtype=torch.float32)
    edge_embeds = edge_embeds.unsqueeze(dim=0)
    hidden_prev = torch.zeros(1, 1, hidden_size)
    output, h_n = time_model(edge_embeds, hidden_prev)

    predicts=[]
    input = h_n[0][0]
    input = input.unsqueeze(dim=0)
    logists = classification(input)
    _, predicts_test = torch.max(logists, 1)
    predicts.append(predicts_test.data.numpy()[0])

    for param in params:
        param.requires_grad = True

    return graph_label,predicts,_.item()

def apply_model(list_file,dict_train_file, model_time, classification,optimizer,time_model, edge,hidden_size, device):
    models = [model_time, classification,time_model]
    params = []
    for model in models:
        for param in model.parameters():
            if param.requires_grad:
                params.append(param)

    optimizer.zero_grad()
    for model in models:
        model.zero_grad()

    loss = 0
    edge_aggregator = edge

    for file in list_file:
        labels = dict_train_file[file]['_labels']
        feature_data = dict_train_file[file]['_feats']
        dict_edge = dict_train_file[file]['_edge']

        nodes_embedding=model_time(feature_data,dict_edge)

        #graph label
        graph_label=np.asarray(labels)

        # Convert to edge embedding
        edge_embeds = []
        for key, edges in dict_edge.items():
            for value in edges:
                if edge_aggregator == 'cat':
                    edge_embed = torch.cat(
                        (nodes_embedding[value[0]].unsqueeze(dim=0), nodes_embedding[value[1]].unsqueeze(dim=0)), dim=1)
                elif edge_aggregator == 'mean':
                    edge_embed = (nodes_embedding[value[0]] + nodes_embedding[value[1]]) / 2
                elif edge_aggregator == 'had':
                    edge_embed = nodes_embedding[value[0]].mul(nodes_embedding[value[1]])
                elif edge_aggregator == 'w1':
                    edge_embed = torch.abs(nodes_embedding[value[0]] - nodes_embedding[value[1]])
                elif edge_aggregator == 'w2':
                    edge_embed = torch.abs(nodes_embedding[value[0]] - nodes_embedding[value[1]]).mul(
                        torch.abs(nodes_embedding[value[0]] - nodes_embedding[value[1]]))
                elif edge_aggregator == 'activate':
                    edge_embed = torch.cat(
                        (nodes_embedding[value[0]].unsqueeze(dim=0), nodes_embedding[value[1]].unsqueeze(dim=0)), dim=1)
                    edge_embed = F.relu(edge_embed)
                edge_embed = edge_embed.data.squeeze()
                edge_embed = edge_embed.data.numpy().tolist()
                edge_embeds.append(edge_embed)
        # Feed each edge in turn to the RNN
        edge_embeds = np.asarray(edge_embeds)
        edge_embeds = torch.tensor(edge_embeds, dtype=torch.float32)
        edge_embeds = edge_embeds.unsqueeze(dim=0)
        hidden_prev = torch.zeros(1, 1, hidden_size)
        output, h_n = time_model(edge_embeds, hidden_prev)

        input = h_n[0][0] # graph embedding
        input=input.unsqueeze(dim=0)
        logists = classification(input)
        loss_sup = -torch.sum(logists[range(logists.size(0)), graph_label], 0)
        loss += loss_sup
    loss.backward()


    # Update the parameters of the model and classifier
    for model in models:
        nn.utils.clip_grad_norm_(model.parameters(), 5)
    optimizer.step()

    optimizer.zero_grad()
    for model in models:
        model.zero_grad()
    return model_time, classification, loss,time_model

