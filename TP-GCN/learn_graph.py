# In[1]
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import *
from tqdm import tqdm
from torch_geometric.loader import DataLoader


def evaluate(test_loader,model_time, classification,time_model, edge_agg, device, hidden_size):
    """
    Test the performance of the model
    Parameters:
        datacenter: datacenter object
        graphSage：Well trained model object
        classification: Well trained classificator object
    """
    models = [model_time, classification,time_model]
    for model in models:
        model.eval()
    with torch.no_grad():
        labels_test_all = []  # Stores test set labels
        predicts_test_all = []  # Store the model prediction results
        predicts_socre_all = []  # Store the model prediction scores

        for step, batch in enumerate(tqdm(test_loader)):
            print(batch)
            feat, labels, edges, ids, lengths = batch['feature'], batch['label'], batch['edge'], batch['id'], batch[
                'length']
            for index in range(len(feat)):  # one graph
                length_i = lengths[index]
                feature = feat[index][:length_i[0]].to(device)
                edge_test = edges[index][:length_i[1]].to(device)
                nodes_embedding = model_time(feature, edge_test)

                # Feed each edge in turn to the RNN
                edge_embeds = edge_agg_function(edge_agg, edge_test, nodes_embedding)
                hidden_prev = torch.zeros(1, 1, hidden_size)
                output, h_n = time_model(edge_embeds.unsqueeze(dim=0), hidden_prev)

                # input =nodes_embedding.mean(dim=0).unsqueeze(dim=0)
                input = h_n[0][0].unsqueeze(dim=0)
                logists = classification(input)
                _, predicts_test = torch.max(logists, 1)


                predicts_test_all.append(predicts_test.data[0].cpu())
                labels_test_all.append(labels[index][0].cpu())
                predicts_socre_all.append(_[0].cpu())

    return predicts_test_all,labels_test_all,predicts_socre_all

def train_model(loader,model_time, classification,optimizer,time_model, edge_agg ,hidden_size, device):
    models = [model_time, classification, time_model]
    params = []
    for model in models:
        for param in model.parameters():
            if param.requires_grad:
                params.append(param)

    optimizer.zero_grad()
    for model in models:
        model.zero_grad()

    # criterion = nn.CrossEntropyLoss().to(device)

    loss_all=0
    pbar = tqdm(enumerate(loader), total =len(loader))
    for step, batch in pbar:
        loss=0
        feat, labels, edges, ids,lengths = batch['feature'], batch['label'], batch['edge'], batch['id'],batch['length']
        for index in range(len(feat)):
            length_i=lengths[index]
            feature=feat[index][:length_i[0]].to(device)
            train_edge=edges[index][:length_i[1]].to(device)
            nodes_embedding=model_time(feature,train_edge)


            # Feed each edge in turn to the RNN
            edge_embeds=edge_agg_function(edge_agg,train_edge,nodes_embedding)
            hidden_prev = torch.zeros(1, 1, hidden_size)
            output, h_n = time_model(edge_embeds.unsqueeze(dim=0), hidden_prev)


            input=h_n[0][0].unsqueeze(dim=0)
            logists = classification(input)
            # loss_sup=criterion(logists,labels[index])
            loss_graph = -torch.sum(logists[range(logists.size(0)), labels[index]], 0)
            loss+=loss_graph

        loss.backward()
        loss_all+=loss


        # Update the parameters of the model and classifier
        for model in models:
            nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

    return model_time, classification, loss_all, time_model


def edge_agg_function(edge_agg,edge_list,nodes_embedding):
    # Convert to edge embedding
    edge_embeds = []
    for value in edge_list:
        if edge_agg == 'cat':
            edge_embed = torch.cat(
                (nodes_embedding[value[0]].unsqueeze(dim=0), nodes_embedding[value[1]].unsqueeze(dim=0)), dim=1)
        elif edge_agg == 'mean':
            edge_embed = (nodes_embedding[value[0]] + nodes_embedding[value[1]]) / 2
        elif edge_agg == 'had':
            edge_embed = nodes_embedding[value[0]].mul(nodes_embedding[value[1]])
        elif edge_agg == 'w1':
            edge_embed = torch.abs(nodes_embedding[value[0]] - nodes_embedding[value[1]])
        elif edge_agg == 'w2':
            edge_embed = torch.abs(nodes_embedding[value[0]] - nodes_embedding[value[1]]).mul(
                torch.abs(nodes_embedding[value[0]] - nodes_embedding[value[1]]))
        elif edge_agg == 'activate':
            edge_embed = torch.cat(
                (nodes_embedding[value[0]].unsqueeze(dim=0), nodes_embedding[value[1]].unsqueeze(dim=0)), dim=1)
            edge_embed = F.relu(edge_embed)
        edge_embed = edge_embed.cpu().detach().numpy()
        edge_embeds.append(edge_embed)
    edge_embeds = torch.from_numpy(np.asarray(edge_embeds, dtype=np.float32))
    return edge_embeds
