import time

import torch.utils.data as Data
import numpy as np
import pandas as pd
import os, sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, precision_score, recall_score
from collections import defaultdict

from torch.utils.data import RandomSampler
from tqdm import tqdm


def evaluate(features,dict_edge,label_test,test_index,model_time, classification,time_model, edge, device,hidden_size):
    """
    Test the performance of the model
    Parameters:
        datacenter: datacenter object
        graphSage：Well trained model object
        classification: Well trained classificator object
    """
    predicts_score = []
    predicts = []
    labels = []


    models = [model_time, classification,time_model]
    params = []
    for model in models:
        for param in model.parameters():
            if param.requires_grad:
                param.requires_grad = False
                params.append(param)

    nodes_embedding = model_time(features, dict_edge)

    # Convert to edge embedding
    edge_embeds = []
    for key, edges in dict_edge.items():
        for value in edges:
            if edge == 'cat':
                edge_embed = torch.cat(
                    (nodes_embedding[value[0]].unsqueeze(dim=0), nodes_embedding[value[1]].unsqueeze(dim=0)), dim=1)
            elif edge == 'mean':
                edge_embed = (nodes_embedding[value[0]] + nodes_embedding[value[1]]) / 2  # 依次得到每条时序边的嵌入
            elif edge == 'had':
                edge_embed = nodes_embedding[value[0]].mul(nodes_embedding[value[1]])  # had
            elif edge == 'w1':
                edge_embed = torch.abs(nodes_embedding[value[0]] - nodes_embedding[value[1]])  # w1
            elif edge == 'w2':
                edge_embed = torch.abs(nodes_embedding[value[0]] - nodes_embedding[value[1]]).mul(
                    torch.abs(nodes_embedding[value[0]] - nodes_embedding[value[1]]))  # w2
            elif edge == 'activate':
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


    for i in test_index:
        input = torch.tensor(0.5*output[i]).unsqueeze(dim=0)
        logists = classification(input)  # get classification results
        _, predicts_test = torch.max(logists, 1)
        predicts.append(predicts_test.data.numpy()[0])
        predicts_score.append(_.data.numpy()[0])
        labels.append(label_test[i])


    for param in params:
        param.requires_grad = True

    return predicts,labels,predicts_score

def apply_model(features,dict_edge,train_index,labels,model_time,time_model,classification,optimizer,edge,device,hidden_size):
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

    nodes_embedding=model_time(features,dict_edge)

    # Convert to edge embedding
    edge_embeds = []
    for key, edges in dict_edge.items():
        for value in edges:
            if edge == 'cat':
                edge_embed = torch.cat(
                    (nodes_embedding[value[0]].unsqueeze(dim=0), nodes_embedding[value[1]].unsqueeze(dim=0)), dim=1)
            elif edge == 'mean':
                edge_embed = (nodes_embedding[value[0]] + nodes_embedding[value[1]]) / 2  # 依次得到每条时序边的嵌入
            elif edge == 'had':
                edge_embed = nodes_embedding[value[0]].mul(nodes_embedding[value[1]])  # had
            elif edge == 'w1':
                edge_embed = torch.abs(nodes_embedding[value[0]] - nodes_embedding[value[1]])  # w1
            elif edge == 'w2':
                edge_embed = torch.abs(nodes_embedding[value[0]] - nodes_embedding[value[1]]).mul(
                    torch.abs(nodes_embedding[value[0]] - nodes_embedding[value[1]]))  # w2
            elif edge == 'activate':
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

    for i in train_index:
        input=torch.tensor(0.5*output[i]).unsqueeze(dim=0)
        logists = classification(input)
        loss_sup = -torch.sum(logists[range(logists.size(0)), np.asarray([labels[i]])], 0)
        loss += loss_sup


    loss.backward()
    # Update the parameters of the model and classifier
    for model in models:
        nn.utils.clip_grad_norm_(model.parameters(), 5)
    optimizer.step()

    optimizer.zero_grad()
    for model in models:
        model.zero_grad()
    return model_time, classification,time_model,loss