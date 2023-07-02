import os
import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# from torch_geometric.loader import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F



class DealDataset(Dataset):
    def __init__(self,file_list,dimensionality,time_dim,updater):
        """
        :param file_list:   List of file
        """
        self.file_list=file_list
        self.dimensionality=dimensionality
        self.time_dim=time_dim
        self.updater=updater

    def __getitem__(self, index):
        file = self.file_list[index]
        with open(file) as f:
            feat_node = []  # Store each node feature vector
            label_list = []  # Store label
            node_map = {}
            dict_edge = {}  # Store edge information
            count = 0  # Node number
            id = f.readline().replace("\n", "")
            list_edge = []
            for i in f:
                if i == 'network[son<-parent]=\n':
                    break
                else:
                    label = i.split('=')[1]  # get label
                    label = label.replace("\n", "")
            for line in f:
                if line == 'nodeInfo=\n':
                    break
                list_edge.append(line)

            # get node feature
            for each_sample in f.readlines():
                if each_sample == '\n':
                    break
                sample_clean = each_sample.strip().split(':')
                node_feature = sample_clean[1]
                one_feat_list = node_feature.strip().split(' ')[:]
                node = id + sample_clean[0]  # Map a node name to a node number
                feat_node.append(one_feat_list)
                node_map[node.strip()] = count
                count += 1
            if label == 'Normal':
                label_list.append(1)
            else:
                label_list.append(0)

        for edge in list_edge:  # get edge information
            edge = edge.strip('\n')
            pair = edge.split(",")[0]
            # order = edge.split(",")[1].split(';')[0]
            timestamed = edge.split(",")[1].split(';')[1]
            left = node_map[id + pair.strip().split("<-")[1]]
            right = node_map[id + pair.strip().split("<-")[0]]
            if int(timestamed) not in dict_edge:
                dict_edge[int(timestamed)] = [[left, right, timestamed]]
            else:
                dict_edge[int(timestamed)].append([left, right, timestamed])

        edge_order = []
        for key, value in dict_edge.items():
            random.shuffle(value)
            for i in value:
                edge_order.append(i)

        edge_order = torch.tensor(np.asarray(edge_order, dtype=np.int64))
        label_list = torch.tensor(np.asarray(label_list, dtype=np.int64))
        feat_node = torch.FloatTensor(np.asarray(feat_node, dtype=np.float64))

        sample = {'feature': feat_node, 'label': label_list, 'id': id, 'edge': edge_order,
                  'length_edge': len(edge_order), 'length_feat': len(feat_node)}
        return sample

    def __len__(self):
        return len(self.file_list)

def collate_func(batch_dic):
    #from torch.nn.utils.rnn import pad_sequence
    batch_len=len(batch_dic)
    max_edge_length=max([dic['length_edge'] for dic in batch_dic])
    mask_edge_batch=torch.zeros((batch_len,max_edge_length))
    max_feat_length=max([dic['length_feat'] for dic in batch_dic])
    mask_feat_batch=torch.zeros((batch_len,max_feat_length))
    feat_batch=[]
    label_batch=[]
    id_batch=[]
    edge_batch=[]
    length_batch=[]
    for i in range(len(batch_dic)):
        dic=batch_dic[i]
        feat_batch.append(dic['feature'])
        edge_batch.append(dic['edge'])
        label_batch.append(dic['label'])
        id_batch.append(dic['id'])
        length_batch.append([dic['length_feat'],dic['length_edge']])
        mask_edge_batch[i,:dic['length_edge']]=1
        mask_feat_batch[i,:dic['length_feat']]=1
    res={}
    res['feature']=pad_sequence(feat_batch,batch_first=True)
    res['edge']=pad_sequence(edge_batch,batch_first=True)
    res['label']=label_batch
    res['id']=id_batch
    res['length']=length_batch
    res['mask_edge']=mask_edge_batch
    res['mask_feat']=mask_feat_batch
    return res

def load_data_graph(path_positive,path_negative,divide):
    # positive sample
    list_positive=[]
    dir_positive=[]
    get_file_path(path_positive,list_positive,dir_positive)
    list_positive=shuffle(list_positive)
    random_number=random.randint(0,100)
    list_train, list_test = train_test_split(list_positive, train_size=divide, random_state=random_number)

    # negtive sample
    list_negative=[]
    dir_negative=[]
    get_file_path(path_negative,list_negative,dir_negative)
    list_negative=shuffle(list_negative)
    list_train_negative ,list_test_negative = train_test_split(list_negative, train_size=divide, random_state=random_number)
    list_train+=list_train_negative
    list_test+=list_test_negative

    return list_train,list_test

def get_file_path(root_path, file_list, dir_list):
    """
        Get the names of all files and directories in the directory
    """
    dir_or_files = os.listdir(root_path)
    for dir_file in dir_or_files:
        dir_file_path = os.path.join(root_path, dir_file)
        if os.path.isdir(dir_file_path):
            dir_list.append(dir_file_path)
            get_file_path(dir_file_path, file_list, dir_list)
        else:
            file_list.append(dir_file_path)


def getRandomIndex(n, x):
    index = np.random.choice(np.arange(n), size=x, replace=False)
    return index



