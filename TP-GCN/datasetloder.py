import os
import random
import numpy as np
import torch
from sklearn.utils import shuffle


def get_file_path(root_path, file_list, dir_list):
    # 获取该目录下所有的文件名称和目录名称
    dir_or_files = os.listdir(root_path)
    for dir_file in dir_or_files:
        # 获取目录或者文件的路径
        dir_file_path = os.path.join(root_path, dir_file)
        # 判断该路径为文件还是路径
        if os.path.isdir(dir_file_path):
            dir_list.append(dir_file_path)
            # 递归获取所有文件和目录的路径q
            get_file_path(dir_file_path, file_list, dir_list)
        else:
            file_list.append(dir_file_path)



def load_data_graph(path_positive,path_negative,divide):
    list_train = []  # store pathes of all training files
    list_test = []  # store pathes of all test files

    # positive sample
    list_positive=[]
    dir_positive=[]
    get_file_path(path_positive,list_positive,dir_positive)
    list_positive=shuffle(list_positive)
    for i in list_positive[:int(len(list_positive)*divide)]:
        list_train.append(i)
    for j in list_positive[int(len(list_positive)*divide):]:
        list_test.append(j)

    # negtive sample
    list_negative=[]
    dir_negative=[]
    get_file_path(path_negative,list_negative,dir_negative)
    list_negative=shuffle(list_negative)
    for i in list_negative[:int(len(list_negative) * divide)]:
        list_train.append(i)
    for j in list_negative[int(len(list_negative) * divide):]:
        list_test.append(j)

    return list_train,list_test


class DataCenter(object):
    """ Loading Dataset
            Parameter:
            file_paths:{file_path1,file_path2}
    """

    def __init__(self, list_file):
        """file_paths:{name:root,...,}"""
        super(DataCenter, self).__init__()
        self.list_file = list_file

    def load_Dataset(self):
        dict_file={}
        for file in self.list_file:
            feat_dict= {}  # Store each node feature vector
            label_list = []  # Store label
            node_map = {}
            dict_edge={} # Store edge information
            count = 0  # Node number

            with open(file) as f:
                id = f.readline().replace("\n", "")
                list_edge = []
                for i in f:
                    if i == 'network[son<-parent]=\n':
                        break
                    else:
                        label = i.split('=')[1] # get label
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
                    one_feat_list.append('0') # Add initial timing features
                    node=id + sample_clean[0] # Map a node name to a node number
                    one_feat_list=torch.FloatTensor(np.asarray([one_feat_list],dtype=np.float64))
                    feat_dict[count]=(one_feat_list)
                    node_map[node.strip()] = count
                    count += 1
                if label == 'Anomaly':
                    label_list.append(0)
                else:
                    label_list.append(1)
                    # print(label_list)

            for edge in list_edge: # get edge information
                pair = edge.split(",")[0]
                time = edge.strip('\n').split(",")[1]
                left = node_map[id + pair.strip().split("<-")[1]]
                right = node_map[id + pair.strip().split("<-")[0]]
                if int(time) not in dict_edge:
                    dict_edge[int(time)] = [[left, right]]
                else:
                    dict_edge[int(time)].append([left, right])

            label_list = np.asarray(label_list, dtype=np.int64)
            for key,value in dict_edge.items():
                random.shuffle(value)
                dict_edge[key]=torch.tensor(np.asarray(value))


            dict_one_file={}
            dict_one_file['_feats']=feat_dict
            dict_one_file['_labels']= label_list
            dict_one_file['_edge']= dict_edge
            dict_file[file]=dict_one_file
        return dict_file


def load_data_node(path,divide):  # file_path,devide
    dict_edge={}
    labels=[]
    dict_feature={}
    with open(path,'r+') as fd:
        fd.readline()
        for line in fd:
            line=line.strip('\n').split(',')
            user_id=int(line[0])
            item_id=int(line[1])
            timestampe=float(line[2])
            label=int(line[3])
            edge_feature=line[4:]
            edge_feature=0.5*np.asarray(edge_feature,dtype=np.float64)
            if user_id not in dict_feature:
                dict_feature[user_id]=edge_feature
            else:
                dict_feature[user_id]+= edge_feature
            if item_id not in dict_feature:
                dict_feature[item_id]=edge_feature
            else:
                dict_feature[item_id]+= edge_feature
            if timestampe not in dict_edge:
                dict_edge[timestampe] = [[user_id, item_id]]
            else:
                dict_edge[timestampe].append([user_id, item_id])
            if label==1:
                labels.append(0)
            else:
                labels.append(1)
    for key, value in dict_edge.items():
        value=shuffle(value)
        dict_edge[key] = torch.tensor(np.asarray(value))

    features=[]
    for node, feat in dict_feature.items():
        feat=list(feat)
        feat.append(0)  # Add initial timing features
        features.append(feat)
    features=torch.tensor(np.asarray(features,dtype=np.float32))
    # features=F.normalize(features)


    train_index = np.array(getRandomIndex(len(labels), int(len(labels)*divide))) # select training nodes randomly
    test_index= np.delete(np.arange(len(labels)), train_index)
    return dict_edge, labels, features,train_index,test_index

def getRandomIndex(n, x):
    index = np.random.choice(np.arange(n), size=x, replace=False)
    return index
