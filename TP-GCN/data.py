import numpy as np
import torch
import random

def load_Dataset(file_list):
    dict_file={}
    for file in file_list:
        with open(file) as f:
            # print(file)
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
            timestamed = edge.split(",")[1]
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
        dict_file[id]=[edge_order,feat_node,label_list]
    return dict_file
