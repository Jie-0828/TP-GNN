import argparse
import copy
import math
import time

import torch.optim
from sklearn import metrics

from learn_graph import *
from data import *
from model import *
from classifier import *
from time2vec import *

# Argument and global variables
parser = argparse.ArgumentParser('Interface for TP-GCN experiments on graph classification task')
parser.add_argument('-d', '--data', type=str, help='dataset to use, Forum-java, HDFS, Gowalla or Brightkite', default='Forum-java')
parser.add_argument('--bs', type=int, default=32, help='batch_size')
parser.add_argument('--n_epoch', type=int, default=1, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--node_dim', type=int, default=32, help='Dimentions of the graph embedding')
parser.add_argument('--time_dim', type=int, default=4, help='Dimentions of the time embedding')
parser.add_argument('--edge_agg', type=str, choices=['mean', 'had', 'w1','w2', 'activate'], help='EdgeAgg method', default='mean')
parser.add_argument('--train_radio', type=str,help='the ratio of training sets', default=0.3)
parser.add_argument('--updater', type=str,default='sum', help='Node feature update mode: [sum, gru]')


args = parser.parse_args()
print(args)
batch_size = args.bs
hidden_size = args.node_dim
edge_agg = args.edge_agg

#Random seed
seed=824
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Load dataset
if args.data == 'Forum-java':
    dimensionality = 3
    path_positive = r'C:\Users\Administrator\Desktop\test\p'  # Positive sample path
    path_negative = r'C:\Users\Administrator\Desktop\test\n'  # Negative sample path
elif args.data == 'HDFS':
    dimensionality = 3
    path_positive = r''
    path_negative = r''
elif args.data == 'Gowalla':
    dimensionality = 3
    path_positive = ''
    path_negative = ''
elif args.data == 'Brightkite':
    dimensionality = 3
    path_positive = ''
    path_negative = ''

list_train,list_test=load_data_graph(path_positive,path_negative,args.train_radio) # Get the training and test files path
print('Train:', len(list_train))
print('Test:', len(list_test))

num_labels = 2  # Binary classification task
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the model and initialize it
model_time = Model(dimensionality,args.time_dim,device,args.updater).to(device)  # Create a model""))
if args.updater=='gru':
    time_model= nn.GRU(input_size=dimensionality, hidden_size=hidden_size, num_layers=1, batch_first=True).to(device)
else:
    time_model= nn.GRU(input_size=dimensionality+args.time_dim, hidden_size=hidden_size, num_layers=1, batch_first=True).to(device)
classification = Classification(hidden_size, num_labels).to(device) # Create a classifier

#load datasets
train_dataset = DealDataset(list_train,dimensionality,args.time_dim,args.updater)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,collate_fn=collate_func)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True, collate_fn=collate_func)

models = [model_time, classification,time_model]
params = []
for model in models:
    for param in model.parameters():
        if param.requires_grad:
            params.append(param)

optimizer = torch.optim.Adam(params, lr=args.lr) # Create optimizer

# Training
f=open(str(args.data)+str(args.train_radio)+str(hidden_size)+'.txt', 'a+')
print('Model with Supervised Learning')
for epoch in range(args.n_epoch):
    time.sleep(0.0001)
    print('----------------------EPOCH %d-----------------------' % epoch)
    model_time, classification, loss, time_model = train_model(train_loader,model_time, classification, optimizer,
                                                              time_model, edge_agg, hidden_size, device)
    print('loss:', loss/len(list_train))
    f.write(str(epoch)+' epoch----'+str(loss.data/len(list_train))+'\n')

# Testing
print('Test Start')
test_dataset = DealDataset(list_test,dimensionality,args.time_dim,args.updater)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, shuffle=True, collate_fn=collate_func)

predicts_test_all,labels_test_all,predicts_socre_all = evaluate(test_loader,model_time, classification,time_model, edge_agg, device, hidden_size)

labels = np.array(labels_test_all)
print(labels)
scores = np.array(predicts_test_all)
print(scores)

# Write result
TP = 0
FP = 0
FN = 0
TN = 0
for k in range(0, labels.shape[0]):
    if scores[k] == 1 and labels[k] == 1:
        TP += 1
    if scores[k] == 1 and labels[k] == 0:
        FP += 1
    if scores[k] == 0 and labels[k] == 1:
        FN += 1
    if scores[k] == 0 and labels[k] == 0:
        TN += 1

f.write('TP:'+str(TP)+'\n')
f.write('FP:'+str(FP)+'\n')
f.write('FN:' + str(FN) + '\n')
f.write('TN:' + str(TN) + '\n')
precision = TP / (TP + FP)
recall = TP / (TP + FN)
F1 = 2 * precision * recall / (precision + recall)
f.write('F1:' + str(F1) + '\n')
f.write('Precision:' + str(precision) + '\n')
f.write('Recall:' + str(recall) + '\n')
f.write('AUC:' + str(metrics.roc_auc_score(labels_test_all ,predicts_socre_all))+"\n")
f.write("\n")
f.close()

print('test_f1—macro_all',str(metrics.f1_score(labels_test_all ,scores,average="macro"))+"\n")
print('test_f1—micro_all',str(metrics.f1_score(labels_test_all ,scores,average="micro"))+"\n")
print('test_f1_all',F1)
print('test_p_all',precision)
print('test_r_all', recall)
print("AUC: "+str(metrics.roc_auc_score(labels_test_all ,predicts_socre_all))+"\n")

