import argparse
import math
import time

from sklearn import metrics

import datasetloder
from learn_graph import *
from datasetloder import *
from model import *
from classifier import *
from tqdm import tqdm

# Argument and global variables
parser = argparse.ArgumentParser('Interface for TP-GCN experiments on graph classification task')
parser.add_argument('-d', '--data', type=str, help='dataset to use, Forum-java, HDFS or Gowalla', default='HDFS')
parser.add_argument('--bs', type=int, default=32, help='batch_size')
parser.add_argument('--n_epoch', type=int, default=10, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--node_dim', type=int, default=32, help='Dimentions of the graph embedding')
parser.add_argument('--edge_agg', type=str, choices=['mean', 'had', 'w1','w2', 'activate'], help='EdgeAgg method', default='mean')
parser.add_argument('--divide', type=str,help='the ratio of training sets', default=0.3)


args = parser.parse_args()
dataset = args.data
batch_size = args.bs
epochs = args.n_epoch
learning_rate = args.lr
hidden_size = args.node_dim
edge_agg = args.edge_agg
divide=args.divide

#Random seed
seed=824
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Load dataset
if dataset == 'Forum-java':
    dimensionality = 5
    path_positive = ''  # Positive sample path
    path_negative = ''  # Negative sample path
elif dataset == 'HDFS':
    dimensionality = 4
    path_positive = r'D:\360MoveData\Users\Asus\Desktop\测试\p'
    path_negative = r'D:\360MoveData\Users\Asus\Desktop\测试\n'
elif dataset == 'Gowalla':
    dimensionality = 3
    path_positive = ''
    path_negative = ''

list_train,list_test=datasetloder.load_data_graph(path_positive,path_negative,divide) # Get the training and test files path
print('Train:', len(list_train))
print('Test:', len(list_test))


num_labels = 2  # Binary classification task
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the model and initialize it
model_time = Model(dimensionality, hidden_size,'graph')  # Create a model
rnn_model= nn.RNN(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
classification = Classification(hidden_size, num_labels) # Create a classifier

models = [model_time, classification,rnn_model]
params = []
for model in models:
    for param in model.parameters():
        if param.requires_grad:
            params.append(param)

optimizer = torch.optim.SGD(params, lr=learning_rate) # Create optimizer



# Training
print('Model with Supervised Learning')

for epoch in tqdm(range(epochs)):
    list_train = shuffle(list_train)
    dc = DataCenter(list_train) # Load training file
    dict_train_file=dc.load_Dataset()
    # print(optimizer.state_dict()['param_groups'][0]['lr'])
    time.sleep(0.0001)
    print('----------------------EPOCH %d-----------------------' % epoch)
    i = 0
    loss_all = 0
    for count in range(math.ceil(len(list_train) / batch_size)):
        if len(list_train[i:i + batch_size]) == batch_size:
            model_time, classification, loss, rnn_model = apply_model(list_train[i:i + batch_size], dict_train_file, model_time, classification, optimizer, rnn_model, edge_agg, hidden_size, device)
            loss_all += loss
        else:
            model_time, classification, loss, rnn_model = apply_model(list_train[i:], dict_train_file, model_time, classification, optimizer, rnn_model, edge_agg, hidden_size, device)
            loss_all += loss
        i += batch_size
    print('loss:', loss_all/len(list_train))

# Testing
labels_test_all=[] # Stores test set labels
predicts_test_all=[] # Store the model prediction results
predicts_socre_all=[] # Store the model prediction scores
num = 0
list_test=shuffle(list_test)
dc_test = DataCenter(list_test) # # Load test file
dict_test_file=dc_test.load_Dataset()
for count in tqdm(range(math.ceil(len(list_test) / batch_size))):
    time.sleep(0.0001)
    max_vali_f1 = 0
    if len(list_test[num:num + batch_size]) == batch_size:
        for file in list_test[num:num + batch_size]:
            labels_test, predicts_test ,score= evaluate(file, dict_test_file, model_time, classification, rnn_model, edge_agg, device, 'debug', epoch, hidden_size)
            predicts_socre_all.append(score)
            for label in labels_test:
                labels_test_all.append(label)
            for predict in predicts_test:
                predicts_test_all.append(predict)
    else:
        for file in list_test[num:]:
            labels_test, predicts_test ,score= evaluate(file, dict_test_file, model_time, classification, rnn_model, edge_agg, device, hidden_size)
            predicts_socre_all.append(score)
            for label in labels_test:
                labels_test_all.append(label)
            for predict in predicts_test:
                predicts_test_all.append(predict)
    num += batch_size

labels = np.array(labels_test_all)
scores = np.array(predicts_test_all)

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

precision = TP / (TP + FP)
recall = TP / (TP + FN)
F1 = 2 * precision * recall / (precision + recall)
with open(str(dataset)+str(divide)+'.txt', 'a+')as f:
   f.write(str(hidden_size))
   f.write('\n')
   f.write('TP:'+str(TP)+'\n')
   f.write('FP:'+str(FP)+'\n')
   f.write('FN:' + str(FN) + '\n')
   f.write('TN:' + str(TN) + '\n')
   f.write('F1:' + str(F1) + '\n')
   f.write('Precision:' + str(precision) + '\n')
   f.write('Recall:' + str(recall) + '\n')
   f.write('AUC:' + str(metrics.roc_auc_score(labels_test_all ,predicts_socre_all))+"\n")
   f.write("\n")

print('test_f1_all',F1)
print('test_p_all',precision)
print('test_r_all', recall)
print("AUC: "+str(metrics.roc_auc_score(labels_test_all ,predicts_socre_all))+"\n")

