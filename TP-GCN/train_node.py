from sklearn import metrics

import datasetloder
from learn_node import *
from datasetloder import *
from model import *
from classifier import *
from tqdm import tqdm

# Argument and global variables
parser = argparse.ArgumentParser('Interface for TP-GCN experiments on graph classification task')
parser.add_argument('-d', '--data', type=str, help='dataset to use, MOOC, Wikipedia or Reddit', default='MOOC')
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
if dataset == 'MOOC':
    path = r''  # dataset path
elif dataset == 'Wikipedia':
    path = ''
elif dataset == 'Reddit':
    path = ''

dict_edge, labels,features,train_index,test_index=datasetloder.load_data_node(path,divide) # Get the training and test files path


num_labels = 2  # Binary classification task
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the model and initialize it
model_time = Model(features.shape[1], hidden_size,'node')  # Create a model
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
    new_feat_data=features.clone()
    print(optimizer.state_dict()['param_groups'][0]['lr'])
    time.sleep(0.0001)
    print('----------------------EPOCH %d-----------------------' % epoch)
    model_time, classification,time_model,loss = apply_model(new_feat_data,dict_edge,train_index,labels,model_time,rnn_model,classification,optimizer,edge_agg,device,hidden_size)
    print('loss:', loss/len(train_index))

# Testing
predicts, labels_, predicts_score = evaluate(features,dict_edge,labels,test_index,model_time,classification,rnn_model,edge_agg,device,hidden_size)

labels = np.array(labels_)
print(labels)
scores = np.array(predicts)
print(scores)

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

# Write result
with open(str(dataset)+str(divide)+'.txt', 'a+')as f:
   f.write(str(edge_agg)+'\n')
   f.write('TP:'+str(TP)+'\n')
   f.write('FP:'+str(FP)+'\n')
   f.write('FN:' + str(FN) + '\n')
   f.write('TN:' + str(TN) + '\n')
   f.write('F1:' + str(F1) + '\n')
   f.write('Precision:' + str(precision) + '\n')
   f.write('Recall:' + str(recall) + '\n')
   f.write('AUC:' + str(metrics.roc_auc_score(labels_ ,predicts_score))+"\n")
   f.write("\n")

print('test_f1_all',F1)
print('test_p_all',precision)
print('test_r_all', recall)
print("AUC: "+str(metrics.roc_auc_score(labels_ ,predicts_score))+"\n")

