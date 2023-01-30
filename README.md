# TP-GCN: Dynamic Network Representation Learning for Graph and Node Classification
<!--#### -->
## Introduction
#### Paper link: [TP-GCN: Dynamic Network Representation Learning for Graph and Node Classification](https://XXX)
![Framework](framework.png?raw=true "Network Framework")
## Running the experiments

### Dataset and preprocessing

#### Download the public data
* [Reddit](http://snap.stanford.edu/jodie/reddit.csv)

* [Wikipedia](http://snap.stanford.edu/jodie/wikipedia.csv)

* [MOOC](http://snap.stanford.edu/jodie/mooc.csv)

### Requirements

* python >= 3.9

* Dependency

```{bash}
torch==1.12.0
tqdm==4.64.0
numpy==1.18.5
scikit-learn==0.24.2
```

### Command and configurations

#### Sample commend
* Learning the down-stream task (graph-classification)
```{bash}
# on Forum-java
python -u train_graph.py -d Forum-java --bs 32  --n_dim 32 --edge_agg mean --divide 0.3

# on HDFS
python -u train_graph.py -d HDFS --bs 32  --n_dim 32 --edge_agg mean --divide 0.3

# on Gowalla
python -u train_graph.py -d Gowalla --bs 32  --n_dim 32 --edge_agg mean --divide 0.3
```

* Learning the down-stream task (node-classification)

Node-classification task reuses the network trained previously. Make sure the `prefix` is the same so that the checkpoint can be found under `saved_models`.

```{bash}
#on MOOC
python -u train_node.py -d MOOC --bs 32  --n_dim 32 --edge_agg mean --divide 0.3

# on Wikipedia
python -u train_node.py -d Wikipedia --bs 32  --n_dim 32 --edge_agg mean --divide 0.3

# on Reddit
python -u train_node.py -d Reddit --bs 32  --n_dim 32 --edge_agg mean --divide 0.3
```

#### General flags

```{txt}
optional arguments:
  -d DATA, --data DATA                       data sources to use
  --bs BS                                    batch_size
  --n_epoch N_EPOCH                          number of epochs
  --lr LR                                    learning rate
  --node_dim NODE_DIM                        dimentions of the node embedding
  --edge_agg {mean,had,w1,w2,activate}       EdgeAgg method
  --divide                                   the ratio of training sets
```
