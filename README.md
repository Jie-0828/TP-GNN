# TP-GCN: Dynamic Network Representation Learning for Graph and Node Classification
<!--#### -->
## Introduction
Anomaly detection has been an important problem which asks to identify systems whose behaviours deviate significantly from expectations. In general, system behaviours
can be characterised by the interactions among system components and their sequence. Therefore, a dynamic network model can be established to capture these interactions. In this way, anomaly detection reduces to a classification task that seeks to identify “anomalous dynamic network” from the “standard” ones.

In this paper, we propose Temporal propagation-Graph Convolutional Neural Network (TP-GCN). The model extracts the temporal information and structure information from the dynamic network to realize the final classification task. TP-GCN contains two components: 1)temporal propagation; 2)global temporal information embedding, which involves a novel way of message propagation.

The proposed approach handles both node classification and graph classification task, and has good performance.

<!-- #### Paper link: [TP-GCN: Dynamic Network Representation Learning for Graph and Node Classification](https://XXX) -->
![framework](https://user-images.githubusercontent.com/105060483/215920194-007a6613-db17-455e-8b0c-73bdecde7fa7.png "The framework of TP-GCN")
## Running the experiments

### Dataset and preprocessing

#### Download the public dataset
* [Gowalla](https://snap.stanford.edu/data/loc-gowalla.html)

* [HDFS](https://doi.org/10.5281/zenodo.1144100)

* [Reddit](http://snap.stanford.edu/jodie/reddit.csv)

* [Wikipedia](http://snap.stanford.edu/jodie/wikipedia.csv)

* [MOOC](http://snap.stanford.edu/jodie/mooc.csv)

#### Download the Forum-java dataset
As for the Fourm-java dataset we created in the paper, its contents are available in the [dataset](https://github.com/TP-GCN/TP-GCN/edit/main/dataset) folder

#### Use your own data
See [example](https://github.com/TP-GCN/TP-GCN/edit/main/example) for the required input data format.

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
