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
* Learning the network using link prediction tasks
```{bash}
# t-gat learning on wikipedia data
python -u learn_edge.py -d wikipedia --bs 200 --uniform  --n_degree 20 --agg_method attn --attn_mode prod --gpu 0 --n_head 2 --prefix hello_world

# t-gat learning on reddit data
python -u learn_edge.py -d reddit --bs 200 --uniform  --n_degree 20 --agg_method attn --attn_mode prod --gpu 0 --n_head 2 --prefix hello_world
```

* Learning the down-stream task (node-classification)

Node-classification task reuses the network trained previously. Make sure the `prefix` is the same so that the checkpoint can be found under `saved_models`.

```{bash}
# on wikipedia
python -u learn_node.py -d wikipedia --bs 100 --uniform  --n_degree 20 --agg_method attn --attn_mode prod --gpu 0 --n_head 2 --prefix hello_world

# on reddit
python -u learn_node.py -d reddit --bs 100 --uniform  --n_degree 20 --agg_method attn --attn_mode prod --gpu 0 --n_head 2 --prefix hello_world
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
