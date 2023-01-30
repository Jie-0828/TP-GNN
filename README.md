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
  -h, --help            show this help message and exit
  -d DATA, --data DATA  data sources to use, try wikipedia or reddit
  --bs BS               batch_size
  --prefix PREFIX       prefix to name the checkpoints
  --n_degree N_DEGREE   number of neighbors to sample
  --n_head N_HEAD       number of heads used in attention layer
  --n_epoch N_EPOCH     number of epochs
  --n_layer N_LAYER     number of network layers
  --lr LR               learning rate
  --drop_out DROP_OUT   dropout probability
  --gpu GPU             idx for the gpu to use
  --node_dim NODE_DIM   Dimentions of the node embedding
  --time_dim TIME_DIM   Dimentions of the time embedding
  --agg_method {attn,lstm,mean}
                        local aggregation method
  --attn_mode {prod,map}
                        use dot product attention or mapping based
  --time {time,pos,empty}
                        how to use time information
  --uniform             take uniform sampling from temporal neighbors
```
