### Preprocess the data

#### Graph classification
If you want to use our model to complete the graph classification of dynamic networks, please process the dataset into the following form.
Refer to specific examples [example.txt](https://github.com/TP-GCN/TP-GCN/example.txt).

```{txt}
contents included:
  -ID          each graph has its corresponding unique number
  
  -Label       graph label "Normal" or "Anomaly"
  
  -network[son<-parent]=               
   target node<-source node, time label
   
  -nodeInfo=
   node idx: features
```
All time label starts from `0`.

#### Node classification
For the node classification task, no special processing is required if the three data sets mentioned in the paper are used: MOOC, Wikipedia, and Reddit.

If you want to use your own data, put it into the following form. Refer to specific examples [example.csv](https://github.com/TP-GCN/TP-GCN/example.csv)
The `CSV` file has following columns
```
source node, target node, time label, node label, list_of_features
```
, which represents source node index, target node index, time label, node label and the node features. Same as graph classification dataset, all time label starts from `0`.


