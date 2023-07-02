### Preprocess the data

#### Graph classification
If you want to use our model to complete the graph classification of dynamic networks, please process the dataset into the following form.
Refer to specific examples [example.txt](https://github.com/TP-GCN/TP-GCN/blob/main/example/example.txt).

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


