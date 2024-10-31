import torch

from torch import nn
from ..modules.base_module import BaseModule
from ..utils import create_grid_graph_2d

from torch_geometric.data import Data
from collections import defaultdict

from torch_geometric.nn import max_pool
from torch_geometric.data import Data
from torch_geometric.nn import MultiAggregation
from torch_geometric.nn.conv import SimpleConv

from kmeans_pytorch import kmeans as km

cached_noise = torch.zeros(2, 4096, 64).cuda().half()
cached_noise_prev = torch.zeros(2, 4096, 64).cuda().half()
indices = torch.zeros(1228, dtype = torch.int64).cuda()

labels = torch.zeros(4096, dtype = torch.int32).cuda() 
dict_ = dict()
edge_index = torch.zeros(2, 16128, dtype = torch.int64).cuda()
counts = torch.zeros(4096, dtype = torch.float32).cuda()

len_indices = torch.tensor(1228, dtype = torch.int64).cuda()



def map_labels_to_indices(labels):
    # Initialize a dictionary to store the result
    label_to_indices = defaultdict(list)
    
    # Iterate over the labels and their indices
    for index, label in enumerate(labels):
        label_to_indices[int(label)].append(index)
    
    # Convert defaultdict to a regular dict (optional)
    return dict(label_to_indices)

def create_adjacency_matrix(labels):
    l = len(labels)
    adj_matrix = torch.zeros((l, l), dtype=torch.float32, device=labels.device)

    for i in range(l):
        for j in range(l):
            if labels[i] == labels[j]:
                adj_matrix[i, j] = 1.0

    return adj_matrix


def create_pyg_data(features, adj_matrix):
    edge_index = torch.nonzero(adj_matrix, as_tuple=False).t().contiguous()
    data = Data(x=features, edge_index=edge_index)
    return data


def add_positional_encoding(features, size):
    h, w = size
    pos_enc = torch.zeros_like(features)

    for i in range(h):
        for j in range(w):
            pos_enc[i * w + j, :features.size(1) //2] = i / h
            pos_enc[i * w + j, features.size(1) //2:] = j / w
    

    enhanced_features = torch.cat([features*0.05 + pos_enc], dim=1)
    
    return enhanced_features


count_indices = 0
class QCacheProjOut(BaseModule):
    def __init__(self, proj_out: nn.Linear):
        super(QCacheProjOut, self).__init__(proj_out)
        self.module = None
        self.proj_out = proj_out
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        multi_aggr = MultiAggregation(
        aggrs=['mean','mean','max'],  # List of aggregation schemes
        mode='sum'  # How to combine the results (here, concatenate)
        )
        self.graph_pool = SimpleConv(aggr=multi_aggr, combine_root= 'self_loop').to('cuda')
        self.graph_pool2 = SimpleConv(aggr='add', combine_root= 'sum').to('cuda')
        self.graph_pool4 = SimpleConv(aggr='mean').to('cuda')
        
    
    def clear(self):
        self.cached_noise = None
        self.counts = None
        self.cached_indices = None
        self.plot_traj = []
        self.plot_traj2 = []
        self.plot_traj3 = []
        self.cached_noise_prev = None
        self.cached_grad = None
        self.count_indices = 0
        self.labels = None
        self.c_count = None
        self.cluster_vals = None
        self.dict_ = None
        self.edge_index = None
        self.num_select_nodes = None

        self.inb = []
        
    
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj_out(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
       
      
        hidden_states = self._forward(x)
        cached_noise.copy_(hidden_states)
        cached_noise_prev.copy_(hidden_states)
        return hidden_states    
    
class QCacheProjOut_7(BaseModule):
    def __init__(self, proj_out: nn.Linear):
        super(QCacheProjOut_7, self).__init__(proj_out)
        self.module = None
        self.proj_out = proj_out
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        multi_aggr = MultiAggregation(
        aggrs=['mean','mean','max'],  # List of aggregation schemes
        mode='sum'  # How to combine the results (here, concatenate)
        )
        self.graph_pool = SimpleConv(aggr=multi_aggr, combine_root= 'self_loop').to('cuda')
        self.graph_pool2 = SimpleConv(aggr='add', combine_root= 'sum').to('cuda')
        self.graph_pool4 = SimpleConv(aggr='mean').to('cuda')
        
    
    def clear(self):
        self.cached_noise = None
        self.counts = None
        self.cached_indices = None
        self.plot_traj = []
        self.plot_traj2 = []
        self.plot_traj3 = []
        self.cached_noise_prev = None
        self.cached_grad = None
        self.count_indices = 0
        self.labels = None
        self.c_count = None
        self.cluster_vals = None
        self.dict_ = None
        self.edge_index = None
        self.num_select_nodes = None

        self.inb = []
        
    
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj_out(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
       
      
        hidden_states = self._forward(x)
        
        global cached_noise
        global cached_noise_prev

        delta_noise = hidden_states - cached_noise # vT - v_start

        hidden_states_pos = add_positional_encoding(hidden_states[1], (64,64))
        from sklearn.cluster import KMeans
        
        labels_ =  KMeans(n_clusters=20, random_state=0).fit(hidden_states_pos.cpu().numpy()).labels_
        labels.copy_(torch.tensor(labels_, device=hidden_states.device))

        tmp_dict = map_labels_to_indices(labels)
        for k, v in tmp_dict.items():
            dict_[k] = v
                    
        num_clusters = labels.max().item() + 1 
        cluster_delta = delta_noise[1].clone()
        
        data = create_grid_graph_2d((64,64), node_features=delta_noise[1])
        data = data.to(hidden_states.device)
        edge_index = data.edge_index
      
        pooled_data = max_pool(labels, data)
                  
        cluster_vals = self.graph_pool(pooled_data.x, pooled_data.edge_index)
                   
        cluster_vals = cluster_vals.norm(p = 2, dim = -1)
                    
        top5_clusters = torch.topk(cluster_vals, k=20)[1]
        top5_clusters = top5_clusters.tolist()
        idx_list_all = []
        sum_len = 0
        for i, cluster_id in enumerate(top5_clusters):
                       
            idx =torch.tensor(dict_[cluster_id],  device = hidden_states.device)
            k_ = min(int(1228 // 10 ), len(idx))
            sum_len += k_
            idx_list = torch.topk(cluster_delta[idx].norm(dim = -1), k= k_,)[1]
                        
            idx_list_all.append(idx[idx_list])
            if sum_len > 1228:
                break
                        
        indices.copy_(torch.cat(idx_list_all)[:1228])

        cached_noise_prev.copy_(hidden_states)
        
        return hidden_states


    def forward_8(self, x: torch.Tensor) -> torch.Tensor:
       
        global indices, labels, dict_, edge_index, cached_noise, cached_noise_prev, counts

        hidden_states = self._forward(x)

        delta_noise = hidden_states - cached_noise

        cluster_delta = delta_noise[1].clone()
        data = Data(x=cluster_delta, edge_index=edge_index)
                   
        pooled_data = max_pool(labels, data)             
        cluster_vals = pooled_data.x
                
        counts = torch.zeros(4096, device=hidden_states.device)
        counts.index_add_(0, indices, torch.ones(len(indices), device=hidden_states.device))


        mask = ~torch.isin(torch.arange(4096, device=hidden_states.device), indices)
        rest_indices = torch.arange(4096, device=hidden_states.device)[mask]

        cluster_vals = cluster_vals.norm(p = 2, dim = -1)
    
        top5_clusters = torch.topk(cluster_vals, k=20)[1]
        top5_clusters = top5_clusters.tolist()
        idx_list_all = []
                    
        sum_len = 0
        for i, cluster_id in enumerate(top5_clusters):
                     
            idx = torch.tensor(dict_[cluster_id],  device = hidden_states.device)
            k_ = min(int(1228 // 5 * 0.25), len(idx))
            k_ = int(k_ * (3-i)) if i < 3 else k_
            k_ = min(k_, len(idx))
            idx_list = torch.topk(cluster_delta[idx].norm(dim = -1), k= k_,dim = -1)[1]
        
            idx_list_all.append(idx[idx_list])
            sum_len += len(idx_list)
            if sum_len > 1228 * 0.25:
                break
        indices_m = torch.cat(idx_list_all)[:int(1228 * 0.25)]
                
        counts_ = counts.clone()
                    
        indices_stale = torch.topk(counts_, k= 1228 , largest=False, dim = -1)[1]
        indices_buff = torch.unique(torch.cat([indices_m, indices_stale[:int(1228*0.75)]], dim = -1))
        
        len_buff = len(indices_buff)
        len_indices.copy_(torch.tensor(len_buff, dtype = torch.int64).cuda())
        if len_buff < 1228:
            
            indices.copy_(torch.cat([indices_buff, indices_stale[int(1228*0.75):]], dim = -1)[:1228])
        else:
            indices.copy_(indices_buff)
        topk_rest_indices = rest_indices
   
        hidden_states.index_copy_(1, topk_rest_indices, cached_noise_prev.index_select(1, topk_rest_indices))
        cached_noise_prev.copy_(hidden_states)
        
        return hidden_states
                
    def forward_9(self, x: torch.Tensor) -> torch.Tensor:
       
        global indices, labels, dict_, edge_index, cached_noise, cached_noise_prev, counts

        hidden_states = self._forward(x)

        delta_noise = hidden_states - cached_noise

        cluster_delta = delta_noise[1].clone()
        data = Data(x=cluster_delta, edge_index=edge_index)
                   
        pooled_data = max_pool(labels, data)             
        cluster_vals = pooled_data.x
                
        counts *= 0.5
        counts.index_add_(0, indices[:len_indices], torch.ones(len_indices, device=hidden_states.device))

        mask = ~torch.isin(torch.arange(4096, device=hidden_states.device), indices)
        rest_indices = torch.arange(4096, device=hidden_states.device)[mask]

        cluster_vals = cluster_vals.norm(p = 2, dim = -1)
    
        top5_clusters = torch.topk(cluster_vals, k=20)[1]
        top5_clusters = top5_clusters.tolist()
        idx_list_all = []
                    
        sum_len = 0
        for i, cluster_id in enumerate(top5_clusters):
                     
            idx = torch.tensor(dict_[cluster_id],  device = hidden_states.device)
            k_ = min(int(1228 // 5 * 0.25), len(idx))
            k_ = int(k_ * (3-i)) if i < 3 else k_
            k_ = min(k_, len(idx))
            idx_list = torch.topk(cluster_delta[idx].norm(dim = -1), k= k_,dim = -1)[1]
            idx_list_stale = torch.topk(counts[idx], k = int(k_ *0.5), largest = False)[1]
            idx_list = torch.unique(torch.cat([idx_list,idx_list_stale]), dim = -1)
            idx_list_all.append(idx[idx_list])
            sum_len += len(idx_list)
            if sum_len > 1228 * 0.25:
                break
        indices_m = torch.cat(idx_list_all)[:int(1228 * 0.25)]
                
        counts_ = counts.clone()
                    
        indices_stale = torch.topk(counts_, k= 1228, largest=False, dim = -1)[1]
        indices_buff = torch.unique(torch.cat([indices_m, indices_stale[:int(1228*0.75)]], dim = -1))
        
        len_buff = len(indices_buff)
        len_indices.copy_(torch.tensor(len_buff, dtype = torch.int64).cuda())
        
        if len_buff < 1228:
            
            indices.copy_(torch.cat([indices_buff,  indices_stale[int(1228*0.75):]], dim = -1)[:1228])
        else:
            indices.copy_(indices_buff)    

        topk_rest_indices = rest_indices
   
        hidden_states.index_copy_(1, topk_rest_indices, cached_noise_prev.index_select(1, topk_rest_indices))
        cached_noise_prev.copy_(hidden_states)
        
        return hidden_states