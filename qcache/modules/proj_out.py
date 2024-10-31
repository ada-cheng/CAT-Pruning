import torch
from torch import nn

from ..modules.base_module import BaseModule
from ..utils import QCacheConfig, create_grid_graph_2d

from torch_geometric.data import Data
from collections import defaultdict

from torch_geometric.nn import max_pool
from torch_geometric.data import Data
from torch_geometric.nn import MultiAggregation
from torch_geometric.nn.conv import SimpleConv

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

class QCacheProjOut(BaseModule):
    def __init__(self, proj_out: nn.Linear, qcache_config: QCacheConfig, block_idx: int = None):
        super(QCacheProjOut, self).__init__(proj_out, qcache_config, block_idx)
        self.module = None
        self.qcache_config = qcache_config
        self.proj_out = proj_out
        self.cached_noise = None
        self.counts = None
        self.cached_indices = None
        self.cached_noise_prev = None
        self.cached_grad = None
        self.labels = None
        self.c_count = None
        self.cluster_vals = None
        self.dict_ = None
        self.edge_index = None
        multi_aggr = MultiAggregation(
        aggrs=['mean','mean','max'],  # List of aggregation schemes
        mode='sum'  # How to combine the results (here, concatenate)
        )
        self.graph_pool = SimpleConv(aggr=multi_aggr, combine_root= 'self_loop').to('cuda')
        self.graph_pool2 = SimpleConv(aggr='add', combine_root= 'sum').to('cuda')
        
        
    
    def clear(self):
        self.cached_noise = None
        self.counts = None
        self.cached_indices = None
        self.cached_noise_prev = None
        self.cached_grad = None
        self.labels = None
        self.c_count = None
        self.cluster_vals = None
        self.dict_ = None
        self.edge_index = None
        
    
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj_out(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
       
        qcache_config = self.qcache_config
        stop_idx = qcache_config.stop_idx_dict['proj']
        assert stop_idx > 0
        select_factor = qcache_config.select_factor['proj']
        
        slice_idx = int(select_factor* x.shape[1])

        hidden_states = self._forward(x)

        select_mode = qcache_config.select_mode['proj']
        if select_mode == 'convergence_stale_cpp':
            
            if self.counter > stop_idx :
                if self.cached_noise is None:
                    self.cached_noise = hidden_states.clone()
                    self.cached_noise_prev = hidden_states.clone()
                elif self.cached_indices is None or self.labels is None:
                    delta_noise = hidden_states - self.cached_noise # vT - v_start
                    self.cached_grad = hidden_states - self.cached_noise_prev
                    hidden_states_pos = add_positional_encoding(hidden_states[1], (64,64))
                    from sklearn.cluster import KMeans
                    labels =  KMeans(n_clusters=20, random_state=0).fit(hidden_states_pos.cpu().numpy()).labels_
                    labels = torch.tensor(labels, device=hidden_states.device)
                    self.labels = labels
                    self.dict_ = map_labels_to_indices(labels)
                    num_clusters = labels.max().item() + 1 
                    self.c_count = torch.zeros(num_clusters, device=hidden_states.device)
                    cluster_delta = delta_noise[1].clone()
                    data = create_grid_graph_2d((64,64), node_features=delta_noise[1])
                    data = data.to(hidden_states.device)
                    self.edge_index = data.edge_index
                    pooled_data = max_pool(labels, data)
                    self.cluster_vals = self.graph_pool(pooled_data.x, pooled_data.edge_index)
                    self.cluster_vals = self.cluster_vals.norm(p = 2, dim = -1)
                    top5_clusters = torch.topk(self.cluster_vals, k=20)[1]
                    top5_clusters = top5_clusters.tolist()
                    idx_list_all = []
                    sum_len = 0

                    for i, cluster_id in enumerate(top5_clusters):
                        idx =torch.tensor(self.dict_[cluster_id],  device = hidden_states.device)
                        k_ = min(int(slice_idx // 10 ), len(idx))
                        sum_len += k_
                        idx_list = torch.topk(cluster_delta[idx].norm(dim = -1), k= k_,)[1]
                        idx_list_all.append(idx[idx_list])
                        if sum_len > slice_idx:
                            break
                        
                    indices = torch.cat(idx_list_all)[:slice_idx]
                    # todos : rescale indices to 4096
                    self.c_count[torch.unique(labels[indices])] += 1
                    self.qcache_manager.add_cache_attn(f'{self.counter + 1}', indices)
                    self.cached_indices = indices
                    self.cached_noise_prev = hidden_states.clone()
                else:
                    indices = self.cached_indices
                    delta_noise = hidden_states - self.cached_noise
                    temp = self.cached_grad.clone()
                    self.cached_grad = hidden_states - self.cached_noise_prev
                    num_clusters = self.labels.max().item() + 1 
                    cluster_delta = delta_noise[1].clone()
                    data = Data(x=cluster_delta, edge_index=self.edge_index)
                    pooled_data = max_pool(self.labels, data)
                    self.cluster_vals = pooled_data.x
                    if self.counts is None:
                        self.counts = torch.zeros(4096, device=hidden_states.device)
                    if self.counts is None or self.counts.norm(dim=-1).min() == 0 or self.c_count.max() < 2:
                        self.counts = torch.zeros(4096, device=hidden_states.device)
                        self.gt_counts = torch.zeros(4096, device=hidden_states.device)
                        self.counts.index_add_(0, indices, torch.ones(len(indices), device=hidden_states.device))
                        self.gt_counts.index_add_(0, indices, torch.ones(len(indices), device=hidden_states.device))
                    else:
                        self.counts *= 0.5
                        self.counts.index_add_(0, indices, torch.ones(len(indices), device=hidden_states.device))
                        self.gt_counts.index_add_(0, indices, torch.ones(len(indices), device=hidden_states.device))
                    mask = ~torch.isin(torch.arange(4096, device=hidden_states.device), indices)
                    rest_indices = torch.arange(4096, device=hidden_states.device)[mask]
                    self.cluster_vals = self.cluster_vals.norm(p = 2, dim = -1)
                    top5_clusters = torch.topk(self.cluster_vals, k=20)[1]
                    top5_clusters = top5_clusters.tolist()
                    idx_list_all = []
                    sum_len = 0
                    for i, cluster_id in enumerate(top5_clusters):
                        idx = torch.tensor(self.dict_[cluster_id],  device = hidden_states.device)
                        k_ = min(int(slice_idx // 5 * 0.25), len(idx))
                        k_ = int(k_ * (3-i)) if i < 3 else k_
                        k_ = min(k_, len(idx))
                        idx_list = torch.topk(cluster_delta[idx].norm(dim = -1), k= k_,dim = -1)[1]
                        if self.counter > stop_idx + 1 and i!=0:
                            idx_list_stale = torch.topk(self.counts[idx], k = int(k_ *0.5), largest = False)[1]
                            idx_list = torch.unique(torch.cat([idx_list,idx_list_stale]), dim = -1)
                        idx_list_all.append(idx[idx_list])
                        sum_len += len(idx_list)
                        if sum_len > slice_idx * 0.25:
                            break

                    counts_ = self.counts.clone()
                    indices_stale = torch.topk(counts_, k=int(slice_idx *0.75), largest=False, dim = -1)[1]
                    indices = torch.cat(idx_list_all)[:int(slice_idx * 0.25)]
                    indices = torch.unique(torch.cat([indices, indices_stale], dim = -1))
                    self.c_count[torch.unique(self.labels[indices])]  = self.c_count[torch.unique(self.labels[indices])]  + 1
                    topk_rest_indices = rest_indices#[~torch.isin(rest_indices,  visit_recent)]#&~torch.isin(rest_indices, indices)]
                    hidden_states[:,topk_rest_indices,:] =  self.cached_noise_prev[:,topk_rest_indices,:] 
                    self.qcache_manager.add_cache_attn(f'{self.counter + 1}', indices)
                    self.cached_indices = indices
                    self.cached_noise_prev = hidden_states.clone()
     
        self.counter += 1
        return hidden_states
    