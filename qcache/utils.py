import torch
from packaging import version

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import avg_pool

def create_grid_graph_2d(size, node_features=None):
    """
    Creates a 2D grid graph with given size.
    
    Args:
        size (tuple): Size of the grid (rows, cols).
        
    Returns:
        Data: A PyG Data object representing the grid graph.
    """
    rows, cols = size
    num_nodes = rows * cols
    
    # Create edge_index for 2D grid graph
    edge_index = []
    for r in range(rows):
        for c in range(cols):
            node = r * cols + c
            if r < rows - 1:  # Edge to node below
              
                edge_index.append([node, node + cols])
                edge_index.append([node + cols, node])
            if c < cols - 1:  # Edge to node to the right
                edge_index.append([node, node + 1])
                edge_index.append([node + 1, node])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    data = Data(x=node_features, edge_index=edge_index)
    
    return data


def check_env():
    if version.parse(torch.version.cuda) < version.parse("11.3"):
        # https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html
        raise RuntimeError("NCCL CUDA Graph support requires CUDA 11.3 or above")
    if version.parse(version.parse(torch.__version__).base_version) < version.parse("2.2.0"):
        # https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
        raise RuntimeError(
            "CUDAGraph with NCCL support requires PyTorch 2.2.0 or above. "
            "If it is not released yet, please install nightly built PyTorch with "
            "`pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121`"
        )


def is_power_of_2(n: int) -> bool:
    return (n & (n - 1) == 0) and n != 0


class QCacheConfig:
    def __init__(
        self,
        height: int = 1024,
        width: int = 1024,
        do_classifier_free_guidance: bool = True,
        stop_idx_dict: dict = None,
        select_factor:dict = None,
        select_mode = None,
        rank = 0,
        use_cuda_graph = False,
    ):
        
        self.height = height
        self.width = width
        self.do_classifier_free_guidance = do_classifier_free_guidance
        self.height = height
        self.width = width

        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        self.device = device

        self.stop_idx_dict = stop_idx_dict
        self.select_factor = select_factor
        self.select_mode = select_mode
        self.use_cuda_graph = use_cuda_graph


    


class PatchCacheManager:
    
    def __init__(self, qcache_config):
        self.qcache_config = qcache_config
        self.attn_cache_indices = dict()
        self.ff_cache_indices = dict()
        self.sigmas = None
    
    def add_cache_attn(self, cache_name, cache):
        self.attn_cache_indices[cache_name] = cache
    #    print('add_cache_attn')
    
    def add_cache_ff(self, cache_name, cache):
        self.ff_cache_indices[cache_name] = cache
        
        
    def get_cache_attn(self, cache_name):
        if cache_name not in self.attn_cache_indices:
            return None
        return self.attn_cache_indices[cache_name]
    
    def get_cache_ff(self, cache_name):
        if cache_name not in self.ff_cache_indices:
            return None
        return self.ff_cache_indices[cache_name]
    
    
    def print_cache_attn(self):
        print(self.attn_cache_indices)
    
    def del_cache_attn(self, cache_name):
        if cache_name in self.attn_cache_indices:
            del self.attn_cache_indices[cache_name]
    
    def del_cache_ff(self, cache_name):
        if cache_name in self.ff_cache_indices:
            del self.ff_cache_indices[cache_name]
        
