import torch
from diffusers.models.attention import Attention, FeedForward, GEGLU, GELU, ApproximateGELU
from diffusers.utils import USE_PEFT_BACKEND
from torch import distributed as dist
from torch import nn
from torch.nn import functional as F

from ..modules.base_module import BaseModule
from ..utils import QCacheConfig

from typing import Optional
import time


class QCacheFeedForward(BaseModule):
    def __init__(
        self, ff: FeedForward, qcache_config: QCacheConfig, block_idx: int = None):
        super(QCacheFeedForward, self).__init__(ff, qcache_config, block_idx)
      
        self.net = ff.net
        self.cached_hidden_states = None
        self.module = None
        
    
    def clear(self):
        self.counter = 0
        self.cached_hidden_states = None

    def _forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
     
        for module in self.net:
            hidden_states = module(hidden_states)
            
        return hidden_states

    def forward(self, hidden_states: torch.Tensor, indices:torch.Tensor = None,*args, **kwargs):
        
        b,l,c = hidden_states.shape # 0 uncond, 1 text
        qcache_config = self.qcache_config
        threshold_step = qcache_config.stop_idx_dict['ff']
        select_mode = qcache_config.select_mode['ff']

        if self.counter > threshold_step :

            if select_mode == 'convergence_t_noise':
                indices = self.qcache_manager.get_cache_attn(f'{self.counter}')
                if indices is not None:
                    hidden_states_selected = hidden_states[...,indices,:].clone()
                    self.cached_hidden_states[..., indices,:] = self._forward(hidden_states_selected)
                else:
                    hidden_states = self._forward(hidden_states)
                    self.cached_hidden_states = hidden_states
            else:
                raise NotImplementedError
            
        else:
            hidden_states = self._forward(hidden_states)
            self.cached_hidden_states = hidden_states
            
        self.counter += 1

        return self.cached_hidden_states