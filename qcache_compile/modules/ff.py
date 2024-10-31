
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


cached_hidden_states_list = [torch.zeros(2, 4096, 1536).cuda().half() for _ in range(24) ]

class QCacheFeedForward(BaseModule):
    def __init__(
        self, ff: FeedForward):
        super(QCacheFeedForward, self).__init__(ff)
        self.module = None
        self.net = ff.net

    
    def clear(self):
        pass

    def _forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
     
        for module in self.net:
          
            hidden_states = module(hidden_states)
            
        return hidden_states
    
   # @torch.compile
    def forward(self, hidden_states: torch.Tensor, index_block: torch.Tensor = None,*args, **kwargs):
        

        hidden_states = self._forward(hidden_states)
          
        cached_hidden_states_list[index_block].copy_(hidden_states)
            
        return cached_hidden_states_list[index_block]
    
    def forward_8(self, hidden_states: torch.Tensor, 
                  index_block: torch.Tensor = None, *args, **kwargs):

        
        global cached_hidden_states_list
        from ..modules.proj_out import indices

        
        hidden_states_selected = hidden_states[...,indices,:].clone()

        cached_hidden_states_list[index_block].index_copy_(dim=1, index=indices, source=self._forward(hidden_states_selected))
            
        return cached_hidden_states_list[index_block]