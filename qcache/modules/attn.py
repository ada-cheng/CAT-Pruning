import torch
from diffusers.models.attention import Attention
from diffusers.utils import USE_PEFT_BACKEND
from torch import distributed as dist
from torch import nn
from torch.nn import functional as F

from ..modules.base_module import BaseModule
from ..utils import QCacheConfig

from typing import Optional, List
import time

import math

indices_list = []


class QCacheAttention(BaseModule):
    def __init__(self, module: Attention, qcache_config: QCacheConfig):
        super(QCacheAttention, self).__init__(module, qcache_config)
        self.cached_kv = None
        self.cached_hidden_states = None

    
    def clear(self):
        self.cached_kv = None
        self.cached_hidden_states = None
        self.counter = 0


class QCacheJointAttention(QCacheAttention):
    def __init__(self, module: Attention, qcache_config: QCacheConfig, block_idx: int = None):
        super(QCacheJointAttention, self).__init__(module, qcache_config)
        self.idx = block_idx
        self.counts = None

    
    def _forward(self, hidden_states: torch.FloatTensor, 
                 encoder_hidden_states: torch.FloatTensor or None, 
                 is_full_forward: bool = True, 
                 indices:torch.Tensor = None,
                 is_idx_fixed: bool = False,
                 ) -> torch.FloatTensor:
        
        b = hidden_states.shape[0]
        attn = self.module
        residual = hidden_states

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        context_input_ndim = encoder_hidden_states.ndim
        if context_input_ndim == 4:
            batch_size, channel, height, width = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size = encoder_hidden_states.shape[0]

        query = attn.to_q(hidden_states)
        # `sample` projections.
        if  is_full_forward:
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
            self.cached_kv = torch.cat([key, value], dim=-1)
        else:
            key, value = torch.split(self.cached_kv, self.cached_kv.shape[-1] // 2, dim=-1)
            updated_key = attn.to_k(hidden_states)
            updated_value = attn.to_v(hidden_states)
            for i in range(0,b,2):
                key[i:i+2,indices[i+1], :] = updated_key[i:i+2]
                value[i:i+2,indices[i+1], :] = updated_value[i:i+2]


        # `context` projections.
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
            

        # attention
        query = torch.cat([query, encoder_hidden_states_query_proj], dim=1)
        key = torch.cat([key, encoder_hidden_states_key_proj], dim=1)
        value = torch.cat([value, encoder_hidden_states_value_proj], dim=1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = hidden_states = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )
        
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Split the attention outputs.
        hidden_states, encoder_hidden_states = (
            hidden_states[:, : residual.shape[1]],
            hidden_states[:, residual.shape[1] :],
        )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        if not attn.context_pre_only:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if context_input_ndim == 4:
            encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        
        return hidden_states, encoder_hidden_states
    
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        indices: torch.Tensor = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        qcache_config = self.qcache_config
        b, l, c = hidden_states.shape
        stop_idx = qcache_config.stop_idx_dict['joint_attn']
        select_mode = qcache_config.select_mode['joint_attn']
        
        if self.counter > stop_idx :
            if select_mode == 'convergence_t_noise':
                indices = self.qcache_manager.get_cache_attn(f'{self.counter}')
                if indices is not None:
                    hidden_states_selected = hidden_states[...,indices,:].clone()
                    output = self._forward(hidden_states=hidden_states_selected, 
                                                     encoder_hidden_states=encoder_hidden_states, 
                                                     is_full_forward=False, indices=indices.repeat(b, 1))
                    self.cached_hidden_states.index_copy_(dim=1, index=indices, source=output[0])
                else :
                    
                    output = self._forward(hidden_states, encoder_hidden_states, is_full_forward=True)
                    self.cached_hidden_states = output[0]

        else:
            output = self._forward(hidden_states, encoder_hidden_states, )
            self.cached_hidden_states = output[0]
        
        self.counter += 1
        return (self.cached_hidden_states, output[1])

        