import torch
from diffusers.models.transformers import SD3Transformer2DModel
from diffusers.models.attention_processor import Attention
from diffusers.models.attention import FeedForward
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import AdaLayerNormZero
from torch import distributed as dist, nn

from .base_model import BaseModel
from ..modules.base_module import BaseModule
from ..modules.attn import QCacheJointAttention
from ..modules.ff import QCacheFeedForward
from ..modules.proj_out import QCacheProjOut
from ..utils import QCacheConfig, PatchCacheManager
from typing import Dict, Any, List, Optional, Union



class QCacheSD3Transformer(BaseModel):  # for Patch Parallelism
    def __init__(self, model: SD3Transformer2DModel, qcache_config: QCacheConfig):
        
        #assert isinstance(model, SD3Transformer2DModel)
        #cnt = 0
        #cnt_attn = 0

        for name, module in model.named_modules():
        #    print(f'name: {name}, module: {module}')
            
            if isinstance(module, BaseModule):
                continue
            for subname, submodule in module.named_children():
                if subname == 'proj_out':
                    setattr(module, subname, QCacheProjOut(submodule, qcache_config))
                if isinstance(submodule, Attention):
                    block_idx = name.split('.')[-1]
                    setattr(module, subname, QCacheJointAttention(submodule, qcache_config, block_idx))

                if isinstance(submodule, FeedForward):
                    if subname == 'ff':
                        block_idx = name.split('.')[-1]
                        setattr(module, subname, QCacheFeedForward(submodule, qcache_config, block_idx))
             

        super(QCacheSD3Transformer, self).__init__(model, qcache_config)


        
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        block_controlnet_hidden_states: List = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        record = False,
        indices = None,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:

        qcache_config = self.qcache_config
        if qcache_config.use_cuda_graph and not record:
            static_inputs = self.static_inputs
            static_inputs['hidden_states'].copy_(hidden_states)
            static_inputs['encoder_hidden_states'].copy_(encoder_hidden_states)
            static_inputs['pooled_projections'].copy_(pooled_projections)
            static_inputs['timestep'].copy_(timestep)
         
            if self.counter >= 8:
                self.cuda_graphs[0].replay()
                output = self.static_outputs[0]
            else:
                output = self.model.forward(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    pooled_projections=pooled_projections,
                    timestep=timestep,
                    return_dict=False
                )[0]
               

        else:
            output = self.model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            return_dict=False,
          
        )[0]
            
            
        if record:
            self.static_inputs = {
                    'hidden_states': hidden_states,
                    'encoder_hidden_states': encoder_hidden_states,
                    'pooled_projections': pooled_projections,
                    'timestep': timestep,
                    'indices': indices
                }
            
        
        self.counter += 1
     
        if return_dict:
            return Transformer2DModelOutput(
                sample=output,
            )
        else:
            return (output,)
    
    def clear_model(self):

        for name, module in self.model.named_modules():
            if isinstance(module, QCacheJointAttention):
                module.clear()
            elif isinstance(module, QCacheFeedForward):
                module.clear()
            elif isinstance(module, QCacheProjOut):
                module.clear()

    
    def set_cache_manager(self, pc_manager: PatchCacheManager):
        for name, module in self.model.named_modules():
            if isinstance(module, QCacheJointAttention):
                module.set_cache_manager(pc_manager)
            elif isinstance(module, QCacheFeedForward):
                module.set_cache_manager(pc_manager)
            elif isinstance(module, QCacheProjOut):
                module.set_cache_manager(pc_manager)

        