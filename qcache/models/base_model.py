from diffusers import ConfigMixin, ModelMixin
from torch import nn

from ..modules.base_module import BaseModule
from ..utils import  QCacheConfig


class BaseModel(ModelMixin, ConfigMixin):
    def __init__(self, model: nn.Module, qcache_config: QCacheConfig):
        super(BaseModel, self).__init__()
        self.model = model
        self.qcache_config = qcache_config
        self.comm_manager = None

        self.buffer_list = None
        self.output_buffer = None
        self.counter = 0

        # for cuda graph
        self.static_inputs = None
        self.static_outputs = None
        self.cuda_graphs = None

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def set_counter(self, counter: int = 0):
        self.counter = counter
        for module in self.model.modules():
            if isinstance(module, BaseModule):
                module.set_counter(counter)

    def setup_cuda_graph(self, static_outputs, cuda_graphs):
        self.static_outputs = static_outputs
        self.cuda_graphs = cuda_graphs

    @property
    def config(self):
        return self.model.config

   
