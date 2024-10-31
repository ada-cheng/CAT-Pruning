from torch import nn

from ..utils import QCacheConfig


class BaseModule(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        qcache_config: QCacheConfig,
        block_idx : int = None,
    ):
        super(BaseModule, self).__init__()
        self.module = module
        self.qcache_config = qcache_config
        self.qcache_manager = None

        self.counter = 0

        self.buffer_list = None
        self.idx = block_idx

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def set_counter(self, counter: int = 0):
        self.counter = counter

    def set_cache_manager(self, qcache_manager):
        self.qcache_manager = qcache_manager
