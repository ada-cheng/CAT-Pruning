from torch import nn

class BaseModule(nn.Module):
    def __init__(
        self,
        module: nn.Module,
    ):
        super(BaseModule, self).__init__()
        self.module = module
        self.qcache_manager = None

        self.counter = 0

        self.buffer_list = None

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def set_counter(self, counter: int = 0):
        self.counter = counter

    def set_cache_manager(self, qcache_manager):
        self.qcache_manager = qcache_manager
