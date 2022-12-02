

from .model import UNetCls
from .model import UNetAtt
from .model import Loss

models = {}
models['unet_cls'] = UNetCls
models['unet_att'] = UNetAtt
models['loss'] = Loss
