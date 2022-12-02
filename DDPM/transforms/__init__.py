from .diffusion_transform_cls import DiffusionTransformCls
from .diffusion_transform_att import DiffusionTransformAtt

transforms = {}
transforms['dt_cls'] = DiffusionTransformCls
transforms['dt_att'] = DiffusionTransformAtt
