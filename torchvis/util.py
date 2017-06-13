"""assigning hooks to it"""
from collections import OrderedDict
from enum import Enum, auto
from functools import partial
from typing import Container

import torch.nn as nn
from torch.autograd import Variable


class LayerType(Enum):
    RELU = auto()
    OTHER = auto()


def prepare_module(net: nn.Module, layers_save: Container[str] = ()) -> (dict, list):
    callback_dict = OrderedDict()  # not necessarily ordered, but this can help some readibility.

    forward_hook_remove_func_list = []

    for x, y in net.named_modules():
        if not isinstance(y, nn.Sequential) and y is not net:
            if x in layers_save or isinstance(y, nn.ReLU):
                callback_dict[x] = {'type': LayerType.RELU if isinstance(y, nn.ReLU) else LayerType.OTHER}

                def forward_hook(m, in_, out_, module_name):
                    if callback_dict[module_name]['type'] == LayerType.RELU:
                        assert isinstance(out_, Variable)
                        callback_dict[module_name]['output'] = out_.data.cpu().numpy().copy()
                        print(module_name, callback_dict[module_name]['output'].shape)

                forward_hook_remove_func_list.append(y.register_forward_hook(partial(forward_hook, module_name=x)))

    return callback_dict, forward_hook_remove_func_list
