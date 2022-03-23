from .baseline_net import BaseNet
from .efficient_net import EffNetB3
import torch

def fetch_network(model_name, rot_repr, use_norm_depth=False, use_pretrained=False, pretrained_path=""):
    if(rot_repr == 'SVD'):
        out_features = 12 # 9 for rot + 3 for translation
    elif(rot_repr == '6D'):
        out_features = 9 # 6 for rot + 3 for translation
    else:
        assert(False, "Unknown value for rotation representation in config.py")

    if use_norm_depth:
        in_channels = 7
    else:
        in_channels = 6

    print("Attempting to load", model_name)
    

    if model_name == 'baseline':
        model = BaseNet(in_channels,out_features)
        if use_pretrained:
            model.load_state_dict(torch.load(pretrained_path))
        return model
    elif model_name == 'effnet_b3':
        model = EffNetB3(in_channels,out_features)
        if use_pretrained:
            model.load_state_dict(torch.load(pretrained_path))
        return model
    else:
        assert(False, "Unknown value for backend_network in config.py")

