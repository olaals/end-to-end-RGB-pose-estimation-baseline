from .baseline_net import BaseNet
import torch

def fetch_network(model_name, rot_repr, use_pretrained=False, pretrained_path=""):
    if(rot_repr == 'SVD'):
        out_features = 12 # 9 for rot + 3 for translation
    elif(rot_repr == '6D'):
        out_features = 9 # 6 for rot + 3 for translation
    else:
        assert(False, "Unknown value for rotation representation in config.py")

    if model_name == 'baseline':
        model = BaseNet(6,out_features)
        if use_pretrained:
            model.load_state_dict(torch.load(pretrained_path))
        return model
    elif model_name == 'efficient_net':
        pass
    else:
        assert(False, "Unknown value for backend_network in config.py")

