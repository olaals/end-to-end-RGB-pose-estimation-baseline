import argparse
from importlib import import_module
import os
import sys


def add_config_cli_input():
    description = "Specify which config file to use"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('config', help="Config file name")
    args = parser.parse_args()
    return args

def get_config_from_args(args):
    sys.path.append("configs")
    sys.path.append("old-configs")
    config_name = args.config
    config_path = os.path.splitext(os.path.split(config_name)[-1])[0]
    config_import = import_module(config_path)
    config_dict = config_import.get_config()
    return config_dict

def get_dict_from_cli():
    args = add_config_cli_input()
    cfg_dict = get_config_from_args(args)
    return cfg_dict







if __name__ == '__main__':
    args = add_config_cli_input()
    cfg_dict = get_config_from_args(args)
    print(cfg_dict)
