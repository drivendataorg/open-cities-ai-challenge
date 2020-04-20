import os
import yaml


## define custom tag handler
def join(loader, node):
    seq = loader.construct_sequence(node)
    return ''.join([str(i) for i in seq])


## register the tag handler
yaml.add_constructor('!join', join)


def _update_dict(d: dict, params: dict):
    print("Overwriting config parameters:")
    for k, v in params.items():
        *path, key = k.split(".")
        inner_dict = d
        for path_key in path:
            if inner_dict[path_key] is None:
                inner_dict[path_key] = {}
            inner_dict = inner_dict[path_key]
        old_v = inner_dict.get(key)
        inner_dict[key] = v
        print(f"    ", f"{k} ".ljust(50, '.'), f"{old_v} -> {v}")
    return d


def save_config(config, directory, name='config.yml'):
    os.makedirs(directory, exist_ok=True)
    fp = os.path.join(directory, name)
    with open(fp, 'w') as f:
        yaml.dump(config, f)


def parse_config(**kwargs):
    # get config path
    cfg_path = kwargs["config"]

    # read config
    with open(cfg_path) as cfg:
        cfg_yaml = yaml.load(cfg, Loader=yaml.FullLoader)

    # override passed parameters in config
    update_cfg = _update_dict(cfg_yaml, kwargs)

    return update_cfg
