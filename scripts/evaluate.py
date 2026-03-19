import random
import numpy as np
import torch
import torch._dynamo
torch._dynamo.config.verbose = True
import hydra
from hydra.utils import instantiate
import os
import sys

try:
    import IPython.display as _disp
    _disp.display = lambda *args, **kwargs: None
except Exception:
    pass

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

sys.path.insert(0, '.')
sys.path.insert(0, '..')
sys.path.insert(0, 'src')
sys.path.insert(0, './alfred')

from reibench.alfred_evaluator import AlfredEvaluator
from dotenv import load_dotenv
load_dotenv() 
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
load_dotenv(override=False)
load_dotenv()
gpu = os.getenv("CUDA_VISIBLE_DEVICES")
print(f"Using GPU: {gpu}")


def _load_task_difficulty_into_cfg(cfg, name: str, config_dir: str) -> bool:
    import yaml
    from omegaconf import OmegaConf
    path = os.path.join(config_dir, "task_difficulty", f"{name}.yaml")
    if not os.path.isfile(path):
        return False
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not data:
        return False
    OmegaConf.set_struct(cfg, False)
    try:
        if not hasattr(cfg, "task_difficulty") or cfg.task_difficulty is None:
            cfg["task_difficulty"] = OmegaConf.create({})
        for k, v in data.items():
            if k.startswith("#") or k == "@package _global__":
                continue
            cfg.task_difficulty[k] = v
    finally:
        OmegaConf.set_struct(cfg, True)
    return True


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    from reibench.utils.config_mapper import get_data_types

    random.seed(cfg.planner.random_seed)
    torch.manual_seed(cfg.planner.random_seed)
    np.random.seed(cfg.planner.random_seed)

    data_types = get_data_types(cfg)
    if cfg.name == 'alfred':
        evaluator = AlfredEvaluator(cfg)
    else:
        raise ValueError("Unknown configuration name. Must be 'alfred' or 'wah'.")

    config_dir = os.path.join(_proj_root, "configs")
    for data_type in data_types:
        print(f"Evaluating data_type: {data_type}")
        if _load_task_difficulty_into_cfg(cfg, data_type, config_dir):
            cfg.data_type = None
        else:
            cfg.data_type = data_type
        evaluator.evaluate()
            

    
    
if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA is available! GPU is being used.")
    
    main()  
