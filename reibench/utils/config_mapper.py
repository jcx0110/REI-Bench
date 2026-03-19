from omegaconf import DictConfig


def get_planner_framework(cfg: DictConfig) -> str:
    if hasattr(cfg, 'framework') and hasattr(cfg.framework, 'planner_framework'):
        return cfg.framework.planner_framework
    if hasattr(cfg, 'planner') and hasattr(cfg.planner, 'planner_framework'):
        return cfg.planner.planner_framework
    return "saycan"


def get_model_name(cfg: DictConfig) -> str:
    if hasattr(cfg, 'model') and hasattr(cfg.model, 'model_name'):
        return cfg.model.model_name
    if hasattr(cfg, 'planner') and hasattr(cfg.planner, 'model_name'):
        return cfg.planner.model_name
    return "meta-llama/Llama-3.1-8B-Instruct"


def get_prompting_method(cfg: DictConfig) -> dict:
    result = {
        'aware_hint': False,
        'COT': False,
        'TOCC': False,
        'ICL': False
    }
    if getattr(cfg, 'prompting_method_enabled', True) is False:
        return result
    if hasattr(cfg, 'method'):
        if hasattr(cfg.method, 'aware_hint'):
            result['aware_hint'] = cfg.method.aware_hint
        if hasattr(cfg.method, 'COT'):
            result['COT'] = cfg.method.COT
        if hasattr(cfg.method, 'TOCC'):
            result['TOCC'] = cfg.method.TOCC
        if hasattr(cfg.method, 'ICL'):
            result['ICL'] = cfg.method.ICL
    if hasattr(cfg, 'prompting_method'):
        if hasattr(cfg.prompting_method, 'aware_hint'):
            result['aware_hint'] = cfg.prompting_method.aware_hint
        if hasattr(cfg.prompting_method, 'COT'):
            result['COT'] = cfg.prompting_method.COT
        if hasattr(cfg.prompting_method, 'TOCC'):
            result['TOCC'] = cfg.prompting_method.TOCC
        if hasattr(cfg.prompting_method, 'ICL'):
            result['ICL'] = cfg.prompting_method.ICL
    return result


DATA_TYPE_READABLE_TO_INTERNAL = {
    "standard-explicit": "1-1", "standard-mixed": "2-1", "standard-implicit": "3-1",
    "noised-explicit": "1-2", "noised-mixed": "2-2", "noised-implicit": "3-2",
    "short-explicit": "1-3", "short-mixed": "2-3", "short-implicit": "3-3",
}


def _data_type_to_internal(value: str) -> str:
    if value in DATA_TYPE_READABLE_TO_INTERNAL:
        return DATA_TYPE_READABLE_TO_INTERNAL[value]
    return value


def get_data_type(cfg: DictConfig) -> str:
    raw = None
    if hasattr(cfg, 'task_difficulty'):
        if hasattr(cfg.task_difficulty, 'data_type') and cfg.task_difficulty.data_type:
            raw = str(cfg.task_difficulty.data_type)
        elif hasattr(cfg.task_difficulty, 're_level') and hasattr(cfg.task_difficulty, 'context_type'):
            re_level_map = {'explicit': '1', 'mixed': '2', 'implicit': '3'}
            context_type_map = {'standard': '1', 'noised': '2', 'short': '3'}
            re_level = cfg.task_difficulty.re_level
            context_type = cfg.task_difficulty.context_type
            if re_level in re_level_map and context_type in context_type_map:
                raw = f"{re_level_map[re_level]}-{context_type_map[context_type]}"
    if raw is None and hasattr(cfg, 'data_type') and cfg.data_type is not None:
        raw = str(cfg.data_type)
    if raw is None:
        raw = "1-1"
    return _data_type_to_internal(raw)


def get_data_types(cfg: DictConfig) -> list:
    if hasattr(cfg, 'data_types') and cfg.data_types is not None:
        return list(cfg.data_types)
    return [get_data_type(cfg)]


def convert_re_level_to_number(re_level: str) -> int:
    mapping = {'explicit': 1, 'mixed': 2, 'implicit': 3}
    return mapping.get(re_level, 1)


def convert_context_type_to_number(context_type: str) -> int:
    mapping = {'standard': 1, 'noised': 2, 'short': 3}
    return mapping.get(context_type, 1)


def get_model_config(cfg: DictConfig) -> dict:
    result = {}
    if hasattr(cfg, 'model'):
        if hasattr(cfg.model, 'model_name'):
            result['model_name'] = cfg.model.model_name
        if hasattr(cfg.model, 'use_accelerate_device_map'):
            result['use_accelerate_device_map'] = cfg.model.use_accelerate_device_map
        if hasattr(cfg.model, 'load_in_8bit'):
            result['load_in_8bit'] = cfg.model.load_in_8bit
        if hasattr(cfg.model, 'device'):
            result['device'] = cfg.model.device
        if hasattr(cfg.model, 'hf_auth_token'):
            result['hf_auth_token'] = cfg.model.hf_auth_token
        if hasattr(cfg.model, 'openai_api_key'):
            result['openai_api_key'] = cfg.model.openai_api_key
    if hasattr(cfg, 'planner'):
        if 'model_name' not in result and hasattr(cfg.planner, 'model_name'):
            result['model_name'] = cfg.planner.model_name
        if 'use_accelerate_device_map' not in result and hasattr(cfg.planner, 'use_accelerate_device_map'):
            result['use_accelerate_device_map'] = cfg.planner.use_accelerate_device_map
        if 'device' not in result and hasattr(cfg.planner, 'device'):
            result['device'] = cfg.planner.device
        if 'hf_auth_token' not in result and hasattr(cfg.planner, 'hf_auth_token'):
            result['hf_auth_token'] = cfg.planner.hf_auth_token
        if 'openai_api_key' not in result and hasattr(cfg.planner, 'openai_api_key'):
            result['openai_api_key'] = cfg.planner.openai_api_key
    return result
