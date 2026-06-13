import importlib

import yaml


def load_yaml_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def locate_class(class_path):
    module_name, class_name = class_path.split(":")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def instantiate_from_config(config, **extra_kwargs):
    class_path = config["class_path"]
    init_args = dict(config.get("init_args") or {})
    init_args.update(extra_kwargs)
    return locate_class(class_path)(**init_args)


def first_component_config(config):
    if "class_path" in config:
        return config

    for value in config.values():
        if isinstance(value, dict) and "class_path" in value:
            return value

    raise ValueError("No component with class_path found in config.")

