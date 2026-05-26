import os
import yaml
from typing import Any, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ConfigLoader:
    _instance = None
    _configs: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
            cls._instance._load_all_configs()
        return cls._instance

    def _load_all_configs(self):
        config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")
        config_files = {
            "retrieval": "retrieval_config.yaml",
            "model": "model_config.yaml",
            "chunking": "chunking_config.yaml",
            "cache": "cache_config.yaml"
        }
        
        for name, filename in config_files.items():
            path = os.path.join(config_dir, filename)
            if os.path.exists(path):
                with open(path, "r") as f:
                    self._configs[name] = yaml.safe_load(f)
            else:
                self._configs[name] = {}

    def get(self, key_path: str, arg2: Any = None, arg3: Any = None) -> Any:
        """
        Retrieves a configuration value. Supports:
        1. config_loader.get("category.subkey.path", default_value)
        2. config_loader.get("category", "subkey.path", default_value)
        """
        if "." in key_path:
            parts = key_path.split(".")
            category = parts[0]
            subkeys = parts[1:]
            default = arg2
        else:
            category = key_path
            if isinstance(arg2, str):
                subkeys = arg2.split(".")
                default = arg3
            else:
                subkeys = []
                default = arg2
                
        category_config = self._configs.get(category, {})
        if not subkeys:
            return category_config if category_config else default
            
        val = category_config
        for k in subkeys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

config_loader = ConfigLoader()
