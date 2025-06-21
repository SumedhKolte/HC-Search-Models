from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class DeploymentConfig:
    model_id: str
    model_path: str
    huggingface_token: str
    cache_dir: str = "./cache"
    use_auth: bool = True
    
    def to_dict(self) -> Dict[Any, Any]:
        return {
            "model_id": self.model_id,
            "model_path": self.model_path,
            "cache_dir": self.cache_dir,
            "use_auth": self.use_auth
        }