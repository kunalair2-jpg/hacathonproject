from openenv.client import EnvClient
from typing import Dict, Any, Optional

class EmailTriageEnvClient(EnvClient):
    """
    Client wrapper for the Email Triage Environment.
    Follows the standard EnvClient interface.
    """
    def __init__(self, url: str):
        super().__init__(url=url)
    
    def reset(self, task_name: str = "easy") -> Dict[str, Any]:
        """Resets the environment and returns the initial observation."""
        response = self._request("POST", "/reset", json={"task_name": task_name})
        return response
    
    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Submits an action and returns (observation, reward, done, info)."""
        response = self._request("POST", "/step", json={"action": action})
        return response
        
    def state(self) -> Dict[str, Any]:
        """Returns the internal state of the environment."""
        response = self._request("GET", "/state")
        return response
