# models/assistant.py

from pydantic import BaseModel
from typing import Dict, Any, Optional


class AssistantConfig(BaseModel):
    configurable: Dict[str, Any]
    copilotkit_config: Optional[Dict[str, Any]] = None


class Assistant(BaseModel):
    assistant_id: str
    graph_id: str
    created_at: str
    updated_at: str
    config: AssistantConfig
    metadata: Dict[str, Any]
    node_name: str
    description: Optional[str] = None
