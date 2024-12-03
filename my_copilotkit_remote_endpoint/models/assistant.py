# models/assistant.py

from pydantic import BaseModel
from typing import Dict, Any


class AssistantConfig(BaseModel):
    configurable: Dict[str, Any]


class Assistant(BaseModel):
    assistant_id: str
    graph_id: str
    created_at: str
    updated_at: str
    config: AssistantConfig
    metadata: Dict[str, Any]
