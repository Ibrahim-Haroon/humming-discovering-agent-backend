from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from src.core.model.conversation_edge import ConversationEdge
from src.core.model.conversation_node import ConversationNode


@dataclass
class WorkerContext:
    """Context needed for a conversation worker"""
    phone_number: str  # Target phone number to call
    business_type: str  # Type of business (used as cache key)
    current_node: Optional[ConversationNode]  # Current conversation node
    parent_edge: Optional[ConversationEdge] = None  # Edge that led to current node
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional contextual data
