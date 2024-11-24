from typing import Optional
from dataclasses import dataclass
from src.core.model.conversation_edge import ConversationEdge
from src.core.model.conversation_node import ConversationNode


@dataclass
class WorkerContext:
    """Context needed for a conversation worker"""
    phone_number: str
    business_type: str
    current_node: ConversationNode
    parent_edge: Optional[ConversationEdge] = None
