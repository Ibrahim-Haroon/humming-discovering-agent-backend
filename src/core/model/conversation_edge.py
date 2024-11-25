from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass(frozen=True)
class ConversationEdge:
    """
    Represents a transition between two conversations via a customer prompt.
    Immutable to ensure edge integrity in the graph.
    """
    source_node_id: str  # ID of the source node
    target_node_id: str  # ID of the target node
    response: str  # The customer prompt that led to this conversation
    timestamp: datetime  # When this transition occurred
    metadata: Optional[dict] = None  # Additional metadata about the transition

    def __hash__(self) -> int:
        """Custom hash implementation for edge uniqueness in sets"""
        return hash((self.source_node_id, self.target_node_id, self.response))
