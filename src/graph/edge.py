from uuid import UUID
from datetime import datetime
from dataclasses import dataclass, field
from src.llm.history.llm_message import LlmMessage


@dataclass
class Edge:
    source_node_id: UUID
    target_node_id: UUID
    user_message: LlmMessage
    created_at: datetime = field(default_factory=datetime.now)

    def __hash__(self):
        return hash((self.source_node_id, self.target_node_id))

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        return (self.source_node_id == other.source_node_id and
                self.target_node_id == other.target_node_id)
