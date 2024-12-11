from typing import Optional
from datetime import datetime
from dataclasses import dataclass, field
from src.llm.history.llm_message import LlmMessage


@dataclass
class Node:
    id: str
    decision_point: str
    assistant_message: LlmMessage
    parent_id: Optional[str] = None
    is_initial: bool = False
    is_terminal: bool = False
    depth: int = 0
    created_at: datetime = field(default_factory=datetime.now)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.id == other.id