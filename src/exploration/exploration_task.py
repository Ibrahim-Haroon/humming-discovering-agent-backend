from dataclasses import dataclass
from src.core.model.conversation_node import ConversationNode
from src.exploration.worker.worker_context import WorkerContext


@dataclass
class ExplorationTask:
    """Represents a single exploration task"""
    node: ConversationNode
    depth: int
    context: WorkerContext
