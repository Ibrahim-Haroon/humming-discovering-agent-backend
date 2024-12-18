import re
from uuid import UUID
from difflib import SequenceMatcher
from threading import RLock
from typing import Dict, Set, Optional, List
from src.graph.edge import Edge
from src.graph.node import Node
from src.util.singleton import singleton
from src.llm.models.llm_message import LlmMessage


@singleton
class ConversationGraph:
    def __init__(self, node_similarity_threshold: float = 0.60):
        self.__root_id: Optional[UUID] = None
        self.__nodes: Dict[UUID, Node] = {}
        self.__edges: Set[Edge] = set()
        self.__node_similarity_threshold = node_similarity_threshold
        self.__lock = RLock()

    @property
    def nodes(self):
        return self.__nodes.copy()

    @property
    def edges(self):
        return self.__edges.copy()

    def add_node(self, node: Node):
        with self.__lock:
            if not self.__root_id:
                if not node.is_initial:
                    raise ValueError("Graph must have INITIAL state")
                self.__root_id = node.id
                self.__nodes[node.id] = node
                return

            if node.is_terminal:
                self.__nodes[node.id] = node
                return node.id

            similar_node = self.__find_similar_node(
                node.decision_point,
            )

            if not similar_node:
                self.__nodes[node.id] = node
                return node.id

            return similar_node.id

    def add_edge(self, edge: Edge):
        with self.__lock:
            if edge.source_node_id not in self.__nodes or edge.target_node_id not in self.__nodes:
                raise ValueError("Edge nodes must exist in graph")
            self.__edges.add(edge)

    def build_conversation_history(self, node_id: UUID) -> List[LlmMessage]:
        """
        Builds the conversation history by walking up the graph from the given node to the root.
        Returns list of messages in chronological order (root to current node).
        """
        with self.__lock:
            messages: List[LlmMessage] = []
            current_node_id = node_id

            while current_node_id is not None:
                current_node = self.__nodes[current_node_id]
                messages.append(current_node.assistant_message)

                if current_node.parent_id is not None:
                    edge = next(
                        edge for edge in self.__edges
                        if edge.source_node_id == current_node.parent_id
                        and edge.target_node_id == current_node_id
                    )
                    messages.append(edge.user_message)

                current_node_id = current_node.parent_id

            return list(reversed(messages))

    def __find_similar_node(self, decision_point: str) -> Optional[Node]:
        normalized_input = self.__normalize_text(decision_point)
        for node in self.__nodes.values():
            normalized_node = self.__normalize_text(node.decision_point)
            similarity = SequenceMatcher(None, normalized_input, normalized_node).ratio()
            if similarity >= self.__node_similarity_threshold:
                return node
        return None

    @staticmethod
    def __normalize_text(text: str) -> str:
        return ' '.join((re.sub(r'[^\w\s]', '', text)).lower().split())
