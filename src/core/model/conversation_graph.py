from collections import deque
from threading import RLock
from typing import Dict, Set, Optional, List, Iterator
from .conversation_node import ConversationNode
from .conversation_edge import ConversationEdge
from ..enum.conversation_state import ConversationState
from src.util.singleton import singleton


@singleton
class ConversationGraph:
    """
    Thread-safe graph structure representing the entire conversation tree discovered during exploration.
    Manages nodes and edges while ensuring concurrent access safety.

    :ivar nodes: Dictionary mapping node IDs to ConversationNode objects
    :type nodes: Dict[str, ConversationNode]
    :ivar edges: Set of ConversationEdge objects connecting nodes
    :type edges: Set[ConversationEdge]
    :ivar root_id: ID of the root node
    :type root_id: Optional[str]
    """
    def __init__(self):
        self.nodes: Dict[str, ConversationNode] = {}
        self.edges: Set[ConversationEdge] = set()
        self.root_id: Optional[str] = None
        self._lock = RLock()  # For thread safety

    def add_node(self, node: ConversationNode) -> None:
        """
        Add a new node to the graph with thread-safety.

        :param node: Node to add to the graph
        :type node: ConversationNode
        :raises ValueError: If first node isn't in INITIAL state
        """
        with self._lock:
            if not self.root_id:
                if node.state != ConversationState.INITIAL:
                    raise ValueError("First node must have INITIAL state")
                self.root_id = node.id
            self.nodes[node.id] = node

    def add_edge(self, edge: ConversationEdge) -> None:
        """
        Add a new edge to the graph with thread-safety.

        :param edge: Edge to add to the graph
        :type edge: ConversationEdge
        :raises ValueError: If source or target nodes don't exist in graph
        """
        with self._lock:
            if edge.source_node_id not in self.nodes:
                raise ValueError(f"Source node {edge.source_node_id} not found")
            if edge.target_node_id not in self.nodes:
                raise ValueError(f"Target node {edge.target_node_id} not found")
            self.edges.add(edge)

    def get_node(self, node_id: str) -> Optional[ConversationNode]:
        """
        Retrieve a node by its ID.

        :param node_id: ID of node to retrieve
        :type node_id: str
        :returns: Node if found, None otherwise
        :rtype: Optional[ConversationNode]
        """
        return self.nodes.get(node_id)

    def get_children(self, node_id: str) -> List[ConversationNode]:
        """
        Get all immediate child nodes of a given node.

        :param node_id: ID of parent node
        :type node_id: str
        :returns: List of child nodes
        :rtype: List[ConversationNode]
        """
        children = []
        for edge in self.edges:
            if edge.source_node_id == node_id:
                child = self.nodes.get(edge.target_node_id)
                if child:
                    children.append(child)
        return children

    def get_path_to_node(self, node_id: str) -> List[ConversationEdge]:
        """
        Get sequence of edges forming path from root to specified node.

        :param node_id: Target node ID
        :type node_id: str
        :returns: Ordered list of edges from root to target
        :rtype: List[ConversationEdge]
        """
        if node_id not in self.nodes:
            return []

        path = []
        current = node_id

        while current != self.root_id:
            edge = self.__find_edge_to_node(current)
            if not edge:
                break
            path.append(edge)
            current = edge.source_node_id

        return list(reversed(path))

    def __find_edge_to_node(self, target_id: str) -> Optional[ConversationEdge]:
        """Helper to find edge leading to a node"""
        for edge in self.edges:
            if edge.target_node_id == target_id:
                return edge
        return None

    def iter_bfs(self) -> Iterator[ConversationNode]:
        """
        Iterate through nodes in breadth-first order.

        :returns: Iterator yielding nodes in BFS order
        :rtype: Iterator[ConversationNode]
        """
        if not self.root_id:
            return

        queue = deque([self.root_id])
        seen = {self.root_id}

        while queue:
            current_id = queue.popleft()
            yield self.nodes[current_id]

            for child in self.get_children(current_id):
                if child.id not in seen:
                    seen.add(child.id)
                    queue.append(child.id)

    def iter_dfs(self) -> Iterator[ConversationNode]:
        """
        Iterate through nodes in depth-first order.

        :returns: Iterator yielding nodes in DFS order
        :rtype: Iterator[ConversationNode]
        """
        if not self.root_id:
            return

        stack = [self.root_id]
        seen = {self.root_id}

        while stack:
            current_id = stack.pop()
            yield self.nodes[current_id]

            for child in reversed(self.get_children(current_id)):
                if child.id not in seen:
                    seen.add(child.id)
                    stack.append(child.id)
