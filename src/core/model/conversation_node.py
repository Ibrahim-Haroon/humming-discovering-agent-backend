from dataclasses import dataclass, field
from typing import Set, Optional, Dict, Any
from datetime import datetime
from ..enum.conversation_state import ConversationState
from ..service.node_identifier import NodeIdentifier
from ..service.response_similarity import ResponseSimilarity


@dataclass
class ConversationNode:
    """
    Represents a point in the conversation where the voice AI agent has spoken and is waiting for user input.
    """
    agent_message: str  # The message from the voice AI agent
    state: ConversationState  # Current state of this node
    parent_id: Optional[str] = None  # ID of the parent node
    explored_responses: Set[str] = field(default_factory=set)  # Responses tried
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional data
    created_at: datetime = field(default_factory=datetime.now)
    _id: Optional[str] = None  # Internal node ID

    def __post_init__(self):
        """Initialize node ID and services"""
        self._node_identifier = NodeIdentifier()
        self.__response_similarity = ResponseSimilarity()

        if self._id is None:
            self._id = self._node_identifier.generate_id(self.agent_message, self.parent_id)

        # Validate state transitions
        if self.parent_id is None and self.state != ConversationState.INITIAL:
            raise ValueError("Root node must have INITIAL state")

    @property
    def id(self) -> str:
        """Unique identifier for this node"""
        return self._id

    def add_response(self, response: str) -> bool:
        """
        Add a response that's been tried from this node.

        :param response: The user response to add
        :returns False if a similar response has already been explored
        """
        if self.state.name.startswith('TERMINAL'):
            raise ValueError("Cannot add responses to terminal nodes")

        if self.__response_similarity.find_similar(response, self.explored_responses):
            return False

        self.explored_responses.add(response)
        return True

    def is_terminal(self) -> bool:
        """Check if this node represents an end state"""
        return self.state.name.startswith('TERMINAL')

    def has_similar_response(self, response: str) -> bool:
        """
        Check if a similar response has already been explored

        :param response: Response to check
        :returns True if a similar response exists
        """
        return self.__response_similarity.find_similar(response, self.explored_responses) is not None
