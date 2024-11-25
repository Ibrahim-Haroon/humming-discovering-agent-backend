from dataclasses import dataclass, field
from typing import Set, Optional, Dict, Any
from datetime import datetime
from ..enum.conversation_state import ConversationState
from ..service.node_identifier import NodeIdentifier
from ..service.response_similarity import ResponseSimilarity


@dataclass
class ConversationNode:
    """
    Represents a node in the conversation graph where a voice AI agent has spoken and awaits user input.
    Each node maintains its state, transcript, and keeps track of explored responses.

    :param conversation_transcription: Complete transcript of the conversation at this node
    :type conversation_transcription: str
    :param state: Current state of the conversation (e.g., INITIAL, IN_PROGRESS)
    :type state: ConversationState
    :param parent_id: Identifier of the parent node, None for root node
    :type parent_id: Optional[str]
    :param explored_responses: Set of customer prompts already tried at this node
    :type explored_responses: Set[str]
    :param metadata: Additional contextual data about the node
    :type metadata: Dict[str, Any]
    :param created_at: Timestamp when the node was created
    :type created_at: datetime

    :raises ValueError: If attempting to create a root node (parent_id=None) with non-INITIAL state
    """
    conversation_transcription: str  # The full conversation transcript
    state: ConversationState  # Current state of this node
    parent_id: Optional[str] = None  # ID of the parent node
    explored_responses: Set[str] = field(default_factory=set)  # Customer prompts tried
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional data
    created_at: datetime = field(default_factory=datetime.now)
    _id: Optional[str] = None  # Internal node ID

    def __post_init__(self):
        """Initialize node ID and services"""
        self._node_identifier = NodeIdentifier()
        self.__response_similarity = ResponseSimilarity()

        if self._id is None:
            self._id = self._node_identifier.generate_id(self.conversation_transcription, self.parent_id)

        # Validate state transitions
        if self.parent_id is None and self.state != ConversationState.INITIAL:
            raise ValueError("Root node must have INITIAL state")

    @property
    def id(self) -> str:
        """
        Get the unique identifier for this node.

        :returns: Deterministic hash based on conversation transcript and parent ID
        :rtype: str
        """
        return self._id

    def add_response(self, prompt: str) -> bool:
        """
        Record a customer prompt that's been attempted at this node.

        :param prompt: The customer response to record
        :type prompt: str
        :returns: False if a similar response has already been explored, True if new
        :rtype: bool

        :note: Uses fuzzy matching to determine similarity between responses
        """
        if self.__response_similarity.find_similar(prompt, self.explored_responses):
            return False

        self.explored_responses.add(prompt)
        return True

    def is_terminal(self) -> bool:
        """
        Check if this node represents an end state in the conversation.

        :returns: True if node state starts with 'TERMINAL', False otherwise
        :rtype: bool
        """
        return self.state.name.startswith('TERMINAL')

    def has_similar_response(self, response: str) -> bool:
        """
        Check if a semantically similar response has already been explored.

        :param response: Response to check for similarity
        :type response: str
        :returns: True if a similar response exists in explored_responses
        :rtype: bool

        :note: Uses same similarity matching algorithm as add_response()
        """
        return self.__response_similarity.find_similar(response, self.explored_responses) is not None
