import hashlib
from typing import Optional
from .text_normalizer import TextNormalizer


class NodeIdentifier:
    """Handles generation and management of node identifiers"""

    def __init__(self):
        self.__normalizer = TextNormalizer()

    def generate_id(self, agent_message: str, parent_id: Optional[str] = None) -> str:
        """
        Generates a deterministic node ID based on the agent's message and parent ID.
        The same agent message under the same parent should always generate the same ID.

        :param agent_message: The message from the voice AI agent
        :param parent_id: Optional ID of the parent node
        :return A unique hash string representing this conversation state
        """
        normalized_message = self.__normalizer.normalize(agent_message)
        content_to_hash = f"{normalized_message}:{parent_id or 'root'}"

        return hashlib.sha256(content_to_hash.encode()).hexdigest()
