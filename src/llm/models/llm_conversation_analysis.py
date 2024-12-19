from typing import Optional, List
from dataclasses import dataclass


@dataclass
class LlmConversationAnalysis:
    """
    Represents the analysis of a conversation
    """
    is_terminal: bool
    possible_responses: Optional[List[str]]  # empty if terminal
