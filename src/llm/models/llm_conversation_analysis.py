from typing import Optional, List
from dataclasses import dataclass


@dataclass
class LlmConversationAnalysis:
    is_terminal: bool
    possible_responses: Optional[List[str]]  # empty if terminal
