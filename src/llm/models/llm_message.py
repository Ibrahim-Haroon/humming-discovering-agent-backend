from typing import Literal
from dataclasses import dataclass


@dataclass
class LlmMessage:
    """
    Represents a message so it can be passed in the payload of request to LLM as conversation history
    """
    role: Literal["user", "assistant"]  # Agnostic roles across providers
    content: str
