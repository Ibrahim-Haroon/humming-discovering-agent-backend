from typing import Literal
from dataclasses import dataclass


@dataclass
class LlmMessage:
    role: Literal["user", "assistant"]  # Agnostic roles across providers
    content: str
