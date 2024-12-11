from abc import ABC, abstractmethod
from typing import Optional, List
from src.llm.history.llm_message import LlmMessage


class LlmResponseService(ABC):
    @abstractmethod
    def response(self, role: str, prompt: str, conversation_history: Optional[List[LlmMessage]]) -> str:
        """
        This method is used as a baseline behavior for all LLM Response Services

        :param role: The behavior/persona for the model to inherit
        :param prompt: The task for the model to complete
        :param conversation_history: The conversation history to provide context for the model
        :return: response from LLM
        :rtype: str
        """
        pass
