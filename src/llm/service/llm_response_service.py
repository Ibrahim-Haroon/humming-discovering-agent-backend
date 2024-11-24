from abc import ABC, abstractmethod


class LlmResponseService(ABC):
    @abstractmethod
    def response(self, role: str, prompt: str) -> str:
        """
        This method is used as a baseline behavior for all LLM Response Services

        :param role: The behavior/persona for the model to inherit
        :param prompt: The task for the model to complete
        :return: response from LLM
        :rtype: str
        """
        pass
