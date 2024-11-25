from src.llm.llm_template import LlmTemplate


class LlmPromptContextualizer:
    """
    Handles contextual prompt generation for different conversation stages.
    """

    @staticmethod
    def generate_continuation_prompt(
            context_type: str,
            current_agent_message: str,
            conversation_history: list[tuple[str, str]],
            explored_paths: set[str]
    ) -> str:
        """
        Generate contextual prompt for continuing conversation.

        :param context_type: Type of business/service
        :type context_type: str
        :param current_agent_message: Latest message from agent
        :type current_agent_message: str
        :param conversation_history: List of (prompt, response) pairs
        :type conversation_history: list[tuple[str, str]]
        :param explored_paths: Previously explored conversation paths
        :type explored_paths: set[str]
        :returns: Generated prompt
        :rtype: str
        """
        return LlmTemplate.generate_exploration_response(
            context_type,
            current_agent_message,
            conversation_history,
            explored_paths
        )

    @staticmethod
    def generate_customer_prompt(
            context_type: str,
            conversation_history: list[tuple[str, str]],
            explored_paths: set[str]
    ) -> str:
        """
        Generate prompts in the context of a customer (first person).

        :param context_type: Type of business/service
        :type context_type: str
        :param conversation_history: Previous conversations as (prompt, response) pairs
        :type conversation_history: list[tuple[str, str]]
        :param explored_paths: Previously explored paths
        :type explored_paths: set[str]
        :returns: Generated customer prompt
        :rtype: str
        """
        return LlmTemplate.generate_customer_prompt(
            context_type,
            conversation_history,
            explored_paths
        )

    @staticmethod
    def generate_analysis_prompt(
            context_type: str,
            customer_prompt: str,
            conversation_transcript: str
    ) -> str:
        """
        Generate prompt for analyzing conversation outcome.

        :param context_type: Type of business/service
        :type context_type: str
        :param customer_prompt: Initial customer prompt
        :type customer_prompt: str
        :param conversation_transcript: Full conversation transcript
        :type conversation_transcript: str
        :returns: Analysis prompt
        :rtype: str
        """
        return LlmTemplate.generate_analysis_prompt(
            context_type,
            customer_prompt,
            conversation_transcript
        )
