from src.llm.llm_template import LlmTemplate


class LlmPromptContextualizer:
    @staticmethod
    def generate(
            context_type: str,
            current_agent_message: str,
            conversation_history: list[tuple[str, str]],
            explored_paths: set[str]
    ) -> str:
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
        return LlmTemplate.generate_analysis_prompt(
            context_type,
            customer_prompt,
            conversation_transcript
        )
