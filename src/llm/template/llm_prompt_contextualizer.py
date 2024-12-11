from src.llm.template.llm_template import LlmTemplate


class LlmPromptContextualizer:
    """
    Handles contextual prompt generation for different conversation stages.
    """

    @staticmethod
    def generate_initial_prompt(
            business_type
    ):
        return LlmTemplate.initial_customer_prompt(business_type)
