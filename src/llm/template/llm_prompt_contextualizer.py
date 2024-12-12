from src.llm.template.llm_template import LlmTemplate


class LlmPromptContextualizer:
    """
    Handles contextual prompt generation for different conversation stages.
    """

    @staticmethod
    def generate_initial_prompt(
            business_type: str
    ):
        return LlmTemplate.initial_customer_prompt(business_type)

    @staticmethod
    def generate_response_prompt(
            business_type: str,
            response: str
    ):
        return LlmTemplate.generate_customer_prompt(
            business_type,
            response
        )


    @staticmethod
    def generate_analysis_prompt(
            business_type: str,
            text: str
    ):
        return LlmTemplate.generate_analysis_prompt(
            business_type,
            text
        )