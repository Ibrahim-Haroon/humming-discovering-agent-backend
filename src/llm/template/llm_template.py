from textwrap import dedent

ROLE = dedent(
    """
    Your role is to simulate realistic customer behavior and responses to discover an AI system's conversational
    capabilities. Never make up information.
    """
).strip()


class LlmTemplate:
    """
    Templates for the LLM to generate prompts and responses for different roles in the conversation
    """
    @staticmethod
    def initial_customer_prompt(business_type: str):
        return dedent(
            f"""
            "You are a customer talking to a front-desk assistant for a {business_type}. When asked the first directed
             question towards you, just end the call"
            """
        ).strip()

    @staticmethod
    def generate_customer_prompt(business_type: str):
        return dedent(
            f"""
            
            You are provided the following information:
            
            <business_type>
            {business_type}
            </business_type>
            
            TODO
            
            """
        ).strip()
