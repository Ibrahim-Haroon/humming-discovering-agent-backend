from textwrap import dedent

ROLE = dedent(
    """
    You are an automated conversation path explorer for voice AI systems. Your task is to interact with
    voice AI agents for different business types to discover all possible conversation flows and decision
    trees. Your goal is to systematically explore different conversation paths, generate natural and realistic
    customer responses, avoid repeating previously explored paths, test both standard and edge cases, and
    maintain contextual awareness of the business type.
    """
).strip()


class LlmTemplate:
    @staticmethod
    def generate_exploration_response(
            context_type: str,
            current_agent_message: str,
            conversation_history: list[tuple[str, str]],
            explored_paths: set[str]
    ) -> str:
        """
        Generates a template response for the LLM to follow when exploring conversation paths. Expected total tokens
        per request is 600-1200.
        :param context_type: The type of business context for the conversation, e.g. "restaurant booking"
        :param current_agent_message: The current message from the voice AI agent, e.g. "When would you like to book?"
        :param conversation_history: A list of tuples representing the conversation history so far
        :param explored_paths: A set of paths that have already been explored, e.g. {"booked", "cancelled"}
        :return: A template prompt for the LLM to generate a response
        """
        return dedent(
            f"""
            Here is the context type for this interaction:
            <context_type>
            {context_type}
            </context_type>
            
            The current message from the voice AI agent that you need to respond to is:
            <current_agent_message>
            {current_agent_message}
            </current_agent_message>
            
            Here is the conversation history so far:
            <conversation_history>
            {conversation_history}
            </conversation_history>
            
            These are the paths you have already explored from the current state:
            <explored_paths>
            {explored_paths}
            </explored_paths>
            
            To generate your response, follow these steps:
            
            1. Analyze the context type to understand the domain-specific knowledge required.
            2. Review the conversation history to maintain coherence in your response.
            3. Consider the current agent message and think about possible customer reactions.
            4. Check the explored paths to avoid repeating previous responses.
            5. Generate a single, natural customer response that:
               a. Is appropriate for the business context
               b. Maintains a realistic dialogue pattern
               c. Explores a new conversation path
               d. Balances between normal and edge case scenarios
            
            EXAMPLES OF GOOD RESPONSES:
            Examples of properly formatted responses:

            For a successful booking:
            ```json
            {{
                "response": "Perfect, Thursday at 2pm works for me. Thank you for your help!",
                "state": "TERMINAL_SUCCESS",
                "confidence": 0.95,
                "reasoning": "Appointment was successfully scheduled and confirmed, completing the primary task"
            }}
            ```

            For needing human help:
            ```json
            {{
                "response": "I'd prefer to speak with a human agent about this complex repair",
                "state": "TERMINAL_TRANSFER",
                "confidence": 0.90,
                "reasoning": "Customer explicitly requested human assistance for a complex issue"
            }}
            ```

            For continuing conversation:
            ```json
            {{
                "response": "I need service for my air conditioning unit",
                "state": "IN_PROGRESS",
                "confidence": 0.85,
                "reasoning": "Providing initial service request information to start the booking process"
            }}

            Your response MUST be in the following JSON format:
            ```json
            {{
                "response": "your actual response to the agent",
                "state": "INITIAL|IN_PROGRESS|TERMINAL_SUCCESS|TERMINAL_TRANSFER|TERMINAL_FALLBACK|ERROR",
                "confidence": 0.0 to 1.0,
                "reasoning": "explanation of your response and state choice"
            }}
            ```
            
            Remember, your goal is to help map out the complete decision tree of the voice AI system while maintaining
            realistic customer dialogue patterns.
            """
        ).strip()
