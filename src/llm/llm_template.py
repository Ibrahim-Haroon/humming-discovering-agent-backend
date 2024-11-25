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

CUSTOMER_ROLE = dedent(
    """
    You are a customer interacting with a voice AI agent for a specific business type. Your goal is to engage in
    natural conversations with the agent, ask relevant questions, provide necessary information, and guide the
    conversation towards a successful outcome. Your role is to simulate realistic customer behavior and responses
    to help the AI system learn and improve its conversational capabilities.
    """
).strip()

ANALYSIS_ROLE = dedent(
    """
    You are an AI assistant analyzing conversations between a voice agent and customers.
    Your goal is to explore diverse conversation paths by:
    1. Marking conversations as IN_PROGRESS unless they absolutely require termination
    2. Only using TERMINAL states when:
       - Customer explicitly requests human transfer
       - Agent fails to handle request
       - System errors occur
    3. Treating "successful" interactions as opportunities for follow-up questions
    """
).strip()


class LlmTemplate:
    """
    Templates for the LLM to generate prompts and responses for different roles in the conversation
    """
    @staticmethod
    def generate_analysis_prompt(
        context_type: str,
        customer_prompt: str,
        conversation_transcript: str
    ):
        return dedent(
            f"""
            You will be provided with two pieces of information:
            1. The business context type
            2. The initial prompt from the customer
            3. The full conversation transcript
            
            Here is the business context type:
            <context_type>
            {context_type}
            </context_type>
            
            Here is the initial customer prompt:
            <prompt>
            {customer_prompt}
            </prompt>
            
            Here is the full conversation transcript:
            <transcript>
            {conversation_transcript}
            </transcript>
            
            Analyze the conversation carefully, considering both the initial prompt and the full transcript in the
            context of the business domain. Determine the following:
            
            1. Which state the conversation reached
            2. Why it reached that state (reasoning)
            3. Whether the conversation successfully completed its intended purpose
            4. Confidence level in your analysis (0.0 to 1.0)
            5. Whether the customer's needs were adequately addressed
            
            The possible conversation states are:
            1. TERMINAL_SUCCESS - Customer got what they needed without human intervention
            2. TERMINAL_TRANSFER - Conversation properly concluded with transfer to human agent
            3. TERMINAL_FALLBACK - Voice agent couldn't handle the request and had to end conversation
            4. IN_PROGRESS - Conversation didn't reach a natural conclusion
            5. ERROR - Something went wrong in the conversation flow
            
            Provide your analysis in the following format:
            ```json
            {{
                "response": "Your analysis of the conversation",
                "state": "TERMINAL_SUCCESS|TERMINAL_TRANSFER|TERMINAL_FALLBACK|IN_PROGRESS|ERROR",
                "confidence": 0.0 to 1.0,
                "reasoning": "Detailed explanation of why this state was reached"
            }}
            ```
            
            
            Here are examples of analysis for different conversation states:
            
            1. TERMINAL_SUCCESS example:
                ```json
                {{
                    "response": "Analysis of AC unit information conversation",
                    "state": "TERMINAL_SUCCESS",
                    "confidence": 0.95,
                    "reasoning": "The customer received comprehensive information about AC units, costs, and financing.
                    The conversation reached a natural conclusion when the customer expressed they needed time to
                    think about the options. All customer queries were answered satisfactorily."
                }}
                ```
            
            2. TERMINAL_TRANSFER example:
                ```json
                {{
                    "response": "Analysis of emergency water leak",
                    "state": "TERMINAL_TRANSFER",
                    "confidence": 0.98,
                    "reasoning": "The customer reported an emergency water leak requiring immediate attention. 
                    The agent correctly identified the urgency and initiated a transfer to a human agent. The 
                    conversation concluded appropriately given the emergency nature."
                }}
                ```
            
            3. IN_PROGRESS example:
                ```json
                {{
                    "response": "Analysis of incomplete AC service inquiry",
                    "state": "IN_PROGRESS",
                    "confidence": 0.85,
                    "reasoning": "The conversation did not reach a natural conclusion. The agent was still attempting
                    to gather necessary information about the AC issue, but the conversation ended without resolution
                    or clear next steps."
                }}
                ```
            
            Remember to consider the full context of the conversation, including the initial prompt and the entire
            transcript within the context of the business domain. Your analysis should be thorough and reflect a
            deep understanding of the conversation dynamics and the customer's needs.
            """
        ).strip()

    @staticmethod
    def generate_customer_prompt(
            context_type: str,
            conversation_history: list[tuple[str, str]],
            explored_paths: set[str]
    ):
        return dedent(
            f"""
            You are tasked with generating a new prompt for a voice agent conversation discovery application. Your goal
            is to create a prompt that will explore a different, untested conversation path by analyzing the given 
            conversation history and considering the already explored paths.

            First, review the following inputs:
            
            <context_type>
            {context_type}
            </context_type>
            
            This provides information about the type of business or service the voice agent represents.
            
            <conversation_history>
            {conversation_history}
            </conversation_history>
            
            This is a list of pairs, where Pair[0] is the prompt and Pair[1] is the conversation transcription. Will
            be empty initially.
            
            <explored_paths>
            {explored_paths}
            </explored_paths>
            
            This is a set of paths that have already been explored. Will be empty initially.
            
            To generate a new prompt, follow these steps:
            
            1. Analyze the conversation history:
               - Identify prompts that did not lead to a complete end of the conversation.
               - Determine which aspects of these prompts could be improved.
               - Look for patterns in successful conversations that reached a natural conclusion.
            
            2. Consider the explored paths:
               - Ensure your new prompt will lead to a conversation path that hasn't been tested yet.
               - Think about potential scenarios or customer needs that haven't been addressed in previous 
                 conversations.
            
            3. Generate a new prompt:
               - Based on your analysis, create a prompt that addresses an unexplored conversation path or improves upon
                 an incomplete conversation.
               - Ensure the prompt is relevant to the context type provided.
               - Include specific details that will guide the conversation, such as customer status, reason for calling,
                 and potential questions or concerns.
            
            4. Format and style requirements:
               - Write the prompt in the first person, as if you are the customer speaking directly to the voice agent.
               - Include a clear intent for the call and enough information to start and guide the conversation.
               - Provide a natural way to conclude the conversation, such as expressing satisfaction with the 
                 information received or indicating a need to think about the options presented.
            
            5. Examples:
               Good prompt example:
               "Hi, I'm calling about my water heater. It's not producing hot water consistently, and I'm wondering if
                it needs to be repaired or replaced. I've been a customer for about 3 years. Can you help me understand
                my options and potentially schedule a service appointment?"
            
               Bad prompt example:
               "When the agent asks if I'm a customer, say yes. Then ask about water heater issues. If they offer a
                repair, ask about replacement options too."
            
            Output your generated prompt inside <prompt> tags. Ensure that the prompt is written in the first person
            and provides enough context and information to guide a complete conversation with the voice agent, from
            introduction to conclusion.
            
            <prompt>
            [Your generated prompt goes here]
            </prompt>
            """
        ).strip()

    @staticmethod
    def generate_exploration_response(
            context_type: str,
            current_agent_message: str,
            conversation_history: list[tuple[str, str]],
            explored_paths: set[str]
    ) -> str:
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
