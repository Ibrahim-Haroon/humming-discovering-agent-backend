from textwrap import dedent

CUSTOMER_ROLE = dedent(
    """
    You're an agent testing AI front-desk assistant services, you must instruct another AI with how to respond to the
    AI front-desk assistant service.
     
    """
).strip()

ANALYSIS_ROLE = dedent(
    """
    You're an expert at analyzing customer service transcription calls
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
    def response_customer_prompt(
            business_type: str,
            response: str,
    ):
        return dedent(
            f"""
            You are provided the following information:
            
            <business_type>
            {business_type}
            </business_type>
            
            <response>
            {response}
            </response>
            
            Your task is to generate instructions for an AI customer simulator.
            
            CRITICAL: The instructions must contain EXACTLY:
            1. Base context
            2. ALL previous instructions (except end call parts)
            3. ONE new instruction that uses the provided response
            4. ONE end call instruction
            
            RULES FOR NEW INSTRUCTION:
            - Must use the response provided
            - NO adding extra instructions
            - NO predicting next responses
             
            Example:
            Given response "Yes, I am an existing customer":
            CORRECT: "You are a customer talking to a front-desk assistant for Air Conditioning and Plumbing company.
            When asked if you are an existing customer, say Yes, I'm an existing customer. For any other questions, end
            call"
            WRONG: "...say Yes, I'm an existing customer. When asked about service needed..." (adds extra instruction)
            
            If asked for personal information, just make up something simple.
            
            Complete format should be sentences containing:
            1. "You are a customer talking to a front-desk assistant for {business_type}"
            2. [All previous When/Say instructions] (excluding end call)
            3. "When asked [question], say [response provided]"
            4. "For any other questions, end call"
            
            Return ONLY these parts. Do not add ANY additional instructions beyond what's in the previous instructions
            plus ONE new instruction for the exact response provided. ONE end call in the entire response.
            """
        ).strip()

    @staticmethod
    def transcription_analysis_prompt(business_type: str, transcript: str):
        return dedent(
            f"""
            You are provided the following information:

            <business_type>
            {business_type}
            </business_type>
            
            <transcript>
            {transcript}
            </transcript>
            
            Your task is to analyze transcripts from AI receptionist calls. These transcripts contain mixed dialogue
            that must be carefully separated and analyzed.
            
            CRITICAL: You must follow these exact steps:
            
            1. SEPARATE THE SPEAKERS
            - Front desk assistant statements:
              * Always start with greetings/introductions
              * Usually longer, formal language
              * Example: "hello thank you for calling..."
            
            - Customer statements:
              * Usually short responses
              * Example: "yes, I'm a customer", "no, this is not an emergency", etc.
              * Often will say "hang up", "end call", etc. (this is a meta-instruction, not actual dialogue)
            
            - Closing statements:
              * Anything after "hang up", "goodbye", etc. is just closing
              * Example: "the call has ended goodbye"
              * Ignore these for analysis
            
            2. MAP QUESTION-ANSWER PAIRS
            - Match front desk questions with customer answers.  
            - Mark a question as COMPLETED once answered.  
            - For repeated questions, only the last instance is active.
            
            3. FIND THE LAST ACTIVE QUESTION            
            - Start from the end of the call (ignore "hang up").  
            - Work backward to find the first unanswered question.  
            - If all questions are answered, check for terminal states.  
            
            4. Determine if encountered situation before
            - If a situation has been analyzed before, then immediately return false
            - Ex. Told the agent you're facing a problem, if next time you're asked again about the problem you're
              facing then just return "True|" since outcome will be the same/similar. 
            
            5. DETERMINE IF TERMINAL
            Terminal states (return "True|"):
            - Call transferred to human agent
            - Appointment confirmed 
            - Service completed
            - No unanswered questions asked before call end
            
            Non-terminal states (return "False|response1;response2;etc"):
            - Last statement is a question
            - Question requires customer input

            6. GENERATE RESPONSE OPTIONS
            For non-terminal states only:
            - Must be natural customer speech
            - Must directly answer the last question
            - Must include ALL valid options
            - NEVER include "hang up" or meta-instructions as responses
            - If asked for personal information (name, address, phone, etc), generate exactly ONE generic response 
            - multiple variations of personal details create unnecessary paths
            - Format with EXACT separators: | between terminal flag and responses, ; between responses
                        
            Examples using proper parsing:
            
            Transcript: "hello thank you for calling plumbing this is john are you an existing customer hang up goodbye"
            Parsing:
            Front desk: "hello thank you for calling plumbing this is john are you an existing customer"
            Last statement: "are you an existing customer"
            Response: "False|Yes, I am an existing customer;No, I'm not an existing customer"
            
            Transcript: "hello thank you for calling plumbing are you an existing customer yes i am an existing customer
            is this an emergency end call goodbye"
            Parsing:
            Front desk: "hello thank you for calling anthem air conditioning and plumbing this is olivia speaking are
            you an existing customer"
            Customer: "yes i'm an existing customer"
            Front desk: "is this an emergency"
            Last statement: "is this an emergency"
            Response: "False|Yes, this is an emergency;No, this is not an emergency"
            
            Transcript: "hello thank you for calling anthem air conditioning and plumbing this is olivia speaking are
            you an existing customer yes i am an existing customer is this an emergency no this is not an emergency what
            kind of issue are you facing thank you for your assistance goodbye"
            Parsing:
            Front desk: "hello thank you for calling anthem air conditioning and plumbing this is olivia speaking are
            you an existing customer"
            Customer: "yes i'm an existing customer"
            Front desk: "is this an emergency"
            Customer: "no this is not an emergency"
            Front desk: "what kind of issue are you facing"
            Customer: "thank you for your assistance goodbye"
            Last statement: "what kind of issue are you facing"
            Response: "False|The issue I'm facing is that my hot water is not working"
            
            Transcript: "hello thank you for calling anthem air conditioning and plumbing are you an existing customer
            no i am not an existing customer may i have your name and physical address please thank you for your
            help goodbye"
            Parsing:
            Front desk: "hello thank you for calling anthem air conditioning and plumbing are you an existing customer"
            Customer: "no i am not an existing customer"
            Front desk: "may i have your name and physical address please"
            Last statement: "may i have your name and physical address please"
            Response: "False|My name is --made up name-- and my address is --made up address--"
            
            
            Transcript: "hello thank you for calling plumbing are you an existing customer yes i am an existing customer
            is this an emergency yes this is an emergency transferring you to an agent now goodbye"
            Parsing:
            Front desk: "hello thank you for calling plumbing are you an existing customer"
            Customer: "yes i am an existing customer"
            Front desk: "is this an emergency"
            Customer: "yes this is an emergency"
            Front desk: "transferring you to an agent now goodbye"
            Last statement: "transferring you to an agent now goodbye"
            Response: "True|"

            For non terminal, it won't always be a single or binary response. For example it would be something like
            this if multiple options given:
                "False|Yes, I'm a gold customer;Yes, I'm a silver customer;Yes, I'm a bronze customer"
            
            If there a 0 possible responses, then that means the node is terminal. So if you think the analysis is
            "False|" then that means it's actually "True|"
            
            Provide your analysis in a single line using the exact format specified.
            """
        ).strip()
