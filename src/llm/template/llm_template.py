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
    def generate_customer_prompt(business_type: str, response: str):
        return dedent(
            f"""
            
            You are provided the following information:
            
            <business_type>
            {business_type}
            </business_type>
            
            <response>
            {response}
            </response>
            
            You must go based on the business type, the response you should respond with, and how previous calls went.
            Construct the AI on how to answer the question which the call ended on previously. 
            
            One thing you will notice for the call transcription is that it's a mixture of the front-desk assistant and
            the AI talking simultaneously, which you're instructing on how to respond. You must discern between them 
            both to understand what's happening
            
            For example:
            previous call: hello thank you for calling turbo this is tom speaking are you an existing customer with us
            end call thank you for calling turbo goodbye
            
            question you need to answer: are you an existing customer?
            
            good instructions: You are a customer talking to a front-desk assistant for a {business_type}. When asked
            if you're an existing customer say yes, then just end the call after they ask the next question
        
            You *must* instruct with ending the call after the new question has been answered. Also just used the
            format of `When asked {{X}}, respond with {{Y}} if it makes sense. Do not overcomplicate. 
            """
        ).strip()

    @staticmethod
    def generate_analysis_prompt(business_type: str, text: str):
        return dedent(
            f"""
            You are provided the following information:

            <business_type>
            {business_type}
            </business_type>
            
            <text>
            {text}
            </text>
            
            You must go based on the business type, text which is a transcription of the call, and how previous calls
            went. You need to analyze whether the call went to completion, for example when the front-desk assistant
            hangs up the call. This can be because it transferred to a human, cannot help, or completed everything
            alone.
            
            One thing you will notice for the call transcription is that it's a mixture of the front-desk assistant and
            the AI talking simultaneously, which you're instructing on how to respond. You must discern between them 
            both to understand what's happening
            
            If it's not terminal then you need to determine all possible responses, it can be binary or also non-binary.
            Here are examples of both:
                Non-binary:
                    hello thank you for calling turbo this is tom speaking are you a gold, silver, or bronze customer 
                    with us end call thank you for calling turbo goodbye
                    
                    Response: False|Yes, I'm a gold customer;Yes, I'm a silver customer;Yes, I'm a bronze customer
                Binary:
                    hello thank you for calling turbo this is tom speaking is this an emergency end call thank you for
                    calling turbo goodbye
                    
                    Response: False|Yes, this is an emergency; No, this is not an emergency
            
            Finally here is an example of terminal:
            hello thank you for calling turbo this is tom speaking is this an emergency yes this is an emergency
            transferring to human
            
            Response: True|
                
            You *must* always return your response in this format
                Example for terminal is True:
                True|
                Example for terminal is False:
                False|Yes, I'm an existing customer;No, I'm not an existing customer
                
            There are two splitters so you cannot use them anywhere else: | and ;
            """
        ).strip()
