import re
import json
from dataclasses import dataclass
from src.core.enum.conversation_state import ConversationState


@dataclass
class ParsedLlmResponse:
    response: str
    state: ConversationState
    confidence: float
    reasoning: str


class LlmResponseParser:
    """
    Handles parsing and validation of LLM responses.

    This class is responsible for extracting and validating the necessary information from the raw LLM response. It
    ensures that the response contains the required fields and that the values are within the expected ranges.

    The parsed response is returned as a `ParsedLlmResponse` dataclass, which includes the response text, conversation
    state, confidence level, and reasoning.
    """

    @staticmethod
    def parse_customer_prompt(customer_prompt) -> str:
        """
        Extracts the customer prompt from the raw LLM response.
        :param customer_prompt: Raw response from LLM
        :return: Extracted customer prompt
        :rtype: str
        """
        match = re.search(r'<prompt>(.*?)</prompt>', customer_prompt, re.DOTALL)
        prompt = match.group(1).strip() if match else customer_prompt  # returns LiteralString type if matched
        return str(prompt)

    @staticmethod
    def parse(llm_response: str) -> ParsedLlmResponse:
        """
        Parses a raw LLM response into ParsedLlmResponse.

        :param llm_response: Raw response from LLM
        :return: Parsed and validated response
        :raises ValueError: If response is invalid
        """
        try:
            # Find JSON block in response
            start = llm_response.find('{')
            end = llm_response.rfind('}') + 1

            if start == -1 or end == 0:
                raise ValueError("No JSON found in response")

            json_str = llm_response[start:end]
            data = json.loads(json_str)

            # Validate required fields
            if not all(k in data for k in ['response', 'state', 'confidence', 'reasoning']):
                raise ValueError("Missing required fields in response")

            # Validate state
            try:
                state = ConversationState[data['state']]
            except KeyError:
                raise ValueError(f"Invalid state: {data['state']}")

            # Validate confidence
            confidence = float(data['confidence'])
            if not 0 <= confidence <= 1:
                raise ValueError(f"Invalid confidence value: {confidence}")

            return ParsedLlmResponse(
                response=data['response'].strip(),
                state=state,
                confidence=confidence,
                reasoning=data['reasoning'].strip()
            )

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error parsing LLM response: {str(e)}")