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
    """Handles parsing and validation of LLM responses"""

    @staticmethod
    def parse(llm_response: str) -> ParsedLlmResponse:
        """
        Parses a raw LLM response into structured format

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